"""
BSP-based software 3D renderer for Python Doom (E1M1).

Faithfully translated from linuxdoom-1.10:
  r_main.c   - R_SetupFrame, R_PointToAngle, R_PointToDist,
               R_ScaleFromGlobalAngle, R_InitTextureMapping
  r_bsp.c    - R_RenderBSPNode, R_Subsector, R_AddLine,
               R_ClipSolidWallSegment, R_ClipPassWallSegment, R_CheckBBox
  r_segs.c   - R_StoreWallRange, R_RenderSegLoop
  r_plane.c  - R_FindPlane, R_CheckPlane, R_MakeSpans, R_DrawPlanes, R_MapPlane
  r_draw.c   - R_DrawColumn (pixel column/span drawing)
  tables.c   - finesine, finecosine, finetangent, tantoangle
"""

from __future__ import annotations

import math
import array
from dataclasses import dataclass, field
from typing import Optional, List

import pygame

from doom.defs import (
    # Constants
    FRACBITS, FRACUNIT,
    ANG45, ANG90, ANG180, ANG270, ANG360, ANGLE_MAX,
    FINEANGLES, FINEMASK, ANGLETOFINESHIFT,
    NF_SUBSECTOR, ML_TWOSIDED, ML_DONTPEGTOP, ML_DONTPEGBOTTOM,
    SIL_NONE, SIL_BOTTOM, SIL_TOP, SIL_BOTH,
    # Lookup tables
    finesine, finecosine, finetangent, tantoangle,
    # Fixed-point helpers
    fixed_mul, fixed_div,
    # Map structures
    MapData, Sector, Seg, Subsector, Node,
)

# ---------------------------------------------------------------------------
# Doom lighting constants (from r_local.h)
# ---------------------------------------------------------------------------

LIGHTLEVELS      = 16
NUMCOLORMAPS     = 32
LIGHTSEGSHIFT    = 4        # lightlevel >> 4 gives 0..15 level index
MAXLIGHTSCALE    = 48
LIGHTSCALESHIFT  = 12
MAXLIGHTZ        = 128
LIGHTZSHIFT      = 20
DISTMAP          = 2

# Slope range for SlopeDiv / R_PointToAngle
SLOPERANGE = 2048
SLOPEBITS  = 11
DBITS      = FRACBITS - SLOPEBITS

# How many fine-angle sub-divisions span a full 90-degree clip half
FIELDOFVIEW = 2048   # FINEANGLES/4 = 2048

# Sky flat special index (Doom uses this to detect sky ceiling/floor)
# We'll fill this in after TextureManager is constructed.
SKYFLATNUM = -1      # will be set in Renderer.__init__

# Maximum number of visible wall segments per frame
MAXDRAWSEGS  = 256
# Maximum number of floor/ceiling planes per frame
MAXVISPLANES = 128
# Maximum clip-range entries for solidsegs
MAXSEGS      = 32

# Bounding-box indices (m_bbox.h)
BOXTOP    = 0
BOXBOTTOM = 1
BOXLEFT   = 2
BOXRIGHT  = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClipRange:
    """Represents one entry in the solidsegs occlusion list."""
    first: int = 0
    last:  int = 0


@dataclass
class DrawSeg:
    """Per-wall segment draw record (drawseg_t)."""
    curline:          Optional[Seg] = None
    x1:               int           = 0
    x2:               int           = 0
    scale1:           int           = 0   # fixed-point
    scale2:           int           = 0
    scalestep:        int           = 0
    silhouette:       int           = SIL_NONE
    bsilheight:       int           = 0   # fixed-point
    tsilheight:       int           = 0
    sprtopclip:       Optional[list] = None
    sprbottomclip:    Optional[list] = None
    maskedtexturecol: Optional[list] = None


@dataclass
class Visplane:
    """Accumulates floor/ceiling spans for a single height/texture/light."""
    height:      int = 0   # fixed-point
    picnum:      int = 0
    light_level: int = 0
    minx:        int = 0
    maxx:        int = -1
    top:         list = field(default_factory=list)
    bottom:      list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: slope division (used by R_PointToAngle)
# ---------------------------------------------------------------------------

def _slope_div(num: int, den: int) -> int:
    """
    SlopeDiv from tables.c.
    Returns an index into tantoangle[] for atan(num/den).
    """
    if den < 512:
        return SLOPERANGE
    ans = (num << 3) // (den >> 8)
    return min(ans, SLOPERANGE)


# ---------------------------------------------------------------------------
# Renderer class
# ---------------------------------------------------------------------------

class Renderer:
    """
    BSP-based software 3D renderer matching linuxdoom-1.10 rendering pipeline.

    Usage:
        renderer = Renderer(320, 200, texture_manager, palette, colormap)
        surface  = renderer.render(px, py, pz, angle, map_data)
    """

    def __init__(
        self,
        screen_width:    int,
        screen_height:   int,
        texture_manager,
        palette:         list,   # 256 (r,g,b) tuples
        colormap:        list,   # 34 × 256-byte light maps
    ) -> None:
        self.width  = screen_width
        self.height = screen_height
        self._tm    = texture_manager
        self._pal   = palette     # index → (r, g, b)
        self._cmap  = colormap    # [light_level][palette_index] → palette_index

        # Pre-convert palette to 32-bit ARGB for pygame
        self._pal32 = [
            (255 << 24) | (r << 16) | (g << 8) | b
            for (r, g, b) in palette
        ]

        # --- Projection constants (R_ExecuteSetViewSize) ---
        self.centerx     = screen_width  // 2
        self.centery     = screen_height // 2
        self.centerxfrac = self.centerx << FRACBITS
        self.centeryfrac = self.centery << FRACBITS
        self.projection  = self.centerxfrac   # focal distance in fixed-point

        # --- Texture mapping tables (R_ExecuteSetViewSize, R_InitTextureMapping) ---
        self.yslope:    list = [0] * screen_height
        self.distscale: list = [0] * screen_width

        # viewangletox[fine_angle] → screen x
        self.viewangletox: list = [0] * (FINEANGLES // 2)
        # xtoviewangle[screen_x]  → relative view angle (BAM)
        self.xtoviewangle: list = [0] * (screen_width + 1)
        # clipangle: the half-FOV angle corresponding to screen edge x=0
        self.clipangle: int = 0

        self._init_texture_mapping()
        # NOTE: _init_light_tables() and _init_plane_tables() are called
        # later in __init__ after all attributes are declared.

        # Sky flat number – look up "F_SKY1" flat
        self._sky_flat_idx = self._tm.get_flat_index("F_SKY1")
        # Sky texture index
        self._sky_tex_idx  = self._tm.get_texture_index("SKY1")

        # --- Per-frame state (cleared at the start of each render) ---
        # Solidsegs: occlusion list for solid walls (R_ClearClipSegs)
        self._solidsegs: List[ClipRange] = [ClipRange() for _ in range(MAXSEGS + 2)]
        self._solidsegs_end: int = 0

        # Visplanes (R_ClearPlanes)
        self._visplanes:     List[Visplane] = []
        self._floor_plane:   Optional[Visplane] = None
        self._ceiling_plane: Optional[Visplane] = None

        # Per-column floor/ceiling clip bounds (R_ClearPlanes)
        self._floorclip:   array.array = array.array('h', [0] * screen_width)
        self._ceilingclip: array.array = array.array('h', [0] * screen_width)

        # Span start tracking for R_MakeSpans
        self._spanstart: array.array = array.array('i', [0] * screen_height)

        # Draw segs list
        self._drawsegs: List[DrawSeg] = []

        # Current seg being processed (global state shared across functions)
        self._curline:      Optional[Seg]    = None
        self._frontsector:  Optional[Sector] = None
        self._backsector:   Optional[Sector] = None
        self._rw_angle1:    int = 0   # global angle of seg v1 (BAM)

        # --- Frame-scope wall rendering globals (r_segs.c) ---
        self._rw_x:             int = 0
        self._rw_stopx:         int = 0
        self._rw_normalangle:   int = 0   # BAM
        self._rw_centerangle:   int = 0   # BAM
        self._rw_offset:        int = 0   # fixed-point
        self._rw_distance:      int = 0   # fixed-point
        self._rw_scale:         int = 0   # fixed-point
        self._rw_scalestep:     int = 0   # fixed-point
        self._rw_midtexturemid: int = 0   # fixed-point
        self._rw_toptexturemid: int = 0   # fixed-point
        self._rw_bottomtexturemid: int = 0  # fixed-point

        self._worldtop:    int = 0
        self._worldbottom: int = 0
        self._worldhigh:   int = 0
        self._worldlow:    int = 0

        self._pixhigh:     int = 0
        self._pixlow:      int = 0
        self._pixhighstep: int = 0
        self._pixlowstep:  int = 0
        self._topfrac:     int = 0
        self._topstep:     int = 0
        self._bottomfrac:  int = 0
        self._bottomstep:  int = 0

        self._segtextured:   bool = False
        self._markfloor:     bool = False
        self._markceiling:   bool = False
        self._maskedtexture: bool = False
        self._midtexture:    int = -1
        self._toptexture:    int = -1
        self._bottomtexture: int = -1

        self._walllights: Optional[list] = None

        # --- View state (set each frame by _setup_frame) ---
        self.viewx:     int = 0
        self.viewy:     int = 0
        self.viewz:     int = 0
        self.viewangle: int = 0
        self.viewcos:   int = 0
        self.viewsin:   int = 0

        # --- Plane drawing globals (r_plane.c) ---
        self._basexscale:    int = 0
        self._baseyscale:    int = 0
        self._planeheight:   int = 0
        self._cachedheight:  list = [0] * screen_height
        self._cacheddistance: list = [0] * screen_height
        self._cachedxstep:   list = [0] * screen_height
        self._cachedystep:   list = [0] * screen_height

        # --- Pixel framebuffer ---
        self._screen = pygame.Surface((screen_width, screen_height))
        # Use a raw pixel array for fast writes
        self._pixels: Optional[pygame.PixelArray] = None

        # Scale light tables (per-frame)
        self._scalelight: list = [[None]*MAXLIGHTSCALE for _ in range(LIGHTLEVELS)]
        self._zlight:     list = [[None]*MAXLIGHTZ     for _ in range(LIGHTLEVELS)]

        # Now that all attributes are declared, run remaining init
        self._init_light_tables()
        self._init_plane_tables()

    # -----------------------------------------------------------------------
    # One-time initialisation helpers
    # -----------------------------------------------------------------------

    def _init_texture_mapping(self) -> None:
        """
        Build viewangletox[] and xtoviewangle[] lookup tables.
        Mirrors R_InitTextureMapping in r_main.c.
        """
        w = self.width

        # Focal length: how far the projection plane is such that
        # FIELDOFVIEW fine-angles span exactly viewwidth pixels.
        focallength = fixed_div(
            self.centerxfrac,
            finetangent[FINEANGLES // 4 + FIELDOFVIEW // 2],
        )

        for i in range(FINEANGLES // 2):
            ft = finetangent[i]
            if ft > FRACUNIT * 2:
                t = -1
            elif ft < -FRACUNIT * 2:
                t = w + 1
            else:
                t = fixed_mul(ft, focallength)
                t = (self.centerxfrac - t + FRACUNIT - 1) >> FRACBITS
                t = max(-1, min(w + 1, t))
            self.viewangletox[i] = t

        # Derive xtoviewangle from viewangletox
        for x in range(w + 1):
            i = 0
            while i < FINEANGLES // 2 and self.viewangletox[i] > x:
                i += 1
            self.xtoviewangle[x] = (i << ANGLETOFINESHIFT) - ANG90

        # Fix fence-post cases
        for i in range(FINEANGLES // 2):
            if self.viewangletox[i] == -1:
                self.viewangletox[i] = 0
            elif self.viewangletox[i] == w + 1:
                self.viewangletox[i] = w

        self.clipangle = self.xtoviewangle[0]

    def _init_plane_tables(self) -> None:
        """
        Build yslope[] and distscale[] tables.
        Mirrors R_ExecuteSetViewSize planes section in r_main.c.
        """
        w = self.width
        h = self.height

        for i in range(h):
            dy = ((i - h // 2) << FRACBITS) + FRACUNIT // 2
            dy = abs(dy)
            self.yslope[i] = fixed_div((w * FRACUNIT) // 2, dy)

        for i in range(w):
            fine_idx = self.xtoviewangle[i] >> ANGLETOFINESHIFT
            fine_idx &= FINEMASK
            cosadj = abs(finecosine[fine_idx])
            if cosadj == 0:
                cosadj = 1
            self.distscale[i] = fixed_div(FRACUNIT, cosadj)

    def _init_light_tables(self) -> None:
        """
        Build zlight[LIGHTLEVELS][MAXLIGHTZ] tables.
        scalelight is rebuilt every frame inside _setup_frame based on the
        current view width (not needed for this fixed-resolution port, but
        we build it once here too).
        Mirrors R_InitLightTables + R_ExecuteSetViewSize light section.
        """
        for i in range(LIGHTLEVELS):
            startmap = ((LIGHTLEVELS - 1 - i) * 2) * NUMCOLORMAPS // LIGHTLEVELS
            for j in range(MAXLIGHTZ):
                scale = fixed_div(
                    (self.width // 2 * FRACUNIT),
                    (j + 1) << LIGHTZSHIFT,
                )
                scale >>= LIGHTSCALESHIFT
                level = startmap - scale // DISTMAP
                level = max(0, min(NUMCOLORMAPS - 1, level))
                self._zlight[i][j] = self._cmap[level]

        for i in range(LIGHTLEVELS):
            startmap = ((LIGHTLEVELS - 1 - i) * 2) * NUMCOLORMAPS // LIGHTLEVELS
            for j in range(MAXLIGHTSCALE):
                level = startmap - j * self.width // self.width // DISTMAP
                level = max(0, min(NUMCOLORMAPS - 1, level))
                self._scalelight[i][j] = self._cmap[level]

    # -----------------------------------------------------------------------
    # Public render entry point
    # -----------------------------------------------------------------------

    def render(
        self,
        player_x:     int,  # fixed-point map units
        player_y:     int,
        player_z:     int,  # eye height above floor (fixed-point)
        player_angle: int,  # BAM angle
        map_data:     MapData,
    ) -> pygame.Surface:
        """
        Render one frame from the player's viewpoint and return a Surface.
        """
        self._map = map_data

        # 1 – Setup frame (R_SetupFrame)
        self._setup_frame(player_x, player_y, player_z, player_angle)

        # 2 – Clear per-frame buffers
        self._clear_clip_segs()
        self._clear_planes()
        self._drawsegs = []

        # 3 – Clear screen to black
        self._screen.fill((0, 0, 0))

        # Open pixel array for direct writes
        self._pixels = pygame.PixelArray(self._screen)

        # 4 – BSP traversal (R_RenderBSPNode)
        if map_data.nodes:
            self._render_bsp_node(len(map_data.nodes) - 1)
        elif map_data.subsectors:
            self._render_subsector(0)

        # 5 – Draw floor/ceiling planes (R_DrawPlanes)
        self._draw_planes()

        # Release pixel array
        del self._pixels
        self._pixels = None

        return self._screen

    # -----------------------------------------------------------------------
    # Frame setup (R_SetupFrame)
    # -----------------------------------------------------------------------

    def _setup_frame(self, x: int, y: int, z: int, angle: int) -> None:
        self.viewx     = x
        self.viewy     = y
        self.viewz     = z
        self.viewangle = angle & ANGLE_MAX

        fine = (angle >> ANGLETOFINESHIFT) & FINEMASK
        self.viewsin = finesine[fine]
        self.viewcos = finecosine[fine]

        # Rebuild per-frame scale light tables
        for i in range(LIGHTLEVELS):
            startmap = ((LIGHTLEVELS - 1 - i) * 2) * NUMCOLORMAPS // LIGHTLEVELS
            for j in range(MAXLIGHTSCALE):
                level = startmap - j * self.width // self.width // DISTMAP
                level = max(0, min(NUMCOLORMAPS - 1, level))
                self._scalelight[i][j] = self._cmap[level]

        # basexscale / baseyscale for plane mapping (R_ClearPlanes sets these)
        # They depend on viewangle so we compute here too.

    # -----------------------------------------------------------------------
    # Solidsegs (R_ClearClipSegs / R_ClipSolidWallSegment / R_ClipPassWallSegment)
    # -----------------------------------------------------------------------

    def _clear_clip_segs(self) -> None:
        """R_ClearClipSegs: reset solidsegs to sentinel entries."""
        self._solidsegs[0].first = -0x7FFFFFFF
        self._solidsegs[0].last  = -1
        self._solidsegs[1].first = self.width
        self._solidsegs[1].last  = 0x7FFFFFFF
        self._solidsegs_end = 2  # index of one-past the last valid entry

    def _all_occluded(self) -> bool:
        """Return True when the entire screen is covered by solid walls."""
        return (self._solidsegs_end == 2
                and self._solidsegs[0].last >= self.width - 1)

    def _clip_solid_wall_segment(self, first: int, last: int) -> None:
        """
        R_ClipSolidWallSegment.
        Integrates [first,last] into the sorted solidsegs list,
        calling _store_wall_range for each newly-visible sub-span.
        """
        segs = self._solidsegs
        end  = self._solidsegs_end

        # Find the first range that touches [first, last]
        start_idx = 0
        while segs[start_idx].last < first - 1:
            start_idx += 1

        if first < segs[start_idx].first:
            if last < segs[start_idx].first - 1:
                # Entirely visible – insert a new clippost
                self._store_wall_range(first, last)
                # Insert at start_idx, shift entries up
                for k in range(end, start_idx, -1):
                    segs[k].first = segs[k - 1].first
                    segs[k].last  = segs[k - 1].last
                segs[start_idx].first = first
                segs[start_idx].last  = last
                self._solidsegs_end = end + 1
                return

            # Fragment above start
            self._store_wall_range(first, segs[start_idx].first - 1)
            segs[start_idx].first = first

        # Bottom contained in start?
        if last <= segs[start_idx].last:
            return

        next_idx = start_idx
        while last >= segs[next_idx + 1].first - 1:
            self._store_wall_range(segs[next_idx].last + 1,
                                   segs[next_idx + 1].first - 1)
            next_idx += 1
            if last <= segs[next_idx].last:
                segs[start_idx].last = segs[next_idx].last
                # Crunch
                if next_idx != start_idx:
                    count = end - next_idx - 1
                    for k in range(count):
                        segs[start_idx + 1 + k].first = segs[next_idx + 1 + k].first
                        segs[start_idx + 1 + k].last  = segs[next_idx + 1 + k].last
                    self._solidsegs_end = start_idx + 1 + count
                return

        # Fragment after *next
        self._store_wall_range(segs[next_idx].last + 1, last)
        segs[start_idx].last = last

        # Crunch
        if next_idx != start_idx:
            count = end - next_idx - 1
            for k in range(count):
                segs[start_idx + 1 + k].first = segs[next_idx + 1 + k].first
                segs[start_idx + 1 + k].last  = segs[next_idx + 1 + k].last
            self._solidsegs_end = start_idx + 1 + count

    def _clip_pass_wall_segment(self, first: int, last: int) -> None:
        """
        R_ClipPassWallSegment.
        Calls _store_wall_range for visible portions without modifying solidsegs.
        """
        segs    = self._solidsegs
        start_i = 0
        while segs[start_i].last < first - 1:
            start_i += 1

        if first < segs[start_i].first:
            if last < segs[start_i].first - 1:
                self._store_wall_range(first, last)
                return
            self._store_wall_range(first, segs[start_i].first - 1)

        if last <= segs[start_i].last:
            return

        while last >= segs[start_i + 1].first - 1:
            self._store_wall_range(segs[start_i].last + 1,
                                   segs[start_i + 1].first - 1)
            start_i += 1
            if last <= segs[start_i].last:
                return

        self._store_wall_range(segs[start_i].last + 1, last)

    # -----------------------------------------------------------------------
    # R_PointToAngle – view-relative angle to a world point
    # -----------------------------------------------------------------------

    def _point_to_angle(self, x: int, y: int) -> int:
        """
        R_PointToAngle from r_main.c.
        Returns BAM angle from (viewx,viewy) to (x,y).
        All values fixed-point.
        """
        x -= self.viewx
        y -= self.viewy

        if x == 0 and y == 0:
            return 0

        if x >= 0:
            if y >= 0:
                if x > y:
                    return tantoangle[_slope_div(y, x)]
                else:
                    return ANG90 - 1 - tantoangle[_slope_div(x, y)]
            else:
                y = -y
                if x > y:
                    return -tantoangle[_slope_div(y, x)]
                else:
                    return ANG270 + tantoangle[_slope_div(x, y)]
        else:
            x = -x
            if y >= 0:
                if x > y:
                    return ANG180 - 1 - tantoangle[_slope_div(y, x)]
                else:
                    return ANG90 + tantoangle[_slope_div(x, y)]
            else:
                y = -y
                if x > y:
                    return ANG180 + tantoangle[_slope_div(y, x)]
                else:
                    return ANG270 - 1 - tantoangle[_slope_div(x, y)]

    # -----------------------------------------------------------------------
    # R_PointToDist
    # -----------------------------------------------------------------------

    def _point_to_dist(self, x: int, y: int) -> int:
        """
        R_PointToDist from r_main.c.
        Returns fixed-point distance from view to (x,y).
        """
        dx = abs(x - self.viewx)
        dy = abs(y - self.viewy)

        if dy > dx:
            dx, dy = dy, dx

        if dx == 0:
            return 1

        ratio = fixed_div(dy, dx)
        angle_idx = (tantoangle[ratio >> DBITS] + ANG90) >> ANGLETOFINESHIFT
        angle_idx &= FINEMASK
        sine = finesine[angle_idx]
        if sine == 0:
            return dx
        return fixed_div(dx, sine)

    # -----------------------------------------------------------------------
    # R_ScaleFromGlobalAngle
    # -----------------------------------------------------------------------

    def _scale_from_global_angle(self, visangle: int) -> int:
        """
        R_ScaleFromGlobalAngle from r_main.c.
        Returns wall column scale factor (fixed-point).
        Requires self._rw_normalangle and self._rw_distance to be set.
        """
        anglea = (ANG90 + (visangle - self.viewangle)) & ANGLE_MAX
        angleb = (ANG90 + (visangle - self._rw_normalangle)) & ANGLE_MAX

        sinea = finesine[(anglea >> ANGLETOFINESHIFT) & FINEMASK]
        sineb = finesine[(angleb >> ANGLETOFINESHIFT) & FINEMASK]

        # Both sines are always positive for visible walls
        sinea = max(1, sinea)
        sineb = max(1, sineb)

        num = fixed_mul(self.projection, sineb)
        den = fixed_mul(self._rw_distance, sinea)

        if den == 0:
            return 64 * FRACUNIT

        if den > (num >> 16):
            scale = fixed_div(num, den)
            scale = max(256, min(64 * FRACUNIT, scale))
        else:
            scale = 64 * FRACUNIT

        return scale

    # -----------------------------------------------------------------------
    # R_PointOnSide (for BSP traversal)
    # -----------------------------------------------------------------------

    def _point_on_side(self, x: int, y: int, node: Node) -> int:
        """
        R_PointOnSide from r_main.c.
        Returns 0 (front/same side as segment) or 1 (back).
        """
        if node.dx == 0:
            if x <= node.x:
                return 1 if node.dy > 0 else 0
            return 0 if node.dy > 0 else 1

        if node.dy == 0:
            if y <= node.y:
                return 0 if node.dx < 0 else 1
            return 1 if node.dx < 0 else 0

        dx = x - node.x
        dy = y - node.y

        left  = fixed_mul(node.dy >> FRACBITS, dx)
        right = fixed_mul(dy, node.dx >> FRACBITS)

        return 0 if right < left else 1

    # -----------------------------------------------------------------------
    # R_CheckBBox
    # -----------------------------------------------------------------------

    # checkcoord table from r_bsp.c
    _checkcoord = [
        [3, 0, 2, 1],
        [3, 0, 2, 0],
        [3, 1, 2, 0],
        [0, 0, 0, 0],
        [2, 0, 2, 1],
        [0, 0, 0, 0],
        [3, 1, 3, 0],
        [0, 0, 0, 0],
        [2, 0, 3, 1],
        [2, 1, 3, 1],
        [2, 1, 3, 0],
    ]

    def _check_bbox(self, bbox: list) -> bool:
        """
        R_CheckBBox from r_bsp.c.
        bbox is [top, bottom, left, right] in fixed-point.
        Returns True if any part of bbox is potentially visible.
        """
        boxtop    = bbox[BOXTOP]
        boxbottom = bbox[BOXBOTTOM]
        boxleft   = bbox[BOXLEFT]
        boxright  = bbox[BOXRIGHT]

        # Determine which corner pair to use
        if self.viewx <= boxleft:
            boxx = 0
        elif self.viewx < boxright:
            boxx = 1
        else:
            boxx = 2

        if self.viewy >= boxtop:
            boxy = 0
        elif self.viewy > boxbottom:
            boxy = 1
        else:
            boxy = 2

        boxpos = (boxy << 2) + boxx
        if boxpos == 5:
            return True

        cc = self._checkcoord[boxpos]
        coords = [boxtop, boxbottom, boxleft, boxright]
        x1 = coords[cc[0]]
        y1 = coords[cc[1]]
        x2 = coords[cc[2]]
        y2 = coords[cc[3]]

        angle1 = (self._point_to_angle(x1, y1) - self.viewangle) & ANGLE_MAX
        angle2 = (self._point_to_angle(x2, y2) - self.viewangle) & ANGLE_MAX

        span = (angle1 - angle2) & ANGLE_MAX
        if span >= ANG180:
            return True

        tspan = (angle1 + self.clipangle) & ANGLE_MAX
        if tspan > 2 * self.clipangle:
            tspan = (tspan - 2 * self.clipangle) & ANGLE_MAX
            if tspan >= span:
                return False
            angle1 = self.clipangle

        tspan = (self.clipangle - angle2) & ANGLE_MAX
        if tspan > 2 * self.clipangle:
            tspan = (tspan - 2 * self.clipangle) & ANGLE_MAX
            if tspan >= span:
                return False
            angle2 = (-self.clipangle) & ANGLE_MAX

        fine1 = ((angle1 + ANG90) >> ANGLETOFINESHIFT) & (FINEANGLES // 2 - 1)
        fine2 = ((angle2 + ANG90) >> ANGLETOFINESHIFT) & (FINEANGLES // 2 - 1)
        sx1 = self.viewangletox[fine1]
        sx2 = self.viewangletox[fine2]

        if sx1 == sx2:
            return False
        sx2 -= 1

        segs    = self._solidsegs
        start_i = 0
        while segs[start_i].last < sx2:
            start_i += 1

        if sx1 >= segs[start_i].first and sx2 <= segs[start_i].last:
            return False

        return True

    # -----------------------------------------------------------------------
    # BSP traversal (R_RenderBSPNode)
    # -----------------------------------------------------------------------

    def _render_bsp_node(self, bspnum: int) -> None:
        """R_RenderBSPNode from r_bsp.c."""
        nodes = self._map.nodes

        # Iterative BSP traversal to avoid Python recursion limits
        stack = [bspnum]
        while stack:
            num = stack.pop()

            if num & NF_SUBSECTOR:
                ss = 0 if num == 0xFFFF else (num & ~NF_SUBSECTOR)
                self._render_subsector(ss)
                continue

            if num < 0 or num >= len(nodes):
                continue

            bsp  = nodes[num]
            side = self._point_on_side(self.viewx, self.viewy, bsp)

            # Push back side first (processed later), then front side
            back_child  = bsp.children[side ^ 1]
            front_child = bsp.children[side]

            # Only push back if potentially visible
            if self._check_bbox(bsp.bbox[side ^ 1]):
                stack.append(back_child)

            stack.append(front_child)

    # -----------------------------------------------------------------------
    # R_Subsector
    # -----------------------------------------------------------------------

    def _render_subsector(self, num: int) -> None:
        """R_Subsector from r_bsp.c."""
        subsectors = self._map.subsectors
        segs       = self._map.segs

        if num >= len(subsectors):
            return

        sub = subsectors[num]
        self._frontsector = sub.sector
        if self._frontsector is None:
            return

        fs = self._frontsector

        # Determine floor/ceiling visplanes
        if fs.floor_height < self.viewz:
            self._floor_plane = self._find_plane(
                fs.floor_height, fs.floor_pic, fs.light_level
            )
        else:
            self._floor_plane = None

        if (fs.ceiling_height > self.viewz
                or fs.ceiling_pic == self._sky_flat_idx):
            self._ceiling_plane = self._find_plane(
                fs.ceiling_height, fs.ceiling_pic, fs.light_level
            )
        else:
            self._ceiling_plane = None

        # Process segs
        first = sub.first_line
        count = sub.num_lines
        for i in range(count):
            seg_idx = first + i
            if seg_idx < len(segs):
                self._add_line(segs[seg_idx])

    # -----------------------------------------------------------------------
    # R_AddLine – clip and queue a seg for drawing
    # -----------------------------------------------------------------------

    def _add_line(self, line: Seg) -> None:
        """R_AddLine from r_bsp.c."""
        self._curline = line

        angle1 = self._point_to_angle(line.v1.x, line.v1.y)
        angle2 = self._point_to_angle(line.v2.x, line.v2.y)

        # Backface cull: span from v2→v1 must be < 180°
        span = (angle1 - angle2) & ANGLE_MAX
        if span >= ANG180:
            return

        # Save global angle for R_StoreWallRange
        self._rw_angle1 = angle1

        # Make angles relative to viewangle
        angle1 = (angle1 - self.viewangle) & ANGLE_MAX
        angle2 = (angle2 - self.viewangle) & ANGLE_MAX

        clipangle = self.clipangle

        # Clip left edge
        tspan = (angle1 + clipangle) & ANGLE_MAX
        if tspan > 2 * clipangle:
            tspan = (tspan - 2 * clipangle) & ANGLE_MAX
            if tspan >= span:
                return
            angle1 = clipangle

        # Clip right edge
        tspan = (clipangle - angle2) & ANGLE_MAX
        if tspan > 2 * clipangle:
            tspan = (tspan - 2 * clipangle) & ANGLE_MAX
            if tspan >= span:
                return
            angle2 = (-clipangle) & ANGLE_MAX

        # Map to screen x coordinates
        fine1 = ((angle1 + ANG90) >> ANGLETOFINESHIFT) & (FINEANGLES // 2 - 1)
        fine2 = ((angle2 + ANG90) >> ANGLETOFINESHIFT) & (FINEANGLES // 2 - 1)
        x1 = self.viewangletox[fine1]
        x2 = self.viewangletox[fine2]

        if x1 == x2:
            return

        self._backsector = line.back_sector
        fs = self._frontsector
        bs = self._backsector

        # Decide: solid wall or pass-through window?
        if bs is None:
            # Single-sided line (solid)
            self._clip_solid_wall_segment(x1, x2 - 1)
            return

        # Closed door – acts like solid
        if (bs.ceiling_height <= fs.floor_height
                or bs.floor_height >= fs.ceiling_height):
            self._clip_solid_wall_segment(x1, x2 - 1)
            return

        # Window (two-sided with height differences)
        if (bs.ceiling_height != fs.ceiling_height
                or bs.floor_height != fs.floor_height):
            self._clip_pass_wall_segment(x1, x2 - 1)
            return

        # Reject invisible two-sided lines (triggers etc.)
        sd = line.sidedef
        mid_tex = (sd.mid_texture if sd is not None else -1)
        if (bs.ceiling_pic == fs.ceiling_pic
                and bs.floor_pic == fs.floor_pic
                and bs.light_level == fs.light_level
                and (mid_tex is None or mid_tex < 0)):
            return

        self._clip_pass_wall_segment(x1, x2 - 1)

    # -----------------------------------------------------------------------
    # R_StoreWallRange – compute and draw a visible wall span
    # -----------------------------------------------------------------------

    def _store_wall_range(self, start: int, stop: int) -> None:
        """
        R_StoreWallRange from r_segs.c.
        Called by the clip routines for each visible wall span [start,stop].
        """
        if start > stop:
            return

        line = self._curline
        if line is None:
            return

        sd  = line.sidedef
        ld  = line.linedef
        fs  = self._frontsector
        bs  = self._backsector

        # --- Compute rw_distance ---
        self._rw_normalangle = (line.angle + ANG90) & ANGLE_MAX
        offsetangle = abs(
            ((self._rw_normalangle - self._rw_angle1) & ANGLE_MAX)
        )
        if offsetangle > ANG90:
            offsetangle = ANG90

        distangle = (ANG90 - offsetangle) & ANGLE_MAX
        hyp = self._point_to_dist(line.v1.x, line.v1.y)
        sineval = finesine[(distangle >> ANGLETOFINESHIFT) & FINEMASK]
        self._rw_distance = fixed_mul(hyp, sineval)

        if self._rw_distance < 1:
            self._rw_distance = 1

        # --- Screen x range ---
        self._rw_x    = start
        self._rw_stopx = stop + 1

        # --- Scale at both ends ---
        self._rw_scale = self._scale_from_global_angle(
            self.viewangle + self.xtoviewangle[start]
        )
        rw_scale1 = self._rw_scale

        if stop > start:
            scale2 = self._scale_from_global_angle(
                self.viewangle + self.xtoviewangle[stop]
            )
            self._rw_scalestep = (scale2 - rw_scale1) // (stop - start)
        else:
            scale2 = rw_scale1
            self._rw_scalestep = 0

        # --- World heights (relative to view) ---
        worldtop    = fs.ceiling_height - self.viewz
        worldbottom = fs.floor_height   - self.viewz

        # --- Texture selection ---
        self._midtexture    = -1
        self._toptexture    = -1
        self._bottomtexture = -1
        self._maskedtexture = False

        flags = ld.flags if ld is not None else 0

        if bs is None:
            # Single-sided: middle texture fills the whole wall
            mid_tex_idx = (sd.mid_texture if sd is not None else -1)
            if mid_tex_idx is not None and mid_tex_idx >= 0:
                self._midtexture = mid_tex_idx
                if flags & ML_DONTPEGBOTTOM:
                    tex_h = self._tm.texture_height(mid_tex_idx) * FRACUNIT
                    vtop  = fs.floor_height + tex_h
                    self._rw_midtexturemid = vtop - self.viewz
                else:
                    self._rw_midtexturemid = worldtop
                row_off = sd.row_offset if sd is not None else 0
                self._rw_midtexturemid += row_off
            self._markfloor   = True
            self._markceiling = True
        else:
            worldhigh = bs.ceiling_height - self.viewz
            worldlow  = bs.floor_height   - self.viewz

            # Sky hack: both sides sky → world top matches
            if (fs.ceiling_pic == self._sky_flat_idx
                    and bs.ceiling_pic == self._sky_flat_idx):
                worldtop = worldhigh

            # Determine floor/ceiling mark flags
            self._markfloor = (
                worldlow != worldbottom
                or bs.floor_pic   != fs.floor_pic
                or bs.light_level != fs.light_level
            )
            self._markceiling = (
                worldhigh != worldtop
                or bs.ceiling_pic  != fs.ceiling_pic
                or bs.light_level  != fs.light_level
            )

            # Closed door forces both marks
            if (bs.ceiling_height <= fs.floor_height
                    or bs.floor_height >= fs.ceiling_height):
                self._markceiling = True
                self._markfloor   = True

            # Top texture
            if worldhigh < worldtop:
                top_idx = (sd.top_texture if sd is not None else -1)
                if top_idx is not None and top_idx >= 0:
                    self._toptexture = top_idx
                    if flags & ML_DONTPEGTOP:
                        self._rw_toptexturemid = worldtop
                    else:
                        tex_h = self._tm.texture_height(top_idx) * FRACUNIT
                        vtop  = bs.ceiling_height + tex_h
                        self._rw_toptexturemid = vtop - self.viewz
                    row_off = sd.row_offset if sd is not None else 0
                    self._rw_toptexturemid += row_off

            # Bottom texture
            if worldlow > worldbottom:
                bot_idx = (sd.bottom_texture if sd is not None else -1)
                if bot_idx is not None and bot_idx >= 0:
                    self._bottomtexture = bot_idx
                    if flags & ML_DONTPEGBOTTOM:
                        self._rw_bottomtexturemid = worldtop
                    else:
                        self._rw_bottomtexturemid = worldlow
                    row_off = sd.row_offset if sd is not None else 0
                    self._rw_bottomtexturemid += row_off

            self._worldhigh = worldhigh
            self._worldlow  = worldlow

        # --- rw_offset and light ---
        segtextured = ((self._midtexture    >= 0)
                    or (self._toptexture    >= 0)
                    or (self._bottomtexture >= 0)
                    or self._maskedtexture)
        self._segtextured = segtextured

        if segtextured:
            offsetangle = (self._rw_normalangle - self._rw_angle1) & ANGLE_MAX
            if offsetangle > ANG180:
                offsetangle = (-offsetangle) & ANGLE_MAX
            if offsetangle > ANG90:
                offsetangle = ANG90

            sineval = finesine[(offsetangle >> ANGLETOFINESHIFT) & FINEMASK]
            rw_offset = fixed_mul(hyp, sineval)

            if ((self._rw_normalangle - self._rw_angle1) & ANGLE_MAX) < ANG180:
                rw_offset = -rw_offset

            tex_off = (sd.texture_offset if sd is not None else 0)
            seg_off = line.offset
            self._rw_offset = rw_offset + tex_off + seg_off
            self._rw_centerangle = (ANG90 + self.viewangle - self._rw_normalangle) & ANGLE_MAX

            # Choose wall light table
            lightnum = (fs.light_level >> LIGHTSEGSHIFT)
            # Horizontal/vertical wall brightness tweak
            if line.v1.y == line.v2.y:
                lightnum -= 1
            elif line.v1.x == line.v2.x:
                lightnum += 1
            lightnum = max(0, min(LIGHTLEVELS - 1, lightnum))
            self._walllights = self._scalelight[lightnum]

        # --- Visibility checks for planes ---
        if fs.floor_height >= self.viewz:
            self._markfloor = False
        if (fs.ceiling_height <= self.viewz
                and fs.ceiling_pic != self._sky_flat_idx):
            self._markceiling = False

        # --- Incremental step values (r_segs.c uses >>4 for sub-pixel) ---
        HEIGHTBITS = 12
        HEIGHTUNIT = 1 << HEIGHTBITS

        wt = worldtop    >> 4
        wb = worldbottom >> 4

        self._topstep    = -fixed_mul(self._rw_scalestep, wt)
        self._topfrac    = (self.centeryfrac >> 4) - fixed_mul(wt, self._rw_scale)
        self._bottomstep = -fixed_mul(self._rw_scalestep, wb)
        self._bottomfrac = (self.centeryfrac >> 4) - fixed_mul(wb, self._rw_scale)

        if bs is not None:
            wh = self._worldhigh >> 4
            wl = self._worldlow  >> 4

            if wh < wt:
                self._pixhigh     = (self.centeryfrac >> 4) - fixed_mul(wh, self._rw_scale)
                self._pixhighstep = -fixed_mul(self._rw_scalestep, wh)
            if wl > wb:
                self._pixlow     = (self.centeryfrac >> 4) - fixed_mul(wl, self._rw_scale)
                self._pixlowstep = -fixed_mul(self._rw_scalestep, wl)

        # Extend visplanes to cover this column range
        if self._markceiling and self._ceiling_plane is not None:
            self._ceiling_plane = self._check_plane(
                self._ceiling_plane, start, stop
            )
        if self._markfloor and self._floor_plane is not None:
            self._floor_plane = self._check_plane(
                self._floor_plane, start, stop
            )

        # --- Actually draw the wall columns ---
        self._render_seg_loop()

    # -----------------------------------------------------------------------
    # R_RenderSegLoop – draw each column of a wall segment
    # -----------------------------------------------------------------------

    def _render_seg_loop(self) -> None:
        """
        R_RenderSegLoop from r_segs.c.
        Draws textured wall columns column by column from rw_x to rw_stopx-1.
        Also records floor/ceiling bounds in the visplanes.
        """
        HEIGHTBITS = 12
        HEIGHTUNIT = 1 << HEIGHTBITS

        rw_x          = self._rw_x
        rw_stopx      = self._rw_stopx
        floorclip      = self._floorclip
        ceilingclip    = self._ceilingclip
        height         = self.height
        centery        = self.centery

        topfrac        = self._topfrac
        topstep        = self._topstep
        bottomfrac     = self._bottomfrac
        bottomstep     = self._bottomstep
        pixhigh        = self._pixhigh
        pixlow         = self._pixlow
        pixhighstep    = self._pixhighstep
        pixlowstep     = self._pixlowstep
        rw_scale       = self._rw_scale
        rw_scalestep   = self._rw_scalestep
        rw_centerangle = self._rw_centerangle
        rw_offset      = self._rw_offset
        rw_distance    = self._rw_distance
        markceiling    = self._markceiling
        markfloor      = self._markfloor
        segtextured    = self._segtextured
        midtexture_idx = self._midtexture
        toptexture_idx = self._toptexture
        bottexture_idx = self._bottomtexture
        walllights     = self._walllights
        floor_plane    = self._floor_plane
        ceiling_plane  = self._ceiling_plane

        has_mid = midtexture_idx >= 0
        has_top = toptexture_idx >= 0
        has_bot = bottexture_idx >= 0

        # Fetch texture column data
        if has_mid:
            mid_w, mid_h, mid_cols = self._tm.get_texture(midtexture_idx)
        if has_top:
            top_w, top_h, top_cols = self._tm.get_texture(toptexture_idx)
        if has_bot:
            bot_w, bot_h, bot_cols = self._tm.get_texture(bottexture_idx)

        for x in range(rw_x, rw_stopx):
            # --- Compute vertical bounds ---
            yl = (topfrac + HEIGHTUNIT - 1) >> HEIGHTBITS
            fl_clip = floorclip[x]
            cl_clip = ceilingclip[x]
            if yl < cl_clip + 1:
                yl = cl_clip + 1

            # Record ceiling span
            if markceiling and ceiling_plane is not None:
                ctop    = cl_clip + 1
                cbottom = yl - 1
                if cbottom >= fl_clip:
                    cbottom = fl_clip - 1
                if ctop <= cbottom:
                    ceiling_plane.top[x]    = ctop
                    ceiling_plane.bottom[x] = cbottom

            yh = bottomfrac >> HEIGHTBITS
            if yh >= fl_clip:
                yh = fl_clip - 1

            # Record floor span
            if markfloor and floor_plane is not None:
                ftop    = yh + 1
                fbottom = fl_clip - 1
                if ftop <= cl_clip:
                    ftop = cl_clip + 1
                if ftop <= fbottom:
                    floor_plane.top[x]    = ftop
                    floor_plane.bottom[x] = fbottom

            # --- Texture column ---
            if segtextured:
                fine_angle = ((rw_centerangle + self.xtoviewangle[x]) >> ANGLETOFINESHIFT) & FINEMASK
                texturecolumn = rw_offset - fixed_mul(finetangent[fine_angle], rw_distance)
                texturecolumn >>= FRACBITS

                # Lighting: scale → light index
                if walllights is not None:
                    light_idx = rw_scale >> LIGHTSCALESHIFT
                    light_idx = max(0, min(MAXLIGHTSCALE - 1, light_idx))
                    colormap = walllights[light_idx]
                else:
                    colormap = self._cmap[0]

                dc_iscale = 0xFFFFFFFF // max(1, rw_scale)

            # --- Draw wall tiers ---
            if has_mid:
                # Single-sided middle wall
                if yl <= yh:
                    self._draw_column(
                        x, yl, yh,
                        mid_cols, texturecolumn % mid_w, mid_h,
                        self._rw_midtexturemid, dc_iscale, colormap,
                    )
                ceilingclip[x] = height
                floorclip[x]   = -1
            else:
                # Two-sided: top wall
                if has_top:
                    mid = pixhigh >> HEIGHTBITS
                    pixhigh += pixhighstep
                    if mid >= fl_clip:
                        mid = fl_clip - 1
                    if mid >= yl:
                        self._draw_column(
                            x, yl, mid,
                            top_cols, texturecolumn % top_w, top_h,
                            self._rw_toptexturemid, dc_iscale, colormap,
                        )
                        ceilingclip[x] = mid
                    else:
                        ceilingclip[x] = yl - 1
                else:
                    if markceiling:
                        ceilingclip[x] = yl - 1

                # Two-sided: bottom wall
                if has_bot:
                    mid = (pixlow + HEIGHTUNIT - 1) >> HEIGHTBITS
                    pixlow += pixlowstep
                    if mid <= cl_clip:
                        mid = cl_clip + 1
                    if mid <= yh:
                        self._draw_column(
                            x, mid, yh,
                            bot_cols, texturecolumn % bot_w, bot_h,
                            self._rw_bottomtexturemid, dc_iscale, colormap,
                        )
                        floorclip[x] = mid
                    else:
                        floorclip[x] = yh + 1
                else:
                    if markfloor:
                        floorclip[x] = yh + 1

            rw_scale   += rw_scalestep
            topfrac    += topstep
            bottomfrac += bottomstep

        # Write back mutated locals
        self._topfrac    = topfrac
        self._bottomfrac = bottomfrac
        self._pixhigh    = pixhigh
        self._pixlow     = pixlow
        self._rw_scale   = rw_scale

    # -----------------------------------------------------------------------
    # _draw_column – R_DrawColumn equivalent (palette-indexed textures)
    # -----------------------------------------------------------------------

    def _draw_column(
        self,
        x:           int,
        y1:          int,
        y2:          int,
        columns:     list,   # list of (y_start, pixels) posts per column
        tex_col:     int,    # texture column index (already wrapped)
        tex_height:  int,
        texturemid:  int,    # fixed-point: where the texture origin sits
        dc_iscale:   int,    # fixed-point: pixels-per-texel step
        colormap:    bytearray,
    ) -> None:
        """
        R_DrawColumn: draw a vertical texture-mapped column to the framebuffer.
        """
        if y1 > y2 or x < 0 or x >= self.width:
            return
        y1 = max(0, y1)
        y2 = min(self.height - 1, y2)
        if y1 > y2:
            return

        # Get the column's pixel data
        if tex_col < 0 or tex_col >= len(columns):
            tex_col = tex_col % len(columns) if columns else 0

        col_posts = columns[tex_col] if tex_col < len(columns) else [(0, b'')]

        # Compose a flat bytearray for the full column height
        # (posts are RLE; we need random access)
        col_pix = bytearray(tex_height)
        for (y_start, pixels) in col_posts:
            end = y_start + len(pixels)
            for j, p in enumerate(pixels):
                dest_y = y_start + j
                if 0 <= dest_y < tex_height:
                    col_pix[dest_y] = p

        # Compute texture v coordinate at first screen pixel y1
        # frac = texturemid + (y1 - centery) * dc_iscale   (from R_DrawColumn)
        frac = texturemid + (y1 - self.centery) * dc_iscale

        pix_arr = self._pixels
        pal32   = self._pal32
        mask    = tex_height - 1  # works for power-of-two heights

        # Fallback for non-power-of-two heights
        if tex_height > 0 and (tex_height & (tex_height - 1)) != 0:
            for y in range(y1, y2 + 1):
                texel = (frac >> FRACBITS) % tex_height
                palette_idx = colormap[col_pix[texel]]
                pix_arr[x, y] = pal32[palette_idx]
                frac += dc_iscale
        else:
            # Power-of-two: use bitwise AND for wrap (fast path)
            for y in range(y1, y2 + 1):
                texel = (frac >> FRACBITS) & mask
                palette_idx = colormap[col_pix[texel]]
                pix_arr[x, y] = pal32[palette_idx]
                frac += dc_iscale

    # -----------------------------------------------------------------------
    # Visplane management (r_plane.c)
    # -----------------------------------------------------------------------

    def _clear_planes(self) -> None:
        """R_ClearPlanes: reset all per-frame plane state."""
        w = self.width
        h = self.height

        for i in range(w):
            self._floorclip[i]   = h
            self._ceilingclip[i] = -1

        self._visplanes     = []
        self._floor_plane   = None
        self._ceiling_plane = None

        for i in range(h):
            self._cachedheight[i] = 0

        # basexscale / baseyscale: left-to-right flat mapping direction
        # angle = (viewangle - ANG90) >> ANGLETOFINESHIFT
        angle = ((self.viewangle - ANG90) >> ANGLETOFINESHIFT) & FINEMASK
        cxf = self.centerxfrac
        if cxf == 0:
            cxf = 1
        self._basexscale =  fixed_div(finecosine[angle], cxf)
        self._baseyscale = -fixed_div(finesine[angle],   cxf)

    def _find_plane(self, height: int, picnum: int, light_level: int) -> Visplane:
        """
        R_FindPlane: find or allocate a visplane with the given properties.
        """
        # Sky: normalise so all sky areas share one plane
        if picnum == self._sky_flat_idx:
            height     = 0
            light_level = 0

        for vp in self._visplanes:
            if (vp.height == height
                    and vp.picnum == picnum
                    and vp.light_level == light_level):
                return vp

        if len(self._visplanes) >= MAXVISPLANES:
            # Reuse last (shouldn't happen in E1M1)
            return self._visplanes[-1]

        vp = Visplane(
            height      = height,
            picnum      = picnum,
            light_level = light_level,
            minx        = self.width,
            maxx        = -1,
            top         = [0xFF] * (self.width + 2),
            bottom      = [0x00] * (self.width + 2),
        )
        self._visplanes.append(vp)
        return vp

    def _check_plane(self, pl: Visplane, start: int, stop: int) -> Visplane:
        """
        R_CheckPlane: extend pl to cover [start,stop], or split if overlapping.
        """
        # Determine intersection and union
        if start < pl.minx:
            intrl  = pl.minx
            unionl = start
        else:
            unionl = pl.minx
            intrl  = start

        if stop > pl.maxx:
            intrh  = pl.maxx
            unionh = stop
        else:
            unionh = pl.maxx
            intrh  = stop

        # Check if there's any already-filled slot in the intersection range
        for x in range(intrl, intrh + 1):
            if pl.top[x] != 0xFF:
                break
        else:
            # No conflict – extend the existing plane
            pl.minx = unionl
            pl.maxx = unionh
            return pl

        # Conflict – create a new visplane
        new_pl = Visplane(
            height      = pl.height,
            picnum      = pl.picnum,
            light_level = pl.light_level,
            minx        = start,
            maxx        = stop,
            top         = [0xFF] * (self.width + 2),
            bottom      = [0x00] * (self.width + 2),
        )
        self._visplanes.append(new_pl)
        return new_pl

    # -----------------------------------------------------------------------
    # R_DrawPlanes – floor/ceiling rendering
    # -----------------------------------------------------------------------

    def _draw_planes(self) -> None:
        """
        R_DrawPlanes from r_plane.c.
        After BSP traversal, draw all accumulated floor/ceiling visplanes.
        """
        for pl in self._visplanes:
            if pl.minx > pl.maxx:
                continue

            if pl.picnum == self._sky_flat_idx:
                self._draw_sky_plane(pl)
                continue

            # Regular flat
            flat_data = self._tm.get_flat_by_index(pl.picnum)
            self._draw_flat_plane(pl, flat_data)

    def _draw_sky_plane(self, pl: Visplane) -> None:
        """Draw the sky using the sky texture as a vertical column."""
        if self._sky_tex_idx < 0:
            return
        sky_w, sky_h, sky_cols = self._tm.get_texture(self._sky_tex_idx)
        pal32 = self._pal32
        cmap  = self._cmap[0]   # sky is always full-bright

        for x in range(pl.minx, pl.maxx + 1):
            y1 = pl.top[x]
            y2 = pl.bottom[x]
            if y1 > y2 or y1 == 0xFF:
                continue
            y1 = max(0, y1)
            y2 = min(self.height - 1, y2)
            if y1 > y2:
                continue

            # Sky texture column based on view angle
            angle = (self.viewangle + self.xtoviewangle[x]) >> ANGLETOFINESHIFT
            sky_col = (angle * sky_w) >> (FINEANGLES.bit_length() - 1)
            sky_col = sky_col % sky_w

            # dc_iscale for sky (full texture height maps to screen height)
            # Original: pspriteiscale >> detailshift ≈ FRACUNIT*SCREENWIDTH/viewwidth
            dc_iscale = (FRACUNIT * 200) // sky_h  # approximation
            texturemid = (sky_h // 2) * FRACUNIT

            self._draw_column(
                x, y1, y2,
                sky_cols, sky_col, sky_h,
                texturemid, dc_iscale, cmap,
            )

    def _draw_flat_plane(self, pl: Visplane, flat_data: bytes) -> None:
        """
        Draw a floor or ceiling visplane using horizontal texture-mapped spans.
        Mirrors R_DrawPlanes + R_MapPlane + R_MakeSpans + R_DrawSpan.
        """
        planeheight = abs(pl.height - self.viewz)
        if planeheight < 1:
            planeheight = 1

        # Choose light table
        lightnum = (pl.light_level >> LIGHTSEGSHIFT)
        lightnum = max(0, min(LIGHTLEVELS - 1, lightnum))
        planezlight = self._zlight[lightnum]

        # Sentinel entries for R_MakeSpans
        width = self.width
        top_arr    = pl.top
        bottom_arr = pl.bottom

        # Sentinel: make column minx-1 and maxx+1 be 0xFF (empty)
        sentinel_prev = top_arr[pl.minx - 1] if pl.minx > 0 else 0xFF
        sentinel_next = top_arr[pl.maxx + 1] if pl.maxx < width else 0xFF
        top_arr[pl.minx - 1] = 0xFF
        if pl.maxx < width:
            top_arr[pl.maxx + 1] = 0xFF

        stop = pl.maxx + 1

        # Walk columns, using R_MakeSpans logic
        for x in range(pl.minx, stop + 1):
            # Previous column bounds
            t1 = top_arr[x - 1] if x > 0 else 0xFF
            b1 = bottom_arr[x - 1] if x > 0 else 0x00
            # Current column bounds
            t2 = top_arr[x] if x <= pl.maxx else 0xFF
            b2 = bottom_arr[x] if x <= pl.maxx else 0x00

            self._make_spans(
                x, t1, b1, t2, b2,
                planeheight, planezlight, flat_data,
            )

    def _make_spans(
        self,
        x:           int,
        t1:          int,  # top of previous column (0xFF = empty)
        b1:          int,  # bottom of previous column
        t2:          int,  # top of current column
        b2:          int,  # bottom of current column
        planeheight: int,  # fixed-point abs height of plane above viewz
        planezlight: list,
        flat_data:   bytes,
    ) -> None:
        """
        R_MakeSpans from r_plane.c.
        Emits completed horizontal spans when a row's run ends.
        Starts new runs as needed.
        """
        spanstart = self._spanstart

        # Close spans that ended
        while t1 < t2 and t1 <= b1:
            self._map_plane(t1, spanstart[t1], x - 1,
                            planeheight, planezlight, flat_data)
            t1 += 1
        while b1 > b2 and b1 >= t1:
            self._map_plane(b1, spanstart[b1], x - 1,
                            planeheight, planezlight, flat_data)
            b1 -= 1

        # Start new spans
        while t2 < t1 and t2 <= b2:
            spanstart[t2] = x
            t2 += 1
        while b2 > b1 and b2 >= t2:
            spanstart[b2] = x
            b2 -= 1

    def _map_plane(
        self,
        y:           int,
        x1:          int,
        x2:          int,
        planeheight: int,
        planezlight: list,
        flat_data:   bytes,
    ) -> None:
        """
        R_MapPlane from r_plane.c.
        Compute xstep/ystep for horizontal span at row y and call _draw_span.
        """
        if x1 > x2 or y < 0 or y >= self.height:
            return

        # Compute distance from viewer to the floor/ceiling at row y
        if planeheight != self._cachedheight[y]:
            self._cachedheight[y]  = planeheight
            distance = fixed_mul(planeheight, self.yslope[y])
            self._cacheddistance[y] = distance
            self._cachedxstep[y]   = fixed_mul(distance, self._basexscale)
            self._cachedystep[y]   = fixed_mul(distance, self._baseyscale)
        else:
            distance = self._cacheddistance[y]

        xstep = self._cachedxstep[y]
        ystep = self._cachedystep[y]

        length = fixed_mul(distance, self.distscale[x1])
        angle  = ((self.viewangle + self.xtoviewangle[x1]) >> ANGLETOFINESHIFT) & FINEMASK

        ds_xfrac = self.viewx + fixed_mul(finecosine[angle], length)
        ds_yfrac = -self.viewy - fixed_mul(finesine[angle], length)

        # Light index based on distance
        light_idx = distance >> LIGHTZSHIFT
        light_idx = max(0, min(MAXLIGHTZ - 1, light_idx))
        colormap  = planezlight[light_idx]

        self._draw_span(y, x1, x2, flat_data, colormap, xstep, ystep, ds_xfrac, ds_yfrac)

    def _draw_span(
        self,
        y:        int,
        x1:       int,
        x2:       int,
        flat_data: bytes,
        colormap:  bytearray,
        xstep:     int,
        ystep:     int,
        xfrac:     int,
        yfrac:     int,
    ) -> None:
        """
        R_DrawSpan from r_draw.c.
        Draws a horizontal texture-mapped span for a floor or ceiling.
        flat_data is a 4096-byte 64x64 palette-indexed image.
        """
        if x1 > x2 or y < 0 or y >= self.height:
            return
        x1 = max(0, x1)
        x2 = min(self.width - 1, x2)
        if x1 > x2:
            return

        pix_arr = self._pixels
        pal32   = self._pal32

        # Doom's span formula: spot = ((yfrac>>(16-6)) & (63*64)) + ((xfrac>>16) & 63)
        for x in range(x1, x2 + 1):
            spot = ((yfrac >> (16 - 6)) & (63 * 64)) + ((xfrac >> 16) & 63)
            pixel = flat_data[spot]
            palette_idx = colormap[pixel]
            pix_arr[x, y] = pal32[palette_idx]

            xfrac += xstep
            yfrac += ystep
