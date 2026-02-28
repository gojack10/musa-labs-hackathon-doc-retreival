"""Player controller for Python Doom port.

Movement logic derived from p_user.c (P_MovePlayer, P_Thrust, P_CalcHeight)
and p_map.c (P_TryMove, PIT_CheckLine) from linuxdoom-1.10.
"""
import math
import pygame

from doom.defs import (
    ANG45, ANG90, ANG180, ANG360, ANGLE_MAX,
    ANGLETOFINESHIFT, FINEANGLES, FINEMASK,
    FRACBITS, FRACUNIT,
    finesine, finecosine,
    ML_BLOCKING, ML_TWOSIDED,
    NF_SUBSECTOR,
    PLAYER_RADIUS, PLAYER_VIEWHEIGHT,
    Sector, MapData,
)

def _mu(v):
    """Convert fixed-point value to map units (integer)."""
    if isinstance(v, float):
        return v / FRACUNIT
    return v >> FRACBITS

# Doom movement constants from g_game.c:
#   forwardmove[0] = 0x19 (walk), forwardmove[1] = 0x32 (run)
#   sidemove[0]    = 0x18 (walk), sidemove[1]    = 0x28 (run)
#   angleturn[slow=2] = 320, angleturn[walk=0] = 640, angleturn[run=1] = 1280
# In P_MovePlayer: P_Thrust(player, angle, cmd->forwardmove * 2048)
# So actual fixed-point move = forwardmove * 2048 (== forwardmove * FRACUNIT/32)
# We convert that to float map-units per tic by dividing by FRACUNIT.

FORWARDMOVE_WALK = 0x19   # 25
FORWARDMOVE_RUN  = 0x32   # 50
SIDEMOVE_WALK    = 0x18   # 24
SIDEMOVE_RUN     = 0x28   # 40

# The multiplier used in P_Thrust: cmd->forwardmove * 2048
THRUST_SCALE = 2048  # FRACUNIT / 32 in fixed-point

# angleturn values (applied as angleturn << 16 to the mobj angle each tic)
ANGLETURN_SLOW  = 320
ANGLETURN_WALK  = 640
ANGLETURN_RUN   = 1280

# MAXBOB = 0x100000 (16 pixels in fixed-point)
MAXBOB = 0x100000  # 16 * FRACUNIT

# Max step-up height from p_map.c P_TryMove: 24 * FRACUNIT
MAX_STEP_HEIGHT = 24.0

# Player collision radius (map units)
RADIUS = float(PLAYER_RADIUS)  # 16.0

# Player height for ceiling collision
PLAYER_HEIGHT = 56.0  # standard Doom player height (mobj info)

# VIEWHEIGHT in map units (VIEWHEIGHT = 41 * FRACUNIT in Doom, stored as float here)
VIEWHEIGHT = float(PLAYER_VIEWHEIGHT)  # 41.0


def _fine_angle(bam_angle: int) -> int:
    """Convert BAM angle to fine angle table index."""
    return (bam_angle >> ANGLETOFINESHIFT) & FINEMASK


def _fixed_to_float(f: int) -> float:
    return f / FRACUNIT


class Player:
    """Doom-correct player entity.

    Angles are stored as 32-bit BAM (Binary Angle Measurement):
      0x00000000 = East
      0x40000000 = North (ANG90)
      0x80000000 = West  (ANG180)
      0xC0000000 = South (ANG270)
    """

    def __init__(self, x: float, y: float, angle: int, sector: Sector):
        self.x = x              # map units (float)
        self.y = y              # map units (float)
        self.angle = int(angle) & ANGLE_MAX  # BAM angle

        # View height stuff (mirrors Doom's player_t)
        self.viewheight = VIEWHEIGHT        # current eye height above floor
        self.deltaviewheight = 0.0          # for landing/pain animation

        # Initialize z from sector floor height (fixed-point -> map units)
        floor_h = _mu(sector.floor_height) if sector else 0.0
        self.z = floor_h
        self.viewz = floor_h + self.viewheight

        # Current sector
        self.sector = sector

        # Momentum (fixed-point style, kept as floats in map units/tic)
        self.momx = 0.0
        self.momy = 0.0

        # Bob amount (fixed-point, derived from momentum magnitude squared)
        self.bob = 0.0

        # Tick counter for bob animation (mirrors leveltime usage)
        self._leveltime = 0

        self.on_ground = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        forward: float,
        strafe: float,
        turn: int,
        map_data: MapData,
        running: bool = False,
    ) -> None:
        """Advance player state by one tic.

        Args:
            forward: -1 (backward) to +1 (forward), or 0.
            strafe:  -1 (left) to +1 (right), or 0.
            turn:    pre-computed BAM angle delta to add this tic (may be negative).
            map_data: current level geometry for collision.
            running: True if shift is held (run speed).
        """
        self._leveltime += 1

        # --- Turning (mirrors P_MovePlayer: player->mo->angle += cmd->angleturn<<16) ---
        # The caller passes a pre-computed BAM delta already scaled for one tic.
        self.angle = (self.angle + turn) & ANGLE_MAX

        # --- Thrust (mirrors P_Thrust) ---
        # Doom: cmd->forwardmove is in range [-50..50], multiplied by 2048 fixed-pt.
        # We normalise the caller's [-1..1] range to the Doom integer range first.
        fwd_speed = FORWARDMOVE_RUN if running else FORWARDMOVE_WALK
        str_speed = SIDEMOVE_RUN    if running else SIDEMOVE_WALK

        # Convert to fixed-point move amounts (same as Doom's ticcmd values)
        fwd_move = int(forward * fwd_speed)   # [-50..50] walk or [-32..32]
        str_move = int(strafe  * str_speed)   # sidestep

        # Compute thrust in fixed-point units (per tic), then to floats
        # P_Thrust: momx += FixedMul(move, finecosine[angle>>ANGLETOFINESHIFT])
        #           momy += FixedMul(move, finesine[angle>>ANGLETOFINESHIFT])
        # move here is forwardmove * THRUST_SCALE
        if fwd_move != 0:
            fa = _fine_angle(self.angle)
            fx = fwd_move * THRUST_SCALE
            self.momx += (fx * finecosine[fa]) / (FRACUNIT * FRACUNIT)
            self.momy += (fx * finesine[fa])   / (FRACUNIT * FRACUNIT)

        if str_move != 0:
            # strafe is perpendicular: angle - ANG90
            fa = _fine_angle((self.angle - ANG90) & ANGLE_MAX)
            sx = str_move * THRUST_SCALE
            self.momx += (sx * finecosine[fa]) / (FRACUNIT * FRACUNIT)
            self.momy += (sx * finesine[fa])   / (FRACUNIT * FRACUNIT)

        # Apply friction (Doom's FRICTION = 0xE800 / FRACUNIT ≈ 0.90625)
        # Applied in P_MobjThinker after movement; keeps things feeling right.
        FRICTION = 0.90625
        new_x = self.x + self.momx
        new_y = self.y + self.momy

        # --- Collision (simplified P_TryMove) ---
        if self._try_move(new_x, new_y, map_data):
            self.x = new_x
            self.y = new_y
        else:
            # Wall-slide: try axis-separated moves
            if self._try_move(new_x, self.y, map_data):
                self.x = new_x
                self.momy = 0.0
            elif self._try_move(self.x, new_y, map_data):
                self.y = new_y
                self.momx = 0.0
            else:
                self.momx = 0.0
                self.momy = 0.0

        # Apply friction to bleed off momentum each tic
        self.momx *= FRICTION
        self.momy *= FRICTION

        # Clamp tiny momenta to zero
        if abs(self.momx) < 0.001:
            self.momx = 0.0
        if abs(self.momy) < 0.001:
            self.momy = 0.0

        # --- Update sector / floor height ---
        self._update_sector(map_data)
        floor_h = _mu(self.sector.floor_height) if self.sector else 0.0
        self.z = floor_h
        self.on_ground = True

        # --- View bobbing (mirrors P_CalcHeight) ---
        # bob = (momx*momx + momy*momy) >> 2, clamped to MAXBOB
        # We work in fixed-point equivalents: convert momenta back to fixed first.
        fx_momx = int(self.momx * FRACUNIT)
        fx_momy = int(self.momy * FRACUNIT)
        fx_bob = (fx_momx * fx_momx + fx_momy * fx_momy) >> 2
        if fx_bob > MAXBOB:
            fx_bob = MAXBOB
        self.bob = fx_bob  # stored in fixed-point, used below

        # viewheight tracks PST_LIVE path in P_CalcHeight
        if self.viewheight > VIEWHEIGHT:
            self.viewheight = VIEWHEIGHT
            self.deltaviewheight = 0.0
        if self.viewheight < VIEWHEIGHT / 2:
            self.viewheight = VIEWHEIGHT / 2
            if self.deltaviewheight <= 0:
                self.deltaviewheight = 1.0

        # bob angle: (FINEANGLES/20 * leveltime) & FINEMASK
        bob_angle = (FINEANGLES // 20 * self._leveltime) & FINEMASK
        # bob_offset = FixedMul(bob/2, finesine[bob_angle]) / FRACUNIT
        bob_fp = (self.bob // 2 * finesine[bob_angle]) // FRACUNIT
        bob_offset = bob_fp / FRACUNIT  # convert from fixed to map units

        self.viewz = self.z + self.viewheight + bob_offset

        # Clamp to ceiling (leave 4 units of clearance)
        if self.sector:
            ceil = _mu(self.sector.ceiling_height) - 4.0
            if self.viewz > ceil:
                self.viewz = ceil

    # ------------------------------------------------------------------
    # Turning helpers (called from game loop before update())
    # ------------------------------------------------------------------

    @staticmethod
    def keyboard_turn_delta(left: bool, right: bool, running: bool, held_tics: int) -> int:
        """Compute BAM angle delta for one tic from keyboard arrows.

        Mirrors Doom's two-stage accelerative turning:
          - slow turn (ANGLETURN_SLOW) for first SLOWTURNTICS=6 tics
          - fast turn (ANGLETURN_WALK/RUN) after that
        Returns positive (left/CCW) or negative (right/CW) BAM delta.
        """
        SLOWTURNTICS = 6
        if held_tics < SLOWTURNTICS:
            speed = ANGLETURN_SLOW
        else:
            speed = ANGLETURN_RUN if running else ANGLETURN_WALK

        delta = 0
        if left:
            delta += speed
        if right:
            delta -= speed
        # angleturn is applied as angleturn<<16 in P_MovePlayer
        return delta << 16

    @staticmethod
    def mouse_turn_delta(mouse_dx: int, sensitivity: int = 5) -> int:
        """Compute BAM angle delta from mouse X movement.

        Mirrors G_BuildTiccmd: cmd->angleturn -= mousex * 0x8
        where mousex = ev->data2 * (mouseSensitivity+5) / 10
        """
        # mousex = mouse_dx * (sensitivity + 5) / 10
        mousex = mouse_dx * (sensitivity + 5) // 10
        # angleturn delta = -mousex * 0x8, then << 16 in P_MovePlayer
        return -(mousex * 0x8) << 16

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_move(self, x: float, y: float, map_data: MapData) -> bool:
        """Return True if the player circle (RADIUS) can fit at (x, y).

        Simplified P_TryMove / PIT_CheckLine:
          1. Compute the new subsector / floor+ceiling at the target point.
          2. Check all linedefs whose bounding box overlaps the player bbox.
          3. For blocking single-sided lines, test if the move crosses them.
          4. For two-sided lines, enforce step-height and headroom.
        """
        # Player bounding box at new position
        left   = x - RADIUS
        right  = x + RADIUS
        bottom = y - RADIUS
        top    = y + RADIUS

        # Find floor/ceiling at destination via BSP
        dest_sector = self._sector_at(x, y, map_data)
        if dest_sector is None:
            # No BSP / no sectors: allow movement if there are also no lines to block
            return len(map_data.lines) == 0

        dest_floor   = _mu(dest_sector.floor_height)
        dest_ceiling = _mu(dest_sector.ceiling_height)

        # Headroom must fit player (56 units tall)
        if dest_ceiling - dest_floor < PLAYER_HEIGHT:
            return False

        # Step-up limit: no more than 24 units above current floor
        if dest_floor - self.z > MAX_STEP_HEIGHT:
            return False

        # Check linedefs
        for line in map_data.lines:
            # Quick bbox rejection (mirrors PIT_CheckLine first check)
            # Convert from fixed-point to map units
            lx1, ly1 = _mu(line.v1.x), _mu(line.v1.y)
            lx2, ly2 = _mu(line.v2.x), _mu(line.v2.y)
            line_left   = min(lx1, lx2)
            line_right  = max(lx1, lx2)
            line_bottom = min(ly1, ly2)
            line_top    = max(ly1, ly2)

            if (right  <= line_left  or left  >= line_right or
                    top <= line_bottom or bottom >= line_top):
                continue

            # Do not test further if player bbox is entirely on one side
            # (P_BoxOnLineSide equivalent — if not -1, skip)
            if not self._box_straddles_line(left, bottom, right, top, line):
                continue

            # One-sided line always blocks (no backsector)
            if line.back_sector is None:
                if line.flags & ML_BLOCKING:
                    return False
                # Even without ML_BLOCKING, one-sided lines are solid walls
                return False

            # Two-sided: check ML_BLOCKING flag
            if line.flags & ML_BLOCKING:
                return False

            # Two-sided passability: check opening height and step
            fs = line.front_sector
            bs = line.back_sector
            if fs and bs:
                open_bottom = max(_mu(fs.floor_height),   _mu(bs.floor_height))
                open_top    = min(_mu(fs.ceiling_height), _mu(bs.ceiling_height))
                opening_h   = open_top - open_bottom

                if opening_h < PLAYER_HEIGHT:
                    return False  # doesn't fit through
                if open_bottom - self.z > MAX_STEP_HEIGHT:
                    return False  # step too high

        return True

    def _box_straddles_line(
        self, left: float, bottom: float, right: float, top: float, line
    ) -> bool:
        """Return True if the AABB straddles (crosses) the line.

        Tests corners: if all corners are on the same side, the box does
        not cross the line.  Returns False if box is on one side (skip),
        True if it straddles (must test).
        """
        # Use cross product sign for each corner
        # Convert line data from fixed-point to map units
        dx = _mu(line.dx)
        dy = _mu(line.dy)
        x0 = _mu(line.v1.x)
        y0 = _mu(line.v1.y)

        corners = [
            (left,  bottom),
            (right, bottom),
            (left,  top),
            (right, top),
        ]
        sides = set()
        for cx, cy in corners:
            cross = (cx - x0) * dy - (cy - y0) * dx
            sides.add(0 if cross >= 0 else 1)
        return len(sides) > 1

    def _point_on_line_side(self, x: float, y: float, line) -> int:
        """Return 0 if (x,y) is on the front side, 1 on the back.

        Mirrors P_PointOnLineSide.
        """
        dx = x - _mu(line.v1.x)
        dy = y - _mu(line.v1.y)
        cross = dx * _mu(line.dy) - dy * _mu(line.dx)
        return 0 if cross <= 0 else 1

    def _update_sector(self, map_data: MapData) -> None:
        """Update self.sector to the sector the player currently occupies."""
        s = self._sector_at(self.x, self.y, map_data)
        if s is not None:
            self.sector = s

    def _sector_at(self, x: float, y: float, map_data: MapData):
        """Return the Sector at map position (x, y) using BSP traversal.

        Mirrors R_PointInSubsector from the renderer.
        """
        if not map_data.nodes:
            if map_data.subsectors:
                return map_data.subsectors[0].sector
            return map_data.sectors[0] if map_data.sectors else None

        # Convert player position to fixed-point for comparison with BSP nodes
        fx = int(x * FRACUNIT)
        fy = int(y * FRACUNIT)

        node_id = len(map_data.nodes) - 1
        while True:
            if node_id & NF_SUBSECTOR:
                ss_idx = node_id & ~NF_SUBSECTOR
                if ss_idx < len(map_data.subsectors):
                    return map_data.subsectors[ss_idx].sector
                return None
            node = map_data.nodes[node_id]
            # BSP node coords are in fixed-point, so compare in fixed-point
            dx = fx - node.x
            dy = fy - node.y
            left  = dy * node.dx
            right = dx * node.dy
            side  = 0 if right < left else 1
            node_id = node.children[side]
