"""
WAD file reader, map loader, and texture/flat loader for Python Doom.

Based on the original id Software linuxdoom-1.10 source:
  w_wad.c / w_wad.h  - WAD I/O
  p_setup.c          - Map loading (P_LoadVertexes, P_LoadSectors, etc.)
  r_data.c           - Texture/flat loading (R_InitTextures, R_InitFlats)
  doomdata.h         - On-disk map format structs
  r_defs.h           - patch_t / column_t / post_t formats
"""

import struct
from typing import Optional

from doom.defs import (
    Vertex, Sector, Side, Line, Seg, Subsector, Node, MapThing, MapData,
    ML_TWOSIDED, NF_SUBSECTOR,
    FRACBITS,
)


# ---------------------------------------------------------------------------
# WAD file reader
# ---------------------------------------------------------------------------

class WAD:
    """
    Opens a WAD file and provides lump access by name or index.

    WAD header layout (12 bytes):
        4s  identification  "IWAD" or "PWAD"
        i   numlumps
        i   infotableofs

    Lump directory entry (16 bytes each):
        i   filepos
        i   size
        8s  name  (null-padded, uppercase)
    """

    _HEADER_FMT = "<4sii"
    _HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 12

    _DIR_ENTRY_FMT = "<ii8s"
    _DIR_ENTRY_SIZE = struct.calcsize(_DIR_ENTRY_FMT)  # 16

    def __init__(self, path: str) -> None:
        self.path = path
        with open(path, "rb") as f:
            data = f.read()
        self._data = data

        ident, numlumps, infotableofs = struct.unpack_from(
            self._HEADER_FMT, data, 0
        )

        if ident not in (b"IWAD", b"PWAD"):
            raise ValueError(
                f"Not a valid WAD file: identification is {ident!r}"
            )

        self.ident = ident.decode("ascii")
        self.numlumps = numlumps

        # Parse the lump directory
        self._directory = []  # list of (filepos, size, name_str)
        offset = infotableofs
        for _ in range(numlumps):
            filepos, size, raw_name = struct.unpack_from(
                self._DIR_ENTRY_FMT, data, offset
            )
            # Strip null bytes and whitespace; names are uppercase in WAD
            name = raw_name.rstrip(b"\x00").rstrip().decode("ascii", errors="replace").upper()
            self._directory.append((filepos, size, name))
            offset += self._DIR_ENTRY_SIZE

        # Build name -> last-occurring index map (last takes precedence,
        # matching the original C code's backwards scan behaviour)
        self._name_index: dict[str, int] = {}
        for i, (_, _, name) in enumerate(self._directory):
            self._name_index[name] = i

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_lump(self, name: str) -> int:
        """Return the lump index for *name*, or -1 if not found."""
        return self._name_index.get(name.upper().rstrip("\x00").strip(), -1)

    def read_lump(self, name: str) -> bytes:
        """Read and return the raw bytes of the lump named *name*."""
        idx = self.find_lump(name)
        if idx == -1:
            raise KeyError(f"Lump not found: {name!r}")
        return self.read_lump_index(idx)

    def read_lump_index(self, index: int) -> bytes:
        """Read and return the raw bytes of lump at *index*."""
        filepos, size, _ = self._directory[index]
        return self._data[filepos: filepos + size]

    def lump_size(self, index: int) -> int:
        """Return the byte length of lump at *index*."""
        return self._directory[index][1]

    def lump_name(self, index: int) -> str:
        """Return the name of lump at *index*."""
        return self._directory[index][2]


# ---------------------------------------------------------------------------
# Palette / colormap loaders
# ---------------------------------------------------------------------------

def load_palette(wad: WAD) -> list:
    """
    Read PLAYPAL lump and return the first palette as a list of 256 (r,g,b)
    tuples.  PLAYPAL contains 14 complete palettes; we only need the first.
    """
    data = wad.read_lump("PLAYPAL")
    palette = []
    for i in range(256):
        r = data[i * 3]
        g = data[i * 3 + 1]
        b = data[i * 3 + 2]
        palette.append((r, g, b))
    return palette


def load_colormap(wad: WAD) -> list:
    """
    Read COLORMAP lump and return a list of 34 maps, each a 256-byte
    bytearray (light-diminishing table used by the renderer).
    """
    data = wad.read_lump("COLORMAP")
    maps = []
    for i in range(34):
        start = i * 256
        maps.append(bytearray(data[start: start + 256]))
    return maps


# ---------------------------------------------------------------------------
# Map loader
# ---------------------------------------------------------------------------

# On-disk struct sizes (bytes)
_THING_SIZE    = 10   # x:s, y:s, angle:s, type:s, options:s
_LINEDEF_SIZE  = 14   # v1:s, v2:s, flags:s, special:s, tag:s, sidenum[2]:2s
_SIDEDEF_SIZE  = 30   # textureoffset:s, rowoffset:s, top:8s, bot:8s, mid:8s, sector:s
_VERTEX_SIZE   = 4    # x:s, y:s
_SEG_SIZE      = 12   # v1:s, v2:s, angle:s, linedef:s, side:s, offset:s
_SSECTOR_SIZE  = 4    # numsegs:H, firstseg:H  (unsigned)
_NODE_SIZE     = 28   # x:s, y:s, dx:s, dy:s, bbox[2][4]:8s, children[2]:2H
_SECTOR_SIZE   = 26   # floorheight:s, ceilingheight:s, floor:8s, ceil:8s, light:s, special:s, tag:s


def _decode_name(raw: bytes) -> str:
    """Convert a null-padded 8-byte WAD name to a Python string."""
    return raw.rstrip(b"\x00").rstrip().decode("ascii", errors="replace").upper()


def load_map(wad: WAD, episode: int = 1, mission: int = 1) -> MapData:
    """
    Load all lumps for the map ExMy and return a fully cross-referenced
    MapData object.

    The map marker lump (e.g. "E1M1") is found and lumps at offsets +1
    through +8 are loaded in the order defined in doomdata.h:
        +1 THINGS, +2 LINEDEFS, +3 SIDEDEFS, +4 VERTEXES, +5 SEGS,
        +6 SSECTORS, +7 NODES, +8 SECTORS
    """
    map_name = f"E{episode}M{mission}"
    marker_idx = wad.find_lump(map_name)
    if marker_idx == -1:
        raise KeyError(f"Map lump not found: {map_name!r}")

    def lump(offset: int) -> bytes:
        return wad.read_lump_index(marker_idx + offset)

    md = MapData()

    # ---- VERTEXES (+4) ------------------------------------------------
    # Must be loaded before linedefs/segs; coords stored as signed shorts
    # In the C source: li->x = SHORT(ml->x) << FRACBITS
    # We store as integer map-units (i.e. raw short value * FRACUNIT)
    raw_verts = lump(4)
    n_verts = len(raw_verts) // _VERTEX_SIZE
    for i in range(n_verts):
        x, y = struct.unpack_from("<hh", raw_verts, i * _VERTEX_SIZE)
        md.vertices.append(Vertex(x=x << FRACBITS, y=y << FRACBITS))

    # ---- SECTORS (+8) -------------------------------------------------
    # Loaded before sidedefs so sides can reference sector objects
    raw_sectors = lump(8)
    n_sectors = len(raw_sectors) // _SECTOR_SIZE
    for i in range(n_sectors):
        off = i * _SECTOR_SIZE
        floorheight, ceilingheight = struct.unpack_from("<hh", raw_sectors, off)
        floor_pic_raw = raw_sectors[off + 4: off + 12]
        ceil_pic_raw  = raw_sectors[off + 12: off + 20]
        lightlevel, special, tag = struct.unpack_from("<hhh", raw_sectors, off + 20)

        floor_pic_name = _decode_name(floor_pic_raw)
        ceil_pic_name  = _decode_name(ceil_pic_raw)

        sector = Sector(
            floor_height   = floorheight << FRACBITS,
            ceiling_height = ceilingheight << FRACBITS,
            floor_pic      = 0,  # filled in by caller after flat lookup
            ceiling_pic    = 0,
            light_level    = lightlevel,
            special        = special,
            tag            = tag,
            floor_pic_name = floor_pic_name,
            ceiling_pic_name = ceil_pic_name,
        )
        md.sectors.append(sector)

    # ---- SIDEDEFS (+3) ------------------------------------------------
    raw_sides = lump(3)
    n_sides = len(raw_sides) // _SIDEDEF_SIZE
    for i in range(n_sides):
        off = i * _SIDEDEF_SIZE
        texoff, rowoff = struct.unpack_from("<hh", raw_sides, off)
        top_raw = raw_sides[off + 4: off + 12]
        bot_raw = raw_sides[off + 12: off + 20]
        mid_raw = raw_sides[off + 20: off + 28]
        sector_idx, = struct.unpack_from("<h", raw_sides, off + 28)

        top_name = _decode_name(top_raw)
        bot_name = _decode_name(bot_raw)
        mid_name = _decode_name(mid_raw)

        # Texture index will be resolved by TextureManager; store -1 for now
        side = Side(
            texture_offset   = texoff << FRACBITS,
            row_offset       = rowoff << FRACBITS,
            top_texture      = -1,
            bottom_texture   = -1,
            mid_texture      = -1,
            sector           = md.sectors[sector_idx] if 0 <= sector_idx < n_sectors else None,
            top_texture_name = top_name,
            bottom_texture_name = bot_name,
            mid_texture_name    = mid_name,
        )
        md.sides.append(side)

    # ---- LINEDEFS (+2) ------------------------------------------------
    # sidenum uses -1 (0xFFFF as unsigned short) to mean "no side"
    raw_lines = lump(2)
    n_lines = len(raw_lines) // _LINEDEF_SIZE
    for i in range(n_lines):
        off = i * _LINEDEF_SIZE
        v1_idx, v2_idx, flags, special, tag = struct.unpack_from("<hhhhh", raw_lines, off)
        # sidenum[0] and [1] are stored as signed shorts; -1 means absent
        sn0, sn1 = struct.unpack_from("<hh", raw_lines, off + 10)

        v1 = md.vertices[v1_idx]
        v2 = md.vertices[v2_idx]

        # dx/dy in fixed-point units
        dx = v2.x - v1.x
        dy = v2.y - v1.y

        front_sector: Optional[Sector] = None
        back_sector:  Optional[Sector] = None

        if 0 <= sn0 < n_sides:
            front_sector = md.sides[sn0].sector
        if sn1 != -1 and 0 <= sn1 < n_sides:
            back_sector = md.sides[sn1].sector

        # Bounding box [top, bottom, left, right] in fixed-point
        if v1.x < v2.x:
            bbox_left, bbox_right = v1.x, v2.x
        else:
            bbox_left, bbox_right = v2.x, v1.x
        if v1.y < v2.y:
            bbox_bottom, bbox_top = v1.y, v2.y
        else:
            bbox_bottom, bbox_top = v2.y, v1.y

        line = Line(
            v1           = v1,
            v2           = v2,
            dx           = dx,
            dy           = dy,
            flags        = flags,
            special      = special,
            tag          = tag,
            sidenum      = [sn0, sn1],
            front_sector = front_sector,
            back_sector  = back_sector,
            bbox         = [bbox_top, bbox_bottom, bbox_left, bbox_right],
        )
        md.lines.append(line)

    # ---- SEGS (+5) ---------------------------------------------------
    # angle stored as signed short representing BAM >> 16
    # Convert: angle = (raw_angle & 0xFFFF) << 16
    raw_segs = lump(5)
    n_segs = len(raw_segs) // _SEG_SIZE
    for i in range(n_segs):
        off = i * _SEG_SIZE
        v1_idx, v2_idx, raw_angle, linedef_idx, side_idx, raw_offset = \
            struct.unpack_from("<hhhhhh", raw_segs, off)

        # BAM angle: treat raw as unsigned 16-bit then shift up
        bam_angle = (raw_angle & 0xFFFF) << 16

        # Offset along linedef (fixed-point)
        seg_offset = (raw_offset & 0xFFFF) << 16

        v1 = md.vertices[v1_idx & 0xFFFF] if v1_idx >= 0 else md.vertices[v1_idx]
        v2 = md.vertices[v2_idx & 0xFFFF] if v2_idx >= 0 else md.vertices[v2_idx]

        sidedef:     Optional[Side]   = None
        linedef_obj: Optional[Line]   = None
        front_sec:   Optional[Sector] = None
        back_sec:    Optional[Sector] = None

        if 0 <= linedef_idx < n_lines:
            linedef_obj = md.lines[linedef_idx]
            actual_side = side_idx & 1
            sn = linedef_obj.sidenum[actual_side]
            if 0 <= sn < n_sides:
                sidedef = md.sides[sn]
                front_sec = sidedef.sector

            # Back sector: the other side of the linedef
            if linedef_obj.flags & ML_TWOSIDED:
                other_sn = linedef_obj.sidenum[actual_side ^ 1]
                if 0 <= other_sn < n_sides:
                    back_sec = md.sides[other_sn].sector

        seg = Seg(
            v1           = v1,
            v2           = v2,
            offset       = seg_offset,
            angle        = bam_angle,
            sidedef      = sidedef,
            linedef      = linedef_obj,
            front_sector = front_sec,
            back_sector  = back_sec,
        )
        md.segs.append(seg)

    # ---- SSECTORS (+6) ----------------------------------------------
    # numsegs and firstseg are unsigned shorts
    raw_ss = lump(6)
    n_ss = len(raw_ss) // _SSECTOR_SIZE
    for i in range(n_ss):
        numsegs, firstseg = struct.unpack_from("<HH", raw_ss, i * _SSECTOR_SIZE)
        # Sector is resolved in P_GroupLines equivalent below
        md.subsectors.append(Subsector(
            sector     = None,
            num_lines  = numsegs,
            first_line = firstseg,
        ))

    # ---- NODES (+7) -------------------------------------------------
    # children are unsigned shorts; if bit 15 set → subsector
    raw_nodes = lump(7)
    n_nodes = len(raw_nodes) // _NODE_SIZE
    for i in range(n_nodes):
        off = i * _NODE_SIZE
        x, y, dx, dy = struct.unpack_from("<hhhh", raw_nodes, off)
        # bbox[2][4] – 8 signed shorts
        bbox_raw = struct.unpack_from("<8h", raw_nodes, off + 8)
        # children[2] – 2 unsigned shorts
        c0, c1 = struct.unpack_from("<HH", raw_nodes, off + 24)

        # Store bboxes as [[top,bot,left,right], [top,bot,left,right]]
        # Original ordering: BOXTOP=0,BOXBOTTOM=1,BOXLEFT=2,BOXRIGHT=3
        bbox = [
            [bbox_raw[0] << FRACBITS, bbox_raw[1] << FRACBITS,
             bbox_raw[2] << FRACBITS, bbox_raw[3] << FRACBITS],
            [bbox_raw[4] << FRACBITS, bbox_raw[5] << FRACBITS,
             bbox_raw[6] << FRACBITS, bbox_raw[7] << FRACBITS],
        ]

        node = Node(
            x        = x  << FRACBITS,
            y        = y  << FRACBITS,
            dx       = dx << FRACBITS,
            dy       = dy << FRACBITS,
            bbox     = bbox,
            children = [c0, c1],
        )
        md.nodes.append(node)

    # ---- THINGS (+1) ------------------------------------------------
    raw_things = lump(1)
    n_things = len(raw_things) // _THING_SIZE
    for i in range(n_things):
        off = i * _THING_SIZE
        x, y, angle, thing_type, options = struct.unpack_from("<hhhhh", raw_things, off)
        md.things.append(MapThing(
            x       = x,
            y       = y,
            angle   = angle,
            type    = thing_type,
            options = options,
        ))

    # ---- REJECT (+9) / BLOCKMAP (+10) --------------------------------
    try:
        md.reject = wad.read_lump_index(marker_idx + 9)
    except Exception:
        md.reject = None

    try:
        md.blockmap = wad.read_lump_index(marker_idx + 10)
    except Exception:
        md.blockmap = None

    # ---- Cross-reference: assign sectors to subsectors -------------
    # (mirrors P_GroupLines in p_setup.c)
    for ss in md.subsectors:
        if ss.first_line < n_segs:
            first_seg = md.segs[ss.first_line]
            if first_seg.sidedef is not None:
                ss.sector = first_seg.sidedef.sector

    return md


# ---------------------------------------------------------------------------
# Texture manager
# ---------------------------------------------------------------------------

# Checkerboard pattern dimensions for missing-texture fallback
_CHECKER_SIZE = 64  # 64x64 pixels


def _make_checkerboard(width: int, height: int) -> list:
    """
    Return a checkerboard column list suitable for get_texture() return value.
    Each column is a list of (y_start, pixels) pairs (run-length encoded posts).
    Colours alternate between palette index 0 and 255.
    """
    columns = []
    for x in range(width):
        col_pixels = bytearray()
        for y in range(height):
            if (x // 8 + y // 8) % 2 == 0:
                col_pixels.append(255)
            else:
                col_pixels.append(0)
        # One run covering the whole column
        columns.append([(0, bytes(col_pixels))])
    return columns


class TextureManager:
    """
    Loads and composes DOOM textures and flats from a WAD.

    Textures are composed lazily on first use and cached.  Flats are stored
    as raw 4096-byte blocks (64x64 palettised pixels).
    """

    def __init__(self, wad: WAD) -> None:
        self._wad = wad

        # Maps texture name (uppercase) → texture index
        self._tex_name_to_idx: dict[str, int] = {}

        # Per-texture metadata: [(name, width, height, patches)]
        # patches: list of (originx, originy, lump_index)
        self._textures: list[tuple] = []

        # Composed column cache: index → (width, height, columns) or None
        self._tex_cache: dict[int, tuple] = {}

        # Flat data: index → bytes (4096)
        self._flat_data: list[bytes] = []
        self._flat_name_to_idx: dict[str, int] = {}

        self._load_pnames()
        self._load_textures()
        self._load_flats()

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _load_pnames(self) -> None:
        """
        Load PNAMES lump: 4-byte count followed by count * 8-byte names.
        Resolves each name to its lump index in the WAD.
        """
        data = self._wad.read_lump("PNAMES")
        n_patches, = struct.unpack_from("<i", data, 0)
        self._patch_lumps: list[int] = []
        for i in range(n_patches):
            raw = data[4 + i * 8: 4 + i * 8 + 8]
            name = _decode_name(raw)
            idx = self._wad.find_lump(name)
            self._patch_lumps.append(idx)

    def _load_textures(self) -> None:
        """
        Load TEXTURE1 (and TEXTURE2 if present).

        TEXTURE lump format:
            4 bytes   numtextures
            numtextures * 4 bytes  offsets (from start of lump)

        Each texture record (at its offset):
            8s  name
            2H  masked (boolean), width
            2H  height
            4   columndirectory (OBSOLETE, skip)
            H   patchcount
            patchcount * 10 bytes of patch records:
                hh  originx, originy
                H   patch (index into PNAMES)
                HH  stepdir, colormap (unused for composition)
        """
        self._textures = []
        self._tex_name_to_idx = {}

        for lump_name in ("TEXTURE1", "TEXTURE2"):
            if self._wad.find_lump(lump_name) == -1:
                continue

            data = self._wad.read_lump(lump_name)
            n_tex, = struct.unpack_from("<i", data, 0)

            for i in range(n_tex):
                offset, = struct.unpack_from("<i", data, 4 + i * 4)

                raw_name = data[offset: offset + 8]
                tex_name = _decode_name(raw_name)

                # masked (4 bytes, boolean), width (2), height (2)
                # columndirectory (4 bytes, obsolete pointer – skip)
                # patchcount (2)
                masked, width, height = struct.unpack_from("<iHH", data, offset + 8)
                # skip 4 bytes columndirectory
                patchcount, = struct.unpack_from("<H", data, offset + 18)

                patches = []
                patch_off = offset + 20
                for _ in range(patchcount):
                    originx, originy, patch_idx, _stepdir, _colormap = \
                        struct.unpack_from("<hhHHH", data, patch_off)
                    patch_off += 10
                    if 0 <= patch_idx < len(self._patch_lumps):
                        lump_idx = self._patch_lumps[patch_idx]
                    else:
                        lump_idx = -1
                    patches.append((originx, originy, lump_idx))

                tex_index = len(self._textures)
                self._textures.append((tex_name, width, height, patches))
                # Last definition wins (PWAD override behaviour)
                self._tex_name_to_idx[tex_name] = tex_index

    def _load_flats(self) -> None:
        """
        Load flat lumps between F_START and F_END markers.
        Each flat is exactly 4096 bytes (64x64 raw palette-indexed pixels).
        """
        self._flat_data = []
        self._flat_name_to_idx = {}

        start_idx = self._wad.find_lump("F_START")
        end_idx   = self._wad.find_lump("F_END")

        if start_idx == -1 or end_idx == -1:
            # Try alternate names used in some WADs
            start_idx = self._wad.find_lump("FF_START")
            end_idx   = self._wad.find_lump("FF_END")

        if start_idx == -1 or end_idx == -1:
            return

        flat_index = 0
        for lump_i in range(start_idx + 1, end_idx):
            if self._wad.lump_size(lump_i) == 0:
                continue  # skip marker sub-lumps (FF_START/FF_END inside)
            name = self._wad.lump_name(lump_i)
            raw  = self._wad.read_lump_index(lump_i)
            if len(raw) < 4096:
                raw = raw + b"\x00" * (4096 - len(raw))
            else:
                raw = raw[:4096]
            self._flat_name_to_idx[name] = flat_index
            self._flat_data.append(bytes(raw))
            flat_index += 1

    # ------------------------------------------------------------------
    # Patch composition helpers
    # ------------------------------------------------------------------

    def _read_patch_columns(self, lump_idx: int) -> tuple:
        """
        Parse a patch lump and return (width, height, columns) where
        columns is a list of column data (raw bytes from start of column
        data, i.e. at the columnofs[x] offset within the lump).

        Returns (1, 1, [[…]]) on error.
        """
        if lump_idx < 0:
            return (1, 1, [[]])
        try:
            data = self._wad.read_lump_index(lump_idx)
        except Exception:
            return (1, 1, [[]])

        if len(data) < 8:
            return (1, 1, [[]])

        width, height, _left_offset, _top_offset = struct.unpack_from("<hhhh", data, 0)
        if width <= 0 or height <= 0:
            return (1, 1, [[]])

        # columnofs[width] – array of int32 offsets into the lump
        col_offsets = []
        for c in range(width):
            off_pos = 8 + c * 4
            if off_pos + 4 > len(data):
                col_offsets.append(len(data))
            else:
                col_off, = struct.unpack_from("<I", data, off_pos)
                col_offsets.append(col_off)

        # Return raw column bytes (starting at column offset)
        raw_columns = []
        for c_off in col_offsets:
            raw_columns.append(data[c_off:] if c_off < len(data) else b"\xff")

        return (width, height, raw_columns)

    def _compose_texture(self, tex_index: int) -> tuple:
        """
        Compose a full texture from its patches.

        Returns (width, height, columns) where columns is a list of
        column data – each column is a list of (y_start, pixels) run pairs.
        """
        name, width, height, patches = self._textures[tex_index]

        if width <= 0 or height <= 0:
            return (1, 1, _make_checkerboard(1, 1))

        # Allocate pixel buffer (palette index 0 = transparent/black)
        buf = bytearray(width * height)

        for originx, originy, lump_idx in patches:
            pw, ph, raw_cols = self._read_patch_columns(lump_idx)
            for col_x, raw_col in enumerate(raw_cols):
                dest_x = originx + col_x
                if dest_x < 0 or dest_x >= width:
                    continue

                # Walk the posts in this column
                pos = 0
                while pos < len(raw_col):
                    topdelta = raw_col[pos] if pos < len(raw_col) else 0xFF
                    if topdelta == 0xFF:
                        break
                    pos += 1
                    if pos >= len(raw_col):
                        break
                    length = raw_col[pos]
                    pos += 1
                    # skip pad byte
                    pos += 1
                    for row in range(length):
                        dest_y = originy + topdelta + row
                        if 0 <= dest_y < height:
                            if pos < len(raw_col):
                                buf[dest_x + dest_y * width] = raw_col[pos]
                        pos += 1
                    # skip trailing pad
                    pos += 1

        # Convert flat pixel buffer to column-format list
        columns = []
        for x in range(width):
            col_pixels = bytes(buf[x + y * width] for y in range(height))
            columns.append([(0, col_pixels)])

        return (width, height, columns)

    # ------------------------------------------------------------------
    # Public API – textures
    # ------------------------------------------------------------------

    def get_texture_index(self, name: str) -> int:
        """
        Find a texture by name.  Returns -1 for "-" (no texture) or if not
        found.
        """
        clean = name.strip("\x00").strip().upper()
        if not clean or clean == "-":
            return -1
        return self._tex_name_to_idx.get(clean, -1)

    def get_texture(self, index: int) -> tuple:
        """
        Return (width, height, columns) for the texture at *index*.
        columns is a list of column data: each column is a list of
        (y_start, pixels) run-length posts.

        Returns a checkerboard pattern if the index is invalid.
        """
        if index < 0 or index >= len(self._textures):
            w = _CHECKER_SIZE
            h = _CHECKER_SIZE
            return (w, h, _make_checkerboard(w, h))

        if index in self._tex_cache:
            return self._tex_cache[index]

        result = self._compose_texture(index)
        self._tex_cache[index] = result
        return result

    def texture_width(self, index: int) -> int:
        """Return the width in pixels of the texture at *index*."""
        if index < 0 or index >= len(self._textures):
            return _CHECKER_SIZE
        return self._textures[index][1]

    def texture_height(self, index: int) -> int:
        """Return the height in pixels of the texture at *index*."""
        if index < 0 or index >= len(self._textures):
            return _CHECKER_SIZE
        return self._textures[index][2]

    # ------------------------------------------------------------------
    # Public API – flats
    # ------------------------------------------------------------------

    def get_flat_index(self, name: str) -> int:
        """Return the flat index for *name*, or -1 if not found."""
        clean = name.strip("\x00").strip().upper()
        return self._flat_name_to_idx.get(clean, -1)

    def get_flat(self, name: str) -> bytes:
        """
        Return 4096 bytes of 64x64 raw palette-indexed pixel data for the
        named flat.  Returns a checkerboard pattern if not found.
        """
        idx = self.get_flat_index(name)
        if idx == -1:
            # Generate a 64x64 checkerboard
            data = bytearray(4096)
            for y in range(64):
                for x in range(64):
                    data[y * 64 + x] = 255 if (x // 8 + y // 8) % 2 == 0 else 0
            return bytes(data)
        return self._flat_data[idx]

    def get_flat_by_index(self, index: int) -> bytes:
        """Return flat data by numeric index."""
        if index < 0 or index >= len(self._flat_data):
            return self.get_flat("__MISSING__")
        return self._flat_data[index]

    def resolve_map_flats(self, map_data: MapData) -> None:
        """
        Resolve flat name references in all sectors of *map_data* to
        numeric indices, updating ``sector.floor_pic`` and
        ``sector.ceiling_pic``.

        Call this after constructing a TextureManager and loading a map.
        """
        for sector in map_data.sectors:
            floor_idx = self.get_flat_index(sector.floor_pic_name)
            ceil_idx  = self.get_flat_index(sector.ceiling_pic_name)
            sector.floor_pic   = max(0, floor_idx)
            sector.ceiling_pic = max(0, ceil_idx)

    def resolve_map_textures(self, map_data: MapData) -> None:
        """
        Resolve texture name references in all sidedefs of *map_data* to
        numeric indices (or -1 for "-"), updating ``side.top_texture``,
        ``side.mid_texture``, and ``side.bottom_texture``.

        Call this after constructing a TextureManager and loading a map.
        """
        for side in map_data.sides:
            side.top_texture    = self.get_texture_index(side.top_texture_name)
            side.mid_texture    = self.get_texture_index(side.mid_texture_name)
            side.bottom_texture = self.get_texture_index(side.bottom_texture_name)
