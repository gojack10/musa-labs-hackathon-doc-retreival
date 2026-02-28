"""Shared constants and data structures for Python Doom."""
import math
from dataclasses import dataclass, field
from typing import Optional

# Screen dimensions (original Doom resolution)
SCREENWIDTH = 640
SCREENHEIGHT = 400
ORIGINAL_WIDTH = 320
ORIGINAL_HEIGHT = 200

# Fixed-point (16.16) - we use Python floats but need these for WAD parsing
FRACBITS = 16
FRACUNIT = 1 << FRACBITS

# Angle constants (Binary Angle Measurement - 32-bit unsigned)
ANG45 = 0x20000000
ANG90 = 0x40000000
ANG180 = 0x80000000
ANG270 = 0xC0000000
ANG360 = 0x100000000  # wraps to 0
ANGLE_MAX = 0xFFFFFFFF

# Fine angles for lookup tables
FINEANGLES = 8192
FINEMASK = FINEANGLES - 1
ANGLETOFINESHIFT = 19

# Slope range for R_PointToAngle
SLOPERANGE = 2048
SLOPEBITS = 11
DBITS = FRACBITS - SLOPEBITS

# Game timing
TICRATE = 35

# Map lump ordering
ML_LABEL = 0
ML_THINGS = 1
ML_LINEDEFS = 2
ML_SIDEDEFS = 3
ML_VERTEXES = 4
ML_SEGS = 5
ML_SSECTORS = 6
ML_NODES = 7
ML_SECTORS = 8
ML_REJECT = 9
ML_BLOCKMAP = 10

# LineDef flags
ML_BLOCKING = 1
ML_BLOCKMONSTERS = 2
ML_TWOSIDED = 4
ML_DONTPEGTOP = 8
ML_DONTPEGBOTTOM = 16
ML_SECRET = 32
ML_SOUNDBLOCK = 64
ML_DONTDRAW = 128
ML_MAPPED = 256

# BSP node child indicator
NF_SUBSECTOR = 0x8000

# Silhouette flags
SIL_NONE = 0
SIL_BOTTOM = 1
SIL_TOP = 2
SIL_BOTH = 3

MAXDRAWSEGS = 256

# Player constants
PLAYER_HEIGHT = 41  # 41 map units (original: 41 * FRACUNIT)
PLAYER_VIEWHEIGHT = 41
PLAYER_RADIUS = 16
PLAYER_MAX_MOVE = 30
GRAVITY = 1.0

# Generate lookup tables using math
def _generate_tables():
    """Generate sine, cosine, tangent, and tantoangle lookup tables."""
    sine = [0] * (5 * FINEANGLES // 4)
    for i in range(5 * FINEANGLES // 4):
        angle = (i * 2.0 * math.pi) / FINEANGLES
        sine[i] = int(math.sin(angle) * FRACUNIT)

    # cosine is sine shifted by PI/2 (FINEANGLES/4)
    cosine = sine[FINEANGLES // 4: FINEANGLES // 4 + FINEANGLES]
    # pad if needed
    while len(cosine) < FINEANGLES:
        cosine.append(0)

    tangent = [0] * (FINEANGLES // 2)
    for i in range(FINEANGLES // 2):
        angle = (i - FINEANGLES / 4 + 0.5) * math.pi * 2.0 / FINEANGLES
        t = math.tan(angle)
        tangent[i] = max(-2**31, min(2**31 - 1, int(t * FRACUNIT)))

    tantoangle = [0] * (SLOPERANGE + 1)
    for i in range(SLOPERANGE + 1):
        a = math.atan2(i, SLOPERANGE) * 0x80000000 / math.pi
        tantoangle[i] = int(a) & ANGLE_MAX

    return sine, cosine, tangent, tantoangle

finesine, finecosine, finetangent, tantoangle = _generate_tables()


def fixed_mul(a: int, b: int) -> int:
    return (a * b) >> FRACBITS

def fixed_div(a: int, b: int) -> int:
    if b == 0:
        return 0x7FFFFFFF if a >= 0 else -0x7FFFFFFF
    if (abs(a) >> 14) >= abs(b):
        return 0x7FFFFFFF if (a ^ b) >= 0 else -0x7FFFFFFF
    return int((a << FRACBITS) / b)

def float_to_fixed(f: float) -> int:
    return int(f * FRACUNIT)

def fixed_to_float(f: int) -> float:
    return f / FRACUNIT

def angle_to_fine(angle: int) -> int:
    """Convert BAM angle to fine angle index."""
    return (angle >> ANGLETOFINESHIFT) & FINEMASK


# ─── Map data structures ───

@dataclass
class Vertex:
    x: float
    y: float

@dataclass
class Sector:
    floor_height: float
    ceiling_height: float
    floor_pic: int  # flat index
    ceiling_pic: int  # flat index
    light_level: int
    special: int
    tag: int
    # Runtime
    floor_pic_name: str = ""
    ceiling_pic_name: str = ""
    lines: list = field(default_factory=list)
    thinglist: list = field(default_factory=list)

@dataclass
class Side:
    texture_offset: float
    row_offset: float
    top_texture: int  # -1 = no texture
    bottom_texture: int  # -1 = no texture
    mid_texture: int  # -1 = no texture
    sector: Optional['Sector'] = None
    # Names for lookup
    top_texture_name: str = "-"
    bottom_texture_name: str = "-"
    mid_texture_name: str = "-"

@dataclass
class Line:
    v1: Vertex
    v2: Vertex
    dx: float = 0
    dy: float = 0
    flags: int = 0
    special: int = 0
    tag: int = 0
    sidenum: list = field(default_factory=lambda: [-1, -1])
    front_sector: Optional[Sector] = None
    back_sector: Optional[Sector] = None
    # Bounding box [top, bottom, left, right]
    bbox: list = field(default_factory=lambda: [0, 0, 0, 0])

@dataclass
class Seg:
    v1: Vertex
    v2: Vertex
    offset: float
    angle: int  # BAM angle
    sidedef: Optional[Side] = None
    linedef: Optional[Line] = None
    front_sector: Optional[Sector] = None
    back_sector: Optional[Sector] = None

@dataclass
class Subsector:
    sector: Optional[Sector] = None
    num_lines: int = 0
    first_line: int = 0

@dataclass
class Node:
    x: float
    y: float
    dx: float
    dy: float
    bbox: list = field(default_factory=lambda: [[[0]*4], [[0]*4]])
    children: list = field(default_factory=lambda: [0, 0])

@dataclass
class MapThing:
    x: float
    y: float
    angle: int
    type: int
    options: int

@dataclass
class MapData:
    """Holds all loaded map data."""
    vertices: list = field(default_factory=list)
    sectors: list = field(default_factory=list)
    sides: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    segs: list = field(default_factory=list)
    subsectors: list = field(default_factory=list)
    nodes: list = field(default_factory=list)
    things: list = field(default_factory=list)
    blockmap: Optional[object] = None
    reject: Optional[bytes] = None
