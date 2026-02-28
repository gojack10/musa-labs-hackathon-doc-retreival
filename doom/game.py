"""Main game class for Python Doom port.

Implements the D_DoomLoop / G_Ticker pattern from linuxdoom-1.10:
  - Fixed 35 Hz game tic rate (TICRATE)
  - Input -> ticcmd building -> player update -> render
  - Mouse captured for mouselook (turning only, no vertical)
"""
import pygame
import sys
import time

from doom.defs import (
    SCREENWIDTH, SCREENHEIGHT,
    ANG90, ANG360, ANGLE_MAX,
    FRACBITS, FRACUNIT,
    TICRATE,
)
from doom.player import Player, ANGLETURN_SLOW, ANGLETURN_WALK, ANGLETURN_RUN


# Slow-turn acceleration threshold (mirrors Doom's SLOWTURNTICS = 6)
SLOWTURNTICS = 6


class Game:
    """Top-level game object.  Create once, then call game.run()."""

    def __init__(self, wad_path: str):
        pygame.init()

        # ── Display ──────────────────────────────────────────────────────────
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
        pygame.display.set_caption("DOOM - Python")

        # Show loading screen before any heavy work
        self._show_loading_screen()

        # ── WAD / map loading ────────────────────────────────────────────────
        # Import here so the loading screen is already visible
        from doom.wad import WAD, load_palette, load_colormap, load_map, TextureManager

        self.wad       = WAD(wad_path)
        self.palette   = load_palette(self.wad)
        self.colormap  = load_colormap(self.wad)
        self.textures  = TextureManager(self.wad)

        # Load E1M1
        self.map_data = load_map(self.wad, 1, 1)

        # Resolve texture/flat name references to numeric indices
        self.textures.resolve_map_textures(self.map_data)
        self.textures.resolve_map_flats(self.map_data)

        # ── Player start ─────────────────────────────────────────────────────
        # thing.type == 1 is the Player 1 start marker (mapthing_t)
        player_start = None
        for thing in self.map_data.things:
            if thing.type == 1:
                player_start = thing
                break

        if player_start is None:
            raise RuntimeError("No Player 1 start thing (type 1) found in E1M1!")

        # Locate starting sector via BSP
        start_sector = self._find_sector(player_start.x, player_start.y)

        # Convert degrees (0-359) to BAM:
        # P_SpawnPlayer uses:  mobj->angle = ANG45 * (mthing->angle / 45)
        bam_angle = (ANG90 * (player_start.angle // 45)) // 2  # ANG45 * (angle/45)
        # More precisely: ANG45 = ANG90/2, so ANG45*(deg/45) = (ANG90/2)*(deg/45)
        # Simplify: angle_bam = int(player_start.angle / 360.0 * ANG360) & ANGLE_MAX
        bam_angle = int(player_start.angle / 360.0 * ANG360) & ANGLE_MAX

        self.player = Player(
            x=float(player_start.x),
            y=float(player_start.y),
            angle=bam_angle,
            sector=start_sector,
        )

        # ── Renderer ─────────────────────────────────────────────────────────
        from doom.renderer import Renderer
        self.renderer = Renderer(
            SCREENWIDTH, SCREENHEIGHT,
            self.textures, self.palette, self.colormap,
        )

        # ── Timing ───────────────────────────────────────────────────────────
        self.clock = pygame.time.Clock()
        self.running = True

        # Mouse capture
        self.mouse_captured = True
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

        # Two-stage keyboard turn acceleration state (mirrors Doom's turnheld)
        self._turnheld = 0  # consecutive tics a turn key has been held

        # Font for HUD overlay (created once to avoid per-frame allocation)
        self._hud_font = pygame.font.SysFont("monospace", 16)

    # ── Loading screen ────────────────────────────────────────────────────────

    def _show_loading_screen(self) -> None:
        """White background with title text, shown before WAD loading."""
        self.screen.fill((255, 255, 255))

        font_large = pygame.font.SysFont("arial", 48, bold=True)
        font_small = pygame.font.SysFont("arial", 24)

        title = font_large.render("MUSA LABS THX FOR HOSTING :D", True, (0, 0, 0))
        title_rect = title.get_rect(center=(SCREENWIDTH // 2, SCREENHEIGHT // 2 - 30))
        self.screen.blit(title, title_rect)

        sub = font_small.render("Loading DOOM...", True, (100, 100, 100))
        sub_rect = sub.get_rect(center=(SCREENWIDTH // 2, SCREENHEIGHT // 2 + 40))
        self.screen.blit(sub, sub_rect)

        pygame.display.flip()
        time.sleep(2)  # Display for 2 seconds

    # ── Sector finder (BSP traversal) ─────────────────────────────────────────

    def _find_sector(self, x: float, y: float):
        """Return the Sector at map position (x, y) using BSP tree walk.
        x, y are in map units; BSP node coords are in fixed-point."""
        from doom.defs import NF_SUBSECTOR
        md = self.map_data

        if not md.nodes:
            if md.subsectors:
                return md.subsectors[0].sector
            return md.sectors[0] if md.sectors else None

        # Convert to fixed-point for BSP comparison
        fx = int(x) << FRACBITS
        fy = int(y) << FRACBITS

        node_id = len(md.nodes) - 1
        while True:
            if node_id & NF_SUBSECTOR:
                ss_idx = node_id & ~NF_SUBSECTOR
                if ss_idx < len(md.subsectors):
                    return md.subsectors[ss_idx].sector
                return None
            node = md.nodes[node_id]
            dx = fx - node.x
            dy = fy - node.y
            left  = dy * node.dx
            right = dx * node.dy
            side  = 0 if right < left else 1
            node_id = node.children[side]

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Main game loop.  Returns when the user quits."""
        # Fixed tic accumulator — runs game logic at exactly TICRATE Hz
        # while rendering as fast as vsync/display allows.
        MS_PER_TIC = 1000.0 / TICRATE  # ≈ 28.57 ms
        accumulator = 0.0

        while self.running:
            frame_ms = self.clock.tick(0)  # uncapped render rate, returns ms elapsed
            frame_ms = min(frame_ms, 200)  # safety clamp (e.g. after alt-tab)
            accumulator += frame_ms

            # ── Event processing (D_ProcessEvents) ───────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_TAB:
                        # Toggle mouse capture
                        self.mouse_captured = not self.mouse_captured
                        pygame.event.set_grab(self.mouse_captured)
                        pygame.mouse.set_visible(not self.mouse_captured)

            if not self.running:
                break

            # ── Build ticcmd inputs (G_BuildTiccmd) ──────────────────────────
            keys = pygame.key.get_pressed()
            running_speed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

            # Forward / backward
            forward = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                forward += 1.0
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                forward -= 1.0

            # Strafe
            strafe = 0.0
            if keys[pygame.K_a]:
                strafe -= 1.0
            if keys[pygame.K_d]:
                strafe += 1.0

            # Keyboard turning (two-stage accelerative, mirrors turnheld logic)
            turning_key = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]
            if turning_key:
                self._turnheld += 1
            else:
                self._turnheld = 0

            # Determine turn speed based on held duration
            if self._turnheld < SLOWTURNTICS:
                tspeed = ANGLETURN_SLOW
            elif running_speed:
                tspeed = ANGLETURN_RUN
            else:
                tspeed = ANGLETURN_WALK

            kb_turn = 0
            if keys[pygame.K_LEFT]:
                kb_turn += tspeed
            if keys[pygame.K_RIGHT]:
                kb_turn -= tspeed

            # Mouse turning: cmd->angleturn -= mousex * 0x8 (then <<16 inside player)
            mouse_turn = 0
            if self.mouse_captured:
                mx, _ = pygame.mouse.get_rel()
                mouse_turn = -(mx * 0x8)

            # Combined angleturn (will be <<16 inside Player.update)
            # We pass already-shifted value so Player just adds it directly.
            turn_bam = (kb_turn + mouse_turn) << 16

            # ── Game tic loop ─────────────────────────────────────────────────
            # Run as many tics as have accumulated (G_Ticker / P_Ticker)
            while accumulator >= MS_PER_TIC:
                self.player.update(
                    forward=forward,
                    strafe=strafe,
                    turn=turn_bam,
                    map_data=self.map_data,
                    running=bool(running_speed),
                )
                accumulator -= MS_PER_TIC
                # Only apply mouse delta once per rendered frame, not per tic,
                # to avoid double-counting when multiple tics fire per frame.
                turn_bam = (kb_turn) << 16  # mouse already consumed above

            # ── Render (D_Display) ────────────────────────────────────────────
            # Convert player coords from map units to fixed-point for renderer
            frame = self.renderer.render(
                int(self.player.x) << FRACBITS,
                int(self.player.y) << FRACBITS,
                int(self.player.viewz) << FRACBITS,
                self.player.angle,
                self.map_data,
            )
            self.screen.blit(frame, (0, 0))

            # ── HUD overlay ───────────────────────────────────────────────────
            self._draw_hud()

            pygame.display.flip()

        pygame.quit()
        sys.exit(0)

    def _draw_hud(self) -> None:
        """Render debug HUD: FPS, position, angle."""
        fps = self.clock.get_fps()
        angle_deg = self.player.angle * 360.0 / ANG360

        text = (
            f"FPS: {fps:5.1f}  "
            f"X: {self.player.x:7.1f}  "
            f"Y: {self.player.y:7.1f}  "
            f"Z: {self.player.z:6.1f}  "
            f"ANG: {angle_deg:5.1f}"
        )
        surf = self._hud_font.render(text, True, (255, 255, 0))
        # Drop-shadow for legibility
        shadow = self._hud_font.render(text, True, (0, 0, 0))
        self.screen.blit(shadow, (6, 6))
        self.screen.blit(surf,   (5, 5))
