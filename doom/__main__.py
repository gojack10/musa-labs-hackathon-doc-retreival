"""Entry point for `python -m doom [wad_path]`.

Usage:
    python -m doom
    python -m doom /path/to/doom1.wad

The default WAD path is doom-src/doom1.wad relative to the package root.
"""
import sys
import os


def main() -> None:
    # Default WAD: <repo_root>/doom-src/doom1.wad
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_wad = os.path.join(repo_root, "doom-src", "doom1.wad")

    wad_path = sys.argv[1] if len(sys.argv) > 1 else default_wad

    if not os.path.exists(wad_path):
        print(f"Error: WAD file not found: {wad_path}")
        print("Usage: python -m doom [/path/to/doom.wad]")
        sys.exit(1)

    from doom.game import Game

    game = Game(wad_path)
    game.run()


if __name__ == "__main__":
    main()
