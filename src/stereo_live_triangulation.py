"""Root launcher for stereo live triangulation.

Allows running from project root:
    python stereo_live_triangulation.py --cam1 1 --cam2 2
"""

from scripts.stereo_live_triangulation import main


if __name__ == "__main__":
    main()
