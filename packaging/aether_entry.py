"""จุดเริ่มของ build ด้วย PyInstaller (เทียบเท่า ``python -m aether``)"""

import sys

from aether.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
