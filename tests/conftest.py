import os
import sys

# Add repo root to sys.path so `import memory` (and peers) works everywhere
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ROOT = os.path.abspath(os.path.join(ROOT, '..'))  # go up two levels from tests/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)