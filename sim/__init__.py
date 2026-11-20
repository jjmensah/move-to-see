# ==========================================================
# sim/__init__.py
# ==========================================================
# Experiment layer. Importing ``sim`` puts this checkout's ``src/`` first on ``sys.path`` so that
# ``python -m sim.<module>`` works without an editable install (and each git worktree uses its own code).

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
