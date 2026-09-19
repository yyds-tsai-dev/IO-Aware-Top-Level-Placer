import os, sys
from pathlib import Path

from ioplace.paths import REPO_ROOT as _REPO
_SUBMODULE = _REPO / "third_party" / "DREAMPlace"
DEFAULT_ROOT = str(_SUBMODULE if (_SUBMODULE / "install/dreamplace/PlaceDB.py").is_file()
                   else _REPO.parent / "DREAMPlace")

def setup_dreamplace(root=None):
    root = root or os.environ.get("DREAMPLACE_ROOT", DEFAULT_ROOT)
    for p in (os.path.join(root, "install"), os.path.join(root, "install", "dreamplace")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
