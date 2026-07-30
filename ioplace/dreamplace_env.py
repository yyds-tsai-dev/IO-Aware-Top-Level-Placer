import os, sys

DEFAULT_ROOT = "/nashome/NVL4/vdalab/yyds-dev/DREAMPlace"

def setup_dreamplace(root=None):
    root = root or os.environ.get("DREAMPLACE_ROOT", DEFAULT_ROOT)
    for p in (os.path.join(root, "install"), os.path.join(root, "install", "dreamplace")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root
