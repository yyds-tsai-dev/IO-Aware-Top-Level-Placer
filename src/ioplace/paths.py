"""Locations shared by repository tools and legacy source snapshots."""
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_ROOT.parent if SOURCE_ROOT.name == "src" else SOURCE_ROOT
