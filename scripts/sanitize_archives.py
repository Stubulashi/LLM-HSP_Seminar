"""Strip .env from the local packages and repack them (a safety-maintenance script).

Background: HSP_cloud.zip / HSP_cloud.tar.gz once shipped the .env file (with API keys) inside the package.
This script copies every entry except .env, writes a temporary file and atomically replaces the original archive.

Usage (from the repo root or anywhere):
    python -X utf8 scripts/sanitize_archives.py
"""
from __future__ import annotations

import os
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ZIP_TARGETS = ["HSP_cloud.zip"]
TAR_TARGETS = ["HSP_cloud.tar.gz"]


def _is_env(name: str) -> bool:
    return Path(name).name == ".env"


def sanitize_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as zin:
        names = zin.namelist()
        skipped = [n for n in names if _is_env(n)]
        if not skipped:
            print(f"[skip] {path.name}: no .env")
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if _is_env(item.filename):
                    continue
                zout.writestr(item, zin.read(item.filename))
    os.replace(tmp, path)
    print(f"[ok] {path.name}: stripped {len(skipped)} .env entries; items {len(names)} -> {len(names) - len(skipped)}")


def sanitize_tar(path: Path) -> None:
    with tarfile.open(path, "r:gz") as tin:
        members = tin.getmembers()
        skipped = [m.name for m in members if _is_env(m.name)]
        if not skipped:
            print(f"[skip] {path.name}: no .env")
            return
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tarfile.open(tmp, "w:gz") as tout:
            for m in members:
                if _is_env(m.name):
                    continue
                fileobj = tin.extractfile(m) if m.isfile() else None
                tout.addfile(m, fileobj)
    os.replace(tmp, path)
    print(f"[ok] {path.name}: stripped {len(skipped)} .env members; members {len(members)} -> {len(members) - len(skipped)}")


def main() -> int:
    for name in ZIP_TARGETS + TAR_TARGETS:
        p = ROOT / name
        if not p.exists():
            print(f"[miss] {name}")
            continue
        if p.suffix == ".zip":
            sanitize_zip(p)
        else:
            sanitize_tar(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
