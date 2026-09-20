"""One-command packaging of the minimal file set for uploading to AutoDL (including data/annotated).

Usage (works on local Windows/Linux):
    python -X utf8 scripts/pack_for_cloud.py            # writes ./HSP_cloud.zip
    python -X utf8 scripts/pack_for_cloud.py --out dist/HSP_cloud_32g.zip

Included: code/config/tests + data/annotated + data/manual_equivalence + .env (if present)
Excluded: results logs datasets docs data/raw_stories *.pdf .git __pycache__ etc. (rebuilt on the cloud / not needed).
After uploading to AutoDL: unzip HSP_cloud.zip -d /root/HSP
"""
import argparse, os, sys, zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOP_FILES = ["main.py", "requirements.txt", ".gitignore", ".env"]
TOP_DIRS = ["config", "analysis", "annotation", "data_manager", "experiment",
            "models", "utils", "visualization", "scripts", "tests"]
DATA_SUBDIRS = ["annotated", "manual_equivalence"]
EXCLUDE_DIR_PARTS = ("__pycache__", ".pytest_cache", ".git", "results", "logs",
                     "datasets", "docs", "raw_stories", "_bak", "_backup", "_smoke")
EXCLUDE_EXT = (".pyc", ".pdf", ".zip", ".tmp", ".npy")


def _iter_files(base: str, sub: str):
    full = os.path.join(base, sub)
    if not os.path.isdir(full):
        return
    for dirpath, dirnames, filenames in os.walk(full):
        dirnames[:] = [d for d in dirnames
                       if not any(x in d for x in EXCLUDE_DIR_PARTS)]
        for fn in filenames:
            if fn.endswith(EXCLUDE_EXT):
                continue
            rel = os.path.join(dirpath, fn)
            yield os.path.relpath(rel, base).replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "HSP_cloud.zip"))
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    entries = []
    for f in TOP_FILES:
        if os.path.exists(os.path.join(ROOT, f)):
            entries.append(f)
    for d in TOP_DIRS:
        entries.extend(_iter_files(ROOT, d))
    for sd in DATA_SUBDIRS:
        entries.extend(_iter_files(ROOT, os.path.join("data", sd)))

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(set(entries)):
            z.write(os.path.join(ROOT, rel), arcname=rel)
    size_mb = os.path.getsize(out) / 1e6
    print(f"[pack] wrote {out} ({size_mb:.1f} MB, {len(set(entries))} files)")
    print("[pack] after uploading to AutoDL, run:")
    print("    mkdir -p /root/HSP && unzip -o HSP_cloud.zip -d /root/HSP && cd /root/HSP")
    return 0


if __name__ == "__main__":
    sys.exit(main())
