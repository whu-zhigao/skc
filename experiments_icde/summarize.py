#!/usr/bin/env python3
"""Collects all experiment JSONs into EXPERIMENT_MANIFEST.json for the
paper-number verification pipeline."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, write_json


def main():
    manifest = []
    for p in sorted(RESULTS.rglob("*.json")):
        if p.name in ("EXPERIMENT_MANIFEST.json",):
            continue
        if p.name.startswith("parallel_manifest_"):
            continue
        try:
            obj = json.loads(p.read_text())
        except Exception:
            obj = {}
        manifest.append({"file": str(p.relative_to(RESULTS)),
                         "mtime": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
            "size_b": p.stat().st_size,
            "top_keys": sorted(obj.keys())[:12]})
    write_json("EXPERIMENT_MANIFEST.json",
               {"n_files": len(manifest), "files": manifest})


if __name__ == "__main__":
    main()
