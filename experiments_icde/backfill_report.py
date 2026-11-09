#!/usr/bin/env python3
"""Backfill reporter: paper slot -> results coverage + values.

Reads the machine map (first ```json fence) from BACKFILL_MAP.md, walks
the results tree, and for every slot reports found/missing, extracts
the value(s), and aggregates multi-seed files to mean +/- std. Emits
BACKFILL_REPORT.md + BACKFILL_REPORT.json next to the results root.

Runs green with ZERO results present (everything MISSING, clean report)
-- that is the pre-run state on a fresh node.

Usage:
    python3 backfill_report.py                     # defaults below
    python3 backfill_report.py --results-root DIR --map FILE \\
        --out-dir DIR --quiet
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parent
DEFAULT_MAP = SUITE / ".." / ".." / "paper" / "BACKFILL_MAP.md"
try:                                   # suite convention: common.RESULTS
    sys.path.insert(0, str(SUITE))
    from common import RESULTS as DEFAULT_RESULTS
except Exception:                      # noqa: BLE001  (numpy absent etc.)
    DEFAULT_RESULTS = SUITE / ".." / ".." / ".." / "code" / "results" \
        / "systems"


def load_map(path: Path) -> dict:
    txt = path.read_text()
    m = re.search(r"```json\s*(\{.*?\})\s*```", txt, re.S)
    if not m:
        sys.exit(f"[backfill] no ```json fence in {path}")
    return json.loads(m.group(1))


def resolve(doc, key, filt=None, by=None):
    """Dotted-path resolver. 'a.b[].c' iterates list b; optional filter
    {field, equals} selects items; optional 'by' splits items into
    {by_value: item}. Returns (ok, value)."""
    if not key:
        return True, doc
    cur = doc
    for part in key.split("."):
        if part.endswith("[]"):
            name = part[:-2]
            if not isinstance(cur, dict) or name not in cur:
                return False, None
            cur = cur[name]
            if not isinstance(cur, list):
                return False, None
            if by:                      # split by a sibling field
                bfield = by.split(".")[-1][:-2]
                out = {}
                for item in cur:
                    if isinstance(item, dict) and bfield in item:
                        out[str(item[bfield])] = item
                cur = out
                continue
            if filt:
                sel = [i for i in cur
                       if isinstance(i, dict)
                       and i.get(filt["field"]) == filt["equals"]]
                if not sel:
                    return False, None
                cur = sel[0]
            else:
                return False, None      # bare [] without filter/by
            continue
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def agg_values(values):
    """mean +/- std over numeric values; else return structure."""
    nums = [v for v in values if is_num(v)]
    if nums and len(nums) == len(values) and len(values) > 1:
        mean = sum(nums) / len(nums)
        var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
        return f"{mean:.4g} ± {math.sqrt(var):.3g} (n={len(nums)})"
    if len(values) == 1:
        v = values[0]
        return v if is_num(v) or isinstance(v, str) else \
            json.dumps(v, ensure_ascii=False)[:160]
    return f"{len(values)} non-scalar values " \
           f"(e.g. {json.dumps(values[0], ensure_ascii=False)[:120]})"


def fmt(v):
    if v is None:
        return "--"
    if is_num(v):
        return f"{v:.4g}"
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False)[:160]


def main():
    ap = argparse.ArgumentParser(
        description="paper-slot coverage report over results/systems")
    ap.add_argument("--map", default=str(DEFAULT_MAP),
                    help="BACKFILL_MAP.md path (default: repo "
                         "icde/paper/BACKFILL_MAP.md)")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS),
                    help="results/systems root to walk")
    ap.add_argument("--out-dir", default=None,
                    help="where to write BACKFILL_REPORT.{md,json} "
                         "(default: results root)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the summary line")
    args = ap.parse_args()

    map_path = Path(args.map).resolve()
    root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root
    slots = load_map(map_path)["slots"]

    rows = []
    for s in slots:
        row = {"slot": s["id"], "tex": s.get("tex", ""),
               "status": s.get("status", ""), "exp": s.get("exp"),
               "value": None, "values_by": None, "n_files": 0,
               "sources": [], "note": s.get("note", "")}
        if s.get("exp") is None:
            row["found"] = False
            rows.append(row)
            continue
        files = sorted(glob.glob(str(root / s["glob"]))
                       ) if s.get("glob") else []
        files = [f for f in files if "manifest" not in Path(f).name]
        row["n_files"] = len(files)
        row["sources"] = [str(Path(f).relative_to(root)) for f in files]
        if not files:
            row["found"] = False
            rows.append(row)
            continue
        row["found"] = True
        if s.get("kind") == "figure":
            rows.append(row)
            continue
        vals, by_map, misses = [], {}, 0
        for f in files:
            try:
                doc = json.loads(Path(f).read_text())
            except Exception:                        # noqa: BLE001
                misses += 1
                continue
            ok, v = resolve(doc, s.get("key"), s.get("filter"), s.get("by"))
            if not ok:
                misses += 1
                continue
            if s.get("by") and isinstance(v, dict):
                for k2, item in v.items():
                    leaf_ok, leaf = resolve(
                        item, s["key"].split(".")[-1])
                    if leaf_ok:
                        by_map.setdefault(k2, []).append(leaf)
            elif isinstance(v, list):
                vals.append(v)          # keep whole series
            else:
                vals.append(v)
        if by_map:
            row["values_by"] = {k2: agg_values(v2)
                                for k2, v2 in sorted(by_map.items())}
        if vals:
            if all(isinstance(v, list) for v in vals):
                row["value"] = f"series len={len(vals[0])} " \
                               f"(e.g. {json.dumps(vals[0])[:120]})"
            else:
                row["value"] = agg_values(vals)
        if misses:
            row["note"] = (row["note"] + "; " if row["note"] else "") + \
                f"{misses} file(s) lacked the key"
        rows.append(row)

    n_found = sum(1 for r in rows if r.get("found"))
    n_missing = len(rows) - n_found
    summary = (f"{len(rows)} slots: {n_found} found, {n_missing} missing "
               f"(results root: {root})")

    out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir / "BACKFILL_REPORT.json").write_text(json.dumps(
        {"summary": summary, "map": str(map_path),
         "results_root": str(root), "slots": rows}, indent=2))

    lines = [f"# BACKFILL_REPORT — {summary}", "",
             f"map: `{map_path}`", "",
             "| slot | tex | status | found | value | n_files | sources |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        val = r.get("value")
        if r.get("values_by"):
            val = "; ".join(f"{k}={v}" for k, v in r["values_by"].items())
        src = ", ".join(r["sources"][:2]) + \
            (f" +{len(r['sources']) - 2}" if len(r["sources"]) > 2 else "")
        lines.append(f"| `{r['slot']}` | {r['tex']} | {r['status']} | "
                     f"{'YES' if r.get('found') else 'no'} | "
                     f"{fmt(val) if not isinstance(val, str) else val} | "
                     f"{r['n_files']} | {src} |")
    Path(out_dir / "BACKFILL_REPORT.md").write_text("\n".join(lines) + "\n")

    if not args.quiet:
        print("\n".join(lines))
        print(f"\nwrote {out_dir}/BACKFILL_REPORT.md and .json")
    else:
        print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
