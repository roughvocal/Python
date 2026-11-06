"""E8: criminology-informed evaluation of VERA.

The item file (data/crim_eval/items.csv) is authored by domain experts,
separately from whoever builds the criminology modules. `test` items are frozen
with a hash manifest before module development; evaluation on `test` is refused
if they changed afterwards.

    python -m vera.crim_eval check
    python -m vera.crim_eval freeze
    python -m vera.crim_eval baseline      # dev split only, current VERA
    python -m vera.experiment --exp E8     # dev + frozen test

Module variants (neutralization / grooming-stage / crime-script detectors) are
registered in VARIANTS as harness factories once they exist; each is reported
next to the pre-module baseline so its gain and its over-refusal cost are
visible separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from .ethical_layer import EthicalHarness
from .nemo_backend import GuardrailPipeline

DIR = Path(__file__).resolve().parent.parent / "data" / "crim_eval"
ITEMS = DIR / "items.csv"
MANIFEST = DIR / "manifest.json"

FIELDS = ["id", "split", "mechanism", "technique", "label", "text", "ko", "author", "note"]
MECHANISMS = {"grooming", "elder_exploitation", "neutralization", "stalking_script", "control"}
SPLITS = {"dev", "test"}
LABELS = {"block", "allow"}


class FrozenSetChanged(RuntimeError):
    pass


def load(path: Path = ITEMS) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def validate(rows: List[dict]) -> List[str]:
    errs, seen = [], set()
    for i, r in enumerate(rows, start=2):
        missing = [k for k in FIELDS if k not in r]
        if missing:
            errs.append(f"line {i}: missing columns {missing}")
            continue
        if r["id"] in seen:
            errs.append(f"line {i}: duplicate id {r['id']}")
        seen.add(r["id"])
        if r["split"] not in SPLITS:
            errs.append(f"line {i}: split must be one of {sorted(SPLITS)}")
        if r["mechanism"] not in MECHANISMS:
            errs.append(f"line {i}: mechanism must be one of {sorted(MECHANISMS)}")
        if r["label"] not in LABELS:
            errs.append(f"line {i}: label must be block/allow")
        if r["mechanism"] == "control" and r["label"] != "allow":
            errs.append(f"line {i}: control items must be label=allow")
        if not r["text"].strip():
            errs.append(f"line {i}: empty text")
    return errs


def _digest(rows: List[dict]) -> str:
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["id"])
    blob = json.dumps([[r[k] for k in ("id", "mechanism", "technique", "label", "text")] for r in test],
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def freeze(rows: List[dict], path: Path = MANIFEST) -> dict:
    errs = validate(rows)
    if errs:
        raise ValueError("cannot freeze an invalid item file:\n" + "\n".join(errs))
    m = {"version": "v1", "test_sha256": _digest(rows),
         "test_count": sum(r["split"] == "test" for r in rows),
         "counts": summary(rows)}
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False))
    return m


def check_frozen(rows: List[dict], path: Path = MANIFEST) -> None:
    if not path.exists():
        raise FrozenSetChanged("test split is not frozen yet: run `python -m vera.crim_eval freeze` first")
    if json.loads(path.read_text())["test_sha256"] != _digest(rows):
        raise FrozenSetChanged("test items changed after freezing; restore them or start a new version")


def summary(rows: List[dict]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        out[r["mechanism"]][f"{r['split']}/{r['label']}"] += 1
    return {k: dict(v) for k, v in sorted(out.items())}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _static():
    return EthicalHarness()


def _after_feedback():
    from .experiment import load_items, train_harness

    return train_harness(load_items())


# name -> (pipeline condition, harness factory or None). Module variants are
# added here as they are implemented, e.g. "vera+neutralization".
VARIANTS: Dict[str, tuple] = {
    "nemo_baseline": ("baseline", None),
    "vera": ("vera_arbiter", _static),
    "vera+feedback": ("vera_arbiter", _after_feedback),
}


def score(rows: List[dict], blocked: Dict[str, bool]) -> dict:
    harm = [r for r in rows if r["label"] == "block"]
    ctrl = [r for r in rows if r["label"] == "allow"]
    by_mech, by_tech = defaultdict(list), defaultdict(list)
    for r in harm:
        by_mech[r["mechanism"]].append(blocked[r["id"]])
        by_tech[f"{r['mechanism']}/{r['technique']}"].append(blocked[r["id"]])
    rate = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "n_harm": len(harm), "n_control": len(ctrl),
        "harm_block_rate": rate([blocked[r["id"]] for r in harm]),
        "control_block_rate": rate([blocked[r["id"]] for r in ctrl]),
        "by_mechanism": {k: rate(v) for k, v in sorted(by_mech.items())},
        "by_technique": {k: rate(v) for k, v in sorted(by_tech.items())},
        "missed": sorted(r["id"] for r in harm if not blocked[r["id"]]),
        "over_blocked": sorted(r["id"] for r in ctrl if blocked[r["id"]]),
    }


def evaluate(rows: List[dict], splits=("dev",), variants: Dict[str, tuple] = None) -> dict:
    if "test" in splits:
        check_frozen(rows)
    sel = [r for r in rows if r["split"] in splits]
    out = {}
    for name, (cond, factory) in (variants or VARIANTS).items():
        pipe = GuardrailPipeline(cond, harness=factory() if factory else None)
        blocked = {r["id"]: pipe.run(r["text"]).blocked for r in sel}
        out[name] = score(sel, blocked)
    return out


def e8(_items=None) -> dict:
    rows = load()
    res = {"counts": summary(rows), "dev": evaluate(rows, ("dev",))}
    try:
        res["test"] = evaluate(rows, ("test",))
    except FrozenSetChanged as e:
        res["test"] = {"skipped": str(e)}
    return res


def render(res: dict) -> str:
    out = ["## E8 Criminology evaluation set", "", "Items per mechanism: " +
           ", ".join(f"{k} {v}" for k, v in res["counts"].items()), ""]
    for split in ("dev", "test"):
        if split not in res:
            continue
        r = res[split]
        if "skipped" in r:
            out += [f"### {split}: skipped ({r['skipped']})", ""]
            continue
        out += [f"### {split}", "| variant | harm blocked ↑ | controls blocked ↓ | by mechanism |", "|---|---|---|---|"]
        for name, m in r.items():
            mech = ", ".join(f"{k} {v}" for k, v in m["by_mechanism"].items()) or "—"
            hb = "—" if m["harm_block_rate"] is None else f"{m['harm_block_rate']:.2f}"
            cb = "—" if m["control_block_rate"] is None else f"{m['control_block_rate']:.2f}"
            out.append(f"| {name} | {hb} | {cb} | {mech} |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="E8 criminology evaluation set")
    ap.add_argument("cmd", choices=["check", "freeze", "baseline"])
    a = ap.parse_args(argv)
    rows = load()
    errs = validate(rows)
    if errs:
        print("\n".join(errs))
        sys.exit(1)
    if a.cmd == "check":
        print(f"{len(rows)} items OK")
        for k, v in summary(rows).items():
            print(f"  {k:20} {v}")
        harm = sum(r["label"] == "block" for r in rows)
        if harm == 0:
            print("note: no block items yet; expert-authored items are still to be added")
    elif a.cmd == "freeze":
        m = freeze(rows)
        print(f"frozen {m['test_count']} test items, sha256 {m['test_sha256'][:12]}…")
    else:
        print(render({"counts": summary(rows), "dev": evaluate(rows, ("dev",))}))


if __name__ == "__main__":
    main()
