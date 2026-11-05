"""Experiment runner for the VERA guardrail harness.

Experiments
-----------
E1  harness architecture: baseline NeMo vs. VERA-only vs. stacked vs. arbiter
E2  ethical adjustment: VERA presets (strict / balanced / permissive)
E3  human feedback learning curve (rounds of reviewer feedback on the train
    pool, evaluated on the held-out test split), for stacked and arbiter
E4  feedback ablation: memory / weights / threshold channels alone and together
E5  annotator noise: robustness of the loop to wrong human labels
E6  inline human-in-the-loop: escalations routed to a reviewer at run time
E7  agent orchestration placement (see orchestration.py)

Usage:  python -m vera.experiment [--exp E1,E3] [--out results/run.json]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .ethical_layer import Decision, EthicalHarness, HarnessConfig
from .feedback import FeedbackConfig, FeedbackLoop, SimulatedReviewer
from .nemo_backend import GuardrailPipeline

DATA = Path(__file__).resolve().parent.parent / "data"
HARM_CATS = ("overt_harm", "jailbreak", "ethical_gray")


def load_items(path: Optional[Path] = None) -> List[dict]:
    path = path or DATA / "eval_prompts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(rows: List[dict]) -> Dict:
    """rows: dicts with label ('block'/'allow'), category, blocked (bool)."""
    tp = sum(r["blocked"] and r["label"] == "block" for r in rows)
    fn = sum((not r["blocked"]) and r["label"] == "block" for r in rows)
    fp = sum(r["blocked"] and r["label"] == "allow" for r in rows)
    tn = sum((not r["blocked"]) and r["label"] == "allow" for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["blocked"])
    return {
        "n": len(rows),
        "harm_block_rate": round(rec, 3),  # recall on harmful
        "benign_block_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,  # over-refusal (FPR)
        "precision": round(prec, 3),
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "accuracy": round((tp + tn) / len(rows), 3) if rows else 0.0,
        "block_rate_by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
    }


def evaluate(pipeline: GuardrailPipeline, items: List[dict]) -> Dict:
    rows = []
    t0 = time.time()
    for it in items:
        res = pipeline.run(it["text"])
        rows.append({**it, "blocked": res.blocked, "blocked_by": res.blocked_by})
    m = metrics(rows)
    m["blocked_by"] = dict(_count(r["blocked_by"] for r in rows if r["blocked"]))
    m["errors"] = [
        {"id": r["id"], "label": r["label"], "blocked_by": r["blocked_by"]}
        for r in rows if r["blocked"] != (r["label"] == "block")
    ]
    m["seconds"] = round(time.time() - t0, 2)
    return m


def _count(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


def split(items, name):
    return [i for i in items if i["split"] == name]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def e1_architecture(items):
    test = split(items, "test")
    out = {}
    for cond in ["baseline", "vera_only", "vera", "vera_arbiter"]:
        out[cond] = evaluate(GuardrailPipeline(cond), test)
    return out


def e2_presets(items):
    test = split(items, "test")
    out = {}
    for cond in ["vera", "vera_arbiter"]:
        for preset in ["strict", "balanced", "permissive"]:
            h = EthicalHarness(HarnessConfig(preset=preset))
            out[f"{cond}/{preset}"] = evaluate(GuardrailPipeline(cond, harness=h), test)
    return out


def run_feedback_curve(items, condition="vera_arbiter", rounds=4, batch=10, fb_config=None,
                       noise=0.0, seed=0, sampling="uncertainty"):
    """Give reviewer feedback on `batch` train items per round; evaluate on test."""
    train, test = split(items, "train"), split(items, "test")
    gold = {i["id"]: i["label"] == "block" for i in items}
    reviewer = SimulatedReviewer(gold, noise=noise, seed=seed)
    harness = EthicalHarness()
    loop = FeedbackLoop(harness, fb_config or FeedbackConfig())
    pipe = GuardrailPipeline(condition, harness=harness)
    rng = random.Random(seed)
    pool = list(train)
    rng.shuffle(pool)
    curve = [{"round": 0, "labels": 0, **_slim(evaluate(pipe, test)), "thresholds": list(harness.thresholds)}]
    for r in range(1, rounds + 1):
        if not pool:
            break
        if sampling == "uncertainty":
            # Reviewers look first at what the harness is least sure about.
            b_t, _ = harness.thresholds
            pool.sort(key=lambda it: abs(harness.risk_components(it["text"])[0] - b_t))
        chosen, pool = pool[:batch], pool[batch:]
        for it in chosen:
            a = harness.assess(it["text"], stage="input")
            loop.review(it["id"], a, reviewer.label(it["id"]), principle=it.get("principle"))
        loop.recalibrate()
        m = evaluate(pipe, test)
        curve.append({"round": r, "labels": reviewer.calls, **_slim(m), "thresholds": list(harness.thresholds),
                      "weights": {p.key: round(p.weight, 2) for p in harness.principles}})
    return curve


def _slim(m):
    return {k: m[k] for k in ("harm_block_rate", "benign_block_rate", "f1", "accuracy", "block_rate_by_category")}


def e3_feedback(items):
    return {cond: run_feedback_curve(items, cond) for cond in ["vera", "vera_arbiter"]}


def e4_ablation(items):
    variants = {
        "none": FeedbackConfig(use_memory=False, use_weights=False, use_threshold=False),
        "memory_only": FeedbackConfig(use_memory=True, use_weights=False, use_threshold=False),
        "weights_only": FeedbackConfig(use_memory=False, use_weights=True, use_threshold=False),
        "threshold_only": FeedbackConfig(use_memory=False, use_weights=False, use_threshold=True),
        "all": FeedbackConfig(),
    }
    return {name: run_feedback_curve(items, "vera_arbiter", fb_config=cfg)[-1] for name, cfg in variants.items()}


def e5_noise(items, seeds=(0, 1, 2, 3, 4)):
    out = {}
    for noise in [0.0, 0.1, 0.2, 0.3]:
        finals = [run_feedback_curve(items, "vera_arbiter", noise=noise, seed=s)[-1] for s in seeds]
        out[str(noise)] = {
            k: round(sum(f[k] for f in finals) / len(finals), 3)
            for k in ("harm_block_rate", "benign_block_rate", "f1", "accuracy")
        }
    return out


def train_harness(items, labels=10, seed=0, noise=0.0) -> EthicalHarness:
    """A harness after one round of (uncertainty-sampled) reviewer feedback."""
    gold = {i["id"]: i["label"] == "block" for i in items}
    reviewer = SimulatedReviewer(gold, noise=noise, seed=seed)
    h = EthicalHarness()
    loop = FeedbackLoop(h)
    pool = list(split(items, "train"))
    random.Random(seed).shuffle(pool)
    b_t, _ = h.thresholds
    pool.sort(key=lambda it: abs(h.risk_components(it["text"])[0] - b_t))
    for it in pool[:labels]:
        loop.review(it["id"], h.assess(it["text"]), reviewer.label(it["id"]), it.get("principle"))
    loop.recalibrate()
    h.audit_log.clear()
    h.stats.clear()
    return h


def e6_inline_hitl(items):
    """Escalations go to a human reviewer at run time instead of default-block."""
    test = split(items, "test")
    gold = {i["id"]: i["label"] == "block" for i in items}
    text_to_id = {i["text"]: i["id"] for i in test}
    out = {}
    for harness_kind in ("static", "trained"):
        for with_human in (False, True):
            reviewer = SimulatedReviewer(gold)
            h = EthicalHarness() if harness_kind == "static" else train_harness(items)
            h.reviewer = reviewer.as_inline_reviewer(text_to_id) if with_human else None
            m = evaluate(GuardrailPipeline("vera_arbiter", harness=h), test)
            key = f"arbiter/{harness_kind}/{'human' if with_human else 'default_block'}"
            out[key] = _slim(m) | {"escalations": h.stats["escalations"], "human_reviews": reviewer.calls}
    return out


def e7_orchestration(items):
    from .orchestration import run_placement_experiment

    return {
        "static": run_placement_experiment(EthicalHarness),
        "after_feedback": run_placement_experiment(lambda: train_harness(items)),
    }


EXPERIMENTS = {
    "E1": e1_architecture, "E2": e2_presets, "E3": e3_feedback, "E4": e4_ablation,
    "E5": e5_noise, "E6": e6_inline_hitl, "E7": e7_orchestration,
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _row(name, m, extra=()):
    cats = m.get("block_rate_by_category", {})
    cells = [name, f"{m['harm_block_rate']:.2f}", f"{m['benign_block_rate']:.2f}", f"{m['f1']:.2f}",
             f"{cats.get('ethical_gray', 0):.2f}", f"{cats.get('sensitive_benign', 0):.2f}"]
    cells += [str(m.get(e, "")) for e in extra]
    return "| " + " | ".join(cells) + " |"


HEAD = "| condition | harm blocked ↑ | benign blocked ↓ | F1 ↑ | gray-zone blocked ↑ | sensitive-benign blocked ↓ |"
SEP = "|---|---|---|---|---|---|"


def render_report(res: Dict) -> str:
    out = ["# VERA experiment report", ""]
    if "E1" in res:
        out += ["## E1 Harness architecture (test split)", HEAD, SEP]
        out += [_row(k, v) for k, v in res["E1"].items()] + [""]
    if "E2" in res:
        out += ["## E2 Ethical adjustment presets", HEAD, SEP]
        out += [_row(k, v) for k, v in res["E2"].items()] + [""]
    if "E3" in res:
        out += ["## E3 Human-feedback learning curve"]
        for cond, curve in res["E3"].items():
            out += [f"### {cond}", "| round | labels | harm blocked | benign blocked | F1 | block thr |",
                    "|---|---|---|---|---|---|"]
            for c in curve:
                out.append(f"| {c['round']} | {c['labels']} | {c['harm_block_rate']:.2f} | "
                           f"{c['benign_block_rate']:.2f} | {c['f1']:.2f} | {c['thresholds'][0]:.2f} |")
            out.append("")
    if "E4" in res:
        out += ["## E4 Feedback-channel ablation (arbiter, after all rounds)", HEAD, SEP]
        out += [_row(k, v) for k, v in res["E4"].items()] + [""]
    if "E5" in res:
        out += ["## E5 Annotator noise (arbiter, mean of 5 seeds)",
                "| label noise | harm blocked | benign blocked | F1 |", "|---|---|---|---|"]
        out += [f"| {k} | {v['harm_block_rate']:.2f} | {v['benign_block_rate']:.2f} | {v['f1']:.2f} |"
                for k, v in res["E5"].items()] + [""]
    if "E6" in res:
        out += ["## E6 Inline human-in-the-loop for escalations",
                HEAD + " escalations | human reviews |", SEP + "---|---|"]
        out += [_row(k, v, ("escalations", "human_reviews")) for k, v in res["E6"].items()] + [""]
    if "E7" in res:
        from .orchestration import render_placement_table

        out += ["## E7 Ethical-layer placement in agent orchestration"]
        for kind, r in res["E7"].items():
            out += [f"### harness: {kind}", render_placement_table(r), ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default=",".join(EXPERIMENTS), help="comma-separated, e.g. E1,E3")
    ap.add_argument("--out", default="results/latest.json")
    ap.add_argument("--report", default="results/REPORT.md")
    args = ap.parse_args(argv)
    items = load_items()
    res = {}
    for name in args.exp.split(","):
        t0 = time.time()
        res[name] = EXPERIMENTS[name](items)
        print(f"[{name}] done in {time.time() - t0:.1f}s")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=str))
    report = render_report(res)
    Path(args.report).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
