"""Human-feedback loop for the VERA ethical harness.

Three update channels, each of which can be switched on/off so their effect
can be measured separately (ablation):

* ``memory``    - store human-labelled exemplars; similar future content is
                  pushed toward the human's label (few-shot / case law).
* ``weights``   - multiplicative update of principle weights: up on a missed
                  harm (false negative), down on an over-refusal (false positive).
* ``threshold`` - re-calibrate the block threshold on the accumulated feedback
                  (maximising F-beta, beta>1 favours catching harm).

A simulated reviewer (``SimulatedReviewer``) provides ground-truth labels with
an optional noise rate so the robustness of the loop to annotator error can
be studied.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .ethical_layer import Decision, EthicalAssessment, EthicalHarness


@dataclass
class HumanFeedback:
    item_id: str
    text: str
    stage: str
    harness_decision: str
    harness_risk: float
    human_says_block: bool
    principle: Optional[str] = None
    note: str = ""


@dataclass
class FeedbackConfig:
    use_memory: bool = True
    use_weights: bool = True
    use_threshold: bool = True
    # "all": every reviewed item becomes an exemplar; "disagreements": only
    # items where the human corrected the harness (or resolved an escalation).
    memory_policy: str = "all"
    lr: float = 0.25
    min_weight: float = 0.2
    max_weight: float = 3.0
    f_beta: float = 1.5
    escalate_margin: float = 0.15
    threshold_floor: float = 0.25


@dataclass
class FeedbackLoop:
    harness: EthicalHarness
    config: FeedbackConfig = field(default_factory=FeedbackConfig)
    history: List[HumanFeedback] = field(default_factory=list)
    weight_trace: List[Dict[str, float]] = field(default_factory=list)

    def record(self, fb: HumanFeedback) -> None:
        self.history.append(fb)
        h = self.harness
        blocked = fb.harness_decision == Decision.BLOCK.value
        false_neg = fb.human_says_block and not blocked
        false_pos = (not fb.human_says_block) and blocked

        if self.config.use_memory:
            disagreement = false_neg or false_pos or fb.harness_decision == Decision.ESCALATE.value
            if self.config.memory_policy == "all" or disagreement:
                h.memory.add(fb.text, fb.human_says_block)

        if self.config.use_weights and (false_neg or false_pos):
            raw, _ = h.scorer.score(fb.text, h.principles)
            targets = [fb.principle] if (false_neg and fb.principle) else [
                k for k, v in raw.items() if v > 0 and not h.principle(k).amplifier
            ]
            for key in targets:
                try:
                    p = h.principle(key)
                except KeyError:
                    continue
                factor = (1 + self.config.lr) if false_neg else (1 - self.config.lr)
                p.weight = max(self.config.min_weight, min(self.config.max_weight, p.weight * factor))
        self.weight_trace.append({p.key: round(p.weight, 3) for p in h.principles})

    def review(self, item_id: str, assessment: EthicalAssessment, human_says_block: bool,
               principle: Optional[str] = None) -> HumanFeedback:
        fb = HumanFeedback(
            item_id=item_id,
            text=assessment.text,
            stage=assessment.stage,
            harness_decision=assessment.decision.value,
            harness_risk=assessment.risk,
            human_says_block=human_says_block,
            principle=principle,
        )
        self.record(fb)
        return fb

    def recalibrate(self) -> Optional[float]:
        """Pick the block threshold that maximises F-beta on feedback so far."""
        if not self.config.use_threshold or len(self.history) < 4:
            return None
        h = self.harness
        # Memory is excluded here: it was built from these very items, so including
        # it would make them trivially separable (label leakage).
        risks = [(h.risk_components(fb.text, use_memory=False)[0], fb.human_says_block) for fb in self.history]
        if not any(y for _, y in risks) or all(y for _, y in risks):
            return None
        best_t, best_f = None, -1.0
        b2 = self.config.f_beta ** 2
        for i in range(5, 96):
            t = i / 100
            tp = sum(1 for r, y in risks if r >= t and y)
            fp = sum(1 for r, y in risks if r >= t and not y)
            fn = sum(1 for r, y in risks if r < t and y)
            denom = (1 + b2) * tp + b2 * fn + fp
            f = (1 + b2) * tp / denom if denom else 0.0
            # tie-break toward the middle of the plateau => higher t on ties
            if f > best_f + 1e-9 or (abs(f - best_f) < 1e-9 and best_t is not None and t - best_t <= 0.01):
                best_t, best_f = t, f
        best_t = max(self.config.threshold_floor, best_t)
        h.config.block_threshold = best_t
        h.config.escalate_threshold = max(0.05, best_t - self.config.escalate_margin)
        return best_t

    def save(self, path: str | Path) -> None:
        h = self.harness
        state = {
            "thresholds": list(h.thresholds),
            "weights": {p.key: p.weight for p in h.principles},
            "history": [asdict(f) for f in self.history],
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, harness: EthicalHarness, path: str | Path, config: Optional[FeedbackConfig] = None):
        state = json.loads(Path(path).read_text())
        harness.config.block_threshold, harness.config.escalate_threshold = state["thresholds"]
        for k, w in state["weights"].items():
            harness.principle(k).weight = w
        loop = cls(harness, config or FeedbackConfig())
        for f in state["history"]:
            fb = HumanFeedback(**f)
            loop.history.append(fb)
            if loop.config.use_memory:
                harness.memory.add(fb.text, fb.human_says_block)
        return loop


class SimulatedReviewer:
    """Stands in for a human annotator: returns the gold label, flipped with prob ``noise``."""

    def __init__(self, gold: Dict[str, bool], noise: float = 0.0, seed: int = 0):
        self.gold = gold
        self.noise = noise
        self.rng = random.Random(seed)
        self.calls = 0

    def label(self, item_id: str) -> bool:
        self.calls += 1
        y = self.gold[item_id]
        return (not y) if self.rng.random() < self.noise else y

    def as_inline_reviewer(self, text_to_id: Dict[str, str]):
        """Callable usable as ``EthicalHarness.reviewer`` for live escalations."""

        def _review(a: EthicalAssessment) -> bool:
            for text, iid in text_to_id.items():
                if text in a.text:
                    return self.label(iid)
            return True  # unknown content: fail closed

        return _review
