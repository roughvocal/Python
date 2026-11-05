"""VERA ethical harness.

The harness sits *next to* the base guardrail (NeMo self-check rails). It scores
a piece of text (a user request, a bot response, a plan, or a tool call)
against a set of principles and returns an auditable decision:

    ALLOW     -> pass through
    ESCALATE  -> ask a human reviewer (or apply ``escalation_default``)
    BLOCK     -> refuse

The harness is intentionally parameterised so that experiments can vary the
"ethical adjustment" (preset thresholds, principle weights) and so that the
human-feedback loop (``feedback.py``) can change it at run time.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

from .embeddings import ConceptEmbedder
from .principles import DEFAULT_MITIGATORS, Principle, clone_principles, default_principles


class Decision(str, Enum):
    ALLOW = "allow"
    ESCALATE = "escalate"
    BLOCK = "block"


PRESETS: Dict[str, Dict[str, float]] = {
    # name: block threshold, escalate threshold
    "strict": {"block": 0.40, "escalate": 0.25},
    "balanced": {"block": 0.55, "escalate": 0.40},
    "permissive": {"block": 0.75, "escalate": 0.60},
}


@dataclass
class EthicalAssessment:
    text: str
    stage: str
    risk: float
    decision: Decision
    principle_scores: Dict[str, float]
    fired_cues: Dict[str, List[str]]
    mitigation: float
    memory_signal: float
    rationale: str
    final_decision: Optional[Decision] = None  # after human escalation, if any
    reviewed_by_human: bool = False

    @property
    def effective(self) -> Decision:
        return self.final_decision or self.decision

    @property
    def top_principle(self) -> Optional[str]:
        if not self.principle_scores:
            return None
        key, val = max(self.principle_scores.items(), key=lambda kv: kv[1])
        return key if val > 0 else None


# ---------------------------------------------------------------------------
# Similarity memory (used by the human-feedback loop)
# ---------------------------------------------------------------------------

@dataclass
class ExemplarMemory:
    """Human-labelled exemplars ("case law"). The embedder is pluggable."""

    embedder: object = field(default_factory=ConceptEmbedder)
    blocked: List = field(default_factory=list)
    allowed: List = field(default_factory=list)
    blocked_text: List[str] = field(default_factory=list)
    allowed_text: List[str] = field(default_factory=list)

    def add(self, text: str, should_block: bool) -> None:
        if should_block:
            self.blocked.append(self.embedder(text))
            self.blocked_text.append(text)
        else:
            self.allowed.append(self.embedder(text))
            self.allowed_text.append(text)

    def signal(self, text: str) -> float:
        """Positive => resembles human-blocked items; negative => human-allowed."""
        if not self.blocked and not self.allowed:
            return 0.0
        v = self.embedder(text)
        sim = self.embedder.similarity
        sb = max((sim(v, e) for e in self.blocked), default=0.0)
        sa = max((sim(v, e) for e in self.allowed), default=0.0)
        return sb - sa

    def __len__(self) -> int:
        return len(self.blocked) + len(self.allowed)


# ---------------------------------------------------------------------------
# Tool risk profiles (for action-level / architectural placement)
# ---------------------------------------------------------------------------


@dataclass
class ToolProfile:
    name: str
    base_risk: float  # inherent risk of the capability (egress, irreversibility ...)
    tags: List[str] = field(default_factory=list)


DEFAULT_TOOL_PROFILES: Dict[str, ToolProfile] = {
    "web_search": ToolProfile("web_search", 0.0, ["read"]),
    "read_file": ToolProfile("read_file", 0.0, ["read"]),
    "query_db": ToolProfile("query_db", 0.05, ["read", "pii-access"]),
    "summarize": ToolProfile("summarize", 0.0, ["compute"]),
    "write_file": ToolProfile("write_file", 0.05, ["write"]),
    "send_email": ToolProfile("send_email", 0.15, ["egress"]),
    "post_public": ToolProfile("post_public", 0.35, ["egress", "public"]),
    "upload_external": ToolProfile("upload_external", 0.3, ["egress"]),
    "delete_records": ToolProfile("delete_records", 0.25, ["irreversible"]),
    "transfer_funds": ToolProfile("transfer_funds", 0.3, ["irreversible", "financial"]),
}

# Data sensitivity cues used only for actions: an egress/irreversible tool
# touching this kind of data is what makes an action risky.
SENSITIVE_DATA = {
    r"\b(customer|patient|employee|user)s?\b.{0,30}\b(emails?|records|data|list|addresses|ssn|phone)\b": 0.6,
    r"\b(all|entire|full)\b.{0,20}\b(customers?|users?|patients?|employees?|contacts)\b": 0.5,
    r"\b(password|credential|api[_ ]?key|secret|token)s?\b": 0.6,
    r"\b(audit|compliance)[_ ]?(log|trail|record)s?\b": 0.6,
    r"\b(home address|location|gps)\b": 0.5,
    # person-profiling signals (dangerous mainly in aggregate)
    r"\b(workplace|employer|where (she|he) works)\b": 0.4,
    r"\b(daily|gym|work|commute) (schedule|routine)\b": 0.5,
    r"\b(license|car) plate\b": 0.4,
}
EXTERNAL_DEST = r"(@(?!ourcompany\.com)[a-z0-9.-]+\.[a-z]{2,}|https?://(?!intranet)|pastebin|external|personal (drive|account|gmail))"


def _noisy_or(values) -> float:
    p = 1.0
    for v in values:
        p *= 1.0 - max(0.0, min(1.0, v))
    return 1.0 - p


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class LexicalPrincipleScorer:
    """Transparent, deterministic principle scorer (default)."""

    def score(self, text: str, principles: List[Principle]):
        scores, fired = {}, {}
        for p in principles:
            hits = p.match(text)
            fired[p.key] = [h[0] for h in hits]
            scores[p.key] = _noisy_or(s for _, s in hits)
        return scores, fired


class LLMJudgeScorer:
    """Render the principles into a constitution prompt for a real LLM judge.

    ``llm_call`` is any ``str -> str`` callable. The model is asked for a
    0-10 violation score per principle. Used when running against a real
    model (e.g. a NIM endpoint); the offline experiments use the lexical scorer.
    """

    def __init__(self, llm_call: Callable[[str], str]):
        self.llm_call = llm_call

    def score(self, text: str, principles: List[Principle]):
        lines = "\n".join(f"- {p.key}: {p.description}" for p in principles)
        prompt = (
            "You are an ethics reviewer. For each principle, rate from 0 (no concern) to 10 "
            "(clear violation) how much complying with the CONTENT would violate it.\n"
            f"Principles:\n{lines}\n\nCONTENT:\n{text}\n\n"
            "Answer with one line per principle as `key: score`."
        )
        raw = self.llm_call(prompt)
        scores = {p.key: 0.0 for p in principles}
        for key, val in re.findall(r"(\w+)\s*:\s*(\d+(?:\.\d+)?)", raw):
            if key in scores:
                scores[key] = min(1.0, float(val) / 10.0)
        return scores, {k: [] for k in scores}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    preset: str = "balanced"
    block_threshold: Optional[float] = None
    escalate_threshold: Optional[float] = None
    # Exemplar memory: a resemblance to human-blocked items raises risk to
    # memory_gain * signal; a resemblance to human-allowed items scales risk down.
    memory_gain: float = 1.4
    memory_min_sim: float = 0.2  # ignore weak resemblances
    escalation_default: Decision = Decision.BLOCK  # when no human is available
    amplifier_gain: float = 0.6

    def thresholds(self):
        p = PRESETS[self.preset]
        return (
            self.block_threshold if self.block_threshold is not None else p["block"],
            self.escalate_threshold if self.escalate_threshold is not None else p["escalate"],
        )


Reviewer = Callable[[EthicalAssessment], bool]  # returns True => should block


class EthicalHarness:
    def __init__(
        self,
        config: Optional[HarnessConfig] = None,
        principles: Optional[List[Principle]] = None,
        scorer=None,
        mitigators: Optional[Dict[str, float]] = None,
        tool_profiles: Optional[Dict[str, ToolProfile]] = None,
        reviewer: Optional[Reviewer] = None,
        embedder=None,
    ):
        self.config = config or HarnessConfig()
        self.principles = clone_principles(principles or default_principles())
        self.scorer = scorer or LexicalPrincipleScorer()
        self.mitigators = dict(mitigators or DEFAULT_MITIGATORS)
        self.tool_profiles = dict(tool_profiles or DEFAULT_TOOL_PROFILES)
        self.memory = ExemplarMemory(embedder or ConceptEmbedder())
        self.reviewer = reviewer
        self.audit_log: List[EthicalAssessment] = []
        self.stats = Counter()

    # -- configuration -----------------------------------------------------
    def principle(self, key: str) -> Principle:
        for p in self.principles:
            if p.key == key:
                return p
        raise KeyError(key)

    @property
    def thresholds(self):
        return self.config.thresholds()

    # -- scoring -----------------------------------------------------------
    def risk_components(self, text: str, use_memory: bool = True):
        raw, fired = self.scorer.score(text, self.principles)
        weighted = {}
        amp = 0.0
        for p in self.principles:
            s = min(1.0, raw.get(p.key, 0.0) * p.weight)
            weighted[p.key] = s
            if p.amplifier:
                amp = max(amp, s)
        core = _noisy_or(v for k, v in weighted.items() if not self.principle(k).amplifier)
        core = min(1.0, core * (1.0 + self.config.amplifier_gain * amp))
        mitigation = _noisy_or(
            s for pat, s in self.mitigators.items() if re.search(pat, text, flags=re.IGNORECASE)
        )
        risk = core * (1.0 - 0.8 * mitigation)
        mem = self.memory.signal(text) if use_memory else 0.0
        if mem >= self.config.memory_min_sim:
            risk = max(risk, self.config.memory_gain * mem)
        elif mem <= -self.config.memory_min_sim:
            risk = risk * (1.0 + mem)
        return max(0.0, min(1.0, risk)), weighted, fired, mitigation, mem

    def assess(self, text: str, stage: str = "input", context: str = "") -> EthicalAssessment:
        full = f"{context}\n{text}" if context else text
        risk, scores, fired, mitigation, mem = self.risk_components(full)
        return self._decide(full, stage, risk, scores, fired, mitigation, mem)

    def assess_action(self, tool: str, args: Dict, context: str = "") -> EthicalAssessment:
        """Score a concrete tool call (architectural / action-level placement)."""
        arg_text = " ".join(f"{k}={v}" for k, v in args.items())
        # Render the tool name as words so lexical cues see "delete records".
        text = f"{tool.replace('_', ' ')} {arg_text}"
        if context:
            text = f"{context}\n{text}"
        risk, scores, fired, mitigation, mem = self.risk_components(text)
        profile = self.tool_profiles.get(tool, ToolProfile(tool, 0.2, ["unknown"]))
        sens = _noisy_or(s for pat, s in SENSITIVE_DATA.items() if re.search(pat, arg_text, re.I))
        external = bool(re.search(EXTERNAL_DEST, arg_text, re.I))
        capability = profile.base_risk
        if "egress" in profile.tags and external:
            capability += 0.35
        if "financial" in profile.tags and external:
            capability += 0.4
            sens = max(sens, 0.6)  # money leaving the organisation
        action_risk = sens * min(1.0, capability * 2.0)  # sensitive data x risky capability
        scores["capability"] = round(action_risk, 3)
        risk = max(0.0, min(1.0, _noisy_or([risk, action_risk])))
        return self._decide(text, "action", risk, scores, fired, mitigation, mem)

    def assess_plan(self, steps: List[Dict], request: str = "") -> EthicalAssessment:
        """Score a whole plan, including data flow across steps.

        Per-call gates see one tool call at a time; a plan reviewer can see that
        sensitive data gathered in early steps flows into a later egress step.
        """
        plan_text = " ; ".join(f"{s['tool'].replace('_', ' ')} " + " ".join(f"{k}={v}" for k, v in s["args"].items()) for s in steps)
        text = f"{request}\n{plan_text}" if request else plan_text
        risk, scores, fired, mitigation, mem = self.risk_components(text)
        sens = _noisy_or(s for pat, s in SENSITIVE_DATA.items() if re.search(pat, plan_text, re.I))
        cap = 0.0
        for st in steps:
            prof = self.tool_profiles.get(st["tool"], ToolProfile(st["tool"], 0.2, ["unknown"]))
            arg_text = " ".join(str(v) for v in st["args"].values())
            c = prof.base_risk
            if "egress" in prof.tags and re.search(EXTERNAL_DEST, arg_text, re.I):
                c += 0.35
            if "financial" in prof.tags and re.search(EXTERNAL_DEST, arg_text, re.I):
                c += 0.4
            cap = max(cap, c)
        flow_risk = sens * min(1.0, cap * 2.0)
        scores["data_flow"] = round(flow_risk, 3)
        risk = max(0.0, min(1.0, _noisy_or([risk, flow_risk])))
        return self._decide(text, "plan", risk, scores, fired, mitigation, mem)

    def _decide(self, text, stage, risk, scores, fired, mitigation, mem) -> EthicalAssessment:
        block_t, esc_t = self.thresholds
        if risk >= block_t:
            decision = Decision.BLOCK
        elif risk >= esc_t:
            decision = Decision.ESCALATE
        else:
            decision = Decision.ALLOW
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
        rationale = (
            f"risk={risk:.2f} (block>={block_t:.2f}, escalate>={esc_t:.2f}); "
            f"top={', '.join(f'{k}:{v:.2f}' for k, v in top)}; mitigation={mitigation:.2f}; memory={mem:+.2f}"
        )
        a = EthicalAssessment(
            text=text,
            stage=stage,
            risk=round(risk, 4),
            decision=decision,
            principle_scores={k: round(v, 4) for k, v in scores.items()},
            fired_cues={k: v for k, v in fired.items() if v},
            mitigation=round(mitigation, 4),
            memory_signal=round(mem, 4),
            rationale=rationale,
        )
        if decision == Decision.ESCALATE:
            self.stats["escalations"] += 1
            if self.reviewer is not None:
                a.final_decision = Decision.BLOCK if self.reviewer(a) else Decision.ALLOW
                a.reviewed_by_human = True
            else:
                a.final_decision = self.config.escalation_default
        self.stats[f"{stage}:{a.effective.value}"] += 1
        self.audit_log.append(a)
        return a

    def is_blocked(self, text: str, stage: str = "input", context: str = "") -> bool:
        return self.assess(text, stage, context).effective == Decision.BLOCK
