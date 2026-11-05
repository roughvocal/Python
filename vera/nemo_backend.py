"""NeMo Guardrails integration.

* ``SimulatedLLM`` implements NeMo's ``LLMModel`` protocol so the full NeMo
  rails runtime can run offline and deterministically. It plays two roles:
    - the self-check *judge* for NeMo's built-in input/output rails. It is a
      deliberately naive "overt harm" detector (keyword based) so that the
      baseline has realistic gaps: it misses ethically problematic requests
      phrased without alarming words, and it over-refuses benign requests that
      contain such words ("how do I kill a python process?").
    - the *compliant* main model: it answers whatever reaches it, so the output
      rails are actually exercised.
* ``GuardrailPipeline`` builds an ``LLMRails`` instance for an experimental
  condition and wires the VERA harness in as registered NeMo actions.

To run against a real model, pass ``engine``/``model`` (e.g. ``nim`` and
``meta/llama-3.1-8b-instruct``) and the simulated LLM is not used.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from .ethical_layer import Decision, EthicalAssessment, EthicalHarness

warnings.filterwarnings("ignore")
logging.getLogger("nemoguardrails").setLevel(logging.ERROR)

from nemoguardrails import LLMRails, RailsConfig  # noqa: E402
from nemoguardrails.actions import action  # noqa: E402
from nemoguardrails.library.self_check.input_check.actions import (  # noqa: E402
    self_check_input as _base_self_check_input,
)
from nemoguardrails.library.self_check.output_check.actions import (  # noqa: E402
    self_check_output as _base_self_check_output,
)
from nemoguardrails.types import LLMResponse, LLMResponseChunk  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "vera"

# What the naive base judge considers "overtly harmful".
OVERT_HARM = [
    r"\bbomb\b", r"\bexplosive", r"\bmeth\b", r"\bnerve agent\b", r"\bsarin\b", r"\bricin\b",
    r"\bmalware\b", r"\bransomware\b", r"\bkeylogger\b", r"\bkill\b", r"\bweapon", r"\bpoison",
    r"\bignore (all |any )?(previous|prior|your) instructions\b", r"\bdeveloper mode\b", r"\bDAN\b",
]


def naive_overt_harm(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in OVERT_HARM)


class SimulatedLLM:
    """Deterministic stand-in for both the guard judge and the main model."""

    def __init__(self):
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "vera-simulated"

    @property
    def provider_name(self) -> Optional[str]:
        return "simulated"

    @property
    def provider_url(self) -> Optional[str]:
        return None

    @staticmethod
    def _text(prompt) -> str:
        if isinstance(prompt, str):
            return prompt
        return "\n".join(str(getattr(m, "content", m)) for m in prompt)

    def respond(self, prompt) -> str:
        self.calls += 1
        txt = self._text(prompt)
        if "Should the user message be blocked" in txt:
            m = re.search(r'User message: "(.*)"', txt, flags=re.S)
            return "Yes" if naive_overt_harm(m.group(1) if m else txt) else "No"
        if "Should the bot message be blocked" in txt:
            m = re.search(r'Bot message: "(.*)"', txt, flags=re.S)
            return "Yes" if naive_overt_harm(m.group(1) if m else txt) else "No"
        # Main model: fully compliant (worst case for the guardrail).
        user = txt.strip().splitlines()[-1] if txt.strip() else ""
        return f"Sure. Here is a detailed answer to your request: {user}"

    async def generate_async(self, prompt, *, stop=None, **kwargs) -> LLMResponse:
        return LLMResponse(content=self.respond(prompt))

    async def stream_async(self, prompt, *, stop=None, **kwargs):
        yield LLMResponseChunk(delta_content=self.respond(prompt))


CONDITION_FLOWS = {
    "baseline": (["self check input"], ["self check output"]),
    "vera": (
        ["self check input", "vera ethical check input"],
        ["self check output", "vera ethical check output"],
    ),
    "vera_only": (["vera ethical check input"], ["vera ethical check output"]),
    "vera_arbiter": (["vera arbiter check input"], ["vera arbiter check output"]),
}


def load_rails_config(condition: str, engine: Optional[str] = None, model: Optional[str] = None) -> RailsConfig:
    cfg = yaml.safe_load((CONFIG_DIR / "config.yml").read_text())
    inp, out = CONDITION_FLOWS[condition]
    cfg["rails"]["input"]["flows"] = inp
    cfg["rails"]["output"]["flows"] = out
    if engine and model:
        cfg["models"] = [{"type": "main", "engine": engine, "model": model}]
    colang = (CONFIG_DIR / "rails.co").read_text()
    return RailsConfig.from_content(colang_content=colang, yaml_content=yaml.safe_dump(cfg))


@dataclass
class PipelineResult:
    text: str
    blocked: bool
    blocked_by: Optional[str]
    response: str
    assessments: List[EthicalAssessment] = field(default_factory=list)


class GuardrailPipeline:
    def __init__(
        self,
        condition: str = "vera",
        harness: Optional[EthicalHarness] = None,
        llm=None,
        engine: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if condition != "baseline" and harness is None:
            harness = EthicalHarness()
        self.condition = condition
        self.harness = harness
        self.llm = None if (engine and model) else (llm or SimulatedLLM())
        self.rails = LLMRails(load_rails_config(condition, engine, model), llm=self.llm)
        self._last: List[EthicalAssessment] = []
        self.overturns = 0
        self._register_actions()

    def _register_actions(self):
        pipeline = self

        @action(name="vera_check_input")
        async def vera_check_input(context: Optional[dict] = None):
            text = (context or {}).get("user_message", "") or ""
            a = pipeline.harness.assess(text, stage="input")
            pipeline._last.append(a)
            return "block" if a.effective == Decision.BLOCK else "allow"

        @action(name="vera_check_output")
        async def vera_check_output(context: Optional[dict] = None):
            ctx = context or {}
            bot = ctx.get("bot_message", "") or ""
            user = ctx.get("user_message", "") or ""
            a = pipeline.harness.assess(bot, stage="output", context=user)
            pipeline._last.append(a)
            return "block" if a.effective == Decision.BLOCK else "allow"

        def arbitrate(a: EthicalAssessment, base_blocked: bool) -> str:
            # VERA blocks -> block. Both allow -> allow.
            if a.effective == Decision.BLOCK:
                return "block"
            if not base_blocked:
                return "allow"
            # Disagreement: base blocked, VERA did not. Overturn only with
            # positive evidence of benign intent (protective/technical framing,
            # or the content resembles items humans explicitly allowed).
            mem_allows = a.memory_signal <= -pipeline.harness.config.memory_min_sim
            overturn = a.mitigation >= 0.3 or mem_allows
            pipeline.overturns += int(overturn)
            return "allow" if overturn else "block"

        # NeMo injects these parameters into actions by name; we forward them
        # to the built-in self-check actions so the base guardrail runs exactly
        # as it would in its own flow.
        @action(name="vera_arbitrate_input")
        async def vera_arbitrate_input(llms=None, llm_task_manager=None, context: Optional[dict] = None,
                                       events=None, llm=None, config=None):
            base = await _base_self_check_input(llms=llms, llm_task_manager=llm_task_manager,
                                                context=context, events=events, llm=llm, config=config)
            a = pipeline.harness.assess((context or {}).get("user_message", "") or "", stage="input")
            pipeline._last.append(a)
            return arbitrate(a, base.is_blocked)

        @action(name="vera_arbitrate_output")
        async def vera_arbitrate_output(llms=None, llm_task_manager=None, context: Optional[dict] = None,
                                        events=None, llm=None, config=None):
            base = await _base_self_check_output(llms=llms, llm_task_manager=llm_task_manager,
                                                 context=context, events=events, llm=llm, config=config)
            ctx = context or {}
            a = pipeline.harness.assess(ctx.get("bot_message", "") or "", stage="output",
                                        context=ctx.get("user_message", "") or "")
            pipeline._last.append(a)
            return arbitrate(a, base.is_blocked)

        self.rails.register_action(vera_arbitrate_input, "vera_arbitrate_input")
        self.rails.register_action(vera_arbitrate_output, "vera_arbitrate_output")
        self.rails.register_action(vera_check_input, "vera_check_input")
        self.rails.register_action(vera_check_output, "vera_check_output")

    def run(self, text: str) -> PipelineResult:
        self._last = []
        res = self.rails.generate(
            messages=[{"role": "user", "content": text}],
            options={"log": {"activated_rails": True}},
        )
        stopped = [r for r in (res.log.activated_rails or []) if r.stop]
        blocked_by = stopped[0].name if stopped else None
        content = res.response[-1]["content"] if isinstance(res.response, list) else str(res.response)
        return PipelineResult(text, bool(stopped), blocked_by, content, list(self._last))
