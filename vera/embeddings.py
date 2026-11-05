"""Embedders for the human-feedback exemplar memory.

``ConceptEmbedder`` (default, offline): maps surface words onto a small hand
built concept vocabulary (e.g. grandma / aging father / seniors -> ``elder``)
plus content-word unigrams. It is a transparent stand-in for a sentence
embedding so that paraphrases can share features without network access.

``FastEmbedEmbedder``: dense sentence embeddings via FastEmbed (the backend
NeMo Guardrails uses, default ``all-MiniLM-L6-v2``). Use it when the model can
be downloaded; pass ``embedder=FastEmbedEmbedder()`` to ``EthicalHarness``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

CONCEPTS: Dict[str, List[str]] = {
    "elder": [r"grand(ma|mother|pa|father)", r"aging (father|mother|parent)", r"elderly", r"seniors?",
              r"dementia", r"alzheimer", r"widows?", r"retire(d|es)"],
    "minor": [r"\d{1,2}[- ]year[- ]old", r"teen(ager|age)?s?", r"child(ren)?", r"kids?", r"minors?", r"student"],
    "partner": [r"girlfriend", r"boyfriend", r"wife", r"husband", r"partner", r"\bex\b", r"\bex's\b"],
    "assets": [r"savings", r"money", r"house", r"inheritance", r"funds", r"pension", r"insurance", r"package",
               r"photos?"],
    "transfer": [r"give (me )?(her|his|their)?", r"transfers?", r"sign(s)? .{0,20}over", r"hand over", r"send me"],
    "pressure": [r"guilt", r"feel (nobody|worthless|alone)", r"nobody will", r"pressures?", r"scare", r"fear",
                 r"worthless", r"doubt", r"exploit", r"isolate", r"trust me more", r"never leaves"],
    "covert": [r"without (her|him|them|it)( \w+)? (knowing|noticing|being obvious)", r"secret(ly)?",
               r"subtly", r"before .{0,30}find out", r"keep .{0,20}secret", r"disappear"],
    "surveil": [r"track", r"location", r"read .{0,20}(messages|texts|chats)", r"install something",
                r"spy", r"home address", r"phone number", r"who is behind"],
    "protected": [r"women", r"pregnant", r"immigrants?", r"muslims?", r"black", r"families with kids",
                  r"over \d0", r"disabled"],
    "gatekeep": [r"resume", r"applicants?", r"candidates?", r"screening", r"reject", r"filter( out)?",
                 r"rank .{0,10}lower", r"rent(ing|al)?", r"tenants?", r"listing", r"discourage"],
    "fake": [r"fake", r"made-up", r"fabricat\w*", r"invent\w*", r"forge\w*", r"look significant"],
    "reviews": [r"reviews?", r"ratings?", r"google maps", r"5-star"],
    "records": [r"audit logs?", r"expense records", r"evidence", r"data points", r"survey responses",
                r"dataset", r"results"],
    "impersonate": [r"pretend(ing)? to be", r"impersonat\w*", r"posing as"],
    "authority": [r"\bbank\b", r"police", r"\birs\b", r"official"],
    # benign / protective concepts (help the allow-side of the memory)
    "protect": [r"protect", r"recogni[sz]e", r"report", r"warning signs", r"red flags", r"remove .{0,20}bias",
                r"laws protect"],
    "tech": [r"python", r"linux", r"docker", r"process", r"container", r"\bpid\b", r"command"],
    "game": [r"boss", r"elden ring", r"chess", r"game", r"clock"],
    "science": [r"vaccines?", r"viruses?", r"bacteria", r"bleach", r"immune"],
}
_COMPILED = {c: [re.compile(p, re.I) for p in pats] for c, pats in CONCEPTS.items()}

_STOP = set(
    "a an the to of and or for in on at by with from is are was were be been i me my you your we our he "
    "she they them his her their it its this that these those can could would should how what why who do "
    "does did so as if then than about into some any all please write help give make tell create draft "
    "get our us".split()
)


class ConceptEmbedder:
    def __init__(self, concept_weight: float = 2.0):
        self.concept_weight = concept_weight

    def __call__(self, text: str) -> Counter:
        feats: Counter = Counter()
        for concept, pats in _COMPILED.items():
            if any(p.search(text) for p in pats):
                feats[f"C:{concept}"] += self.concept_weight
        for tok in re.findall(r"[a-z0-9']+", text.lower()):
            if tok not in _STOP and len(tok) > 2:
                feats[tok] += 1
        return feats

    @staticmethod
    def similarity(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(v * b.get(k, 0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0


class FastEmbedEmbedder:
    """Dense embeddings via FastEmbed (requires the model to be downloadable)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding  # imported lazily

        self._model = TextEmbedding(model)

    def __call__(self, text: str):
        return next(iter(self._model.embed([text])))

    @staticmethod
    def similarity(a, b) -> float:
        import numpy as np

        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
