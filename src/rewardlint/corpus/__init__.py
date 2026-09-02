"""The case corpus.

Scope: **answer-checking verifiers** -- functions that decide whether a
completion states the right short answer (a number, an expression, a
multiple-choice letter, a yes/no). That covers most of math and short-form
RLVR, and every exploit in :mod:`.exploits` applies to any text verifier
including an LLM judge.

Out of scope for now: execution-based code verifiers, which take a patch and a
test suite rather than a completion and a reference, and need a sandbox. See
the roadmap.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from ..case import Case
from . import exploits as _exploits
from . import math_cases as _math

#: Cases whose expected verdict is a defensible judgement call rather than a
#: fact. ``5 meters`` for a reference of ``5`` is right if units are optional
#: and wrong if they are not; ``3.14`` for ``3.14159`` depends on your tolerance.
#: They are excluded from the headline rates by default and reported separately,
#: because scoring a tool on questions that have no single right answer is how
#: benchmarks stop meaning anything.
CONTESTED_TAG = "contested"

ALL_CASES: List[Case] = _math.ALL + _exploits.ALL
ATTACKS: Dict[str, str] = dict(_exploits.ATTACKS)


def _validate() -> None:
    seen = set()
    for case in ALL_CASES:
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)


_validate()


def load(
    domains: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    include_contested: bool = False,
    attacks: Optional[Sequence[str]] = None,
) -> List[Case]:
    """Select cases from the corpus.

    By default this returns everything except the contested cases.
    """
    cases: Iterable[Case] = ALL_CASES
    if domains:
        wanted = set(domains)
        cases = [c for c in cases if c.domain in wanted]
    if categories:
        wanted = set(categories)
        cases = [c for c in cases if c.category in wanted]
    if attacks:
        wanted = set(attacks)
        cases = [c for c in cases if c.attack in wanted]
    if not include_contested:
        cases = [c for c in cases if CONTESTED_TAG not in c.tags]
    return list(cases)


def contested() -> List[Case]:
    return [c for c in ALL_CASES if CONTESTED_TAG in c.tags]


def stats() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for case in ALL_CASES:
        counts[case.category] = counts.get(case.category, 0) + 1
    counts["total"] = len(ALL_CASES)
    counts["contested"] = len(contested())
    return counts


__all__ = ["ALL_CASES", "ATTACKS", "CONTESTED_TAG", "contested", "load", "stats"]
