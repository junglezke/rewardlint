"""Run a reward function over the corpus and score the reward function."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import corpus as _corpus
from .adapter import AdapterError, CallPlan, call, plan, probe, to_score
from .case import Case

#: A completion scoring above this is treated as accepted. Verifiers that
#: return a graded score rather than 0/1 can override it.
DEFAULT_THRESHOLD = 0.0


@dataclass
class Outcome:
    """What the reward function did on one case."""

    case: Case
    score: Optional[float]
    accepted: Optional[bool]
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        """True when the verifier agreed with the corpus."""
        return self.error is None and self.accepted == self.case.expect

    @property
    def failure_mode(self) -> Optional[str]:
        if self.passed or self.error is not None:
            return None
        return "false positive" if self.case.expect is False else "false negative"


@dataclass
class Audit:
    """The verdict on a reward function."""

    name: str
    outcomes: List[Outcome]
    call_plan: str
    threshold: float
    contested: List[Outcome] = field(default_factory=list)
    version: str = "0.1.0"

    # -- headline rates ----------------------------------------------------

    @property
    def should_accept(self) -> List[Outcome]:
        return [o for o in self.outcomes if o.case.expect]

    @property
    def should_reject(self) -> List[Outcome]:
        return [o for o in self.outcomes if not o.case.expect]

    @property
    def false_positives(self) -> List[Outcome]:
        """Wrong or degenerate completions the verifier accepted.

        This is the reward-hacking surface: every one of these is a gradient
        that trains the policy toward something you did not want.
        """
        return [o for o in self.should_reject if o.error is None and o.accepted]

    @property
    def false_negatives(self) -> List[Outcome]:
        """Correct completions the verifier rejected.

        Thrown-away learning signal, and in GRPO also thrown-away rollouts:
        a group where the right answer scores zero has less variance to rank.
        """
        return [o for o in self.should_accept if o.error is None and not o.accepted]

    @property
    def errors(self) -> List[Outcome]:
        """Cases where the verifier raised. A grader that crashes on a
        malformed completion returns nothing, and most training loops read
        'nothing' as zero reward."""
        return [o for o in self.outcomes if o.error is not None]

    @property
    def false_positive_rate(self) -> float:
        graded = [o for o in self.should_reject if o.error is None]
        return len(self.false_positives) / len(graded) if graded else 0.0

    @property
    def false_negative_rate(self) -> float:
        graded = [o for o in self.should_accept if o.error is None]
        return len(self.false_negatives) / len(graded) if graded else 0.0

    @property
    def exploits_accepted(self) -> List[Outcome]:
        return [o for o in self.false_positives if o.case.category == "exploit"]

    @property
    def attacks_that_work(self) -> Dict[str, List[Outcome]]:
        """Successful exploits grouped by strategy.

        Grouped because 'the shotgun strategy works' is one fact to act on,
        not six scattered failures.
        """
        grouped: Dict[str, List[Outcome]] = {}
        for outcome in self.exploits_accepted:
            grouped.setdefault(outcome.case.attack or "unlabelled", []).append(outcome)
        return grouped

    @property
    def passed(self) -> bool:
        return not self.false_positives and not self.false_negatives and not self.errors

    def by_category(self) -> Dict[str, Dict[str, int]]:
        table: Dict[str, Dict[str, int]] = {}
        for outcome in self.outcomes:
            row = table.setdefault(outcome.case.category, {"total": 0, "failed": 0})
            row["total"] += 1
            if not outcome.passed:
                row["failed"] += 1
        return table

    @property
    def is_graded(self) -> bool:
        """True when the verifier returns partial credit rather than 0/1.

        Shaped rewards -- open-r1's ``tag_count_reward`` pays 0.25 per correctly
        formed tag -- are common, and against the default threshold of 0 *any*
        partial credit reads as acceptance. That would report every shaped
        reward as permissive, which is a threshold artefact rather than a
        finding, so the report has to say so and ask for a threshold.
        """
        return any(
            o.score is not None and 0.0 < o.score < 1.0 for o in self.outcomes
        )

    @property
    def graded_note(self) -> str:
        if not self.is_graded:
            return ""
        scores = sorted({round(o.score, 4) for o in self.outcomes if o.score is not None})
        preview = ", ".join(str(v) for v in scores[:6]) + (" ..." if len(scores) > 6 else "")
        return (
            f"This verifier returns partial credit (scores seen: {preview}), and the "
            f"threshold is {self.threshold}, so anything above zero counts as accepted. "
            "For a shaped reward that is a threshold artefact rather than a finding -- "
            "re-run with `--threshold` set to the score you would treat as success, or "
            "read the per-case scores rather than the rates."
        )

    @property
    def profile(self) -> str:
        """Which kind of verifier this is, from the shape of its failures.

        Added after auditing verl's GSM8K scorer, which reports a 98%
        false-negative rate on this corpus. That is not a defect: it requires
        the ``#### N`` answer format, so it correctly refuses every completion
        that does not use it. Reporting that as a bug would be wrong, and a tool
        that cannot tell the two apart is worse than no tool.

        Returns one of ``permissive``, ``format_strict``, ``balanced``.
        """
        if self.exploits_accepted or self.false_positive_rate > 0.10:
            return "permissive"
        if self.false_negative_rate > 0.50:
            return "format_strict"
        return "balanced"

    @property
    def interpretation(self) -> str:
        """How to read this result. The number that matters depends on the profile."""
        if self.is_graded:
            return self.graded_note
        if self.profile == "permissive":
            return (
                "Permissive verifier: it accepts completions that do not deserve reward. "
                "The false-positive rate and the accepted exploits are the numbers that "
                "matter; a policy will find them."
            )
        if self.profile == "format_strict":
            return (
                "Format-strict verifier: it accepts no exploits, and refuses most correct "
                "answers because it requires one specific output shape. If that shape is "
                "mandated by your prompt, this false-negative rate is your format "
                "requirement rather than a defect -- but measure how often your model "
                "actually complies, because every non-compliant rollout becomes silent "
                "zero reward. Re-run with `--category exploit` for the format-independent "
                "half of the corpus."
            )
        return (
            "Balanced verifier: no exploits accepted, and it recognises correct answers "
            "across surface forms."
        )

    @property
    def headline(self) -> str:
        if self.passed:
            return f"{self.name}: clean on {len(self.outcomes)} cases."
        parts = []
        if self.exploits_accepted:
            attacks = ", ".join(sorted(self.attacks_that_work))
            parts.append(f"{len(self.exploits_accepted)} exploits accepted ({attacks})")
        if self.false_positive_rate:
            parts.append(f"{self.false_positive_rate:.0%} false-positive rate")
        if self.false_negative_rate:
            parts.append(f"{self.false_negative_rate:.0%} false-negative rate")
        if self.errors:
            parts.append(f"{len(self.errors)} crashes")
        return f"{self.name}: " + "; ".join(parts) + "."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "call_plan": self.call_plan,
            "threshold": self.threshold,
            "n_cases": len(self.outcomes),
            "passed": self.passed,
            "headline": self.headline,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "n_exploits_accepted": len(self.exploits_accepted),
            "attacks_that_work": sorted(self.attacks_that_work),
            "by_category": self.by_category(),
            "profile": self.profile,
            "is_graded": self.is_graded,
            "interpretation": self.interpretation,
            "failures": [
                {
                    "id": o.case.id,
                    "category": o.case.category,
                    "attack": o.case.attack,
                    "mode": o.failure_mode,
                    "reference": o.case.reference,
                    "completion": _truncate(o.case.completion),
                    "score": o.score,
                    "why": o.case.why,
                }
                for o in self.outcomes
                if not o.passed
            ],
        }

    def print(self, fmt: str = "terminal") -> None:
        from .report import emit, render

        emit(render(self, fmt))


def audit(
    reward_fn: Callable,
    name: Optional[str] = None,
    cases: Optional[Sequence[Case]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    include_contested: bool = False,
    domains: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
) -> Audit:
    """Run ``reward_fn`` over the corpus and report where it disagrees.

    Args:
        reward_fn: your verifier. Almost any signature works -- see
            :mod:`rewardlint.adapter`.
        threshold: a score strictly above this counts as accepted.
        include_contested: include cases whose expected verdict is a judgement
            call. Off by default; they are reported separately.
    """
    call_plan: CallPlan = probe(reward_fn, plan(reward_fn))
    selected = list(cases) if cases is not None else _corpus.load(
        domains=domains, categories=categories, include_contested=False
    )

    outcomes = [_run_case(reward_fn, call_plan, case, threshold) for case in selected]
    contested_outcomes = (
        [_run_case(reward_fn, call_plan, c, threshold) for c in _corpus.contested()]
        if include_contested or cases is None
        else []
    )

    return Audit(
        name=name or getattr(reward_fn, "__name__", "reward_fn"),
        outcomes=outcomes,
        call_plan=call_plan.describe(),
        threshold=threshold,
        contested=contested_outcomes,
    )


def _run_case(fn: Callable, call_plan: CallPlan, case: Case, threshold: float) -> Outcome:
    try:
        raw = call(fn, call_plan, case.completion, case.reference, case.prompt)
        score = to_score(raw)
    except AdapterError as exc:
        return Outcome(case=case, score=None, accepted=None, error=f"adapter: {exc}")
    except Exception as exc:
        # A verifier that raises on a malformed completion is a real finding:
        # most training loops turn an exception into a zero, or into a crash
        # halfway through an epoch.
        return Outcome(case=case, score=None, accepted=None, error=f"{type(exc).__name__}: {exc}")
    return Outcome(case=case, score=score, accepted=score > threshold)


def _truncate(text: str, limit: int = 160) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 3] + "..."
