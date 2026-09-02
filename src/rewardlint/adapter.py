"""Call an arbitrary reward function without making the user rewrite it.

There is no standard signature for a verifier. Across TRL, verl, open-r1,
Math-Verify and a hundred hand-rolled graders you will find at least:

    f(completion, reference)
    f(prediction, ground_truth)
    f(response, answer)
    f(completions, **kwargs) -> list[float]        # the TRL reward-function protocol
    f(solution_str, ground_truth, extra_info)      # verl's compute_score
    f(*, model_output, gold, question)

Asking people to write an adapter before they can run the tool is asking them
not to run the tool. So this module inspects the signature and works out how to
call it, and says clearly what it decided so a wrong guess is visible rather
than silently producing a nonsense report.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Parameter names that mean "the model's output", most specific first.
COMPLETION_NAMES = (
    "completion",
    "completions",
    "solution_str",
    "model_output",
    "prediction",
    "predict_str",
    "response",
    "output",
    "pred",
    "generated",
    "text",
    "answer",
)

#: Parameter names that mean "the correct answer".
REFERENCE_NAMES = (
    "reference",
    "ground_truth",
    "gold",
    "gold_answer",
    "target",
    "label",
    "solution",
    "gt",
    "answer_gt",
    "reference_answer",
    "expected",
    # Last, and only reachable when nothing else claimed the role: `answer`
    # means the gold value in about as many codebases as it means the model's.
    "answer",
)

#: Parameter names that mean "the problem statement".
PROMPT_NAMES = ("prompt", "prompts", "question", "problem", "query", "input", "instruction")

#: Names whose presence means the function expects a batch, not one example.
_PLURAL = {"completions", "prompts"}


class AdapterError(ValueError):
    """Raised when a reward function's signature cannot be worked out."""


@dataclass
class CallPlan:
    """How :func:`call` will invoke the reward function."""

    completion_param: Optional[str]
    reference_param: Optional[str]
    prompt_param: Optional[str]
    positional: bool
    batched: bool

    def describe(self) -> str:
        if self.positional:
            order = ["completion", "reference"]
            if self.prompt_param:
                order.insert(0, "prompt")
            return f"positional: f({', '.join(order)})"
        bits = []
        if self.prompt_param:
            bits.append(f"{self.prompt_param}=prompt")
        if self.completion_param:
            bits.append(f"{self.completion_param}=completion")
        if self.reference_param:
            bits.append(f"{self.reference_param}=reference")
        suffix = "  [batched: passes and expects lists]" if self.batched else ""
        return "keyword: f(" + ", ".join(bits) + ")" + suffix


def _match(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """First candidate present in ``names``, honouring candidate priority order."""
    lowered = {n.lower(): n for n in names}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def plan(fn: Callable) -> CallPlan:
    """Work out how to call ``fn``."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError) as exc:  # builtins, some C callables
        raise AdapterError(f"cannot inspect {fn!r}: {exc}") from exc

    params = [
        p
        for p in signature.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    names = [p.name for p in params]
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())

    completion = _match(names, COMPLETION_NAMES)
    # A parameter can only fill one role.
    reference = _match([n for n in names if n != completion], REFERENCE_NAMES)
    prompt = _match(names, PROMPT_NAMES)

    # `answer` is genuinely ambiguous: it means "the model's answer" in about as
    # many codebases as it means "the gold answer". Read it as the completion
    # only when something else is clearly the reference; otherwise hand it the
    # reference role and look again for a completion.
    if completion == "answer" and reference is None:
        reference = "answer"
        completion = _match([n for n in names if n != "answer"], COMPLETION_NAMES)

    batched = bool(completion in _PLURAL or (completion is None and "completions" in names))

    if completion and reference:
        return CallPlan(completion, reference, prompt, positional=False, batched=batched)

    # The TRL protocol: f(completions, **kwargs), with the reference arriving
    # through a dataset column forwarded as a keyword.
    if completion in _PLURAL and has_var_kw:
        return CallPlan(completion, reference, prompt, positional=False, batched=True)

    required = [p for p in params if p.default is inspect.Parameter.empty]
    if len(required) >= 2:
        # Two unnamed-but-required parameters: assume (completion, reference),
        # which is the dominant convention. Reported in the plan so a user can
        # see the assumption and pass an explicit adapter if it is wrong.
        return CallPlan(None, None, None, positional=True, batched=False)

    raise AdapterError(
        f"could not work out how to call {getattr(fn, '__name__', fn)!r} "
        f"with parameters {names}. Wrap it:\n\n"
        "    audit(lambda completion, reference: my_fn(completion, reference))"
    )


def call(fn: Callable, plan_: CallPlan, completion: str, reference: str, prompt: str = "") -> Any:
    """Invoke the reward function for one case and return its raw result."""
    if plan_.positional:
        args: List[Any] = [completion, reference]
        return fn(*args)

    kwargs: Dict[str, Any] = {}
    if plan_.completion_param:
        kwargs[plan_.completion_param] = [completion] if plan_.batched else completion
    if plan_.reference_param:
        kwargs[plan_.reference_param] = [reference] if plan_.batched else reference
    if plan_.prompt_param:
        kwargs[plan_.prompt_param] = [prompt] if plan_.batched else prompt
    return fn(**kwargs)


def to_score(result: Any) -> float:
    """Coerce whatever the reward function returned into a single number.

    Handles: bool, int/float, a one-element sequence (the batched protocol),
    and the ``{"score": ...}`` dict that several frameworks return.
    """
    if isinstance(result, bool):
        return 1.0 if result else 0.0
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for key in ("score", "reward", "value", "acc", "correct"):
            if key in result:
                return to_score(result[key])
        raise AdapterError(f"reward function returned a dict with no score key: {sorted(result)}")
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        items = list(result)
        if len(items) == 1:
            return to_score(items[0])
        raise AdapterError(
            f"reward function returned {len(items)} values for one example; "
            "rewardlint grades one completion at a time"
        )
    # numpy / torch scalars
    item = getattr(result, "item", None)
    if callable(item):
        try:
            return float(item())
        except (TypeError, ValueError):
            pass
    raise AdapterError(f"cannot read a score from {type(result).__name__}: {result!r}")
