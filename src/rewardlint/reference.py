"""Reference verifiers: the patterns people actually write, plus one that holds up.

The first four are not strawmen. Every one of them appears in real RLVR
codebases, and each is a reasonable-looking twenty lines. Auditing them is the
fastest way to see the shape of the problem, because they sit at different
points on the same trade-off:

* loose matching keeps recall and hands you a reward hack,
* strict matching closes the hack and throws away signal from correct answers.

:func:`robust_match` is the constructive half -- a verifier that takes an
explicit position on that trade-off instead of arriving at one by accident.
Copy it, or copy the ideas. The measured rates for all five are in the README.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import List, Optional

__all__ = [
    "REFERENCE_VERIFIERS",
    "boxed_exact",
    "exact_match",
    "last_number",
    "robust_match",
    "substring_match",
]

# --------------------------------------------------------------------------
# The common patterns
# --------------------------------------------------------------------------


def exact_match(completion: str, reference: str) -> float:
    """String equality after stripping. Cannot be gamed; rejects almost everything."""
    return float(completion.strip() == reference.strip())


def substring_match(completion: str, reference: str) -> float:
    """"Does the answer appear anywhere in the output?"

    The single most common hand-rolled verifier, and the single most exploitable.
    """
    return float(reference.strip() in completion)


def last_number(completion: str, reference: str) -> float:
    """Take the last number in the completion and compare numerically.

    A real improvement on substring matching, and the standard GSM8K-style
    heuristic. Still has no idea whether that number was being asserted or
    dismissed.
    """
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion.replace(",", ""))
    if not numbers:
        return 0.0
    try:
        return float(abs(float(numbers[-1]) - float(reference.replace(",", ""))) < 1e-6)
    except ValueError:
        return 0.0


def boxed_exact(completion: str, reference: str) -> float:
    """Extract ``\\boxed{...}`` and compare as strings.

    Correctly refuses most exploits. Also refuses ``\\frac{1}{2}`` when the
    reference says ``1/2``, which is where the thrown-away signal comes from.
    """
    matches = _boxed_contents(completion)
    if len(matches) != 1:
        return 0.0
    return float(matches[0].strip() == reference.strip())


# --------------------------------------------------------------------------
# A verifier that takes a position
# --------------------------------------------------------------------------

#: Phrases that introduce a stated answer. Ordered longest-first so that
#: "the final answer is" wins over "answer is".
_ANSWER_MARKERS = (
    r"the\s+final\s+answer\s+is",
    r"final\s+answer\s*[:=]",
    r"the\s+answer\s+is",
    r"answer\s*[:=]",
    r"####",
)

_NEGATION = re.compile(r"\b(?:not|isn'?t|aren'?t|never|cannot|can'?t|rule[sd]?\s+out)\b", re.I)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def robust_match(completion: str, reference: str) -> float:
    """Extract a single asserted answer, then compare it semantically.

    Three rules, in order of how much they matter:

    1. **Require exactly one asserted answer.** Multiple boxed answers, or a
       marker followed by a list of candidates, are refused as ambiguous. This
       is what closes the shotgun exploit, and no amount of better normalisation
       substitutes for it.
    2. **Grade the conclusion, not the transcript.** Reasoning inside
       ``<think>`` is stripped before extraction, and a negated span is refused.
       "The answer is not 42" is not an assertion that the answer is 42.
    3. **Compare values, not strings.** ``1/2``, ``\\frac{1}{2}``, ``0.5`` and
       ``2/4`` are one answer written four ways.

    The deliberate cost: a correct answer stated in bare prose with no marker
    (``The product is **42**``) is refused. That is a real false negative and it
    is the price of rule 1. If your task cannot pay it, prompt for
    ``\\boxed{}`` -- but then measure how often the model complies, because
    every non-compliant rollout becomes silent zero reward.
    """
    body = _THINK_BLOCK.sub(" ", completion)

    boxed = _boxed_contents(body)
    if len(boxed) > 1:
        return 0.0  # ambiguous: the shotgun defence
    if len(boxed) == 1:
        return float(_values_equal(boxed[0], reference))

    span = _marked_answer(body)
    if span is not None:
        if _NEGATION.search(span):
            return 0.0
        candidates = _candidates(span)
        if len(candidates) != 1:
            return 0.0
        return float(_values_equal(candidates[0], reference))

    # No marker: accept only a completion that is itself a bare answer.
    stripped = body.strip()
    if stripped and len(stripped) <= 64 and "\n" not in stripped:
        return float(_values_equal(stripped, reference))
    return 0.0


# --------------------------------------------------------------------------
# Extraction and normalisation
# --------------------------------------------------------------------------


def _boxed_contents(text: str) -> List[str]:
    """Every ``\\boxed{...}`` payload, brace-matched so nesting survives."""
    out: List[str] = []
    for match in re.finditer(r"\\boxed\s*{", text):
        depth, start = 1, match.end()
        index = start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            out.append(text[start : index - 1])
    return out


def _marked_answer(text: str) -> Optional[str]:
    """Text following the *last* answer marker, up to the end of that statement.

    The last marker, because a model that corrects itself states the answer it
    means last.
    """
    best: Optional[int] = None
    for pattern in _ANSWER_MARKERS:
        for match in re.finditer(pattern, text, re.I):
            if best is None or match.end() > best:
                best = match.end()
    if best is None:
        return None
    tail = text[best:]
    # Stop at a sentence boundary, but not at the decimal point in "3.14".
    stop = re.search(r"(?<!\d)\.(?!\d)|\n\n|$", tail)
    return tail[: stop.start()] if stop else tail


def _candidates(span: str) -> List[str]:
    """Distinct answer-looking tokens in a span. More than one means ambiguous."""
    span = span.strip().strip(".:")
    boxed = _boxed_contents(span)
    if boxed:
        return list(dict.fromkeys(boxed))

    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?", span)
    if numbers:
        return list(dict.fromkeys(n.strip() for n in numbers))

    letters = re.findall(r"\b([A-Da-d])\b", span)
    if letters:
        return list(dict.fromkeys(letter.upper() for letter in letters))

    cleaned = span.strip()
    return [cleaned] if cleaned else []


_LATEX_NOISE = (
    (re.compile(r"\\left|\\right|\\,|\\!|\\;|\\ "), " "),
    (re.compile(r"\\d?frac\s*{([^{}]*)}\s*{([^{}]*)}"), r"(\1)/(\2)"),
    (re.compile(r"\\text\s*{([^{}]*)}"), r"\1"),
    (re.compile(r"\\mathrm\s*{([^{}]*)}"), r"\1"),
    (re.compile(r"\\sqrt\s*{([^{}]*)}"), r"sqrt(\1)"),
    (re.compile(r"\\sqrt\s*(\w)"), r"sqrt(\1)"),
    (re.compile(r"\\pi\b"), "π"),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), "$"),
)


def _normalise(text: str) -> str:
    """Reduce a written answer to a comparable form."""
    out = text.strip()
    out = out.replace("\u2212", "-")  # U+2212 MINUS SIGN
    out = re.sub(r"^\$+|\$+$", "", out.strip()).strip()
    for pattern, replacement in _LATEX_NOISE:
        out = pattern.sub(replacement, out)
    boxed = _boxed_contents(out)
    if len(boxed) == 1:
        out = boxed[0]
    out = out.strip()
    out = re.sub(r"^[a-zA-Z]\s*=\s*", "", out)  # "x = 5" -> "5"
    out = re.sub(r"^\((.*)\)$", r"\1", out.strip())  # "(5)" -> "5"
    out = re.sub(r"^(?:the|a|an)\s+", "", out, flags=re.I)
    out = out.rstrip(".").strip()
    out = re.sub(r"\s+", " ", out)
    return out


def _to_number(text: str) -> Optional[Fraction]:
    """Parse a normalised answer as an exact rational, if it is one."""
    value = text.replace(" ", "")
    if not value:
        return None
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)  # thousands separators
    value = value.lstrip("+")

    # Mixed number: "1 1/2" survives as "11/2" after space removal, so parse
    # it from the pre-strip form instead.
    mixed = re.fullmatch(r"(-?\d+)\s+(\d+)/(\d+)", text.strip())
    if mixed:
        whole, num, den = (int(g) for g in mixed.groups())
        sign = -1 if whole < 0 else 1
        return Fraction(abs(whole) * den + num, den) * sign

    frac = re.fullmatch(r"\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?", value)
    if frac:
        try:
            return Fraction(frac.group(1)) / Fraction(frac.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return Fraction(float(value))  # scientific notation
    except (ValueError, OverflowError):
        return None


#: Relative tolerance for numeric comparison. Sized to accept a decimal
#: truncated at ~10 significant figures (0.3333333333 for 1/3) and to reject a
#: genuine loss of precision (3.14 for 3.14159, a relative error of 5e-4).
_REL_TOL = 1e-8


def _symbolic_key(text: str) -> str:
    r"""Canonical form for a symbolic answer that will not parse as a number.

    Makes implicit multiplication explicit and drops layout whitespace, so
    ``2\sqrt{3}``, ``2 sqrt(3)`` and ``2*sqrt(3)`` compare equal -- and so does
    ``(1, 2)`` against ``(1,2)``.
    """
    out = re.sub(r"\\cdot|\\times", "*", text)
    out = re.sub(r"\s+", "", out)
    # Implicit multiplication: a digit immediately followed by a symbol.
    out = re.sub(r"(?<=[0-9])(?=[A-Za-zπ\\(])", "*", out)
    return out.lower()


def _values_equal(candidate: str, reference: str) -> bool:
    left, right = _normalise(candidate), _normalise(reference)
    if left.lower() == right.lower():
        return True

    left_num, right_num = _to_number(left), _to_number(right)
    if left_num is not None and right_num is not None:
        if left_num == right_num:
            return True
        scale = max(abs(float(right_num)), 1.0)
        return abs(float(left_num) - float(right_num)) <= _REL_TOL * scale

    # Neither side is a number: compare as symbols.
    return _symbolic_key(left) == _symbolic_key(right)


#: Audited in the README. Ordered loosest to strictest.
REFERENCE_VERIFIERS = {
    "substring_match": substring_match,
    "last_number": last_number,
    "exact_match": exact_match,
    "boxed_exact": boxed_exact,
    "robust_match": robust_match,
}
