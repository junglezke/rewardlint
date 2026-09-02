"""ANSI terminal report. No dependencies -- this has to run inside a training image."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import List, Optional

from ..audit import Audit, Outcome
from ..corpus import ATTACKS

_RESET = "\033[0m"
_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[91m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "cyan": "\033[36m",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _supports_unicode() -> bool:
    if os.environ.get("REWARDLINT_ASCII"):
        return False
    encoding = getattr(sys.stdout, "encoding", None) or ""
    try:
        "\u2500\u00b7".encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


class _Painter:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        return "".join(_STYLES.get(s, "") for s in styles) + text + _RESET


def render(
    audit: Audit,
    width: Optional[int] = None,
    color: Optional[bool] = None,
    unicode: Optional[bool] = None,
) -> str:
    width = width or min(shutil.get_terminal_size((96, 24)).columns, 96)
    paint = _Painter(_supports_color() if color is None else color)
    use_unicode = _supports_unicode() if unicode is None else unicode
    rule, dot = ("\u2500", "\u00b7") if use_unicode else ("-", "*")

    out: List[str] = [
        "",
        "  " + paint("rewardlint", "bold", "cyan") + paint(f" {audit.version}", "dim")
        + "  " + paint(audit.name, "bold"),
        "  " + paint(rule * (width - 4), "dim"),
        "  " + paint(f"{len(audit.outcomes)} cases  {dot}  {audit.call_plan}", "dim"),
        "",
    ]

    if audit.passed:
        out += [
            "  " + paint("PASS", "bold", "green") + "  No disagreements with the corpus.",
            "",
        ]
    else:
        out += [
            "  " + paint("RATES", "bold"),
            f"      false positive  {_bar(audit.false_positive_rate, paint, 'red')}  "
            f"{audit.false_positive_rate:>5.0%}   "
            + paint(f"{len(audit.false_positives)} wrong or degenerate completions accepted", "dim"),
            f"      false negative  {_bar(audit.false_negative_rate, paint, 'yellow')}  "
            f"{audit.false_negative_rate:>5.0%}   "
            + paint(f"{len(audit.false_negatives)} correct completions rejected", "dim"),
            "",
        ]
        for chunk in textwrap.wrap(audit.interpretation, width - 6):
            out.append("      " + paint(chunk, "dim"))
        out.append("")

    if audit.errors:
        out += [_head("CRASHES", width, paint, rule), ""]
        out.append(
            "  "
            + paint(
                "Your verifier raised on these. Most training loops turn an exception "
                "into a silent zero reward, which is indistinguishable from a wrong answer.",
                "dim",
            )
        )
        out.append("")
        for outcome in audit.errors[:8]:
            out += _case_block(outcome, width, paint, dot, show_error=True)

    attacks = audit.attacks_that_work
    if attacks:
        out += [_head("EXPLOITS THAT WORK", width, paint, rule), ""]
        for attack, outcomes in sorted(attacks.items()):
            out.append("  " + paint(attack, "bold", "red"))
            blurb = (
                f"{ATTACKS.get(attack, '')} {len(outcomes)} case(s) accepted. A policy that "
                "finds this gets paid for it, and every metric you watch will look like progress."
            )
            for chunk in textwrap.wrap(blurb, width - 8):
                out.append("      " + paint(chunk, "dim"))
            for outcome in outcomes:
                out += _case_block(outcome, width, paint, dot, indent=6)
            out.append("")

    other_fp = [o for o in audit.false_positives if o.case.category != "exploit"]
    if other_fp:
        out += [_head("WRONG ANSWERS ACCEPTED", width, paint, rule), ""]
        for outcome in other_fp:
            out += _case_block(outcome, width, paint, dot)

    if audit.false_negatives:
        out += [_head("CORRECT ANSWERS REJECTED", width, paint, rule), ""]
        out.append(
            "  "
            + paint(
                "Thrown-away learning signal. In GRPO these also shrink group variance, "
                "so they cost rollouts as well as accuracy.",
                "dim",
            )
        )
        out.append("")
        for outcome in audit.false_negatives[:20]:
            out += _case_block(outcome, width, paint, dot)
        if len(audit.false_negatives) > 20:
            out.append(
                "  " + paint(f"... and {len(audit.false_negatives) - 20} more", "dim")
            )
            out.append("")

    out += [_head("BY CATEGORY", width, paint, rule), ""]
    for category, row in sorted(audit.by_category().items()):
        total = row["total"]
        passed = total - row["failed"]
        style = "green" if row["failed"] == 0 else "yellow"
        tally = paint(f"{passed}/{total}", style)
        out.append(f"      {category:<14} {tally} passed")
    out.append("")

    if audit.contested:
        disagreements = [o for o in audit.contested if not o.passed]
        out += [_head("CONTESTED", width, paint, rule), ""]
        out.append(
            "  "
            + paint(
                "Cases where the right verdict is a judgement call, excluded from the rates "
                "above. Your answer may legitimately differ from the corpus.",
                "dim",
            )
        )
        out.append("")
        for outcome in audit.contested:
            mark = "agrees" if outcome.passed else "differs"
            out.append(
                f"      {outcome.case.id:<24} {paint(mark, 'dim')}  "
                + paint(outcome.case.why, "dim")
            )
        if disagreements:
            out.append("")
        out.append("")

    return "\n".join(out)


def _bar(value: float, paint: _Painter, style: str, width: int = 20) -> str:
    filled = int(round(min(max(value, 0.0), 1.0) * width))
    return paint("#" * filled, style) + paint("." * (width - filled), "dim")


def _case_block(
    outcome: Outcome,
    width: int,
    paint: _Painter,
    dot: str,
    indent: int = 4,
    show_error: bool = False,
) -> List[str]:
    pad = " " * indent
    case = outcome.case
    body = width - indent - 4
    lines = [f"{pad}{paint(case.id, 'bold')}"]
    for chunk in textwrap.wrap(case.why, body):
        lines.append(f"{pad}  {paint(chunk, 'dim')}")
    lines.append(f"{pad}  reference:  {paint(_show(case.reference, body), 'cyan')}")
    lines.append(f"{pad}  completion: {paint(_show(case.completion, body), 'cyan')}")
    if show_error:
        for chunk in textwrap.wrap(f"raised {outcome.error}", body):
            lines.append(f"{pad}  {paint(chunk, 'red')}")
    else:
        verdict = "accepted" if outcome.accepted else "rejected"
        expected = "accept" if case.expect else "reject"
        note = f"your verifier {verdict} it (score {outcome.score}); it should {expected}"
        lines.append(f"{pad}  {paint(note, 'dim')}")
    lines.append("")
    return lines


def _show(text: str, limit: int) -> str:
    shown = text.replace("\n", "\\n").replace("\t", "\\t")
    if len(shown) > limit:
        shown = shown[: limit - 3] + "..."
    return repr(shown)


def _head(label: str, width: int, paint: _Painter, rule: str) -> str:
    text = f"{rule * 2} {label} "
    return "  " + paint(text + rule * max(width - len(text) - 4, 0), "dim")
