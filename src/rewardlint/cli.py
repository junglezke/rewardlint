"""Command line interface.

    rewardlint audit my_project.rewards:accuracy_reward
    rewardlint compare
    rewardlint corpus --category exploit
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from typing import Callable, List, Optional

from . import __version__
from .report import emit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rewardlint",
        description="Find out what your reward function actually accepts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  rewardlint audit my_project.rewards:accuracy_reward\n"
            "  rewardlint audit ./rewards.py:check --format markdown\n"
            "  rewardlint compare               # audit the built-in reference verifiers\n"
            "  rewardlint corpus --category exploit\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"rewardlint {__version__}")
    sub = parser.add_subparsers(dest="command")

    audit_cmd = sub.add_parser("audit", help="audit a reward function")
    audit_cmd.add_argument(
        "target",
        help="module:function, or path/to/file.py:function",
    )
    audit_cmd.add_argument("-f", "--format", default="terminal", choices=["terminal", "markdown", "json"])
    audit_cmd.add_argument("-o", "--output", help="write the report to a file")
    audit_cmd.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="a score strictly above this counts as accepted (default: 0.0)",
    )
    audit_cmd.add_argument("--category", nargs="+", help="restrict to these case categories")
    audit_cmd.add_argument("--include-contested", action="store_true")
    audit_cmd.add_argument(
        "--fail-on",
        default="exploit",
        choices=["never", "exploit", "any"],
        help="exit non-zero when: an exploit is accepted (default), or any case fails",
    )
    audit_cmd.add_argument("--no-color", action="store_true")

    compare = sub.add_parser("compare", help="audit the built-in reference verifiers")
    compare.add_argument("-f", "--format", default="terminal", choices=["terminal", "markdown"])

    corpus_cmd = sub.add_parser("corpus", help="list the corpus")
    corpus_cmd.add_argument("--category", nargs="+")
    corpus_cmd.add_argument("--attack", nargs="+")
    corpus_cmd.add_argument("--full", action="store_true", help="show completions in full")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "audit":
        return _cmd_audit(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "corpus":
        return _cmd_corpus(args)
    parser.print_help()
    return 0


def load_target(target: str) -> Callable:
    """Import ``module:function`` or ``path/to/file.py:function``."""
    if ":" not in target:
        raise ValueError(
            f"{target!r} is missing the function name. Use module:function, "
            "e.g. my_project.rewards:accuracy_reward"
        )
    module_ref, _, attr = target.rpartition(":")

    if module_ref.endswith(".py"):
        path = os.path.abspath(module_ref)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no such file: {module_ref}")
        spec = importlib.util.spec_from_file_location("_rewardlint_target", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_ref}")
        module = importlib.util.module_from_spec(spec)
        # Let the target import its own package-relative modules.
        sys.path.insert(0, os.path.dirname(path))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    else:
        sys.path.insert(0, "")
        try:
            module = importlib.import_module(module_ref)
        finally:
            sys.path.pop(0)

    try:
        return getattr(module, attr)
    except AttributeError:
        available = [n for n in dir(module) if not n.startswith("_") and callable(getattr(module, n))]
        raise AttributeError(
            f"{module_ref} has no attribute {attr!r}. Callables found: {', '.join(available[:15])}"
        ) from None


def _cmd_audit(args) -> int:
    from .audit import audit
    from .report import render

    try:
        fn = load_target(args.target)
    except Exception as exc:
        print(f"rewardlint: {exc}", file=sys.stderr)
        return 2

    try:
        result = audit(
            fn,
            name=args.target,
            threshold=args.threshold,
            categories=args.category,
            include_contested=args.include_contested,
        )
    except Exception as exc:
        print(f"rewardlint: {exc}", file=sys.stderr)
        return 2

    if args.format == "terminal" and args.no_color:
        from .report.terminal import render as render_terminal

        text = render_terminal(result, color=False)
    else:
        text = render(result, args.format)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {args.output}")
    else:
        emit(text)

    if args.fail_on == "never":
        return 0
    if args.fail_on == "exploit":
        return 1 if (result.exploits_accepted or result.errors) else 0
    return 0 if result.passed else 1


def _cmd_compare(args) -> int:
    from .audit import audit
    from .reference import REFERENCE_VERIFIERS

    rows = []
    for name, fn in REFERENCE_VERIFIERS.items():
        result = audit(fn, name=name)
        rows.append(
            (
                name,
                result.false_positive_rate,
                result.false_negative_rate,
                len(result.exploits_accepted),
                ", ".join(sorted(result.attacks_that_work)) or "-",
            )
        )

    if args.format == "markdown":
        print("| verifier | false positive | false negative | exploits accepted | attacks that work |")
        print("|---|---|---|---|---|")
        for name, fp, fn_rate, n, attacks in rows:
            print(f"| `{name}` | {fp:.0%} | {fn_rate:.0%} | {n} | {attacks} |")
        return 0

    print()
    print(f"  {'verifier':<18}{'FP':>6}{'FN':>7}{'exploits':>10}   attacks that work")
    print("  " + "-" * 78)
    for name, fp, fn_rate, n, attacks in rows:
        print(f"  {name:<18}{fp:>5.0%}{fn_rate:>7.0%}{n:>10}   {attacks}")
    print()
    print("  FP = wrong or degenerate completions accepted (the reward-hacking surface)")
    print("  FN = correct completions rejected (thrown-away learning signal)")
    print()
    return 0


def _cmd_corpus(args) -> int:
    from . import corpus

    cases = corpus.load(categories=args.category, attacks=args.attack, include_contested=True)
    for case in cases:
        verdict = "accept" if case.expect else "REJECT"
        completion = case.completion if args.full else _short(case.completion)
        emit(
            "\n".join(
                [
                    f"{case.id:<28} {case.category:<12} must {verdict}",
                    f"    reference:  {case.reference!r}",
                    f"    completion: {completion!r}",
                    f"    why:        {case.why}",
                    "",
                ]
            )
        )
    print(f"{len(cases)} cases  |  totals: {corpus.stats()}")
    return 0


def _short(text: str, limit: int = 90) -> str:
    shown = text.replace("\n", "\\n")
    return shown if len(shown) <= limit else shown[: limit - 3] + "..."


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
