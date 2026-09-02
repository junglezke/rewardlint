"""Tests for rewardlint.

The reference-verifier rates are asserted as exact numbers. They are published
in the README, so a silent change to the corpus or to the matching logic that
moves them is a documentation bug, and this suite is what turns it into a
failing build.
"""

from __future__ import annotations

import json

import pytest

from rewardlint import corpus
from rewardlint.adapter import AdapterError, CallPlan, plan, to_score
from rewardlint.audit import audit
from rewardlint.case import Case
from rewardlint.cli import main
from rewardlint.reference import (
    REFERENCE_VERIFIERS,
    boxed_exact,
    exact_match,
    last_number,
    robust_match,
    substring_match,
)
from rewardlint.report import render

# -- corpus -----------------------------------------------------------------


def test_case_ids_are_unique():
    ids = [c.id for c in corpus.ALL_CASES]
    assert len(ids) == len(set(ids))


def test_every_case_explains_itself():
    """A case whose rationale is unwritten cannot be argued with."""
    for case in corpus.ALL_CASES:
        assert case.why.strip(), case.id
        assert len(case.why) > 15, f"{case.id}: rationale is too thin to be useful"


def test_every_exploit_names_its_attack():
    for case in corpus.ALL_CASES:
        if case.category == "exploit":
            assert case.attack, case.id
            assert case.attack in corpus.ATTACKS, f"{case.id}: attack not described"


def test_category_invariants_are_enforced():
    with pytest.raises(ValueError, match="must expect rejection"):
        Case(id="x", category="exploit", domain="d", reference="1", completion="1",
             expect=True, why="a rationale long enough to pass validation")
    with pytest.raises(ValueError, match="must expect acceptance"):
        Case(id="x", category="equivalence", domain="d", reference="1", completion="1",
             expect=False, why="a rationale long enough to pass validation")
    with pytest.raises(ValueError, match="unknown category"):
        Case(id="x", category="nonsense", domain="d", reference="1", completion="1",
             expect=True, why="a rationale long enough to pass validation")


def test_contested_cases_are_excluded_by_default():
    default = corpus.load()
    assert all(corpus.CONTESTED_TAG not in c.tags for c in default)
    assert len(corpus.load(include_contested=True)) > len(default)
    assert corpus.contested(), "the contested set should not be empty"


def test_filters():
    exploits = corpus.load(categories=["exploit"])
    assert exploits and all(c.category == "exploit" for c in exploits)
    shotgun = corpus.load(attacks=["shotgun"])
    assert shotgun and all(c.attack == "shotgun" for c in shotgun)


# -- adapter ----------------------------------------------------------------


@pytest.mark.parametrize(
    "fn, expect_completion, expect_reference",
    [
        (lambda completion, reference: 1.0, "completion", "reference"),
        (lambda prediction, ground_truth: 1.0, "prediction", "ground_truth"),
        (lambda response, gold: 1.0, "response", "gold"),
        (lambda solution_str, ground_truth: 1.0, "solution_str", "ground_truth"),
        (lambda output, target: 1.0, "output", "target"),
        (lambda model_output, expected: 1.0, "model_output", "expected"),
    ],
)
def test_adapter_recognises_common_signatures(fn, expect_completion, expect_reference):
    p = plan(fn)
    assert p.completion_param == expect_completion
    assert p.reference_param == expect_reference
    assert not p.positional


def test_adapter_handles_the_trl_batched_protocol():
    def reward(completions, ground_truth, **kwargs):
        return [float(c == g) for c, g in zip(completions, ground_truth)]

    p = plan(reward)
    assert p.batched
    result = audit(reward, name="trl-style")
    assert result.false_positive_rate == 0.0  # exact matching accepts nothing wrong


def test_adapter_reads_answer_as_the_reference_when_nothing_else_claims_it():
    """`answer` is ambiguous across codebases; it must not be mistaken for the
    completion when it is clearly the gold value."""
    p = plan(lambda completion, answer: 1.0)
    assert p.completion_param == "completion"
    assert p.reference_param == "answer"


def test_adapter_falls_back_to_positional():
    def reward(a, b):
        return float(a == b)

    p = plan(reward)
    assert p.positional
    assert "positional" in p.describe()


def test_adapter_refuses_an_unreadable_signature():
    with pytest.raises(AdapterError, match="could not work out how to call"):
        plan(lambda x: 1.0)


def test_call_plan_describes_itself():
    described = CallPlan("completion", "reference", None, False, False).describe()
    assert "completion=completion" in described


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, 1.0),
        (False, 0.0),
        (1, 1.0),
        (0.25, 0.25),
        ([0.5], 0.5),
        ((1.0,), 1.0),
        ({"score": 0.75}, 0.75),
        ({"reward": 1}, 1.0),
    ],
)
def test_score_coercion(value, expected):
    assert to_score(value) == expected


def test_score_coercion_rejects_a_batch():
    with pytest.raises(AdapterError, match="returned 3 values"):
        to_score([1.0, 0.0, 1.0])


def test_score_coercion_rejects_a_dict_without_a_score():
    with pytest.raises(AdapterError, match="no score key"):
        to_score({"explanation": "looks fine"})


# -- audit mechanics --------------------------------------------------------


def test_a_crashing_verifier_is_reported_not_propagated():
    """A grader that raises on malformed input is a finding, not a tool failure:
    most training loops turn the exception into a silent zero."""

    def brittle(completion, reference):
        return float(completion[0] == reference[0])  # IndexError on ""

    result = audit(brittle)
    assert result.errors
    assert any("IndexError" in o.error for o in result.errors)
    assert len(result.outcomes) == len(corpus.load())


def test_a_verifier_that_accepts_everything_is_all_false_positives():
    result = audit(lambda completion, reference: 1.0, name="always")
    assert result.false_positive_rate == 1.0
    assert result.false_negative_rate == 0.0
    assert len(result.exploits_accepted) == len(corpus.load(categories=["exploit"]))


def test_a_verifier_that_rejects_everything_is_all_false_negatives():
    result = audit(lambda completion, reference: 0.0, name="never")
    assert result.false_positive_rate == 0.0
    assert result.false_negative_rate == 1.0


def test_threshold_is_respected():
    graded = audit(lambda completion, reference: 0.5, threshold=0.9)
    assert graded.false_positive_rate == 0.0
    lenient = audit(lambda completion, reference: 0.5, threshold=0.1)
    assert lenient.false_positive_rate == 1.0


def test_attacks_are_grouped_not_scattered():
    result = audit(substring_match)
    grouped = result.attacks_that_work
    assert "shotgun" in grouped
    assert len(grouped["shotgun"]) > 1


def test_headline_is_one_line():
    for fn in REFERENCE_VERIFIERS.values():
        assert "\n" not in audit(fn).headline


# -- the reference verifiers ------------------------------------------------


def test_substring_match_is_exploitable():
    """The most common hand-rolled verifier, and the reason this tool exists."""
    assert substring_match("The answer could be 40, 41, 42, 43, or 44.", "42") == 1.0
    assert substring_match("The answer is definitely not 42.", "42") == 1.0


def test_exact_match_is_safe_and_useless():
    result = audit(exact_match)
    assert result.false_positive_rate == 0.0
    assert result.false_negative_rate > 0.9


def test_last_number_is_fooled_by_negation():
    assert last_number("The answer is definitely not 42.", "42") == 1.0


def test_boxed_exact_refuses_multiple_boxes():
    assert boxed_exact("\\boxed{41} \\boxed{42}", "42") == 0.0
    assert boxed_exact("\\boxed{42}", "42") == 1.0


@pytest.mark.parametrize(
    "completion, reference",
    [
        ("42", "42"),
        ("\\boxed{42}", "42"),
        ("1/2", "0.5"),
        ("\\frac{1}{2}", "1/2"),
        ("2/4", "1/2"),
        ("0.3333333333", "1/3"),
        ("−5", "-5"),
        ("1,000", "1000"),
        ("2\\sqrt{3}", "2*sqrt(3)"),
        ("Answer: 42", "42"),
        ("#### 42", "42"),
        ("<think>maybe 41</think>\n\nThe answer is \\boxed{42}.", "42"),
    ],
)
def test_robust_match_accepts_correct_answers(completion, reference):
    assert robust_match(completion, reference) == 1.0, completion


@pytest.mark.parametrize(
    "completion",
    [
        "",
        "\\boxed{}",
        "The answer could be 40, 41, 42, 43, or 44.",
        "\\boxed{41} \\boxed{42} \\boxed{43}",
        "The answer is definitely not 42.",
        "<think>6 * 7 = 42.</think>\n\nThe answer is 43.",
        "0.05",
        "142",
    ],
)
def test_robust_match_rejects_wrong_and_degenerate_answers(completion):
    assert robust_match(completion, "42") == 0.0, completion


def test_robust_match_documents_its_one_false_negative():
    """The price of requiring an explicit assertion. Stated in the docstring and
    in the README rather than hidden."""
    result = audit(robust_match)
    assert [o.case.id for o in result.false_negatives] == ["fmt.markdown_heading"]


#: Published in the README. A change here is a documentation bug.
EXPECTED_RATES = {
    "substring_match": (0.38, 0.52, 9),
    "last_number": (0.05, 0.50, 2),
    "exact_match": (0.00, 0.96, 0),
    "boxed_exact": (0.00, 0.91, 0),
    "robust_match": (0.00, 0.02, 0),
}


@pytest.mark.parametrize("name", sorted(EXPECTED_RATES))
def test_published_rates_have_not_drifted(name):
    fp, fn, exploits = EXPECTED_RATES[name]
    result = audit(REFERENCE_VERIFIERS[name], name=name)
    assert result.false_positive_rate == pytest.approx(fp, abs=0.01), "README table is stale"
    assert result.false_negative_rate == pytest.approx(fn, abs=0.01), "README table is stale"
    assert len(result.exploits_accepted) == exploits
    assert not result.errors


# -- verifier profile -------------------------------------------------------


def test_profile_separates_permissive_from_format_strict():
    """The distinction this tool would be useless without.

    A verifier that requires `#### N` refuses most of the corpus. That is its
    format requirement, not a defect, and calling it a bug would be wrong.
    """
    assert audit(substring_match).profile == "permissive"
    assert audit(robust_match).profile == "balanced"
    assert audit(exact_match).profile == "format_strict"
    assert audit(boxed_exact).profile == "format_strict"


def test_format_strict_interpretation_points_at_the_exploit_subset():
    text = audit(boxed_exact).interpretation
    assert "--category exploit" in text
    assert "not a defect" in text or "rather than a defect" in text


def test_permissive_interpretation_names_the_real_risk():
    assert "policy will find them" in audit(substring_match).interpretation


def test_profile_appears_in_the_json_report():
    payload = json.loads(render(audit(substring_match), "json"))
    assert payload["profile"] == "permissive"
    assert payload["interpretation"]


def test_partial_credit_is_recognised_not_reported_as_permissive():
    """A shaped reward against a zero threshold reads as accepting everything.
    That is a threshold artefact, and the report has to say so."""

    def shaped(completion, reference):
        return 0.25 if reference in completion else 0.0

    result = audit(shaped)
    assert result.is_graded
    assert "partial credit" in result.interpretation
    assert "--threshold" in result.interpretation
    # And the threshold actually resolves it.
    assert audit(shaped, threshold=0.9).false_positive_rate == 0.0


def test_binary_verifiers_are_not_flagged_as_graded():
    assert not audit(robust_match).is_graded
    assert not audit(substring_match).is_graded


# -- the TRL chat protocol --------------------------------------------------


def _openr1_style_format_reward(completions, **kwargs):
    """Shaped like open-r1's reward functions: TRL chat-format completions."""
    return [1.0 if "<answer>" in c[0]["content"] else 0.0 for c in completions]


def test_chat_format_is_detected_by_probing_not_guessed():
    """Nothing in `f(completions, **kwargs)` says whether the elements are
    strings or message lists. Guessing wrong reports every case as a crash,
    which looks like the user's bug and is ours."""
    result = audit(_openr1_style_format_reward)
    assert not result.errors, [o.error for o in result.errors[:3]]
    assert "chat" in result.call_plan


def test_plain_string_functions_are_left_alone():
    result = audit(substring_match)
    assert "chat" not in result.call_plan


# -- reports and CLI --------------------------------------------------------


@pytest.mark.parametrize("fmt", ["terminal", "markdown", "json"])
def test_renderers_produce_output(fmt):
    text = render(audit(substring_match), fmt)
    assert isinstance(text, str) and len(text) > 200


def test_json_report_is_valid():
    payload = json.loads(render(audit(substring_match), "json"))
    assert payload["false_positive_rate"] > 0
    assert "shotgun" in payload["attacks_that_work"]
    assert payload["failures"]


def test_terminal_report_has_an_ascii_mode():
    from rewardlint.report.terminal import render as render_terminal

    render_terminal(audit(robust_match), color=False, unicode=False).encode("ascii", "replace")


def test_clean_verifier_renders_a_pass():
    text = render(audit(robust_match, name="robust"), "markdown")
    assert "correct completions rejected" in text or "False negative" in text


def test_unknown_format_is_rejected():
    with pytest.raises(KeyError, match="unknown format"):
        render(audit(exact_match), "pdf")


def test_cli_compare_exits_zero(capsys):
    assert main(["compare"]) == 0
    assert "substring_match" in capsys.readouterr().out


def test_cli_compare_markdown(capsys):
    assert main(["compare", "-f", "markdown"]) == 0
    assert "| `substring_match` |" in capsys.readouterr().out


def test_cli_corpus_lists_cases(capsys):
    assert main(["corpus", "--category", "exploit"]) == 0
    out = capsys.readouterr().out
    assert "x.shotgun.list" in out
    assert "must REJECT" in out


def test_cli_audit_gates_on_exploits(capsys):
    assert main(["audit", "rewardlint.reference:substring_match", "-f", "json"]) == 1
    assert main(["audit", "rewardlint.reference:robust_match", "-f", "json"]) == 0


def test_cli_audit_fail_on_any(capsys):
    assert main(["audit", "rewardlint.reference:robust_match", "-f", "json", "--fail-on", "any"]) == 1
    assert main(["audit", "rewardlint.reference:robust_match", "-f", "json", "--fail-on", "never"]) == 0


def test_cli_reports_a_bad_target_without_a_traceback(capsys):
    assert main(["audit", "nope.module:fn"]) == 2
    assert "rewardlint:" in capsys.readouterr().err


def test_cli_suggests_callables_when_the_attribute_is_missing(capsys):
    assert main(["audit", "rewardlint.reference:no_such_fn"]) == 2
    assert "robust_match" in capsys.readouterr().err


def test_cli_loads_a_function_from_a_file(tmp_path, capsys):
    module = tmp_path / "my_rewards.py"
    module.write_text(
        "def check(completion, reference):\n    return float(reference in completion)\n",
        encoding="utf-8",
    )
    assert main(["audit", f"{module}:check", "-f", "json", "--fail-on", "never"]) == 0
    assert json.loads(capsys.readouterr().out)["false_positive_rate"] > 0


def test_cli_writes_to_a_file(tmp_path, capsys):
    target = tmp_path / "report.md"
    main(["audit", "rewardlint.reference:last_number", "-f", "markdown",
          "-o", str(target), "--fail-on", "never"])
    assert target.read_text(encoding="utf-8").startswith("## rewardlint")


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()
