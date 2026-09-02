"""Numeric and symbolic answer cases.

Two failure directions, and they hurt differently.

**False negatives** (rejecting a correct answer) throw away learning signal and
actively teach the policy that a right answer was wrong. In GRPO they also
shrink group variance, so they cost you rollouts as well as accuracy.

**False positives** (accepting a wrong answer) are worse: they are the gradient
that trains a reward hack.

Every case below is a surface form that a language model actually produces.
"""

from __future__ import annotations

from typing import List

from ..case import Case


def _eq(cid: str, reference: str, completion: str, why: str, **kw) -> Case:
    return Case(
        id=cid,
        category="equivalence",
        domain="math",
        reference=reference,
        completion=completion,
        expect=True,
        why=why,
        **kw,
    )


def _ne(cid: str, reference: str, completion: str, why: str, **kw) -> Case:
    return Case(
        id=cid,
        category="distinction",
        domain="math",
        reference=reference,
        completion=completion,
        expect=False,
        why=why,
        **kw,
    )


def _fmt(cid: str, reference: str, completion: str, why: str, **kw) -> Case:
    return Case(
        id=cid,
        category="format",
        domain="math",
        reference=reference,
        completion=completion,
        expect=True,
        why=why,
        **kw,
    )


EQUIVALENCE: List[Case] = [
    # -- numeric surface forms ---------------------------------------------
    _eq("eq.int.plain", "42", "42", "the trivial case; if this fails nothing else matters"),
    _eq("eq.int.whitespace", "42", "  42  ", "models pad answers with whitespace constantly"),
    _eq("eq.int.trailing_period", "42", "42.", "a sentence-final period is not a decimal point"),
    _eq("eq.int.thousands_sep", "1000", "1,000", "grouping separators are a formatting choice"),
    _eq("eq.int.thousands_ref", "1,000", "1000", "and the reference may carry them instead"),
    _eq("eq.int.leading_zeros", "42", "042", "leading zeros do not change an integer"),
    _eq("eq.int.plus_sign", "42", "+42", "an explicit plus sign is redundant, not wrong"),
    _eq("eq.float.trailing_zero", "0.5", "0.50", "trailing zeros are not significant here"),
    _eq("eq.float.leading_dot", "0.5", ".5", "a bare decimal point is common in model output"),
    _eq("eq.float.int_valued", "3", "3.0", "3.0 and 3 are the same number"),
    _eq("eq.float.sci_notation", "1000", "1e3", "scientific notation for a round number"),
    _eq("eq.float.sci_upper", "0.001", "1E-3", "uppercase exponent marker"),
    _eq("eq.neg.unicode_minus", "-5", "\u22125", "U+2212 MINUS SIGN, emitted by LaTeX-aware models"),
    _eq("eq.neg.spaced", "-5", "- 5", "a space after the sign is a tokenisation artefact"),
    _eq("eq.zero.negative", "0", "-0", "negative zero is zero"),
    # -- fractions and radicals --------------------------------------------
    _eq("eq.frac.slash", "0.5", "1/2", "a fraction and its decimal are the same value"),
    _eq("eq.frac.latex", "1/2", "\\frac{1}{2}", "LaTeX fraction, the default for a math-tuned model"),
    _eq("eq.frac.dfrac", "1/2", "\\dfrac{1}{2}", "\\dfrac is \\frac with different spacing"),
    _eq("eq.frac.unreduced", "1/2", "2/4", "an unreduced fraction is still the same number"),
    _eq("eq.frac.repeating", "1/3", "0.3333333333", "a truncated decimal for a repeating fraction"),
    _eq("eq.frac.mixed", "3/2", "1 1/2", "mixed-number notation"),
    _eq("eq.radical.latex", "2*sqrt(3)", "2\\sqrt{3}", "LaTeX radical vs a plain-text one"),
    _eq("eq.radical.spacing", "\\sqrt{2}", "\\sqrt 2", "LaTeX braces are optional for one token"),
    _eq("eq.pi.symbol", "\\pi", "\u03c0", "the Unicode symbol and the LaTeX macro"),
    _eq("eq.pi.coefficient", "2\\pi", "2 \\pi", "spacing around a coefficient"),
    # -- wrappers the model adds -------------------------------------------
    _eq("eq.wrap.dollars", "5", "$5$", "inline math delimiters"),
    _eq("eq.wrap.boxed", "5", "\\boxed{5}", "\\boxed is how every math-RL recipe asks for the answer"),
    _eq("eq.wrap.boxed_ref", "\\boxed{5}", "5", "and the reference may be the boxed one"),
    _eq("eq.wrap.text", "5", "\\text{5}", "\\text wrapping from a verbose model"),
    _eq("eq.wrap.parens", "5", "(5)", "a parenthesised answer; models wrap results in brackets constantly"),
    _eq("eq.wrap.equals", "5", "x = 5", "restating the variable is not a different answer"),
    _eq("eq.wrap.units", "5", "5 meters", "a unit the reference omitted", tags=("contested",)),
    _eq("eq.wrap.percent_sign", "50\\%", "50%", "escaped vs literal percent sign"),
    # -- non-numeric answers ------------------------------------------------
    _eq("eq.bool.case", "yes", "Yes", "capitalisation of a yes/no answer"),
    _eq("eq.bool.punct", "yes", "Yes.", "with sentence punctuation"),
    _eq("eq.choice.case", "A", "a", "multiple-choice letter case"),
    _eq("eq.choice.parens", "A", "(A)", "the letter as a parenthesised option"),
    _eq("eq.text.article", "the derivative", "derivative", "a leading article does not change which concept was named"),
    _eq("eq.tuple.spacing", "(1,2)", "(1, 2)", "spacing inside a coordinate pair"),
    _eq("eq.set.order", "{1,2,3}", "{3,2,1}", "set membership is unordered", tags=("contested",)),
]

DISTINCTION: List[Case] = [
    _ne("ne.magnitude.decimal", "0.5", "0.05", "an order-of-magnitude error is a wrong answer"),
    _ne("ne.magnitude.int", "10", "100", "a factor of ten apart"),
    _ne("ne.transposed", "12", "21", "transposed digits; a substring matcher can be fooled here"),
    _ne("ne.sign", "5", "-5", "opposite sign -- the classic sign-error answer, and it must not pass"),
    _ne("ne.frac.inverted", "1/2", "2/1", "an inverted fraction is a different number"),
    _ne("ne.precision", "3.14159", "3.14", "a truncation that loses required precision", tags=("contested",)),
    _ne("ne.off_by_one", "42", "43", "off by one, the most common genuine arithmetic slip"),
    _ne("ne.prefix", "5", "52", "5 is a prefix of 52; a prefix match accepts this"),
    _ne("ne.substring", "42", "142", "42 is a substring of 142; a substring match accepts this"),
    _ne("ne.digit_inside", "7", "17", "the reference digit appears inside a longer number"),
    _ne("ne.bool.negated", "yes", "no", "the opposite answer to a yes/no question"),
    _ne("ne.choice.wrong", "A", "B", "a different multiple-choice option"),
    _ne("ne.empty_vs_zero", "0", "", "an empty answer is not zero"),
    _ne("ne.units_wrong", "5 meters", "5 feet", "same number, wrong unit"),
]

FORMAT: List[Case] = [
    _fmt(
        "fmt.chain_of_thought",
        "42",
        "Let me work through this step by step.\n\nFirst, 6 times 7.\n6 * 7 = 42.\n\n"
        "Therefore the answer is \\boxed{42}.",
        "the shape of essentially every reasoning-model completion",
    ),
    _fmt(
        "fmt.think_tags",
        "42",
        "<think>\nI need 6 * 7. That is 42.\n</think>\n\nThe answer is 42.",
        "R1-style completions put reasoning in <think> tags",
    ),
    _fmt(
        "fmt.markdown_heading",
        "42",
        "## Solution\n\nThe product is **42**.",
        "markdown formatting around the answer",
    ),
    _fmt(
        "fmt.answer_label",
        "42",
        "Answer: 42",
        "an explicit answer label",
    ),
    _fmt(
        "fmt.final_answer_label",
        "42",
        "#### 42",
        "the GSM8K answer delimiter",
    ),
    _fmt(
        "fmt.multiline_latex",
        "42",
        "\\begin{align}\nx &= 6 \\times 7 \\\\\n  &= 42\n\\end{align}\n\n\\boxed{42}",
        "a LaTeX derivation ending in the answer",
    ),
    _fmt(
        "fmt.self_correction",
        "42",
        "The answer is 41. Wait, let me recheck: 6 * 7 = 42. The answer is \\boxed{42}.",
        "self-correction is common and the *last* answer is the claimed one",
        tags=("contested",),
    ),
    _fmt(
        "fmt.trailing_prose",
        "42",
        "\\boxed{42}\n\nLet me know if you would like me to explain any step.",
        "assistant-style sign-off after the answer",
    ),
    _fmt(
        "fmt.code_block",
        "42",
        "```python\nresult = 6 * 7\nprint(result)  # 42\n```\n\nThe answer is 42.",
        "an answer computed in a code block and then stated",
    ),
]

ALL: List[Case] = EQUIVALENCE + DISTINCTION + FORMAT
