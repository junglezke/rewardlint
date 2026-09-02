"""The unit of the corpus: one (reference, completion, expected verdict) triple.

Every case states *why* it exists. A test case whose rationale is not written
down cannot be argued with, and this corpus is entirely a collection of
opinions about what a verifier should and should not accept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

#: What a case is testing.
#:
#: ``equivalence``  the completion is correct in a different surface form.
#:                  Rejecting it is a false negative: real learning signal
#:                  thrown away, and a model punished for being right.
#: ``distinction``  the completion is genuinely wrong. Accepting it is a false
#:                  positive: the verifier teaching the policy a falsehood.
#: ``exploit``      the completion is a degenerate strategy a policy can find
#:                  by gradient descent without solving anything. Accepting it
#:                  is a reward hack waiting to happen.
#: ``format``       the completion is correct but wrapped in realistic model
#:                  output -- prose, markdown, reasoning. Rejecting it means
#:                  your verifier only works on completions it will never see.
CATEGORIES = ("equivalence", "distinction", "exploit", "format")


@dataclass(frozen=True)
class Case:
    """One graded example."""

    id: str
    category: str
    domain: str
    reference: str
    completion: str
    #: True when a correct verifier accepts this completion.
    expect: bool
    why: str
    #: For exploits, the name of the strategy being attempted.
    attack: Optional[str] = None
    #: Optional problem statement, for verifiers that take one.
    prompt: str = ""
    tags: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.category == "distinction" and self.expect:
            raise ValueError(f"{self.id}: a 'distinction' case must expect rejection")
        if self.category == "exploit" and self.expect:
            raise ValueError(f"{self.id}: an 'exploit' case must expect rejection")
        if self.category in ("equivalence", "format") and not self.expect:
            raise ValueError(f"{self.id}: an '{self.category}' case must expect acceptance")

    @property
    def failure_mode(self) -> str:
        """What it costs you when a verifier gets this case wrong."""
        return "false positive" if self.expect is False else "false negative"

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "category": self.category,
            "domain": self.domain,
            "reference": self.reference,
            "completion": self.completion,
            "expect": self.expect,
            "why": self.why,
            "attack": self.attack,
            "prompt": self.prompt,
            "tags": list(self.tags),
        }
