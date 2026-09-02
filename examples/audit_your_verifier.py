"""Audit a reward function you just wrote.

Run: python examples/audit_your_verifier.py

The grader below is deliberately a *reasonable* attempt -- extract the last
boxed answer, compare it to the reference. Run it and see which of the nine
exploit strategies still get through.
"""

import re

from rewardlint import audit

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def my_reward(completion: str, reference: str) -> float:
    """Pull the last boxed answer and compare it to the reference."""
    boxes = _BOXED.findall(completion)
    if not boxes:
        return 0.0
    return float(boxes[-1].strip() == reference.strip())


result = audit(my_reward)
print(result.headline)
print()
result.print()

# Machine-readable, for CI or a dashboard.
print("attacks that work:", sorted(result.attacks_that_work) or "none")
print("exit code with --fail-on exploit:", 1 if result.exploits_accepted else 0)
