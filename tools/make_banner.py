"""Regenerate docs/assets/report.svg, the banner in the README.

    python tools/make_banner.py

Kept in the repo so the README image is reproducible rather than a screenshot
nobody can regenerate after the report format changes.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from ansi2svg import render  # noqa: E402

from rewardlint import audit  # noqa: E402
from rewardlint.reference import substring_match  # noqa: E402
from rewardlint.report.terminal import render as render_report  # noqa: E402

lines = render_report(
    audit(substring_match, name="my_project.rewards:accuracy_reward"),
    width=84,
    color=True,
    unicode=True,
).split("\n")

head = lines[1:14]
exploits = next(i for i, line in enumerate(lines) if "EXPLOITS THAT WORK" in line)
body = lines[exploits : exploits + 13]

svg = render("\n".join(head + body).rstrip(), "rewardlint audit my_project.rewards:accuracy_reward")
out = pathlib.Path(__file__).parent.parent / "docs" / "assets" / "report.svg"
out.write_text(svg, encoding="utf-8")
print(f"wrote {out}")
