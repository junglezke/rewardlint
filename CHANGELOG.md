# Changelog

## [0.1.0] — 2026-10-04

First public release.

### Added

- 87-case corpus across four categories: exploits (24), equivalence (40),
  distinction (14) and format (9). Four cases are marked contested and excluded
  from the headline rates.
- Nine exploit strategies, including three aimed at LLM-judge and rubric
  graders: prompt injection, forged grader output, and verbosity bias.
- Signature-inferring adapter covering the `f(completion, reference)`,
  `f(prediction, ground_truth)`, verl `f(solution_str, ground_truth)` and TRL
  batched `f(completions, **kwargs)` conventions, reporting the convention it
  chose so a wrong guess is visible.
- Five reference verifiers spanning the loose/strict trade-off, with their
  measured rates asserted in the test suite and published in the README.
- `robust_match`: 0% false positives, 2% false negatives, no exploits accepted.
- Terminal, Markdown and JSON reports; `audit`, `compare` and `corpus` CLI
  commands with `--fail-on` for CI gating.
- Verifier profile classification (permissive / format-strict / balanced), partial-credit
  detection, and TRL chat-format probing -- all three added after auditing real verifiers
  from verl and open-r1.
