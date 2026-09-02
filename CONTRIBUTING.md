# Contributing to rewardlint

## The most valuable contribution

**An exploit that worked on a real verifier and is not in the corpus.** Open an
issue with the completion and, if you can, what the grader was doing. You do not
need to write any code — a strategy that actually beat a production grader is
worth more than a hundred synthetic variations, and turning it into a case is
five minutes of work for a maintainer.

## Second: a case you think is wrong

Some of these are genuinely contestable. Is `5 meters` correct when the reference
says `5`? Is `3.14` close enough to `3.14159`? If your verifier disagrees with the
corpus and you think your verifier is right, say so. Either the case moves into
the `contested` set or the corpus gets more accurate; both are wins.

## Adding a case

One entry in `src/rewardlint/corpus/`. Every case must carry:

- a stable `id` in the existing namespace (`eq.*`, `ne.*`, `fmt.*`, `x.<attack>.*`),
- the right `category` — the dataclass enforces that exploits and distinctions
  expect rejection and that equivalence and format cases expect acceptance,
- a `why` that a reader can disagree with. This corpus is a collection of
  opinions about what a verifier should accept; an opinion with no reason
  attached cannot be argued with or improved.

New exploits also need an entry in `ATTACKS` describing the strategy, because the
report groups by attack: "the shotgun strategy works" is one actionable fact, not
six scattered failures.

## Adding a reference verifier

Only if it represents a pattern people actually write. The point of
`REFERENCE_VERIFIERS` is to show where real code sits on the loose/strict
trade-off, not to collect strawmen.

Note that the measured rates for all of them are asserted as exact values in
`tests/test_rewardlint.py` and published in the README. If your change moves
them, update both — a stale README is a bug.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests examples
```

Please keep the package dependency-free. It has to install next to whatever a
user's training image already pins.
