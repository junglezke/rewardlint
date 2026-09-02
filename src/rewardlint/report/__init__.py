"""Report renderers."""

from __future__ import annotations

import json
from typing import Callable, Dict


def render(audit, fmt: str = "terminal") -> str:
    renderers: Dict[str, Callable] = {
        "terminal": _terminal,
        "text": _terminal,
        "markdown": _markdown,
        "md": _markdown,
        "json": lambda a: json.dumps(a.to_dict(), indent=2, default=str),
    }
    try:
        return renderers[fmt](audit)
    except KeyError:
        raise KeyError(f"unknown format {fmt!r}; choose from {', '.join(sorted(renderers))}") from None


def _terminal(audit):
    from .terminal import render as r

    return r(audit)


def _markdown(audit):
    from .markdown import render as r

    return r(audit)


def emit(text: str) -> None:
    """Print text a terminal may not be able to encode.

    Reports can carry codepoints a legacy console code page cannot represent.
    Losing a glyph is acceptable; raising UnicodeEncodeError instead of showing
    the report is not.
    """
    import sys

    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(text)
