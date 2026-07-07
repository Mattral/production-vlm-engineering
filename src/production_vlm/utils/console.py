"""Console output helper: upgrades to ``rich`` formatting if installed, else plain stdlib prints.

Every example imports this instead of importing ``rich`` directly, so
the repo's core examples never hard-fail in a minimal environment —
they just look a little plainer.
"""

from __future__ import annotations

try:
    from rich.console import Console as _RichConsole
    from rich.table import Table as _RichTable

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


_KNOWN_TAGS = (
    "bold",
    "/bold",
    "red",
    "/red",
    "green",
    "/green",
    "yellow",
    "/yellow",
    "cyan",
    "/cyan",
    "magenta",
    "/magenta",
    "blue",
    "/blue",
    "dim",
    "/dim",
    "bold cyan",
    "/bold cyan",
    "bold green",
    "/bold green",
    "bold red",
    "/bold red",
)


def _strip_rich_markup(text: str) -> str:
    """Remove only known rich-style markup tags, leaving literal brackets (e.g. `[ml]`) untouched."""
    out = text
    for tag in _KNOWN_TAGS:
        out = out.replace(f"[{tag}]", "")
    return out


class Console:
    def __init__(self) -> None:
        self._impl = _RichConsole() if _HAS_RICH else None

    def print(self, text: str = "") -> None:
        if self._impl is not None:
            self._impl.print(text)
        else:
            print(_strip_rich_markup(text))

    def rule(self, title: str = "") -> None:
        if self._impl is not None:
            self._impl.rule(title)
        else:
            plain = _strip_rich_markup(title)
            bar = "=" * max(10, 70 - len(plain))
            print(f"\n{'=' * 4} {plain} {bar}")

    def table(self, title: str, columns: list[str], rows: list[list[str]]) -> None:
        if self._impl is not None and _HAS_RICH:
            t = _RichTable(title=title)
            for col in columns:
                t.add_column(col)
            for row in rows:
                t.add_row(*[str(c) for c in row])
            self._impl.print(t)
        else:
            print(f"\n{title}")
            widths = [max(len(str(c)) for c in [col, *[r[i] for r in rows]]) for i, col in enumerate(columns)]
            header = "  ".join(c.ljust(w) for c, w in zip(columns, widths, strict=True))
            print(header)
            print("-" * len(header))
            for row in rows:
                print("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)))
