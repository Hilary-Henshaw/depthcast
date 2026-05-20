"""Replace a sentinel-delimited section of the README with stdin.

Finds the block between ``<!-- BEGIN:[section] -->`` and
``<!-- END:[section] -->`` in the README and replaces its contents with
whatever is read from standard input. Writes the README in place.

App-independent. Typical use with the benchmark script::

    python scripts/benchmark.py | \\
        python scripts/inject_readme_section.py --section benchmarks

Exits non-zero with a clear message if the sentinels are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_DEFAULT_README = Path(__file__).resolve().parents[1] / "README.md"


def inject(readme_path: Path, section: str, content: str) -> None:
    """Replace the named sentinel block in ``readme_path`` with content.

    Args:
        readme_path: Path to the README file to edit in place.
        section: Sentinel name used in the BEGIN/END markers.
        content: Replacement text placed between the markers.

    Raises:
        FileNotFoundError: If the README does not exist.
        ValueError: If either sentinel marker is missing or malformed.
    """
    if not readme_path.is_file():
        raise FileNotFoundError(f"README not found: {readme_path}")

    begin = f"<!-- BEGIN:{section} -->"
    end = f"<!-- END:{section} -->"
    text = readme_path.read_text(encoding="utf-8")

    begin_index = text.find(begin)
    end_index = text.find(end)
    if begin_index == -1 or end_index == -1:
        raise ValueError(
            f"Could not find sentinels for section '{section}'. "
            f"Expected both {begin} and {end} in {readme_path}."
        )
    if end_index < begin_index:
        raise ValueError(
            f"Sentinel order is wrong for section '{section}': "
            f"{end} appears before {begin}."
        )

    body = content.strip("\n")
    updated = (
        text[: begin_index + len(begin)]
        + "\n"
        + body
        + "\n"
        + text[end_index:]
    )
    readme_path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, read stdin, and inject the section."""
    parser = argparse.ArgumentParser(
        description="Inject stdin into a README sentinel block."
    )
    parser.add_argument(
        "--section",
        required=True,
        help="Sentinel name, e.g. 'benchmarks'.",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=_DEFAULT_README,
        help="Path to the README (defaults to the project README).",
    )
    args = parser.parse_args(argv)

    content = sys.stdin.read()
    if not content.strip():
        parser.error("No content received on stdin.")

    try:
        inject(args.readme, args.section, content)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        f"Injected section '{args.section}' into {args.readme}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
