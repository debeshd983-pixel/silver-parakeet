"""Bump the package version in every place it is recorded.

Usage:
    python scripts/bump_version.py patch        # 1.0.0 -> 1.0.1
    python scripts/bump_version.py minor        # 1.0.0 -> 1.1.0
    python scripts/bump_version.py major        # 1.0.0 -> 2.0.0
    python scripts/bump_version.py 1.2.3        # set an explicit version
    python scripts/bump_version.py patch --dry-run

Files kept in sync:
    pyproject.toml          [project] version = "..."
    sahu65/__init__.py      __version__ = "..."

Why this exists: PyPI releases are immutable, so `ci.yml` publishes with
skip-existing: true and a push that does not change the version uploads nothing.
Shipping a change therefore means bumping the version first - doing it by hand in
two files is how they drift apart. After bumping, commit and push to main; the
pipeline publishes automatically.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "sahu65" / "__init__.py"

# (file, pattern, replacement template). Exactly one match required per file.
TARGETS = (
    (PYPROJECT, re.compile(r'(?m)^version = "[^"]*"$'), 'version = "{version}"'),
    (PACKAGE_INIT, re.compile(r'(?m)^__version__ = "[^"]*"$'), '__version__ = "{version}"'),
)

VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_current(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        sys.exit(f"ERROR: no version line found in {path.relative_to(ROOT)}")
    return match.group(0).split('"')[1]


def compute(current: str, spec: str) -> str:
    if VERSION_RE.fullmatch(spec):
        return spec
    match = VERSION_RE.fullmatch(current)
    if match is None:
        sys.exit(f"ERROR: current version {current!r} is not X.Y.Z; pass an explicit version")
    major, minor, patch = (int(part) for part in match.groups())
    if spec == "major":
        return f"{major + 1}.0.0"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    if spec == "patch":
        return f"{major}.{minor}.{patch + 1}"
    sys.exit(f"ERROR: {spec!r} is not major, minor, patch, or an X.Y.Z version")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", help="major | minor | patch | explicit X.Y.Z")
    parser.add_argument("--dry-run", action="store_true", help="print changes, write nothing")
    args = parser.parse_args()

    currents = {path: read_current(path, pattern) for path, pattern, _ in TARGETS}
    unique = set(currents.values())
    if len(unique) > 1:
        # Healable: both files are rewritten from the pyproject value below, but say so.
        listed = ", ".join(f"{path.name}={ver}" for path, ver in currents.items())
        print(f"WARNING: version lines disagree: {listed}")
    current = currents[PYPROJECT]
    new = compute(current, args.spec)

    for path, pattern, template in TARGETS:
        text = path.read_text(encoding="utf-8")
        updated, count = pattern.subn(template.format(version=new), text, count=1)
        if count != 1:
            sys.exit(f"ERROR: expected exactly one version line in {path.relative_to(ROOT)}, found {count}")
        if args.dry_run:
            print(f"would write {path.relative_to(ROOT)}: {current} -> {new}")
            continue
        path.write_text(updated, encoding="utf-8", newline="")
        print(f"updated {path.relative_to(ROOT)}: {current} -> {new}")

    if args.dry_run:
        print("dry run - nothing written")
    else:
        print(f"\nnext: git add -A && git commit -m 'sahu65 {new}' && git push")
        print("a push to main builds and publishes sahu65 " + new + " automatically")
    return 0


if __name__ == "__main__":
    sys.exit(main())