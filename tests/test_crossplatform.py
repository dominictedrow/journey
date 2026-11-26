"""Cross-platform guard: no hard-coded POSIX paths or os.path in src/ (CLAUDE.md)."""

import re

from qt.common.paths import REPO_ROOT, visible_files

SRC = REPO_ROOT / "src"
BAD = [
    (re.compile(r"""["'](/Users|/Volumes|/home|/tmp|/var|/etc)/"""), "hard-coded absolute path"),
    (re.compile(r"\bos\.path\."), "os.path (use pathlib)"),
    (re.compile(r"""["'][A-Za-z]:\\\\"""), "hard-coded Windows path"),
]


def test_no_hardcoded_paths():
    problems = []
    for py in (p for p in SRC.rglob("*.py") if not p.name.startswith("._")):
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            for rx, why in BAD:
                if rx.search(line):
                    problems.append(f"{py.relative_to(REPO_ROOT)}:{n}: {why}")
    assert not problems, "\n".join(problems)


def test_no_shell_scripts():
    scripts = REPO_ROOT / "scripts"
    shells = [p.name for p in visible_files(scripts, "*") if p.suffix in {".sh", ".bash", ".zsh"}]
    assert not shells
