import tempfile
from pathlib import Path

import git


def clone_repo(url: str) -> Path:
    tmp = tempfile.mkdtemp(prefix="toolmaker_")
    git.Repo.clone_from(url, tmp, depth=1)
    return Path(tmp)


def read_repo_context(path: Path, max_lines: int = 150) -> str:
    parts = []

    for name in ["README.md", "README.rst", "README.txt", "README"]:
        readme = path / name
        if readme.exists():
            lines = readme.read_text(errors="ignore").splitlines()[:max_lines]
            parts.append(f"=== {name} ===\n" + "\n".join(lines))
            break

    for py in sorted(path.glob("*.py"))[:4]:
        lines = py.read_text(errors="ignore").splitlines()[:max_lines]
        parts.append(f"=== {py.name} ===\n" + "\n".join(lines))

    # Also check a src/ or main package dir
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir() and not subdir.name.startswith(".") and subdir.name not in ("tests", "docs", "examples"):
            for py in sorted(subdir.glob("*.py"))[:3]:
                lines = py.read_text(errors="ignore").splitlines()[:max_lines]
                parts.append(f"=== {subdir.name}/{py.name} ===\n" + "\n".join(lines))
            break

    return "\n\n".join(parts)
