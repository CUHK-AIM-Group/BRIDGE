from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "__pycache__"}
PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]"),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
    "credential-like token": re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[:=]\s*[^\s,]+"),
    "unredacted cohort placeholder": re.compile(r"(?i)private[_ -]?(?:cohort|site)[_ -]?name"),
    "subject identifier": re.compile(r"(?i)\b(?:RID|PTID|patient)[_-]?\d{3,}\b"),
}


def main() -> None:
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        if path.suffix.lower() in {".jpg", ".png", ".pdf", ".pt", ".nii", ".gz"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(f"{path.relative_to(ROOT)}: {name}: {match.group(0)!r}")
    if findings:
        raise SystemExit("\n".join(findings))
    print("Sensitive-information scan passed.")


if __name__ == "__main__":
    main()
