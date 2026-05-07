"""Lightweight prose guard for the ICML manuscript."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


VERBATIM_RE = re.compile(
    r"\\begin\{(?P<env>lstlisting|verbatim\*?)\}.*?\\end\{(?P=env)\}",
    re.DOTALL,
)


@dataclass(frozen=True)
class Finding:
    severity: str
    path: Path
    line: int
    rule: str
    text: str


BANNED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("not-X-but-Y", re.compile(r"\bnot\b[^.\n;:]{0,120}\bbut\b", re.IGNORECASE)),
    (
        "not-only-but",
        re.compile(r"\bnot\s+only\b[^.\n;:]{0,120}\bbut\b", re.IGNORECASE),
    ),
    (
        "not-just-but",
        re.compile(r"\bnot\s+just\b[^.\n;:]{0,120}\bbut\b", re.IGNORECASE),
    ),
    (
        "not-merely-but",
        re.compile(r"\bnot\s+merely\b[^.\n;:]{0,120}\bbut\b", re.IGNORECASE),
    ),
    ("comma-not", re.compile(r",\s+not\b", re.IGNORECASE)),
    ("not-rather", re.compile(r"\bnot\b[^.\n;:]{0,120}\brather\b", re.IGNORECASE)),
    ("rather-than", re.compile(r"\brather\s+than\b", re.IGNORECASE)),
)

FORBIDDEN_CLAIMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "mistral-replicates",
        re.compile(r"\bMistral\b[^.\n]{0,120}\breplicat", re.IGNORECASE),
    ),
    (
        "mistral-closes-l1",
        re.compile(r"\bMistral\b[^.\n]{0,120}\bcloses?\s+L1\b", re.IGNORECASE),
    ),
    (
        "simid-improves-truthfulness",
        re.compile(
            r"\bSIMID\b[^.\n]{0,120}\bimproves?\s+truthfulness\b", re.IGNORECASE
        ),
    ),
    (
        "saes-fail",
        re.compile(
            r"\bSAEs?\b[^.\n]{0,80}\bfail(?:ed|s)?\b|\bfail(?:ed|s)?\b[^.\n]{0,80}\bSAEs?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "truthfulqa-transfer",
        re.compile(r"\bTruthfulQA\b[^.\n]{0,120}\btransfer(?:s|red)?\b", re.IGNORECASE),
    ),
    (
        "jailbreak-settled",
        re.compile(
            r"\bjailbreak\b[^.\n]{0,120}\b(settled|ground truth)\b", re.IGNORECASE
        ),
    ),
    (
        "wrong-entity-mechanism",
        re.compile(r"\bwrong-entity\b[^.\n]{0,120}\bmechanism\b", re.IGNORECASE),
    ),
    (
        "readouts-identify-steering-targets",
        re.compile(
            r"\breadouts?\b[^.\n]{0,120}\bidentify\b[^.\n]{0,80}\bsteering targets?\b",
            re.IGNORECASE,
        ),
    ),
)

TRUTHFULNESS_QUALIFIERS = (
    "TruthfulQA",
    "MC",
    "multiple-choice",
    "answer",
    "open",
    "generation",
    "factual",
    "direction",
    "signal",
    "readout",
    "surface",
    "claim",
)

STEERING_QUALIFIERS = (
    "activation",
    "operator",
    "surface",
    "gate",
    "control",
    "audit",
    "target",
    "intervention",
    "Gemma",
    "FaithEval",
    "claim",
    "method",
)


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def line_excerpt(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def mask_span(chars: list[str], start: int, end: int) -> None:
    for index in range(start, end):
        if chars[index] != "\n":
            chars[index] = " "


def is_escaped_percent(line: str, index: int) -> bool:
    slash_count = 0
    cursor = index - 1
    while cursor >= 0 and line[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def mask_latex_non_prose(text: str) -> str:
    chars = list(text)

    offset = 0
    for line in text.splitlines(keepends=True):
        line_end = offset + len(line)
        for index, char in enumerate(line):
            if char == "%" and not is_escaped_percent(line, index):
                mask_span(chars, offset + index, line_end)
                break
        offset = line_end

    comment_masked_text = "".join(chars)
    for match in VERBATIM_RE.finditer(comment_masked_text):
        mask_span(chars, match.start(), match.end())

    return "".join(chars)


def add_match(
    findings: list[Finding],
    severity: str,
    path: Path,
    text: str,
    match: re.Match[str],
    rule: str,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            path=path,
            line=line_number(text, match.start()),
            rule=rule,
            text=line_excerpt(text, match.start()),
        )
    )


def has_nearby_qualifier(
    text: str, start: int, end: int, qualifiers: tuple[str, ...]
) -> bool:
    window = text[max(0, start - 90) : min(len(text), end + 90)].lower()
    return any(qualifier.lower() in window for qualifier in qualifiers)


def check_path(path: Path) -> tuple[list[Finding], list[Finding]]:
    failures: list[Finding] = []
    warnings: list[Finding] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        failures.append(Finding("FAIL", path, 1, "file-not-found", str(error)))
        return failures, warnings
    except UnicodeDecodeError as error:
        failures.append(Finding("FAIL", path, 1, "unicode-decode-error", str(error)))
        return failures, warnings

    prose_text = mask_latex_non_prose(text)

    for match in re.finditer("\u2014", prose_text):
        add_match(failures, "FAIL", path, text, match, "unicode-em-dash")
    for match in re.finditer(r"---", prose_text):
        add_match(failures, "FAIL", path, text, match, "latex-em-dash")

    for rule, pattern in BANNED_PATTERNS:
        for match in pattern.finditer(prose_text):
            add_match(failures, "FAIL", path, text, match, rule)

    for rule, pattern in FORBIDDEN_CLAIMS:
        for match in pattern.finditer(prose_text):
            add_match(failures, "FAIL", path, text, match, rule)

    for match in re.finditer(r"\btruthfulness\b", prose_text, re.IGNORECASE):
        if not has_nearby_qualifier(
            prose_text, match.start(), match.end(), TRUTHFULNESS_QUALIFIERS
        ):
            add_match(
                warnings,
                "WARN",
                path,
                text,
                match,
                "truthfulness-without-nearby-qualifier",
            )

    for match in re.finditer(r"\bsteering\b", prose_text, re.IGNORECASE):
        if not has_nearby_qualifier(
            prose_text, match.start(), match.end(), STEERING_QUALIFIERS
        ):
            add_match(
                warnings, "WARN", path, text, match, "steering-without-nearby-scope"
            )

    return failures, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["paper/icml/main.tex"], help="LaTeX files to check"
    )
    args = parser.parse_args()

    failures: list[Finding] = []
    warnings: list[Finding] = []
    for raw_path in args.paths:
        path = Path(raw_path)
        path_failures, path_warnings = check_path(path)
        failures.extend(path_failures)
        warnings.extend(path_warnings)

    for finding in failures + warnings:
        print(
            f"{finding.severity} {finding.path}:{finding.line}: {finding.rule}: {finding.text}"
        )

    if failures:
        print(
            f"icml prose check failed: {len(failures)} failure(s), {len(warnings)} warning(s)",
            file=sys.stderr,
        )
        return 1

    print(f"icml prose check passed: 0 failure(s), {len(warnings)} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
