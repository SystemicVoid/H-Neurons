import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review the 3,500-entry TriviaQA consistency batch and write a markdown report."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/gemma3_4b/consistency_samples.jsonl"),
        help="Input JSONL batch file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/batch3500_review.md"),
        help="Output markdown report path.",
    )
    return parser.parse_args()


from utils import normalize_answer  # noqa: E402


@dataclass
class Entry:
    qid: str
    question: str
    responses: list[str]
    judges: list[str]
    ground_truth: list[str]


def load_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = json.loads(line)
            if len(payload) != 1:
                raise ValueError(
                    f"Line {line_number} does not contain exactly one QID object."
                )

            qid, content = next(iter(payload.items()))
            required = {"question", "responses", "judges", "ground_truth"}
            missing = required - set(content)
            if missing:
                raise ValueError(
                    f"Line {line_number} missing fields: {sorted(missing)}"
                )

            responses = content["responses"]
            judges = content["judges"]
            if len(responses) != 10 or len(judges) != 10:
                raise ValueError(
                    f"Line {line_number} has responses={len(responses)} and judges={len(judges)}; expected 10 each."
                )

            entries.append(
                Entry(
                    qid=str(qid),
                    question=str(content["question"]),
                    responses=[str(response) for response in responses],
                    judges=[str(judge) for judge in judges],
                    ground_truth=[str(alias) for alias in content["ground_truth"]],
                )
            )
    return entries


def qid_sort_key(qid: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", qid)
    if match:
        return (int(match.group(1)), qid)
    return (10**18, qid)


def word_count(text: str) -> int:
    return len(text.split())


def nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("Cannot compute percentiles for an empty list.")
    if percentile <= 0:
        return values[0]
    if percentile >= 100:
        return values[-1]
    rank = math.ceil((percentile / 100) * len(values))
    return values[max(0, rank - 1)]


def markdown_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def render_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_report(entries: list[Entry], input_path: Path) -> str:
    total_entries = len(entries)

    all_correct_entries = [
        entry for entry in entries if all(judge == "true" for judge in entry.judges)
    ]
    all_incorrect_entries = [
        entry for entry in entries if all(judge == "false" for judge in entry.judges)
    ]
    mixed_entries = [
        entry
        for entry in entries
        if entry not in all_correct_entries and entry not in all_incorrect_entries
    ]
    uncertain_entries = [
        entry
        for entry in entries
        if any(judge == "uncertain" for judge in entry.judges)
    ]

    all_responses: list[tuple[str, str]] = []
    word_counts: list[int] = []
    long_responses: list[tuple[str, int, str]] = []
    multiline_by_qid: dict[str, int] = {}
    multiline_total = 0

    loophole_rows: list[list[str]] = []
    loophole_qids: set[str] = set()

    mixed_rows: list[list[str]] = []
    varying_all_correct: list[tuple[str, int, list[str]]] = []

    passing_total = 0
    passing_true = 0
    passing_false = 0

    yield_curve_rows: list[list[str]] = []
    cumulative_all_correct = 0
    milestone_set = {500, 1000, 1500, 2000, 2500, 3000, 3500}

    for index, entry in enumerate(entries, start=1):
        if all(judge == "true" for judge in entry.judges):
            cumulative_all_correct += 1
        if index in milestone_set:
            yield_curve_rows.append([str(index), str(cumulative_all_correct)])

        if (
            len(set(entry.judges)) == 1
            and "uncertain" not in entry.judges
            and "error" not in entry.judges
        ):
            passing_total += 1
            if entry.judges[0] == "true":
                passing_true += 1
            elif entry.judges[0] == "false":
                passing_false += 1

        normalized_responses = sorted(
            {normalize_answer(response) for response in entry.responses}
        )

        if entry in mixed_entries:
            mixed_rows.append(
                [
                    entry.qid,
                    str(entry.judges.count("true")),
                    str(entry.judges.count("false")),
                    str(entry.judges.count("uncertain")),
                    str(len(normalized_responses)),
                ]
            )

        if entry in all_correct_entries and len(normalized_responses) > 1:
            varying_all_correct.append(
                (entry.qid, len(normalized_responses), normalized_responses)
            )

        norm_ground_truth = [
            (alias, normalize_answer(alias)) for alias in entry.ground_truth
        ]

        for response in entry.responses:
            all_responses.append((entry.qid, response))

            response_word_count = word_count(response)
            word_counts.append(response_word_count)
            if response_word_count > 15:
                long_responses.append((entry.qid, response_word_count, response))

            if "\n" in response:
                multiline_total += 1
                multiline_by_qid[entry.qid] = multiline_by_qid.get(entry.qid, 0) + 1

        if entry in all_correct_entries:
            for response in entry.responses:
                if word_count(response) <= 3:
                    continue
                norm_response = normalize_answer(response)
                for original_alias, norm_alias in norm_ground_truth:
                    if norm_alias and norm_alias in norm_response:
                        loophole_qids.add(entry.qid)
                        loophole_rows.append(
                            [
                                entry.qid,
                                markdown_escape(original_alias),
                                str(word_count(response)),
                                markdown_escape(
                                    json.dumps(response, ensure_ascii=False)
                                ),
                            ]
                        )
                        break
                else:
                    continue
                break

    sorted_word_counts = sorted(word_counts)
    percentiles = {
        "p5": nearest_rank(sorted_word_counts, 5),
        "p10": nearest_rank(sorted_word_counts, 10),
        "p25": nearest_rank(sorted_word_counts, 25),
        "p50": nearest_rank(sorted_word_counts, 50),
        "p75": nearest_rank(sorted_word_counts, 75),
        "p90": nearest_rank(sorted_word_counts, 90),
        "p95": nearest_rank(sorted_word_counts, 95),
        "p99": nearest_rank(sorted_word_counts, 99),
        "max": sorted_word_counts[-1],
    }

    mixed_rows.sort(key=lambda row: qid_sort_key(row[0]))
    loophole_rows.sort(key=lambda row: qid_sort_key(row[0]))
    long_responses.sort(key=lambda item: (qid_sort_key(item[0]), item[1], item[2]))
    varying_all_correct.sort(key=lambda item: (-item[1], qid_sort_key(item[0])))

    report: list[str] = []
    report.append("# Batch 3500 Review")
    report.append("")
    report.append(f"Reviewed file: `{input_path}`  ")
    report.append(
        "Batch size: 3,500 questions x 10 responses each = 35,000 judged responses"
    )
    report.append("")
    report.append("## 1. Basic Counts")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["Total entries", str(total_entries)],
                ["All-correct", str(len(all_correct_entries))],
                ["All-incorrect", str(len(all_incorrect_entries))],
                ["Mixed", str(len(mixed_entries))],
                ["Uncertain-containing", str(len(uncertain_entries))],
                [
                    "All-correct + all-incorrect + mixed",
                    str(
                        len(all_correct_entries)
                        + len(all_incorrect_entries)
                        + len(mixed_entries)
                    ),
                ],
            ],
        )
    )
    report.append("")
    report.append("## 2. Substring Loophole Quantification")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["All-correct entries", str(len(all_correct_entries))],
                [
                    "All-correct entries with a >3-word substring-match response",
                    str(len(loophole_qids)),
                ],
            ],
        )
    )
    report.append("")
    if loophole_rows:
        report.extend(
            render_table(
                ["QID", "Matched alias", "Words", "Example response"],
                loophole_rows,
            )
        )
    else:
        report.append("| QID | Matched alias | Words | Example response |")
        report.append("| --- | --- | --- | --- |")
        report.append("| None | None | 0 | None |")
    report.append("")
    report.append("## 3. Response Length Distribution")
    report.append("")
    report.extend(
        render_table(
            ["Percentile", "Word count"],
            [[label, str(value)] for label, value in percentiles.items()],
        )
    )
    report.append("")
    if long_responses:
        report.extend(
            render_table(
                ["QID", "Words", "Response"],
                [
                    [
                        qid,
                        str(words),
                        markdown_escape(json.dumps(response, ensure_ascii=False)),
                    ]
                    for qid, words, response in long_responses
                ],
            )
        )
    else:
        report.append("| QID | Words | Response |")
        report.append("| --- | --- | --- |")
        report.append("| None | 0 | None |")
    report.append("")
    report.append("## 4. Multiline Response Count")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [["Responses containing a newline", str(multiline_total)]],
        )
    )
    report.append("")
    if multiline_by_qid:
        report.extend(
            render_table(
                ["QID", "Multiline responses"],
                [
                    [qid, str(multiline_by_qid[qid])]
                    for qid in sorted(multiline_by_qid, key=qid_sort_key)
                ],
            )
        )
    else:
        report.append("| QID | Multiline responses |")
        report.append("| --- | --- |")
        report.append("| None | 0 |")
    report.append("")
    report.append("## 5. Mixed-Label Breakdown")
    report.append("")
    if mixed_rows:
        report.extend(
            render_table(
                [
                    "QID",
                    "True count",
                    "False count",
                    "Uncertain count",
                    "Distinct normalized responses",
                ],
                mixed_rows,
            )
        )
    else:
        report.append(
            "| QID | True count | False count | Uncertain count | Distinct normalized responses |"
        )
        report.append("| --- | --- | --- | --- | --- |")
        report.append("| None | 0 | 0 | 0 | 0 |")
    report.append("")
    report.append("## 6. All-Correct Yield Curve")
    report.append("")
    report.extend(
        render_table(["Milestone", "Cumulative all-correct"], yield_curve_rows)
    )
    report.append("")
    report.append("## 7. Judge Agreement Sanity Check on All-Correct")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["All-correct entries", str(len(all_correct_entries))],
                [
                    "All-correct entries with >1 distinct normalized response",
                    str(len(varying_all_correct)),
                ],
            ],
        )
    )
    report.append("")
    if varying_all_correct:
        report.extend(
            render_table(
                ["QID", "Distinct normalized responses", "Normalized forms"],
                [
                    [qid, str(count), markdown_escape("; ".join(forms))]
                    for qid, count, forms in varying_all_correct[:20]
                ],
            )
        )
    else:
        report.append("| QID | Distinct normalized responses | Normalized forms |")
        report.append("| --- | --- | --- |")
        report.append("| None | 0 | None |")
    report.append("")
    report.append("## 8. Downstream Usability")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["Total passing entries", str(passing_total)],
                ["Passing-true", str(passing_true)],
                ["Passing-false", str(passing_false)],
            ],
        )
    )
    report.append("")
    report.append("## Totals")
    report.append("")
    report.extend(
        render_table(
            ["Metric", "Count"],
            [
                ["Total responses", str(len(all_responses))],
                ["Expected responses", str(total_entries * 10)],
            ],
        )
    )
    report.append("")

    return "\n".join(report)


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input_path)
    report = build_report(entries, args.input_path)
    args.output_path.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
