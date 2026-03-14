# Batch 100 Review

Reviewed file: `data/gemma3_4b_TriviaQA_consistency_samples.jsonl`  
Batch size: 100 questions x 10 responses each = 1,000 judged responses

## Summary

| Check | Status | Notes |
| --- | --- | --- |
| 1. Data integrity | PASS | 100/100 lines parsed as valid JSON; each line has exactly one QID object with `question`, `responses`, `judges`, `ground_truth`; all `responses` and `judges` lists have length 10. |
| 2. Response quality | FAIL | In a 10-entry spot sample, 7/10 looked like plausible short answers; 3/10 had obvious format problems (full-sentence answers or multiline output). |
| 3. Judge accuracy | PASS | Requested spot check of 5 all-correct and 5 all-incorrect entries matched the stored `ground_truth` aliases. Caveat: one separate all-correct item (`tc_138`) is an obvious substring-match false positive. |
| 4. Distribution stats | PASS | All-correct: 33, all-incorrect: 55, mixed: 12, uncertain: 0. |
| 5. Consistency (10/10 same label) | FAIL | 88/100 entries have the same judge label across all 10 samples; 12/100 are mixed. |
| 6. Red flags | FAIL | No empty responses or encoding corruption, but there are 10 multiline responses, 23 long/explanatory responses, and one clear judge loophole example. |

## 1. Data Integrity

- Valid JSON lines: 100/100
- Missing required fields: 0
- Wrong list lengths (`responses != 10` or `judges != 10`): 0
- Empty responses: 0
- Replacement-character / encoding corruption (`�`): 0
- Control-character corruption: 0

Example valid shape:

```json
{"tc_x": {"question": "...", "responses": ["..."], "judges": ["true"], "ground_truth": ["..."]}}
```

## 2. Response Quality Spot Sample

Spot sample chosen to cover both clean and suspicious cases.

| QID | Example response(s) | Verdict |
| --- | --- | --- |
| `tc_8` | `Portugal` | Good short answer |
| `tc_10` | `Chicago Bears` | Good short answer |
| `tc_17` | `William Golding` | Good short answer |
| `tc_31` | `DZ` | Good short answer |
| `tc_44` | `Insulin` | Good short answer |
| `tc_3` | `York`, `Chelsea`, `Warrington` | Short-answer format is fine, but content is inconsistent |
| `tc_15` | `Japan`, `United Kingdom`, `Germany` | Short-answer format is fine, but content is inconsistent |
| `tc_32` | `He was shot and killed by Lee Harvey Oswald.` | Bad format: full sentence, not answer-only |
| `tc_21` | `Joan Molinsky is better known as a financial therapist.` | Bad format: explanatory sentence, not answer-only |
| `tc_138` | `Henry James` + newline + another person | Bad format: multiline / extra answer appended |

Result: 7/10 sampled entries were plausible short-answer outputs; 3/10 were not.

## 3. Judge Accuracy Spot Check

Checked against the stored `ground_truth` aliases in the file, not against external fact sources.

### All-correct sample

| QID | Response | Ground-truth alias match | Stored label |
| --- | --- | --- | --- |
| `tc_8` | `Portugal` | Yes (`Portugal`) | all `true` |
| `tc_10` | `Chicago Bears` | Yes (`Chicago Bears`) | all `true` |
| `tc_17` | `William Golding` | Yes (`Golding`) | all `true` |
| `tc_31` | `DZ` | Yes (`DZ`) | all `true` |
| `tc_44` | `Insulin` | Yes (`insulin`) | all `true` |

### All-incorrect sample

| QID | Response(s) | Why incorrect vs stored `ground_truth` | Stored label |
| --- | --- | --- | --- |
| `tc_1` | `Eugene O'Neill`, `Ernest Hemingway` | No alias overlap with `Sinclair Lewis` aliases | all `false` |
| `tc_11` | `Portugal` | No alias overlap with stored Norway aliases | all `false` |
| `tc_16` | `David`, `Herbert` | No alias overlap with `Walter` aliases | all `false` |
| `tc_23` | `China` | No alias overlap with stored Italy aliases | all `false` |
| `tc_27` | `1918` | No alias overlap with `1914` aliases | all `false` |

Spot-check result: 10/10 sampled entries agree with the stored judge labels.

Caveat: `tc_138` shows a rule-judge loophole. Every response starts with `Henry James` but then appends an unrelated second name on a new line, and all 10 are still labeled `true`. The current judge behaves like a substring lock: once a gold alias appears anywhere in the response, extra junk does not matter.

## 4. Distribution Stats

- All-correct: 33
- All-incorrect: 55
- Mixed: 12
- Uncertain: 0

Raw label totals across all 1,000 samples:

- `true`: 376
- `false`: 624
- `uncertain`: 0

## 5. Consistency

- 10/10 same label: 88 entries
- Not 10/10 same label: 12 entries

Mixed-label examples:

- `tc_3`: Judi Dench birthplace responses vary between `York` and incorrect cities
- `tc_15`: ISDN question alternates between `Japan`, `United Kingdom`, `Germany`
- `tc_139`: North Pole submarine question mixes `USS Nautilus` with incorrect submarines
- `tc_145`: Iron Triangle question has 1 `true` (`Vietnam`) and 9 `false`

## 6. Red Flags

Positive:

- Empty responses: 0
- Encoding corruption found: 0
- `uncertain` labels: 0

Problems:

- Multiline responses: 10 total, all in `tc_138`
- Long / explanatory responses (>8 words): 23 total
- Worst offenders for long responses: `tc_32` (10), `tc_21` (7), `tc_19` (6)

Examples:

- `tc_32`: `He was shot and killed by Lee Harvey Oswald.`
- `tc_21`: `Joan Molinsky is better known as a financial therapist.`
- `tc_138`: `Henry James` followed by a second, unrelated name on a new line

## Bottom Line

The file is structurally clean and usable for analysis, but not clean enough to treat as a high-confidence consistency set without filtering. The main issues are mixed-label questions, several non-short-form generations, and a judge rule that can over-credit responses that contain a correct alias plus extra wrong content.
