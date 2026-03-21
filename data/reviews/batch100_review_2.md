# Batch 100 Review — Second Pass

Reviewed file: `data/gemma3_4b_TriviaQA_consistency_samples.jsonl`
Batch size: 100 questions × 10 responses = 1,000 judged responses

---

## 1. Rule Judge Substring Loophole Quantification

Of the 33 all-correct entries, **2 QIDs** contain at least one response where a ground_truth alias matches as a substring but the full response is >3 words. These account for **20 individual responses**.

| QID | # responses >3 words | Response (verbatim) | Matched alias (normalized) |
|---|---|---|---|
| `tc_115` | 10/10 | `And that's the way it is.` (with/without quotes) | `and that s way it is` |
| `tc_138` | 10/10 | `Henry James\nJuan Gris`, `Henry James\nJulio Cortázar`, `Henry James\nHans Holbein the Younger`, etc. | `henry james` |

`tc_115`: All 10 responses are the correct answer itself (a well-known catchphrase), which happens to be 6 words. The alias is the full response. No extra wrong content appended.

`tc_138`: All 10 responses contain the correct answer on line 1, followed by an unrelated second name on a new line. The substring match fires on `henry james` and ignores the appended content.

---

## 2. Response Length Distribution

Across all 1,000 responses:

| Percentile | Word count |
|---|---|
| p25 | 1 |
| p50 | 2 |
| p75 | 2 |
| p95 | 4 |
| max | 12 |

**Responses with >10 words (1 total):**

| QID | resp# | Word count | Verbatim |
|---|---|---|---|
| `tc_21` | 3 | 12 | `Joan Molinsky is better known as a luxury travel advisor and author.` |

---

## 3. Paper's Reported Accuracy vs. Batch Rate

The paper (`original-paper-markdown-converted.md`, §6.1.1) does not report a per-model TriviaQA accuracy rate. It states:

> "This strict filtering yields a high-quality contrastive set of 1,000 fully correct and 1,000 fully incorrect examples."

The paper does not disclose how many total TriviaQA questions were sampled to obtain those 2,000 consistent entries, nor what fraction were all-correct vs. all-incorrect per model.

**Table 1** reports hallucination *detection* accuracy (classifier performance), not raw QA accuracy. The Gemma-3-4B row in Table 1 shows detection accuracy of 76.9% on TriviaQA — this is the classifier's ability to distinguish faithful vs. hallucinatory responses, not the model's QA accuracy.

| Metric | Value |
|---|---|
| Batch all-correct rate | 33/100 (33%) |
| Paper's reported Gemma-3-4B TriviaQA QA accuracy | Not reported |
| Paper's reported Gemma-3-4B TriviaQA detection accuracy (Table 1) | 76.9% |

No direct comparison is possible.

---

## 4. Downstream Schema Compatibility

`extract_answer_tokens.py` (lines 97–135) reads each JSONL line as `data[qid]` and accesses:

| Field | Expected type | Present in JSONL | Type in JSONL | Match |
|---|---|---|---|---|
| `judges` | `list` | Yes | `list[str]` (len 10) | Yes |
| `responses` | `list` | Yes | `list[str]` (len 10) | Yes |
| `question` | `str` | Yes | `str` | Yes |
| `ground_truth` | — | Yes | `list[str]` | Not accessed by downstream (harmless extra field) |

Schema check across all 100 entries: **0 mismatches**. All field names, nesting (`{qid: {fields...}}`), and types match what `extract_answer_tokens.py` expects.

---

## 5. Mixed-Label Entries

12 entries have non-unanimous judge labels:

| QID | true | false | uncertain | Distinct normalized responses |
|---|---|---|---|---|
| `tc_3` | 4 | 6 | 0 | `chelsea`, `chesham bois`, `leatherhead`, `richmond`, `warrington`, `york` |
| `tc_15` | 6 | 4 | 0 | `germany`, `japan`, `united kingdom` |
| `tc_18` | 2 | 8 | 0 | `closed gear system`, `closed gearbox`, `first functional electric starter motor for automobiles`, `first mass produced electric starter motor`, `first pneumatic tire`, `first windshield wiper`, `starting handle for rear axle`, `starting mechanism for automobile transmissions`, `windshield wiper` |
| `tc_26` | 7 | 3 | 0 | `harry`, `prince edward` |
| `tc_30` | 3 | 7 | 0 | `geri halliwell`, `melanie brown`, `victoria beckham` |
| `tc_77` | 8 | 2 | 0 | `thursday`, `tuesday` |
| `tc_97` | 1 | 9 | 0 | `awakenings`, `dead poets society`, `fisher king`, `hook`, `regarding you`, `sixth sense` |
| `tc_100` | 1 | 9 | 0 | `arkansas`, `kentucky`, `missouri`, `oregon`, `south dakota` |
| `tc_103` | 6 | 4 | 0 | `bone china pottery`, `bone china tableware` |
| `tc_113` | 4 | 6 | 0 | `luxury hotels and department stores`, `luxury hotels and resorts`, `luxury hotels and restaurants` |
| `tc_139` | 3 | 7 | 0 | `uss ike`, `uss ike ssn 718`, `uss ike ssn 728`, `uss nautilus`, `uss theodore roosevelt` |
| `tc_145` | 1 | 9 | 0 | `bangkok thailand`, `detroit michigan`, `los angeles california`, `new york city`, `vietnam` |

---

## 6. Sampling Parameter Fidelity

Paper's consistency filtering parameters (`original-paper-markdown-converted.md`, §6.1.1, line 219):

> "sampling 10 distinct responses using probabilistic decoding parameters (`temperature=1.0`, `top_k=50`, `top_p=0.9`)"

The paper does not specify `max_new_tokens` / `max_tokens` for the consistency filtering step. (The `max_length=512` mentioned in §6.2 applies to the perturbation experiments' open-ended generation, not to consistency filtering.)

| Parameter | Paper (§6.1.1) | Script (`collect_responses.py:165–171`) | Match |
|---|---|---|---|
| `temperature` | 1.0 | 1.0 | Yes |
| `top_k` | 50 | 50 | Yes |
| `top_p` | 0.9 | 0.9 | Yes |
| `max_new_tokens` | Not specified | 50 | N/A |
| `sample_num` | 10 | 10 (default) | Yes |
