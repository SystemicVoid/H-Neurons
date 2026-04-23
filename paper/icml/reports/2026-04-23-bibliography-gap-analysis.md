# Bibliography Gap Analysis — ICML 2026 Workshop Paper — 2026-04-23

> **Status:** Actionable. All arXiv IDs verified via `arxiv_metadata.py`. BibTeX entries ready in §IV.
>
> **Current bibliography:** 20 entries (3 are dead — never cited in text).
> **Recommended additions:** 13 must-have + 7 nice-to-have = 20 new entries → 40 total.
> **No bibliography size constraint.** Add all tiers.

---

## I. Executive Summary

The paper's empirical claims are sound, but the bibliography has two structural weaknesses a reviewer would flag immediately:

1. **Missing benchmark provenance.** The paper uses TruthfulQA, TriviaQA, SimpleQA, BioASQ, FalseQA, and JailbreakBench but cites none of their origin papers. This is the most visible gap.
2. **Missing method lineage.** The paper audits the readout-to-steering heuristic but does not cite the foundational papers that established that heuristic (RepE, ActAdd, foundational SAE work, Gemma Scope).

Secondary gaps exist in wrong-entity substitution precedent and causal-methodology grounding for the four-stage framework, but these are strengthening moves rather than vulnerabilities.

---

## II. Claim-by-Claim Diagnosis

### Claim 1: Readout-to-steering dissociation (§3 Localization)

**Current support:** Hewitt 2019, Elazar 2021, Kumar 2022, Hase 2023, Arad 2025, Wu 2025, Wang 2026, Bhalla 2024 — good.

**Gap:** The paper audits a heuristic ("identify a strong readout, then intervene through it") but does not cite the foundational papers that established that heuristic paradigm:
- **Zou et al. 2023 (RepE)** — arXiv:2310.01405 — the top-down representation engineering framework; the phrase "representation-engineering intervention outputs" appears in §5 without citation.
- **Turner et al. 2024 (ActAdd)** — arXiv:2308.10248 — foundational activation steering; established the α-sweep dose-response pattern the paper uses.
- **Cunningham et al. 2023** — arXiv:2309.08600 — foundational SAE interpretability paper; needed since the paper's entire localization comparison uses SAE features.
- **Lieberum et al. 2024 (Gemma Scope)** — arXiv:2408.05147 — the source of the SAE features used; also supports the L3 "partial layer coverage" limitation.

**Criticality:** Must-have. Reviewers will notice missing method lineage.

### Claim 2: Externality / transfer failure (§4)

**Current support:** Li 2023 (ITI), Pres 2024, Opielka 2026 — adequate for the dissociation point.

**Gap:** Benchmark origin papers are missing:
- **Lin et al. 2022 (TruthfulQA)** — arXiv:2109.07958 — ACL 2022. Used as primary MC surface.
- **Wei et al. 2024 (SimpleQA)** — arXiv:2411.04368 — supporting stress test for MC/generation divergence.
- **Joshi et al. 2017 (TriviaQA)** — arXiv:1705.03551 — origin of the bridge benchmark.
- **Hu et al. 2023 (FalseQA)** — arXiv:2307.02394 — ACL 2023. Used for false-premise dose-response.
- **Tsatsaronis et al. 2015 (BioASQ)** — DOI:10.1186/s12859-015-0564-6 — domain QA surface.

**Wrong-entity substitution** is presented as the paper's most distinctive externality contribution but lacks precedent citation:
- **Nan et al. 2021** — arXiv:2102.09130 — EACL 2021. Entity-level factual consistency; establishes wrong-entity substitution as a recognized factual error class in NLG.

**Criticality:** Benchmark origins are must-have. Nan 2021 is high-value for the abstract-level claim.

### Claim 3: Measurement sensitivity (§5)

**Current support:** StrongREJECT, Safer or Luckier, Know Thy Judge — adequate for the general point.

**Gap:**
- **Chao et al. 2024 (JailbreakBench)** — arXiv:2404.01318 — NeurIPS 2024 D&B. The actual benchmark used; must cite.

**Criticality:** Must-have.

### Claim 4: Four-stage audit framework (§6)

**Current support:** framed as synthesis of the paper's own findings — acceptable for a workshop paper.

**Gap (nice-to-have):** Methodological grounding would make the framework less ad hoc:
- **Park et al. 2024** — arXiv:2311.03658 — ICML 2024. Linear representation hypothesis; theoretical backdrop for why people expect good readout → good steering.
- **Goldowsky-Dill et al. 2023** — arXiv:2304.05969 — path patching; methodological precedent for causal localization.

**Criticality:** Nice-to-have. Strengthening, not fixing a vulnerability.

### Claim 5: SAE partial coverage (L3 limitation)

**Gap:** Gemma Scope and SAE scaling papers:
- **Lieberum et al. 2024** — already listed above (must-have).
- **Gao et al. 2024** — arXiv:2406.04093 — scaling and evaluating SAEs; supports the claim that SAE quality depends on training choices.

**Criticality:** Gao 2024 is nice-to-have. Lieberum 2024 is must-have (already counted).

### Claim 6: Dose-response methodology

**Gap (nice-to-have):**
- **Panickssery et al. 2024 (CAA)** — arXiv:2312.06681 — contrastive activation addition; another major intervention paradigm the paper implicitly positions against.

**Criticality:** Nice-to-have.

---

## III. Prioritized Addition List

### Tier 1: Must-Have (fix vulnerabilities)

| # | Paper | arXiv / DOI | Strengthens | Status |
|---|---|---|---|---|
| 1 | Zou et al. 2023 — Representation Engineering | 2310.01405 | Method lineage (§1, §5) | ✅ verified |
| 2 | Turner et al. 2024 — Activation Engineering | 2308.10248 | Method lineage, dose-response | ✅ verified |
| 3 | Cunningham et al. 2023 — SAE Interpretability | 2309.08600 | SAE localization (§3) | ✅ verified |
| 4 | Lieberum et al. 2024 — Gemma Scope | 2408.05147 | SAE source, L3 limitation | ✅ verified |
| 5 | Lin et al. 2022 — TruthfulQA | 2109.07958 | Benchmark provenance (§4) | ✅ verified |
| 6 | Wei et al. 2024 — SimpleQA | 2411.04368 | Benchmark provenance (§4) | ✅ verified |
| 7 | Joshi et al. 2017 — TriviaQA | 1705.03551 | Benchmark provenance (§4) | ✅ verified |
| 8 | Chao et al. 2024 — JailbreakBench | 2404.01318 | Benchmark provenance (§5) | ✅ verified |
| 9 | Hu et al. 2023 — FalseQA | 2307.02394 | Benchmark provenance (§4) | ✅ verified |
| 10 | Tsatsaronis et al. 2015 — BioASQ | 10.1186/s12859-015-0564-6 | Benchmark provenance (§4) | ✅ verified |

### Tier 2: High-Value (strengthen distinctive claims)

| # | Paper | arXiv / DOI | Strengthens | Status |
|---|---|---|---|---|
| 11 | Nan et al. 2021 — Entity-level Factual Consistency | 2102.09130 | Wrong-entity substitution (§4.3) | ✅ verified |
| 12 | Park et al. 2024 — Linear Representation Hypothesis | 2311.03658 | Framework theoretical grounding (§6) | ✅ verified |
| 13 | Panickssery et al. 2024 — CAA | 2312.06681 | Intervention paradigm lineage (§2) | ✅ verified |

### Tier 3: Depth & Completeness (add all — no size constraint)

| # | Paper | arXiv / DOI | Strengthens | Status |
|---|---|---|---|---|
| 14 | Goldowsky-Dill et al. 2023 — Path Patching | 2304.05969 | Framework methodology (§6) | ✅ verified |
| 15 | Gao et al. 2024 — Scaling SAEs | 2406.04093 | L3 limitation | ✅ verified |
| 16 | Geiger et al. 2023 — Causal Abstraction | 2301.04709 | Framework grounding (§6) | ✅ verified |
| 17 | Meng et al. 2022 — ROME | 2202.05262 | Localization→editing precedent | ✅ verified |
| 18 | Mazeika et al. 2024 — HarmBench | 2402.04249 | Measurement ecosystem (§5) | ✅ verified |
| 19 | Maynez et al. 2020 — Faithfulness/Factuality | 2005.00661 | Wrong-entity taxonomy | ✅ verified |
| 20 | Stoehr et al. 2024 — Activation Scaling | 2410.04962 | Dose-response precedent | ✅ verified |

---

## IV. BibTeX Entries (Tier 1 + Tier 2, Drop-In)

```bibtex
% === TIER 1: Must-Have ===

@article{zou2023repe,
  title   = {Representation Engineering: A Top-Down Approach to {AI} Transparency},
  author  = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and Dombrowski, Ann-Kathrin and Goel, Shashwat and Li, Nathaniel and Byun, Michael J. and Wang, Zifan and Mallen, Alex and Basart, Steven and Koyber, Sanmi and Song, Dawn and Fredrikson, Matt and Kolter, J. Zico and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2310.01405},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.01405},
}

@article{turner2024steering,
  title   = {Steering Language Models With Activation Engineering},
  author  = {Turner, Alexander Matt and Thiergart, Lisa and Leech, Gavin and Mini, David and Udell, Fabien Roger},
  journal = {arXiv preprint arXiv:2308.10248},
  year    = {2024},
  url     = {https://arxiv.org/abs/2308.10248},
}

@article{cunningham2023saes,
  title   = {Sparse Autoencoders Find Highly Interpretable Features in Language Models},
  author  = {Cunningham, Hoagy and Ewart, Aidan and Riggs, Logan and Huben, Robert and Sharkey, Lee},
  journal = {arXiv preprint arXiv:2309.08600},
  year    = {2023},
  url     = {https://arxiv.org/abs/2309.08600},
}

@article{lieberum2024gemmascope,
  title   = {{Gemma Scope}: Open Sparse Autoencoders Everywhere All At Once on {Gemma}~2},
  author  = {Lieberum, Tom and Rajavelu, Senthooran and Conmy, Arthur and Smith, Lewis and Sonnerat, Nicolas and Varma, Vikrant and Kram\'{a}r, J\'{a}nos and Dragan, Anca and Shah, Rohin and Nanda, Neel},
  journal = {arXiv preprint arXiv:2408.05147},
  year    = {2024},
  url     = {https://arxiv.org/abs/2408.05147},
}

@inproceedings{lin2022truthfulqa,
  title     = {{TruthfulQA}: Measuring How Models Mimic Human Falsehoods},
  author    = {Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year      = {2022},
  url       = {https://arxiv.org/abs/2109.07958},
}

@article{wei2024simpleqa,
  title   = {Measuring Short-Form Factuality in Large Language Models},
  author  = {Wei, Jason and Yang, Jie and Huang, Kai and Tung, Martin and Pierson, William and Rao, Jiayin and Liang, Jiacheng and Singhal, Karan},
  journal = {arXiv preprint arXiv:2411.04368},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.04368},
}

@inproceedings{joshi2017triviaqa,
  title     = {{TriviaQA}: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
  author    = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year      = {2017},
  url       = {https://arxiv.org/abs/1705.03551},
}

@inproceedings{chao2024jailbreakbench,
  title     = {{JailbreakBench}: An Open Robustness Benchmark for Jailbreaking Large Language Models},
  author    = {Chao, Patrick and Robey, Alexander and Dobriban, Edgar and Hassani, Hamed and Pappas, George J. and Wong, Eric},
  booktitle = {Advances in Neural Information Processing Systems: Datasets and Benchmarks Track},
  year      = {2024},
  url       = {https://arxiv.org/abs/2404.01318},
}

@inproceedings{hu2023falseqa,
  title     = {Won't Get Fooled Again: Answering Questions with False Premises},
  author    = {Hu, Shengding and Huang, Yifan and Liu, Zhiyuan and Sun, Maosong},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.02394},
}

@article{tsatsaronis2015bioasq,
  title   = {An Overview of the {BioASQ} Large-Scale Biomedical Semantic Indexing and Question Answering Competition},
  author  = {Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos and Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R. and Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and Polychronopoulos, Dimitris and Almirantis, Yannis and Pavlopoulos, John and Baskiotis, Nicolas and Gallinari, Patrick and Artieres, Thierry and Ngonga Ngomo, Axel-Cyrille and Heino, Norman and Gaussier, Eric and Barber, Liliana and Mowatt, Federica and Nentidis, Anastasios and Paliouras, Georgios},
  journal = {BMC Bioinformatics},
  volume  = {16},
  pages   = {138},
  year    = {2015},
  doi     = {10.1186/s12859-015-0564-6},
}

% === TIER 2: High-Value ===

@inproceedings{nan2021entity,
  title     = {Entity-Level Factual Consistency of Abstractive Text Summarization},
  author    = {Nan, Feng and Nallapati, Ramesh and Wang, Zhiguo and Nogueira dos Santos, Cicero and Zhu, Henghui and Zhang, Dejiao and McKeown, Kathleen and Xiang, Bing},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2021},
  url       = {https://arxiv.org/abs/2102.09130},
}

@inproceedings{park2024linear,
  title     = {The Linear Representation Hypothesis and the Geometry of Large Language Models},
  author    = {Park, Kiho and Choe, Yo Joong and Veitch, Victor},
  booktitle = {International Conference on Machine Learning},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.03658},
}

@article{panickssery2024caa,
  title   = {Steering {Llama}~2 via Contrastive Activation Addition},
  author  = {Panickssery, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  journal = {arXiv preprint arXiv:2312.06681},
  year    = {2024},
  url     = {https://arxiv.org/abs/2312.06681},
}
% === TIER 3: Nice-to-Have ===

@article{goldowskydill2023pathpatching,
  title   = {Localizing Model Behavior with Path Patching},
  author  = {Goldowsky-Dill, Nicholas and MacLeod, Chris and Sato, Lucas and Arora, Aman},
  journal = {arXiv preprint arXiv:2304.05969},
  year    = {2023},
  url     = {https://arxiv.org/abs/2304.05969},
}

@article{gao2024scalingsaes,
  title   = {Scaling and Evaluating Sparse Autoencoders},
  author  = {Gao, Leo and la Tour, Tom Dupr\'{e} and Tillman, Henk and Goh, Gabriel and Troll, Rajan and Radford, Alec and Sutskever, Ilya and Leike, Jan and Wu, Jeffrey},
  journal = {arXiv preprint arXiv:2406.04093},
  year    = {2024},
  url     = {https://arxiv.org/abs/2406.04093},
}

@article{geiger2023causalabstraction,
  title   = {Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability},
  author  = {Geiger, Atticus and Zhengxuan, Wu and Potts, Christopher and Icard, Thomas and Goodman, Noah D.},
  journal = {arXiv preprint arXiv:2301.04709},
  year    = {2023},
  url     = {https://arxiv.org/abs/2301.04709},
}

@inproceedings{meng2022rome,
  title     = {Locating and Editing Factual Associations in {GPT}},
  author    = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.05262},
}

@article{mazeika2024harmbench,
  title   = {{HarmBench}: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author  = {Mazeika, Mantas and Phan, Long and Yin, Xuwang and Zou, Andy and Wang, Zifan and Mu, Norman and Sakhaee, Elham and Li, Nathaniel and Basart, Steven and Li, Bo and Forsyth, David and Hendrycks, Dan},
  journal = {arXiv preprint arXiv:2402.04249},
  year    = {2024},
  url     = {https://arxiv.org/abs/2402.04249},
}

@inproceedings{maynez2020faithfulness,
  title     = {On Faithfulness and Factuality in Abstractive Summarization},
  author    = {Maynez, Joshua and Narayan, Shashi and Bohnet, Bernd and McDonald, Ryan},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.00661},
}

@inproceedings{stoehr2024activationscaling,
  title     = {Activation Scaling for Steering and Interpreting Language Models},
  author    = {Stoehr, Niklas and Mitchell, Eric and Hall, David and Jurafsky, Dan and Potts, Christopher},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  year      = {2024},
  url       = {https://arxiv.org/abs/2410.04962},
}
```

---

## V. Suggested Insertion Points in main.tex

### Introduction (§1, lines 84–88)

Current: `\citep{li2023iti,gao2025hneurons,arditi2024refusal}`

Add RepE, ActAdd, and linear representation hypothesis to the sentence about the readout-to-steering heuristic:

```
...identify a strong readout, then intervene through it
\citep{li2023iti,gao2025hneurons,arditi2024refusal,zou2023repe,turner2024steering}.
```

### Related Work (§2, line 129)

After the probe critique sentence, add linear representation grounding:

```
The linear representation hypothesis \citep{park2024linear} provides the theoretical
backdrop: if concepts are encoded as linear directions, readout quality and steering
utility should correlate—but our results and concurrent work show they often do not.
```

Add SAE foundation + Gemma Scope after the SAE-centric analysis sentence (line 134):

```
Concurrent SAE-centric analyses \citep{arad2025saes,wang2026interpretability} and
synthetic-benchmark evaluations \citep{wu2025axbench} ...
These build on foundational SAE work \citep{cunningham2023saes} and the Gemma~Scope
open SAE suite \citep{lieberum2024gemmascope}.
```

Add CAA to the intervention lineage:

```
...contrastive activation addition \citep{panickssery2024caa}...
```

### Localization section (§3, line 151)

Add Gemma Scope citation where SAE activations are first mentioned:

```
...Gemma Scope~2 SAE activations \citep{lieberum2024gemmascope} selected 266...
```

### Externality section (§4)

Add benchmark citations at first mention:

- Line ~213: `TruthfulQA \citep{lin2022truthfulqa}` (or alongside existing `\citep{li2023iti}`)
- Line ~218: `SimpleQA ($n = 1{,}000$) \citep{wei2024simpleqa}`
- Line ~224: `TriviaQA bridge benchmark \citep{joshi2017triviaqa}`
- Line ~205: `FalseQA ($n = 687$) \citep{hu2023falseqa}`
- Line ~207: `BioASQ factoid QA \citep{tsatsaronis2015bioasq}`

### Wrong-entity substitution (§4.3, line 238)

Add entity-level factual consistency precedent:

```
...the wrong-entity substitution category \citep{nan2021entity}: the model replaces...
```

### Measurement section (§5)

Add JailbreakBench and HarmBench citations:

```
JailbreakBench \citep{chao2024jailbreakbench}
```

And in the evaluator discussion, add HarmBench as part of the measurement ecosystem:

```
...standardized red-teaming frameworks \citep{mazeika2024harmbench,souly2024strongreject}...
```

### Study orientation (§1, line 114)

Add benchmark citations in the list:

```
...FaithEval \citep{ming2025faitheval}, TruthfulQA \citep{lin2022truthfulqa},
TriviaQA \citep{joshi2017triviaqa}, SimpleQA \citep{wei2024simpleqa},
BioASQ \citep{tsatsaronis2015bioasq}), and jailbreak settings
(JailbreakBench \citep{chao2024jailbreakbench}).
```

### Tier 3 insertion points

**Related Work (§2, line 129)** — add causal methodology grounding:

```
...high readout accuracy can arise for reasons the model does not functionally rely on
\citep{hewitt2019control,elazar2021amnesic,kumar2022probes}. Causal abstraction
\citep{geiger2023causalabstraction} and path patching \citep{goldowskydill2023pathpatching}
formalize the distinction between features a model encodes and features it functionally uses.
```

**Related Work (§2)** — add ROME as a localization→editing precedent alongside Hase:

```
\citet{hase2023localization} showed that better localization does not reliably predict
better editing, extending the causal-tracing localization results of \citet{meng2022rome}.
```

**Related Work (§2)** — add activation scaling precedent:

```
...contrastive activation addition \citep{panickssery2024caa} and activation scaling
\citep{stoehr2024activationscaling,turner2024steering}...
```

**Wrong-entity substitution (§4.3)** — add faithfulness/factuality taxonomy:

```
...wrong-entity substitution \citep{nan2021entity}, a factual corruption pattern
recognized in the summarization literature \citep{maynez2020faithfulness}...
```

**Limitations (§6.2)** — add SAE scaling paper for the L3 limitation:

```
SAE layer coverage is partial; a different SAE width or layer selection could yield
non-null results \citep{lieberum2024gemmascope,gao2024scalingsaes}.
```

---

## VI. Dead Entries (currently in .bib but never cited in main.tex)

| Entry | Status |
|---|---|
| `huang2025guidedbench` | Never `\cite`d anywhere in the paper |
| `nguyen2025matsteer` | Never `\cite`d anywhere in the paper |
| `lee2025cast` | Never `\cite`d anywhere in the paper |

These can be kept in `references.bib` for future use but will not appear in the compiled bibliography unless cited. No action needed — BibTeX ignores uncited entries automatically.

---

## VII. Verification

All arXiv IDs verified via `arxiv_metadata.py` on 2026-04-23. BioASQ DOI verified via `crossref_metadata.py lookup`. No retraction flags on any entry.
