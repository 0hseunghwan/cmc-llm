# cmc-llm: Reproducing CMC and Replacing the Final Reranker with an LLM

This repository is a fork of [yc-song/cmc](https://github.com/yc-song/cmc), the code for
*Comparing Neighbors Together Makes it Easy: Jointly Comparing Multiple Candidates for Efficient and Effective Retrieval*
(Song et al., 2024, [arXiv:2405.12801](https://arxiv.org/abs/2405.12801)).

The plan has two parts. First, reproduce CMC on ZeSHEL. Second, swap the final-stage cross-encoder (CE)
for an LLM and measure where the resulting **BE → CMC → LLM** pipeline lands on accuracy and cost.

> **Status:** planning. No results yet. The sections under [Results](#results) will be filled in as stages finish.

## Research questions

1. **Does an LLM pick better than a fine-tuned cross-encoder?** Given the same candidates, does it beat the CE's 65.9 ZeSHEL accuracy?
2. **How much does CMC contribute in front of the LLM?** Compare giving the LLM *k* candidates straight from the bi-encoder (BE) against giving it *k* candidates after CMC reranking.
3. **How many candidates should the LLM see?** Measure how accuracy and token cost change with *k*.

The scope is ZeSHEL only. It has the largest gap between CMC and CE in the paper (59.2 vs 65.9), and the task depends on
reading entity descriptions, so any benefit from an LLM should be easy to see. Other datasets and KGQA come later.

## Branches

| Branch | Contents |
|---|---|
| `main` | Upstream default branch. Its README describes CMC, but the code is the older hard-negatives EL codebase (Zhang & Stratos, 2021) and does not support the README's flags (e.g. `--type_model=extend_multi_dot`). |
| `submission` | Upstream CMC code at submission time (2024-01-24). Matches the README commands. **Base for this work.** |
| `master` | Later upstream CMC code (2024-04-23). Adds candidate-generation scripts such as `save_candidates.sh` and `candidates_postprocess.py`. |
| `experiments` | All changes in this project. Branched from `submission`. |

## Stage 0: Setup

Three files that the experiments need are not in the repository. Everything after this stage depends on them.
If a file can't be found within two days, ask on the upstream issue tracker or email the first author.

- [ ] **Environment.** `environment.yml` pins Python 3.8, `torch==1.13.0` and `transformers==3.1.0`, built for `linux-64`. It has to be created on a Linux GPU machine. The training script calls `wandb.init()` directly, so log in to wandb or set `WANDB_MODE=offline`.
- [ ] **ZeSHEL data** from [lajanugen/zeshel](https://github.com/lajanugen/zeshel). The loader expects `documents/*.json` and `mentions/{train,val,test}.json` under the data directory. `utils/preprocess_zeshel.py` may be needed to get there.
- [ ] **BE candidate file** (`./data/cands_top1024/candidates_{train,val,test}.json`, one JSON line per mention with `candidates`, `labels` and `scores`). The paper uses the ZeSHEL bi-encoder from Yadav et al. (2022), with recall@64 = 87.95. No download link is given, so it may have to be generated with `main_retriever.py` and `save_candidates.py`.
- [ ] **Encoder checkpoint** (`./cocondenser.bin`). Table 4 shows the best ZeSHEL result for CMC comes from MS MARCO coCondenser initialization (Gao & Callan, 2022). The loader prefixes each key with `mention_encoder.`/`entity_encoder.` and calls `load_state_dict(strict=False)`. If the file's keys start with `bert.`, **nothing loads and no error is raised**. Check that keys look like `embeddings.word_embeddings.weight`.
- [ ] **Cross-encoder checkpoint** for configurations A and B: a CE trained on ZeSHEL that reproduces the paper's 65.9.
- [ ] **GPU memory.** One epoch took about 4 hours on an A100 in the paper.

## Stage 1: Reproduction

Run the README commands unchanged before modifying anything. If the baseline is off, later differences can't be
attributed to my changes rather than to the environment.

Targets (ZeSHEL test set):

| Item | Paper | Source |
|---|---|---|
| BE recall@64 | 87.95 | Tables 1, 8 |
| BE + CMC(64) R@1 | 59.22 | Table 8 |
| BE + CMC(512) recall@64 | 91.51 | Tables 1, 8 |
| BE + CMC(512) recall@16 | 86.32 | Table 8 |
| CMC final reranker accuracy | 59.2 ± 0.3 | Table 3 |

**Pass criterion:** CMC R@1 within ±1 percentage point of the paper. The paper reports a standard deviation of 0.3,
so a gap above 1 point means a configuration difference.

If the numbers don't match, check in this order:

1. **Encoder initialization.** In Table 4, test accuracy ranges from 55.3 to 59.2 depending on initialization. Confirm the coCondenser checkpoint actually loaded.
2. **BE candidates.** Confirm my BE gives recall@64 = 87.95. Different candidates change everything downstream.
3. **Seeds.** The paper averages five seeds. Run at least three and report mean and standard deviation.
4. **Accuracy definition.** The paper uses both unnormalized (all test mentions) and normalized (only mentions whose gold is among the candidates) accuracy. Make sure the comparison uses the same one.

A failed reproduction still gets recorded: the setting, the number obtained and the gap to the paper go into the results table.

## Stage 2: Export CMC outputs

Write CMC rankings to disk so the LLM stage is decoupled. Prompts and *k* can then change without rerunning CMC.

Only saving code is added to the existing evaluation loop. One JSONL line per mention:

```json
{"query_id": "...", "mention_context": "...", "gold_id": "...",
 "be_ranking": ["id1", "id2"], "be_scores": [0.0, 0.0],
 "cmc_ranking": ["id7", "id1"], "cmc_scores": [0.0, 0.0]}
```

- `be_ranking`: BE top 512. `cmc_ranking`: top 64 after CMC reranks those 512.
- A separate lookup file maps entity IDs to titles and descriptions.
- Export both the validation and test splits. Prompt tuning uses validation only.
- **Check:** recall@16 and recall@64 recomputed from the JSONL must match Stage 1. A mismatch means a bug in the export code.

## Stage 3: Main experiment (BE → CMC → LLM)

Four configurations hand the same number of candidates *k* to the final stage. They differ only in who selects the
candidates and what the final stage is.

| Config | Candidate selection | Final stage | Role |
|---|---|---|---|
| A | BE top *k* | CE | Standard pipeline in the paper |
| B | BE 512 → CMC → top *k* | CE | Paper's proposed pipeline (reproduced) |
| C | BE top *k* | LLM | LLM without CMC |
| D | BE 512 → CMC → top *k* | LLM | **Proposed here** |

- **Q1 (LLM vs CE):** B vs D. Only the final stage differs.
- **Q2 (contribution of CMC):** C vs D. Only the presence of CMC differs.
- **Q3 (best *k*):** sweep *k* in D.

Variables:

| Variable | Values | Notes |
|---|---|---|
| *k* | 4, 8, 16, 32 | Descriptions are at most 128 tokens, so *k* = 32 is about 4k prompt tokens |
| Candidate order | By score, random | Separates real judgment from a bias toward early options |
| LLM | Start with one | Add one model of a different size once results come in |

**Prompt.** The input is the context with the mention marked, followed by a numbered list of candidates (title +
description). The model outputs a single number. Any other output counts as wrong, and its rate is logged separately.
Wording is tuned on 200 validation examples only and then frozen. Temperature is 0. The LLM is not fine-tuned. The CE is,
and that difference is stated when interpreting results.

**Why the order experiment matters.** With score-ordered input, an LLM that always answers "1" already gets CMC's
R@1 (59.2). Accuracy has to hold up under random order before we can say the LLM is reading and judging. A large gap
between the two conditions means the LLM is leaning on the ranking.

**Cost control.** The test set has about 10k mentions. All four *k* values × two orders would be 80k LLM calls. First
run every combination on a fixed-seed sample of 1,000 test mentions, then confirm the best one or two settings on the
full test set.

**First run:** B and D at *k* = 16. That pair answers Q1 and decides how far to widen the rest of the grid.

## Stage 4: Additional experiments (optional)

In order of increasing cost. None of these results appear in the paper.

| # | Experiment | Method | Question | Extra cost |
|---|---|---|---|---|
| 1 | Two-pass CMC | BE 512 → CMC → 64 → same CMC → *k* | Does a second comparison over a narrower set improve the top ranks? | Inference only |
| 2 | Block cross-candidate attention | Attention mask so candidates see only the query, not each other | Does CMC's gain come from comparing candidates? | 1 retraining |
| 3 | Training candidate count | Train with 8, 16, 32, 64 candidates | How few candidates are enough to train with? | 3 retrainings |

Expectations:

- **Two-pass CMC:** probably a small gain. Narrowing from 512 to 64 changed R@1 by only 0.14 points (Table 8). Small changes at the top still matter when the LLM gets only 4–8 candidates, so this is attached to configuration D.
- **Attention blocking:** the paper suggests the gain likely comes from cross-candidate interaction but doesn't test it. If accuracy holds with attention blocked, the gain comes from somewhere else.
- **Training candidate count:** the paper fixes 64 at training and varies only inference (Table 8).

A null result is still worth reporting. It would suggest CMC is limited by compressing each candidate into a single
vector, not by how it compares candidates.

## Metrics

The primary metric is **unnormalized accuracy**: the share of all test mentions whose final answer is correct. It matches
the definition in Tables 2 and 3 of the paper, so numbers are directly comparable.

Recall@k and normalized accuracy separate the two failure sources. A miss with the gold absent from the candidates is
an earlier-stage failure. A miss with the gold present is a final-stage failure. From the paper's numbers, the CE gets
about 25 points wrong even when the gold is in its candidates. How much of that gap an LLM closes is the substance of Q1.

| Metric | Definition | Question it answers |
|---|---|---|
| Unnormalized accuracy | Share of all mentions with a correct final answer | End-to-end pipeline performance |
| Recall@k | Share of mentions whose gold is among the *k* candidates passed to the final stage | Upper bound for the final stage |
| Normalized accuracy | Accuracy over mentions whose gold is among the *k* candidates | Selection ability of the final stage |
| Input tokens | Mean LLM prompt tokens per mention | Cost |
| Latency | Mean final-stage time per mention | Speed |
| Format error rate | Share of LLM outputs that are not a valid number | Prompt stability |

## Error analysis

Pair B and D per mention into four cells: both correct, CE only, LLM only, both wrong. Read 30 LLM-only and 30
CE-only cases and label their types. Also break results down by ZeSHEL test domain (Forgotten Realms, Lego, Star Trek,
YuGiOh).

## Logging rules

- Each experiment gets one folder with its config file, seed, commit hash and results.
- Any experiment that retrains CMC uses at least three seeds and reports mean ± standard deviation.
- Every raw LLM response is saved, so scoring can change and errors can be analyzed without new calls.
- Prompts are never changed after looking at test results.

## Timeline

About 7–9 weeks in total. This depends on GPU availability and on how quickly the Stage 0 files can be obtained.

| Stage | Estimate | Done when | If blocked |
|---|---|---|---|
| 0. Setup | 1 week | Environment, data, candidates and checkpoint in place; training starts without errors | Contact the authors after two days without the files |
| 1. Reproduction | 2 weeks | CMC R@1 within ±1 point of the paper | After two weeks, record the gap and use my numbers as the baseline |
| 2. Export | 0.5 weeks | Recall recomputed from JSONL matches Stage 1 | |
| 3. Main experiment | 2–3 weeks | Full table of 4 configs × 4 *k* × 2 orders on the 1k sample; best setting confirmed on the full test set | If LLM cost is too high, use only *k* = 8 and 16 |
| 4. Additional | 1–2 weeks | Two-pass CMC done; the other two optional | Skip if out of time |
| 5. Write-up | 1 week | This README has the reproduction table, main table and error analysis summary | |

The two-week limit on Stage 1 matters most. Every comparison here runs in the same environment, so conclusions hold
even if the reproduced numbers aren't identical to the paper's.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| LLM is less accurate than the CE | Q1 answer is "no" | Report it as is: how far an un-fine-tuned LLM falls short of a fine-tuned CE, and the error types. Q2 and Q3 are unaffected. |
| Reproduced numbers differ from the paper | No direct comparison to published numbers | Use my numbers as the baseline. All A–D comparisons happen in the same environment. |
| LLM already knows ZeSHEL entities | Answers from memory, not reading | ZeSHEL comes from fan wikis, and popular franchises may be in pretraining data. Run a control where the LLM gets only the mention and no candidates. |
| LLM relies on candidate order | Accuracy just copies the CMC ranking | Random-order condition (Stage 3). |
| LLM cost and time | Can't run the full grid | Narrow down on the 1k sample first. Consider local open models. |
| Candidate file or checkpoint unavailable | Blocked at Stage 0 | Contact the authors. Training the BE myself adds about 1–2 weeks. |
| Prior work already covers this | Less novelty | Before Stage 3, search "LLM listwise reranking entity linking" and "zero-shot entity linking LLM". Add any matching setting as a baseline. |

Prior work on LLMs as rerankers is extensive. What is new here is limited to the combination with CMC and the controlled
comparison on ZeSHEL.

## Results

*To be filled in.*

### Reproduction

| Item | Paper | This repo | Notes |
|---|---|---|---|
| BE recall@64 | 87.95 | – | |
| BE + CMC(64) R@1 | 59.22 | – | |
| BE + CMC(512) recall@64 | 91.51 | – | |
| BE + CMC(512) recall@16 | 86.32 | – | |
| CMC final reranker accuracy | 59.2 ± 0.3 | – | |

### Main experiment

| Config | *k* | Order | Recall@k | Unnorm. acc. | Norm. acc. | Tokens / query | Format errors |
|---|---|---|---|---|---|---|---|
| A | | | | | | | |
| B | | | | | | | |
| C | | | | | | | |
| D | | | | | | | |

### Error analysis

*Summary of LLM-only and CE-only cases.*

## How to run

From the upstream README (`submission` branch).

```bash
conda env create --file environment.yml
```

Training:

```bash
python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.1 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=1 --save_topk_nce --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_top1024 --type_bert=base --B=2 --gradient_accumulation_steps=2 --gpus=0 --model=./cocondenser.bin --cocondenser --distill_training --alpha=0.2 --identity_bert --fixed_initial_weight
```

Evaluation:

```bash
python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.1 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=1 --save_topk_nce --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_top1024 --type_bert=base --B=2 --gradient_accumulation_steps=2 --gpus=0 --model=./cocondenser.bin --cocondenser --distill_training --alpha=0.2 --identity_bert --fixed_initial_weight --training_finished --resume_training --run_id=[run_id]
```

Commands for CMC export (Stage 2) and LLM reranking (Stage 3) will be added here.

## Limitations

- The LLM is not fine-tuned, while the CE is.
- The LLM may have prior knowledge of ZeSHEL entities.
- Results cover a single dataset.

## References

- Song et al. 2024. *Comparing Neighbors Together Makes it Easy: Jointly Comparing Multiple Candidates for Efficient and Effective Retrieval.* [arXiv:2405.12801](https://arxiv.org/abs/2405.12801). All paper numbers and table references above come from v2.
- [yc-song/cmc](https://github.com/yc-song/cmc): original code and run commands.
- Logeswaran et al. 2019. *Zero-Shot Entity Linking by Reading Entity Descriptions.* [lajanugen/zeshel](https://github.com/lajanugen/zeshel).
- Yadav et al. 2022. *Efficient Nearest Neighbor Search for Cross-Encoder Models using Matrix Factorization.* (ZeSHEL bi-encoder.)
- Gao & Callan 2022. *Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval.* (coCondenser.)
- Zhang & Stratos 2021. *Understanding Hard Negatives in Noise Contrastive Estimation.*

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
