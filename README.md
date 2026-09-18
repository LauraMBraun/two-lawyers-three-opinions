# Two Lawyers, Three Opinions (Replication code)

Code for the workshop paper *Two Lawyers, Three Opinions: Can Output-Based LLM Uncertainty Find the Cases
Where Experts Disagree?* (UncertaiNLP @ EMNLP 2026).

The paper evaluates two output-based uncertainty signals -- **sampling entropy** and
**verbalized probability** -- against expert disagreement on German social media posts
annotated for punishability under the StGB.

**The corpus in this repository carries no content.** The KISTRA data used in the paper
cannot be released for data-protection reasons. `data/dummy_posts.csv` holds 5 dummy data samples
with invented three-expert annotations. 

## Quickstart

```bash
uv sync
```


```bash
./run_all.sh Qwen/Qwen3-0.6B   
```

### Models

In this replication package we use with QWEN3 0.6B by default a smaller options that fit a laptop GPU.
The paper uses Qwen3 8B/32B and Gemma 4 12B/31B at the recommended temperatures (0.6/0.7 and 1.0).

Small models might fail to emit the required JSON. Samples that do not parse are dropped
and counted in `n_valid_samples`; a post with zero valid samples is excluded downstream.
If most samples fail, raise `--max-tokens` or drop to `--thinking off`.

## Layout

```
data/dummy_posts.csv   synthetic posts + three expert section labels each
src/prompts.py         system prompt and user template, verbatim from the appendix
src/run_inference.py   n samples per post -> one row per post
src/evaluate.py        every statistic the paper reports
src/figures.py         the paper's plots
results/<run>/         checkpoint_full.csv, meta.json, metrics.json
figures/               pdf + png
```

## What gets computed

`run_inference.py` aggregates the samples per post the way the paper does: the **verdict**
is the majority of the samples (a 5/5 split counts as not punishable), the **verbalized
probability** is the mean of the reported `p_punishable` values, and **sampling entropy**
is the binary entropy of the share of punishable judgments. As an uncertainty score the
verbalized probability enters as its closeness to 0.5.

`evaluate.py` writes `metrics.json` per run:

| key | paper |
| --- | --- |
| `classification` | P/R/F1 overall and split by contested vs. unanimous (Fig. 1, 7) |
| `uncertainty.error_detection` | AUROC/AUPRC for the model's own errors (Fig. 3) |
| `uncertainty.disagreement_detection` | AUROC/AUPRC for contested posts, with the prevalence baseline (Fig. 5) |
| `uncertainty.soft_label_fit` | Brier and Jensen-Shannon divergence against the human soft label (Fig. 11) |
| `stratified_disagreement` | the same, split by majority class (Fig. 9) |
| `deferral` | class composition of the retained half, lift over a majority-class baseline, share of errors and contested posts deferred |
| `risk_coverage` | selective risk at every coverage level (Fig. 3) |

With only 5 posts some statistics are undefined... an AUROC needs both classes present,
an F1 on the contested subset needs a true positive in it. Those come back as `null` and
the affected figure is skipped.

## Citation

```bibtex
@inproceedings{braun2026twolawyers,
  title     = {Two Lawyers, Three Opinions: Can Output-Based {LLM} Uncertainty Find the Cases Where Experts Disagree?},
  author    = {Braun, Laura and Assenmacher, Matthias and Hohenadler, Martin and Kauermann, Göran},
  booktitle = {Proceedings of the third Workshop on Uncertainty-Aware NLP (UncertaiNLP)},
  year      = {2026}
}
```
