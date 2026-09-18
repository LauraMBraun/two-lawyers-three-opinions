#!/usr/bin/env python3
"""Reproduce the paper's figures from the run directories.

    uv run src/figures.py                 # all figures into figures/
    uv run src/figures.py --only f1 auroc

Figures are keyed by the names used in the paper:
  f1          stratified F1, unanimous vs. contested       (Fig. 1)
  sampling    sampled share grouped by human soft label    (Fig. 2)
  risk        risk-coverage curves for both signals        (Fig. 3)
  calibration verbalized P(punishable) vs. soft label      (Fig. 4)
  auroc       contested-vs-unanimous AUROC and AUPRC       (Fig. 5)
  strata      the same, split by majority class            (Fig. 9)
  instability within-item SD of the verbalized probability (Fig. 8)
  distrib     Brier and JSD against the human soft label   (Fig. 11)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import load_run, safe_auc

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "figures"
PALETTE = ["#2b7a8c", "#174d5a", "#e39b8a", "#c1553a",
           "#4d9dbf", "#1f5f75", "#cbb994", "#8a6d3b"]
GRAY = "#555555"


def load_runs(results: Path) -> list[tuple[str, pd.DataFrame]]:
    runs = []
    for meta_path in sorted(results.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        label = f"{meta['model'].removeprefix('hf:')} · {meta['thinking'].upper()}"
        runs.append((label, load_run(meta_path.parent)))
    if not runs:
        raise SystemExit(f"no runs found under {results}")
    return runs


def _bars(ax, x, values, colors, filled, width):
    kw = (dict(color=colors) if filled
          else dict(facecolor="none", edgecolor=colors, linewidth=1.6))
    ax.bar(x, values, width, zorder=2, **kw)


def _finish(fig, name):
    FIGDIR.mkdir(exist_ok=True)
    path = FIGDIR / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ------------------------------------------------------------------ Figure 1
def fig_f1(runs):
    labels = [c for c, _ in runs]
    x, w = np.arange(len(runs)), 0.38
    fig, ax = plt.subplots(figsize=(1.3 * len(runs) + 2, 3.4))
    for off, key, filled in [(-w / 2, False, True), (+w / 2, True, False)]:
        vals = []
        for _, df in runs:
            sub = df[df["contested"].astype(bool) == key]
            if sub.empty:
                vals.append(np.nan)
                continue
            tp = ((sub["pred"] == 1) & (sub["gold"] == 1)).sum()
            fp = ((sub["pred"] == 1) & (sub["gold"] == 0)).sum()
            fn = ((sub["pred"] == 0) & (sub["gold"] == 1)).sum()
            vals.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan)
        _bars(ax, x + off, vals, PALETTE[:len(runs)], filled, w)
        for xi, v in zip(x + off, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=GRAY)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, label="experts unanimous"),
                       plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=GRAY,
                                     label="experts contested")],
              frameon=False, fontsize=8, loc="upper left")
    _finish(fig, "f1_stratified")


# ------------------------------------------------------------------ Figure 2
def fig_sampling(runs):
    buckets = [(0.0, "0/3"), (1 / 3, "1/3"), (2 / 3, "2/3"), (1.0, "3/3")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for ax, (val, name) in zip(axes, buckets):
        for (label, df), color in zip(runs, PALETTE):
            sub = df[np.isclose(df["gold_strafbar_frac"], val, atol=0.02)]
            if sub.empty:
                continue
            counts = sub["frac_strafbar_true"].value_counts(normalize=True).sort_index()
            ax.plot(counts.index, counts.values, marker="o", ms=3,
                    lw=1, color=color, label=label)
        ax.set_title(f"{name} punishable (n = {int(np.isclose(runs[0][1]['gold_strafbar_frac'], val, atol=0.02).sum())})",
                     fontsize=9)
        ax.set_xlim(-0.05, 1.05)
    axes[0].set_ylabel("relative frequency")
    fig.supxlabel("Sampled share of 'punishable' judgments", fontsize=9)
    axes[-1].legend(frameon=False, fontsize=7)
    _finish(fig, "sampling_vs_human")


# ------------------------------------------------------------------ Figure 3
def fig_risk(runs):
    fig, axes = plt.subplots(2, 1, figsize=(6, 5.4), sharex=True)
    for ax, (sig, col) in zip(axes, [("sampling entropy", "u_entropy"),
                                     ("verbalized probability", "u_verbalized")]):
        for (label, df), color in zip(runs, PALETTE):
            g = df.dropna(subset=[col])
            if g.empty:
                continue
            rng = np.random.default_rng(0)
            order = np.argsort(g[col].to_numpy(float) + rng.random(len(g)) * 1e-9)
            err = g["error"].to_numpy(int)[order]
            cov = np.arange(1, len(err) + 1) / len(err)
            auc = safe_auc(g["error"].to_numpy(int), g[col].to_numpy(float))
            ax.plot(cov, np.cumsum(err) / np.arange(1, len(err) + 1),
                    color=color, lw=1.2,
                    label=f"{label} ({auc:.2f})" if not np.isnan(auc) else label)
        ax.set_ylabel(sig, fontsize=9)
        ax.legend(frameon=False, fontsize=7, title="AUROC", title_fontsize=7)
    axes[1].set_xlabel("Coverage (lowest-uncertainty items retained first)")
    fig.supylabel("Selective risk (error rate)", fontsize=9)
    _finish(fig, "risk_coverage")


# ------------------------------------------------------------------ Figure 4
def fig_calibration(runs):
    n = len(runs)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows),
                             squeeze=False, sharey=True)
    buckets = [(0.0, "0/3"), (1 / 3, "1/3"), (2 / 3, "2/3"), (1.0, "3/3")]
    for ax, (label, df), color in zip(axes.ravel(), runs, PALETTE):
        xs, meds, q1, q3 = [], [], [], []
        for i, (val, _) in enumerate(buckets):
            sub = df[np.isclose(df["gold_strafbar_frac"], val, atol=0.02)]
            if sub.empty:
                continue
            v = sub["verbalized_p_true"].dropna()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).normal(0, .04, len(v)),
                       v, s=8, alpha=.5, color=color)
            xs.append(i); meds.append(v.median())
            q1.append(v.quantile(.25)); q3.append(v.quantile(.75))
        if xs:
            ax.fill_between(xs, q1, q3, color="0.85", zorder=0)
            ax.plot(xs, meds, "k-o", ms=4, lw=1.2)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels([b[1] for b in buckets])
        ax.set_ylim(-0.05, 1.05)
    for ax in axes.ravel()[len(runs):]:
        ax.axis("off")
    fig.supxlabel("Human soft label (share punishable)", fontsize=9)
    fig.supylabel("Verbalized P(punishable)", fontsize=9)
    _finish(fig, "calibration_verbalized")


# ------------------------------------------------------------------ Figure 5
def fig_auroc(runs):
    labels = [c for c, _ in runs]
    x, w = np.arange(len(runs)), 0.42
    fig, axes = plt.subplots(2, 1, figsize=(1.3 * len(runs) + 2, 5.2), sharex=True)
    base = float(np.mean([df["contested"].astype(bool).mean() for _, df in runs]))

    for ax, (metric, kind, rnd, ylim) in zip(
            axes, [("AUROC", "roc", 0.5, (0.4, 1.0)),
                   ("AUPRC", "pr", base, (0.0, 0.8))]):
        for off, (sig, col), filled in [(-w / 2, ("sampling", "u_entropy"), True),
                                        (+w / 2, ("verbalized", "u_verbalized"), False)]:
            vals = [safe_auc(df["contested"].astype(bool).to_numpy(int),
                             df[col].to_numpy(float), kind) for _, df in runs]
            _bars(ax, x + off, vals, PALETTE[:len(runs)], filled, w)
            for xi, v in zip(x + off, vals):
                if not np.isnan(v):
                    ax.text(xi, v + 0.015, f"{v:.2f}", ha="center",
                            fontsize=7, color=GRAY)
        ax.axhline(rnd, color=GRAY, lw=0.8, ls=":")
        ax.set_ylabel(metric)
        ax.set_ylim(*ylim)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=GRAY,
                                          label="sampling entropy"),
                            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=GRAY,
                                          label="verbalized probability")],
                   frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    _finish(fig, "disagreement_auroc_auprc")


# ------------------------------------------------------------------ Figure 9
def fig_strata(runs):
    labels = [c for c, _ in runs]
    x, w = np.arange(len(runs)), 0.42
    fig, axes = plt.subplots(1, 2, figsize=(2.0 * len(runs) + 3, 3.4), sharey=True)
    strata = [("majority-negative (0/3 vs 1/3)", lambda d: d["gold_strafbar_frac"] <= 1 / 3),
              ("majority-positive (2/3 vs 3/3)", lambda d: d["gold_strafbar_frac"] >= 2 / 3)]
    for ax, (title, mask_fn) in zip(axes, strata):
        for off, col, filled in [(-w / 2, "u_entropy", True),
                                 (+w / 2, "u_verbalized", False)]:
            vals = []
            for _, df in runs:
                sub = df[mask_fn(df)]
                y = sub["contested"].astype(bool).to_numpy(int)
                vals.append(safe_auc(y, sub[col].to_numpy(float))
                            if len(np.unique(y)) == 2 else np.nan)
            _bars(ax, x + off, vals, PALETTE[:len(runs)], filled, w)
        ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0.3, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylabel("AUROC")
    _finish(fig, "disagreement_auroc_by_majority")


# ------------------------------------------------------------------ Figure 8
def fig_instability(runs):
    fig, ax = plt.subplots(figsize=(1.2 * len(runs) + 2, 3.2))
    data = [df["verbalized_sd"].dropna().to_numpy() for _, df in runs]
    parts = ax.violinplot([d for d in data if len(d)], showmedians=True)
    for body, color in zip(parts["bodies"], PALETTE):
        body.set_facecolor(color)
        body.set_alpha(.7)
    ax.set_xticks(range(1, len([d for d in data if len(d)]) + 1))
    ax.set_xticklabels([c for (c, _), d in zip(runs, data) if len(d)],
                       rotation=45, ha="right")
    ax.set_ylabel("Within-item SD of\nverbalized P(punishable)", fontsize=9)
    _finish(fig, "verbalized_instability")


# ----------------------------------------------------------------- Figure 11
def fig_distrib(runs):
    labels = [c for c, _ in runs]
    x, w = np.arange(len(runs)), 0.42
    from evaluate import jsd
    fig, axes = plt.subplots(2, 1, figsize=(1.3 * len(runs) + 2, 4.8), sharex=True)
    for ax, metric in zip(axes, ["brier", "jsd"]):
        for off, col, filled in [(-w / 2, "frac_strafbar_true", True),
                                 (+w / 2, "verbalized_p_true", False)]:
            vals = []
            for _, df in runs:
                g = df.dropna(subset=[col, "gold_strafbar_frac"])
                v, s = g[col].to_numpy(float), g["gold_strafbar_frac"].to_numpy(float)
                vals.append(np.mean((v - s) ** 2) if metric == "brier" else jsd(v, s))
            _bars(ax, x + off, vals, PALETTE[:len(runs)], filled, w)
        ax.set_ylabel("Brier" if metric == "brier" else "JSD [bits]")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=GRAY,
                                          label="sampling distribution"),
                            plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=GRAY,
                                          label="verbalized probability")],
                   frameon=False, fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    _finish(fig, "distributional_metrics")


FIGURES = {"f1": fig_f1, "sampling": fig_sampling, "risk": fig_risk,
           "calibration": fig_calibration, "auroc": fig_auroc,
           "strata": fig_strata, "instability": fig_instability,
           "distrib": fig_distrib}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    a = ap.parse_args()

    runs = load_runs(Path(a.results))
    print(f"[figures] {len(runs)} run(s): {', '.join(c for c, _ in runs)}")
    for name in (a.only or FIGURES):
        try:
            FIGURES[name](runs)
        except Exception as exc:                      # a tiny dummy run can be degenerate
            print(f"  [skip] {name}: {exc}")


if __name__ == "__main__":
    main()
