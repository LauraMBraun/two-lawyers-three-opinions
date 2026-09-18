#!/usr/bin/env python3
"""Every number the paper reports, for one run directory or all of them.

    uv run src/evaluate.py                       # all runs under results/
    uv run src/evaluate.py --run results/mock_on

Writes metrics.json per run and results/summary.csv across runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 42


# ------------------------------------------------------------------- helpers
def load_run(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "checkpoint_full.csv")
    for c in ["strafbar_pred", "gold_strafbar_majority", "contested"]:
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False})
    for c in ["gold_strafbar_frac", "frac_strafbar_true", "entropy_strafbar_bits",
              "verbalized_p_true", "verbalized_sd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 5/5 splits count as not punishable (footnote 3)
    df["pred"] = df["strafbar_pred"].fillna(False).astype(bool)
    df["gold"] = df["gold_strafbar_majority"].astype(bool)
    df["error"] = df["pred"] != df["gold"]
    df["u_entropy"] = df["entropy_strafbar_bits"]
    df["u_verbalized"] = 1 - 2 * (df["verbalized_p_true"] - 0.5).abs()
    return df


def boot_ci(fn, *arrays, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap over items."""
    arrays = [np.asarray(a) for a in arrays]
    k = len(arrays[0])
    if k < 3:
        return {"point": None, "lo": None, "hi": None, "n": int(k)}
    rng = np.random.default_rng(seed)
    point = fn(*arrays)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:          # a resample with only one class
            continue
        if v is not None and not np.isnan(v):
            boots.append(v)
    if not boots:
        return {"point": _r(point), "lo": None, "hi": None, "n": int(k)}
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": _r(point), "lo": _r(lo), "hi": _r(hi), "n": int(k)}


def _r(v, d=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), d)


def prf(pred, gold):
    tp = float(((pred == 1) & (gold == 1)).sum())
    fp = float(((pred == 1) & (gold == 0)).sum())
    fn = float(((pred == 0) & (gold == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and prec + rec else np.nan
    return prec, rec, f1


def safe_auc(y, s, metric="roc"):
    y, s = np.asarray(y, float), np.asarray(s, float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    return (roc_auc_score(y, s) if metric == "roc"
            else average_precision_score(y, s))


def jsd(p, q):
    """Jensen-Shannon divergence in bits between two Bernoulli distributions."""
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    q = np.clip(np.asarray(q, float), 1e-12, 1 - 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))

    return float(np.mean(0.5 * kl(p, m) + 0.5 * kl(q, m)))


# ----------------------------------------------------------------- the metrics
def classification(df: pd.DataFrame) -> dict:
    out = {}
    for name, sub in [("overall", df),
                      ("unanimous", df[~df["contested"].astype(bool)]),
                      ("contested", df[df["contested"].astype(bool)])]:
        if sub.empty:
            continue
        p, g = sub["pred"].to_numpy(int), sub["gold"].to_numpy(int)
        out[name] = {
            "n": int(len(sub)),
            "precision": boot_ci(lambda a, b: prf(a, b)[0], p, g),
            "recall": boot_ci(lambda a, b: prf(a, b)[1], p, g),
            "f1": boot_ci(lambda a, b: prf(a, b)[2], p, g),
        }
    return out


def uncertainty(df: pd.DataFrame) -> dict:
    err = df["error"].to_numpy(int)
    contested = df["contested"].astype(bool).to_numpy(int)
    out = {"error_detection": {}, "disagreement_detection": {}, "soft_label_fit": {}}

    for sig, col in [("sampling", "u_entropy"), ("verbalized", "u_verbalized")]:
        s = df[col].to_numpy(float)
        out["error_detection"][sig] = {
            "auroc": boot_ci(lambda y, x: safe_auc(y, x, "roc"), err, s),
            "auprc": boot_ci(lambda y, x: safe_auc(y, x, "pr"), err, s),
        }
        out["disagreement_detection"][sig] = {
            "auroc": boot_ci(lambda y, x: safe_auc(y, x, "roc"), contested, s),
            "auprc": boot_ci(lambda y, x: safe_auc(y, x, "pr"), contested, s),
            "baseline_auprc": _r(contested.mean()),
        }

    soft = df["gold_strafbar_frac"].to_numpy(float)
    for sig, col in [("sampling", "frac_strafbar_true"),
                     ("verbalized", "verbalized_p_true")]:
        v = df[col].to_numpy(float)
        out["soft_label_fit"][sig] = {
            "brier": _r(np.mean((v - soft) ** 2)),
            "jsd_bits": _r(jsd(v, soft)),
            "mean_abs_gap": _r(np.mean(np.abs(v - soft))),
        }

    # Do the two signals rank items the same way?
    both = df[["u_entropy", "u_verbalized"]].dropna()
    constant = both["u_entropy"].nunique() < 2 or both["u_verbalized"].nunique() < 2
    out["signal_rank_correlation"] = (
        None if len(both) < 3 or constant
        else _r(both["u_entropy"].corr(both["u_verbalized"], method="spearman")))
    return out


def stratified_disagreement(df: pd.DataFrame) -> dict:
    """Detection within majority-negative (0/3 vs 1/3) and majority-positive (2/3 vs 3/3)."""
    out = {}
    for name, mask in [("majority_negative", df["gold_strafbar_frac"] <= 1 / 3),
                       ("majority_positive", df["gold_strafbar_frac"] >= 2 / 3)]:
        sub = df[mask]
        y = sub["contested"].astype(bool).to_numpy(int)
        if len(sub) < 3 or len(np.unique(y)) < 2:
            out[name] = {"n": int(len(sub)), "note": "too few items or one class only"}
            continue
        out[name] = {"n": int(len(sub)), "baseline_auprc": _r(y.mean())}
        for sig, col in [("sampling", "u_entropy"), ("verbalized", "u_verbalized")]:
            s = sub[col].to_numpy(float)
            out[name][sig] = {
                "auroc": boot_ci(lambda a, b: safe_auc(a, b, "roc"), y, s),
                "auprc": boot_ci(lambda a, b: safe_auc(a, b, "pr"), y, s),
            }
    return out


def deferral(df: pd.DataFrame, coverage=0.5, seed=0) -> dict:
    """Class-composition diagnostics for the selective-prediction claim.

    Entropy over n binary samples takes few distinct values, so ties are heavy:
    break them at random rather than by row order, which would otherwise let
    corpus ordering decide what gets retained.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for sig, col in [("sampling", "u_entropy"), ("verbalized", "u_verbalized")]:
        g = df.dropna(subset=[col])
        n = len(g)
        k = int(round(coverage * n))
        if k in (0, n):
            continue
        order = np.argsort(g[col].to_numpy(float) + rng.random(n) * 1e-9, kind="stable")
        retained, deferred = g.iloc[order[:k]], g.iloc[order[k:]]

        pos_share = float(retained["gold"].mean())
        baseline = min(pos_share, 1 - pos_share)      # always predict majority class
        retained_err = float(retained["error"].mean())
        n_err = int(g["error"].sum())

        out[sig] = {
            "coverage": coverage,
            "full_error": _r(g["error"].mean()),
            "retained_error": _r(retained_err),
            "retained_pos_share": _r(pos_share),
            "baseline_error_retained": _r(baseline),
            "lift_vs_baseline": _r(retained_err / baseline) if baseline > 0 else None,
            "frac_errors_deferred": _r(1 - retained["error"].sum() / n_err) if n_err else None,
            "recall_pos_retained": _r(retained["gold"].sum() / g["gold"].sum())
                                   if g["gold"].sum() else None,
            "contested_recall_deferred": _r(deferred["contested"].astype(bool).sum()
                                            / g["contested"].astype(bool).sum())
                                         if g["contested"].astype(bool).sum() else None,
        }
    return out


def risk_coverage(df: pd.DataFrame, col: str, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    g = df.dropna(subset=[col])
    n = len(g)
    order = np.argsort(g[col].to_numpy(float) + rng.random(n) * 1e-9, kind="stable")
    err = g["error"].to_numpy(int)[order]
    return [{"coverage": _r((i + 1) / n), "risk": _r(err[:i + 1].mean())}
            for i in range(n)]


# --------------------------------------------------------------------- driver
def evaluate_run(run_dir: Path) -> dict:
    df = load_run(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    m = {
        "meta": meta,
        "descriptive": {
            "n_posts": int(len(df)),
            "n_contested": int(df["contested"].astype(bool).sum()),
            "share_contested": _r(df["contested"].astype(bool).mean()),
            "share_punishable_majority": _r(df["gold"].mean()),
            "n_without_valid_sample": int((df["n_valid_samples"] == 0).sum()),
            "mean_within_item_sd": _r(df["verbalized_sd"].mean()),
            "verdict_agreement_verbalized": _r(
                ((df["verbalized_p_true"] >= 0.5) == df["pred"]).mean()),
        },
        "classification": classification(df),
        "uncertainty": uncertainty(df),
        "stratified_disagreement": stratified_disagreement(df),
        "deferral": deferral(df),
        "risk_coverage": {sig: risk_coverage(df, col) for sig, col in
                          [("sampling", "u_entropy"), ("verbalized", "u_verbalized")]},
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m


def flatten(m: dict) -> dict:
    u, c, d = m["uncertainty"], m["classification"], m["deferral"]
    row = {
        "condition": f"{m['meta']['model'].removeprefix('hf:')} / {m['meta']['thinking']}",
        "n_posts": m["descriptive"]["n_posts"],
        "share_contested": m["descriptive"]["share_contested"],
        "f1": c["overall"]["f1"]["point"],
        "f1_unanimous": c.get("unanimous", {}).get("f1", {}).get("point"),
        "f1_contested": c.get("contested", {}).get("f1", {}).get("point"),
    }
    for sig in ("sampling", "verbalized"):
        row[f"err_auroc_{sig}"] = u["error_detection"][sig]["auroc"]["point"]
        row[f"dis_auroc_{sig}"] = u["disagreement_detection"][sig]["auroc"]["point"]
        row[f"dis_auprc_{sig}"] = u["disagreement_detection"][sig]["auprc"]["point"]
        row[f"brier_{sig}"] = u["soft_label_fit"][sig]["brier"]
        row[f"jsd_{sig}"] = u["soft_label_fit"][sig]["jsd_bits"]
        if sig in d:
            row[f"lift_{sig}"] = d[sig]["lift_vs_baseline"]
            row[f"errors_deferred_{sig}"] = d[sig]["frac_errors_deferred"]
    row["dis_auprc_baseline"] = u["disagreement_detection"]["sampling"]["baseline_auprc"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="single run directory")
    ap.add_argument("--results", default=str(ROOT / "results"))
    a = ap.parse_args()

    results = Path(a.results)
    runs = ([Path(a.run)] if a.run
            else sorted(p.parent for p in results.glob("*/checkpoint_full.csv")))
    if not runs:
        raise SystemExit(f"no runs found under {results}")

    rows = []
    for run in runs:
        print(f"[eval] {run.name}")
        rows.append(flatten(evaluate_run(run)))

    summary = pd.DataFrame(rows)
    summary.to_csv(results / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\n[done] {results / 'summary.csv'}")


if __name__ == "__main__":
    main()
