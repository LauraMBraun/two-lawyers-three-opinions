#!/usr/bin/env python3
"""Draw n samples per post and aggregate them into one row per post.

    uv run src/run_inference.py --model Qwen/Qwen3-0.6B --thinking on

Writes results/<model>_<thinking>/checkpoint_full.csv plus meta.json.
One row per post, with the columns the evaluation and figure scripts expect.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from prompts import SECTIONS, build_messages

ROOT = Path(__file__).resolve().parent.parent
JSON_RE = re.compile(r"\{[^{}]*\"p_punishable\"[^{}]*\}", re.S)


# --------------------------------------------------------------------- parsing
def normalize_section(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip().replace("§§", "§")
    s = re.sub(r"\s+", " ", s)
    if s == "§ 86":                      # collapse § 86 into § 86a, as in the paper
        s = "§ 86a"
    return s if s in SECTIONS else "Anderer"


def parse_reply(raw: str) -> dict | None:
    """Pull the last JSON object out of a reply; None if nothing parses."""
    matches = JSON_RE.findall(raw or "")
    for block in reversed(matches):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        p = obj.get("p_punishable")
        verdict = obj.get("punishable")
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() == "true"
        if not isinstance(p, (int, float)) or not isinstance(verdict, bool):
            continue
        return {"p": max(0, min(100, int(p))),
                "punishable": verdict,
                "section": normalize_section(obj.get("section"))}
    return None


_HF_CACHE: dict = {}


def _load_hf(model_id: str):
    """Load tokenizer and model once per process."""
    if model_id in _HF_CACHE:
        return _HF_CACHE[model_id]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(f"missing dependency: {exc.name}. Run: uv sync")

    print(f"[info] loading {model_id} (first run downloads weights)")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    print(f"[info] device: {next(model.parameters()).device}")
    _HF_CACHE[model_id] = (tok, model)
    return tok, model


def sample_hf(text: str, model_id: str, thinking: bool, temperature: float,
              max_tokens: int, n: int) -> list[dict | None]:
    """Draw all n samples for one post in a single batched generate() call."""
    tok, model = _load_hf(model_id)        
    import torch
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:                                   
        prompt = tok.apply_chat_template(build_messages(text),
                                         enable_thinking=thinking, **kwargs)
    except TypeError:
        prompt = tok.apply_chat_template(build_messages(text), **kwargs)

    batch = tok([prompt] * n, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**batch, do_sample=True, temperature=temperature,
                             top_p=0.95, max_new_tokens=max_tokens,
                             pad_token_id=tok.pad_token_id)
    replies = tok.batch_decode(out[:, batch["input_ids"].shape[1]:],
                               skip_special_tokens=True)
    return [parse_reply(r) for r in replies]


# ------------------------------------------------------------------ aggregation
def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def aggregate(samples: list[dict]) -> dict:
    """Ten samples -> one row. Verdict is the majority, probability the mean."""
    valid = [s for s in samples if s is not None]
    n = len(valid)
    if n == 0:
        return {"n_valid_samples": 0, "strafbar_pred": pd.NA,
                "frac_strafbar_true": pd.NA, "entropy_strafbar_bits": pd.NA,
                "verbalized_p_true": pd.NA, "verbalized_sd": pd.NA,
                "paragraph_pred": pd.NA, "paragraph_agreement": pd.NA,
                "entropy_paragraph_bits": pd.NA}

    frac = sum(s["punishable"] for s in valid) / n
    probs = [s["p"] / 100.0 for s in valid]
    mean_p = sum(probs) / n
    sd = (sum((x - mean_p) ** 2 for x in probs) / n) ** 0.5

    counts = Counter(s["section"] or "Kein Paragraph" for s in valid)
    top_sec, top_n = counts.most_common(1)[0]
    para_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    # 5/5 splits count as not punishable (footnote 3 in the paper)
    return {"n_valid_samples": n,
            "strafbar_pred": bool(frac > 0.5),
            "frac_strafbar_true": frac,
            "entropy_strafbar_bits": binary_entropy(frac),
            "verbalized_p_true": mean_p,
            "verbalized_sd": sd,
            "paragraph_pred": top_sec,
            "paragraph_agreement": top_n / n,
            "entropy_paragraph_bits": para_entropy}


def gold_columns(row: pd.Series) -> dict:
    """Three expert section labels -> binary soft label and majority vote."""
    votes = [row["expert_a"], row["expert_b"], row["expert_c"]]
    punishable = [v.strip() != "Kein Paragraph" for v in votes]
    frac = sum(punishable) / 3
    distinct = len({normalize_section(v) for v in votes})
    return {"gold_strafbar_frac": frac,
            "gold_strafbar_majority": frac > 0.5,
            "contested": 0 < frac < 1,
            "paragraphen_agreement": 1.0 if distinct == 1 else (2 / 3 if distinct == 2 else 1 / 3)}


# --------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Hugging Face repo id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=0, help="first N posts (0 = all)")
    ap.add_argument("--data", default=str(ROOT / "data" / "dummy_posts.csv"))
    ap.add_argument("--out", default=None, help="run directory (default: results/<model>_<thinking>)")
    a = ap.parse_args()

    posts = pd.read_csv(a.data)
    if a.limit:
        posts = posts.head(a.limit)

    tag = a.model.removeprefix("hf:").replace(":", "-").replace("/", "-")
    run_dir = Path(a.out) if a.out else ROOT / "results" / f"{tag}_{a.thinking}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] {a.model} thinking={a.thinking} temp={a.temperature} "
          f"posts={len(posts)} samples={a.n_samples}")

    rows = []
    bar = tqdm(list(posts.reset_index(drop=True).iterrows()),
               desc="posts", unit="post", dynamic_ncols=True)
    for i, post in bar:
        bar.set_description(f"post {i + 1}/{len(posts)}")
        bar.set_postfix_str(f"{post['post_id']} generating…", refresh=True)
        t0 = time.perf_counter()
        samples = sample_hf(post["text"], a.model, a.thinking == "on",
                            a.temperature, a.max_tokens, a.n_samples)
        row = {"title": post["post_id"], "text": post["text"]}
        row |= gold_columns(post)
        row |= aggregate(samples)
        rows.append(row)
        dt = time.perf_counter() - t0
        n_valid = row["n_valid_samples"]
        bar.set_postfix_str(
            f"{post['post_id']} valid={n_valid}/{a.n_samples} "
            f"p={row['verbalized_p_true']:.2f} {dt:.0f}s"
            if n_valid else f"{post['post_id']} no valid sample {dt:.0f}s",
            refresh=True)
        if n_valid < a.n_samples:
            bar.write(f"  [warn] {post['post_id']}: only {n_valid}/{a.n_samples} "
                      f"samples returned parseable JSON")
    bar.close()

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "checkpoint_full.csv", index=False)
    meta = {"model": a.model, "thinking": a.thinking, "temperature": a.temperature,
            "n_samples": a.n_samples, "n_posts": len(df), "data": str(a.data)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] {run_dir / 'checkpoint_full.csv'}")


if __name__ == "__main__":
    main()
