# -*- coding: utf-8 -*-
"""
Perplexity evaluation with optional override of the first `start_size` tokens.

- Baseline: uses dataset tokens as-is.
- Override mode: replace the first `start_size` tokens of every sample with
  user-specified tokens via --override_start_text or --override_start_ids.

Compatible with Streaming-LLM (StartRecentKVCache) even when that class
does not provide a `.reset()` method.
"""

import os
from typing import List, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from streaming_llm.utils import parse_args, load
from streaming_llm.kv_cache import StartRecentKVCache


# ==========================================================
# Utility functions
# ==========================================================

def select_kv_seq_dims(model) -> Tuple[int, int]:
    """Infer key/value seq dimensions for the model architecture."""
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    k_seq_dim = v_seq_dim = 2
    if "mpt" in model_type:
        k_seq_dim = v_seq_dim = 3
    return k_seq_dim, v_seq_dim


def get_override_ids(tokenizer, args, start_size: int) -> Optional[List[int]]:
    """
    Return list of token IDs to override the first `start_size` tokens.
    Priority: --override_start_ids > --override_start_text.
    """
    ids = None
    if getattr(args, "override_start_ids", None):
        raw = [x.strip() for x in args.override_start_ids.split(",")]
        ids = [int(x) for x in raw if x != ""]
    elif getattr(args, "override_start_text", None):
        ids = tokenizer.encode(args.override_start_text, add_special_tokens=False)

    if ids is None:
        return None

    # Adjust length
    if len(ids) < start_size:
        eos = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        ids = ids + [eos] * (start_size - len(ids))
    else:
        ids = ids[:start_size]
    return ids


def iter_texts(dataset, text_key_hint: Optional[str] = "text"):
    """Yield raw text strings from HuggingFace dataset."""
    for sample in dataset:
        if text_key_hint and text_key_hint in sample:
            yield sample[text_key_hint]
        else:
            yield str(sample)


# ==========================================================
# Main evaluation
# ==========================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parse_args()

    # --------------------------
    # Load dataset
    # --------------------------
    data = load_dataset(args.dataset_name, args.task, split=args.split)
    if args.num_samples is not None:
        data = data.select(range(min(len(data), args.num_samples)))

    # --------------------------
    # Load model/tokenizer
    # --------------------------
    model, tokenizer = load(args.model_name_or_path)
    model.to(device)
    model.eval()

    # --------------------------
    # Prepare override IDs
    # --------------------------
    override_ids = get_override_ids(tokenizer, args, args.start_size)

    # --------------------------
    # Settings
    # --------------------------
    use_sr_cache = bool(getattr(args, "enable_start_recent_kv_cache", False))
    loss_fn = CrossEntropyLoss(reduction="none")
    token_nlls = []
    evaluated_tokens = 0

    os.makedirs(args.output_dir, exist_ok=True)
    nll_path = os.path.join(args.output_dir, "nlls.txt")

    # --------------------------
    # Main loop
    # --------------------------
    with open(nll_path, "w", encoding="utf-8") as nll_f:
        for text in tqdm(iter_texts(data), total=len(data)):
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            input_ids_full = enc["input_ids"][0]

            if input_ids_full.numel() < args.start_size + 1:
                continue

            # ---- optional override ----
            if override_ids is not None and args.start_size > 0:
                input_ids_full = input_ids_full.clone()
                input_ids_full[:args.start_size] = torch.tensor(
                    override_ids, dtype=input_ids_full.dtype
                )

            # ---- per-sample cache init ----
            past_key_values = None
            cache_manager = None
            if use_sr_cache:
                k_seq_dim, v_seq_dim = select_kv_seq_dims(model)
                cache_manager = StartRecentKVCache(
                    start_size=args.start_size,
                    recent_size=args.recent_size,
                    k_seq_dim=k_seq_dim,
                    v_seq_dim=v_seq_dim,
                )

            # ---- teacher forcing ----
            with torch.no_grad():
                for t in range(1, input_ids_full.numel()):
                    cur_inp = input_ids_full[t - 1 : t].unsqueeze(0).to(device)  # (1,1)
                    target = input_ids_full[t : t + 1].to(device)               # (1,)
                    target = target.long()  # ensure proper dtype for CE

                    out = model(input_ids=cur_inp, use_cache=True, past_key_values=past_key_values)
                    logits = out.logits[:, -1, :]  # (1, vocab)

                    if cache_manager is not None:
                        past_key_values = cache_manager(out.past_key_values)
                    else:
                        past_key_values = out.past_key_values

                    # IMPORTANT: keep logits as (1, vocab) to match target shape (1,)
                    nll = loss_fn(logits, target).mean()
                    token_nlls.append(nll.detach().cpu())
                    nll_f.write(f"{float(nll):.8f}\n")

                    evaluated_tokens += 1
                    if args.num_eval_tokens is not None and evaluated_tokens >= args.num_eval_tokens:
                        break

            if args.num_eval_tokens is not None and evaluated_tokens >= args.num_eval_tokens:
                break

    # --------------------------
    # Compute perplexity
    # --------------------------
    if len(token_nlls) == 0:
        print("No tokens evaluated.")
        ppl_value = float("inf")
    else:
        mean_nll = torch.stack(token_nlls).mean()
        ppl_value = float(torch.exp(mean_nll))

    print(f"Perplexity: {ppl_value:.6f}")
    with open(os.path.join(args.output_dir, "ppl.txt"), "w", encoding="utf-8") as wf:
        wf.write(f"{ppl_value:.6f}\n")


# ==========================================================
# Entrypoint
# ==========================================================

if __name__ == "__main__":
    main()
