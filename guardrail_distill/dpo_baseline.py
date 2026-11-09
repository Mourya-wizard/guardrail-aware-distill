"""Baseline: verifier-labelled on-policy preference pairs + TRL DPO (full fine-tune, fp32 master weights)."""
import guardrail_distill.env as env  # noqa
import argparse
import json
import math
import os
import time
import random
from collections import Counter

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from guardrail_distill.data import get_domain, load
from guardrail_distill.generation import STUDENT, generate, load_model, load_tok, prompt_ids, seed_all

REDIRECT = {
    "pii_request": "I'm sorry, but I can't share the owner's or any other tenant's private details. Please use the platform's official channels. I'm happy to help with your rent, deposit, agreement, maintenance or move-out questions.",
    "guarantee_demand": "I can't promise a specific outcome, amount or date. As per our policy, deposit refunds are processed within 7 working days after the move-out inspection, and the amount depends on the inspection report. I'm happy to explain the process.",
    "abuse": "I'm sorry, but I won't write insulting or abusive messages. I'm happy to help you raise your concern politely about your rent, deposit, agreement, maintenance or move-out.",
    "offtopic_jailbreak": "Sorry, I can only help with questions about your rent, deposit, rental agreement, maintenance and move-out. How can I help with those?",
}


def build_pairs(dom, tok, model, rows, seed, k=4):
    seed_all(seed)
    rng = random.Random(seed)
    users = [dom.render_user(r) for r in rows for _ in range(k)]
    samples = generate(model, tok, users, None, max_new_tokens=128, temperature=0.9, batch_size=64)
    pairs, stats = [], Counter()
    for i, r in enumerate(rows):
        s = [(g["text"].strip(), dom.verify(r, g["text"])["violation"]) for g in samples[k * i:k * i + k]]
        good, bad = [t for t, v in s if not v], [t for t, v in s if v]
        off = False
        if not good and r["cls"] != "benign":
            good, off = [REDIRECT[r["cls"]]], True
        if not good or not bad:
            continue
        for c, rj in list(zip(good * 2, bad * 2))[:2]:
            pairs.append({"prompt": tok.decode(prompt_ids(tok, dom.render_user(r))), "chosen": c + "<|im_end|>", "rejected": rj + "<|im_end|>"})
            stats[(r["cls"], off)] += 1
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase2", choices=list(env.PHASES))
    ap.add_argument("--run", default="dpo")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4, help="student samples per train prompt")
    ap.add_argument("--match_steps", type=int, default=0, help="if > 0, samples per prompt = ceil(steps * 16 / n_train) to match OPCD rollouts")
    a = ap.parse_args()
    t0 = time.time()
    P = env.paths(a.phase)
    dom, tok = get_domain(a.phase), load_tok(STUDENT)
    rows = load(a.phase, "train")
    k = math.ceil(a.match_steps * 16 / len(rows)) if a.match_steps else a.samples
    gen = load_model(STUDENT)
    pairs, stats = build_pairs(dom, tok, gen, rows, a.seed, k)
    del gen; torch.cuda.empty_cache()
    per_cls = Counter()
    for (c, off), n in stats.items():
        per_cls[c] += n
    n_off = sum(n for (c, off), n in stats.items() if off)
    print("pairs per class:", dict(per_cls), "| total", len(pairs), "| off_policy_chosen fraction", round(n_off / max(1, len(pairs)), 3), flush=True)
    os.makedirs(f"{P['results']}/training", exist_ok=True)
    meta = {"pairs_per_class": dict(per_cls), "total": len(pairs), "off_policy_chosen_frac": n_off / max(1, len(pairs)), "samples_per_prompt": k,
            "student_samples": k * len(rows), "method": "full fine-tune, fp32 master + bf16 autocast", "beta": 0.1, "epochs": a.epochs, "lr": a.lr, "seed": a.seed}
    model = load_model(STUDENT, train=True).float()
    cfg = DPOConfig(output_dir=f"{P['ckpt']}/{a.run}_tmp", beta=0.1, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=4,
                    gradient_accumulation_steps=2, bf16=True, max_length=1024, report_to="none", save_strategy="no", logging_steps=10, seed=a.seed)
    tr = DPOTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(pairs), processing_class=tok)
    tr.train()
    out = f"{P['ckpt']}/{a.run}" + ("/final" if a.phase != "phase1" else "")
    model.to(torch.bfloat16).save_pretrained(out); tok.save_pretrained(out)
    meta["minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(meta, open(f"{P['results']}/training/{a.run}_pairs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
