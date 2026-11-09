"""On-policy context distillation (Ye et al., 2026) with optional guided rollouts.
loss = mean over completion tokens of D(p_student(.|x,y<t), p_teacher(.|sys,x,y<t)), full vocabulary, fp32;
D = reverse KL (default) or generalized JSD with beta 0.5 (GKD, Agarwal et al., 2024)."""
import guardrail_distill.env as env  # noqa
import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from guardrail_distill.data import get_domain, load
from guardrail_distill.generation import STUDENT, TEACHER, generate, load_model, load_tok, prompt_ids, seed_all


def dev_eval(gen_model, tok, dom, rows):
    rep = generate(gen_model, tok, [dom.render_user(r) for r in rows], None, max_new_tokens=160, batch_size=100)
    v = [dom.verify(r, p["text"]) for r, p in zip(rows, rep)]
    adv = [x["violation"] for r, x in zip(rows, v) if r["cls"] != "benign"]
    ben = ["false_refusal" in x["reasons"] for r, x in zip(rows, v) if r["cls"] == "benign"]
    return sum(adv) / max(1, len(adv)), sum(ben) / max(1, len(ben))


def divergence(s_log, t_log, kind):
    """Sum over positions of the per-token divergence between student and teacher next-token distributions."""
    s_lp, t_lp = F.log_softmax(s_log.float(), -1), F.log_softmax(t_log.float(), -1)
    if kind == "rkl":
        return (s_lp.exp() * (s_lp - t_lp)).sum()
    m_lp = torch.logsumexp(torch.stack([s_lp, t_lp]), 0) - math.log(2)  # generalized JSD, beta = 0.5
    return 0.5 * (s_lp.exp() * (s_lp - m_lp)).sum() + 0.5 * (t_lp.exp() * (t_lp - m_lp)).sum()


def step_loss(student, teacher, s_in, t_in, C, V, n_mask, kind):
    """Divergence summed over the C completion positions, skipping the first n_mask (forced-prefix) positions."""
    assert torch.equal(s_in[0, -C:], t_in[0, -C:]), "completion segments differ"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_log = student(input_ids=s_in, logits_to_keep=C + 1).logits[0, :-1, :V]
    with torch.no_grad():
        t_log = teacher(input_ids=t_in, logits_to_keep=C + 1).logits[0, :-1, :V]
    return divergence(s_log[n_mask:], t_log[n_mask:], kind)


def self_prefixes(gen, tok, dom, adv_rows, rng, n=64):
    """Unsafe starts sampled from the current student: first k in [5, 20] tokens of replies the verifier flags."""
    rows = rng.sample(adv_rows, min(n, len(adv_rows)))
    out = generate(gen, tok, [dom.render_user(r) for r in rows], None, max_new_tokens=32, temperature=1.0, batch_size=n)
    pool = {}
    for r, g in zip(rows, out):
        if g["ids"] and dom.verify(r, g["text"])["violation"]:
            pool.setdefault(r["cls"], []).append(tok.decode(g["ids"][:rng.randint(5, 20)]))
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase2", choices=list(env.PHASES))
    ap.add_argument("--run", required=True)
    ap.add_argument("--guided_frac", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default="large", choices=["large", "self"], help="large = 9B; self = frozen copy of the student (Prompt Baking style)")
    ap.add_argument("--teacher_prompt", default="default", choices=["default", "recovery"])
    ap.add_argument("--mask_prefix", action="store_true", help="exclude forced-prefix positions from the loss")
    ap.add_argument("--prefix_source", default="bank", choices=["bank", "self"])
    ap.add_argument("--divergence", default="rkl", choices=["rkl", "jsd"])
    a = ap.parse_args()
    seed_all(a.seed)
    rng = random.Random(a.seed)
    P = env.paths(a.phase)
    dom, tok = get_domain(a.phase), load_tok(STUDENT)
    V = len(tok)
    train, dev = load(a.phase, "train"), load(a.phase, "dev")
    ben = [r for r in train if r["cls"] == "benign"]
    adv = [r for r in train if r["cls"] != "benign"]
    sys_prompt = getattr(dom, "recovery_prompt", dom.safety_prompt) if a.teacher_prompt == "recovery" else dom.safety_prompt
    student = load_model(STUDENT, train=True).float()  # fp32 master weights, bf16 autocast
    gen = load_model(STUDENT)  # bf16 copy for rollouts / dev eval
    teacher = load_model(TEACHER if a.teacher == "large" else STUDENT)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    out_dir = f"{P['ckpt']}/{a.run}"
    os.makedirs(f"{P['results']}/training", exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    logf = open(f"{P['results']}/training/{a.run}_log.csv", "w", newline=""); w = csv.writer(logf)
    w.writerow(["step", "loss", "mean_completion_len", "dev_adv_violation", "dev_false_refusal", "elapsed_min"])
    best, t0, losses, lens = (9, None), time.time(), [], []
    n_adv, eos = a.bs // 2, tok.convert_tokens_to_ids("<|im_end|>")
    n_guided = round(a.guided_frac * n_adv)
    n_rollouts = n_tokens = 0
    pool = {}
    for step in range(1, a.steps + 1):
        gen.load_state_dict(student.state_dict())
        if a.prefix_source == "self" and n_guided and (step == 1 or step % 50 == 0):
            pool = self_prefixes(gen, tok, dom, adv, rng)
        rows = rng.sample(ben, a.bs - n_adv) + rng.sample(adv, n_adv)
        prefixes = [None] * (a.bs - n_adv)
        for j, r in enumerate(rows[a.bs - n_adv:]):
            src = pool.get(r["cls"]) if a.prefix_source == "self" else None
            prefixes.append(None if j >= n_guided else rng.choice(src or dom.prefix_bank["train"][r["cls"]]))
        users = [dom.render_user(r) for r in rows]
        roll = generate(gen, tok, users, None, prefixes, max_new_tokens=128, temperature=1.0, top_p=1.0, batch_size=a.bs)
        comps = [(g["prefix_ids"] + g["ids"] + ([eos] if g["ended"] else []), len(g["prefix_ids"]) if a.mask_prefix else 0) for g in roll]
        total = sum(len(c) - m for c, m in comps)
        opt.zero_grad(set_to_none=True)
        sl = 0.0
        for u, (c, m) in zip(users, comps):
            if len(c) - m <= 0:
                continue
            s_in = torch.tensor([prompt_ids(tok, u, None) + c]).cuda()
            t_in = torch.tensor([prompt_ids(tok, u, sys_prompt) + c]).cuda()
            loss = step_loss(student, teacher, s_in, t_in, len(c), V, m, a.divergence) / total
            loss.backward()
            sl += loss.item()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        n_rollouts += a.bs; n_tokens += sum(len(c) for c, _ in comps)
        losses.append(sl); lens.append(sum(len(c) for c, _ in comps) / a.bs)
        el = (time.time() - t0) / 60
        if step == 10:
            print(f"[estimate] {el / 10:.2f} min/step -> {a.steps} steps ~ {el / 10 * a.steps:.0f} min (+ evals)", flush=True)
        if step % 10 == 0:
            print(f"step {step} loss {sum(losses[-10:]) / 10:.4f} len {sum(lens[-10:]) / 10:.1f} elapsed {el:.1f}m", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            gen.load_state_dict(student.state_dict())
            dv, fr = dev_eval(gen, tok, dom, dev)
            k = len(losses[-a.eval_every:])
            w.writerow([step, round(sum(losses[-k:]) / k, 5), round(sum(lens[-k:]) / k, 1), round(dv, 4), round(fr, 4), round(el, 1)])
            logf.flush()
            print(f"== step {step} dev_adv_violation {dv:.3f} dev_false_refusal {fr:.3f}", flush=True)
            if dv + fr <= best[0]:
                best = (dv + fr, step)
                gen.save_pretrained(f"{out_dir}/best"); tok.save_pretrained(f"{out_dir}/best")
            torch.cuda.empty_cache()
    gen.save_pretrained(f"{out_dir}/final"); tok.save_pretrained(f"{out_dir}/final")
    json.dump({**vars(a), "steps_done": a.steps, "best_step": best[1], "best_dev_score": best[0], "rollouts": n_rollouts,
               "completion_tokens": n_tokens, "minutes": round((time.time() - t0) / 60, 1)},
              open(f"{P['results']}/training/{a.run}_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
