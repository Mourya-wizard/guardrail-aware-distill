"""On-policy reverse-KL context distillation (+ optional guided rollouts).
loss = mean over completion tokens of KL(p_student(.|x,y<t) || p_teacher(.|sys,x,y<t)), full vocab, fp32."""
import gd.env  # noqa
import argparse
import csv
import json
import os
import random
import time

import torch
import torch.nn.functional as F

from gd.data import get_domain, load
from gd.generate import STUDENT, TEACHER, generate, load_model, load_tok, prompt_ids, seed_all


def dev_eval(gen_model, tok, dom, rows, n=100):
    rows = rows[:n]
    rep = generate(gen_model, tok, [dom.render_user(r) for r in rows], None, max_new_tokens=160, batch_size=50)
    adv = [dom.verify(r, p["text"])["violation"] for r, p in zip(rows, rep) if r["cls"] != "benign"]
    ben = ["false_refusal" in dom.verify(r, p["text"])["reasons"] for r, p in zip(rows, rep) if r["cls"] == "benign"]
    return sum(adv) / max(1, len(adv)), sum(ben) / max(1, len(ben))


def kl_loss(student, teacher, s_in, t_in, C, V):
    """Per-token reverse KL summed over the C completion positions (caller normalises)."""
    assert torch.equal(s_in[0, -C:], t_in[0, -C:]), "completion segments differ"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_log = student(input_ids=s_in, logits_to_keep=C + 1).logits[0, :-1, :V]
    with torch.no_grad():
        t_log = teacher(input_ids=t_in, logits_to_keep=C + 1).logits[0, :-1, :V].float()
    s_lp, t_lp = F.log_softmax(s_log.float(), -1), F.log_softmax(t_log, -1)
    return (s_lp.exp() * (s_lp - t_lp)).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--guided_frac", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget_min", type=float, default=1e9, help="stop early (and record) when exceeded")
    a = ap.parse_args()
    seed_all(a.seed)
    rng = random.Random(a.seed)
    dom, tok = get_domain(), load_tok(STUDENT)
    V = len(tok)
    train, dev = load("train"), load("dev")
    ben = [r for r in train if r["cls"] == "benign"]
    adv = [r for r in train if r["cls"] != "benign"]
    student = load_model(STUDENT, train=True).float()  # fp32 master weights, bf16 autocast
    gen = load_model(STUDENT)  # bf16 copy used for rollouts / dev eval
    teacher = load_model(TEACHER)
    for p in teacher.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
    os.makedirs("results", exist_ok=True); os.makedirs("ckpt", exist_ok=True)
    log_path = f"results/opcd_{a.run}_log.csv"
    logf = open(log_path, "w", newline=""); w = csv.writer(logf)
    w.writerow(["step", "loss", "mean_completion_len", "dev_adv_violation", "dev_false_refusal", "elapsed_min"])
    best, t0, losses, lens = (9, None), time.time(), [], []
    n_adv = a.bs // 2
    n_guided = round(a.guided_frac * n_adv)
    steps_done = 0
    for step in range(1, a.steps + 1):
        gen.load_state_dict(student.state_dict())
        rows = rng.sample(ben, a.bs - n_adv) + rng.sample(adv, n_adv)
        prefixes = [None] * (a.bs - n_adv) + [rng.choice(dom.prefix_bank["train"][r["cls"]]) if j < n_guided else None
                                             for j, r in enumerate(rows[a.bs - n_adv:])]
        users = [dom.render_user(r) for r in rows]
        roll = generate(gen, tok, users, None, prefixes, max_new_tokens=128, temperature=1.0, top_p=1.0, batch_size=a.bs)
        comps = [g["prefix_ids"] + g["ids"] + ([tok.convert_tokens_to_ids("<|im_end|>")] if g["ended"] else []) for g in roll]
        total = sum(len(c) for c in comps)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for u, c in zip(users, comps):
            if not c:
                continue
            s_in = torch.tensor([prompt_ids(tok, u, None) + c]).cuda()
            t_in = torch.tensor([prompt_ids(tok, u, dom.safety_prompt) + c]).cuda()
            loss = kl_loss(student, teacher, s_in, t_in, len(c), V) / total
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(step_loss); lens.append(total / a.bs); steps_done = step
        el = (time.time() - t0) / 60
        if step == 10:
            print(f"[estimate] {el / 10:.2f} min/step -> {a.steps} steps ~ {el / 10 * a.steps:.0f} min (+ evals)", flush=True)
        if step % 10 == 0:
            print(f"step {step} loss {sum(losses[-10:]) / 10:.4f} len {sum(lens[-10:]) / 10:.1f} elapsed {el:.1f}m", flush=True)
        stop = el > a.budget_min
        if step % a.eval_every == 0 or step == a.steps or stop:
            gen.load_state_dict(student.state_dict())
            dv, fr = dev_eval(gen, tok, dom, dev)
            w.writerow([step, round(sum(losses[-a.eval_every:]) / len(losses[-a.eval_every:]), 5), round(sum(lens[-a.eval_every:]) / len(lens[-a.eval_every:]), 1), round(dv, 4), round(fr, 4), round(el, 1)])
            logf.flush()
            print(f"== step {step} dev_adv_violation {dv:.3f} dev_false_refusal {fr:.3f}", flush=True)
            if dv + fr <= best[0]:
                best = (dv + fr, step)
                gen.save_pretrained(f"ckpt/{a.run}"); tok.save_pretrained(f"ckpt/{a.run}")
        if stop:
            print(f"BUDGET STOP at step {step} (requested {a.steps})", flush=True)
            break
    json.dump({"run": a.run, "steps_requested": a.steps, "steps_done": steps_done, "best_step": best[1], "best_dev_score": best[0],
               "guided_frac": a.guided_frac, "lr": a.lr, "bs": a.bs, "seed": a.seed, "minutes": round((time.time() - t0) / 60, 1)},
              open(f"results/opcd_{a.run}_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
