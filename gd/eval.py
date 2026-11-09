import gd.env  # noqa
import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch

from gd.data import get_domain, load
from gd.generate import STUDENT, TEACHER, generate, load_model, load_tok, prompt_ids, seed_all

CONDS = {  # cond -> (model, use_safety_prompt)
    "student_none": (STUDENT, False), "student_prompt": (STUDENT, True), "teacher_prompt": (TEACHER, True),
    "opcd": ("ckpt/opcd", False), "opcd_gr": ("ckpt/opcd_gr", False), "dpo": ("ckpt/dpo", False)}


def rate(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def evaluate(model, tok, domain, rows, system, seed=0, n_latency=50, bs=32, tag=""):
    seed_all(seed)
    users = [domain.render_user(r) for r in rows]
    rep = generate(model, tok, users, system, max_new_tokens=160, batch_size=bs)
    adv_idx = [i for i, r in enumerate(rows) if r["cls"] != "benign"]
    rng = random.Random(seed)
    pre = {i: rng.choice(domain.prefix_bank["test"][rows[i]["cls"]]) for i in adv_idx}
    rec = generate(model, tok, [users[i] for i in adv_idx], system, [pre[i] for i in adv_idx], max_new_tokens=160, batch_size=bs)

    viol, per_cls, refus, ontop, recov = [], defaultdict(list), [], [], defaultdict(list)
    dump = []
    for i, r in enumerate(rows):
        v = domain.verify(r, rep[i]["text"])
        dump.append({"id": r["id"], "cls": r["cls"], "reply": rep[i]["text"], **v})
        if r["cls"] == "benign":
            refus.append("false_refusal" in v["reasons"]); ontop.append(bool(v["on_topic"]))
        else:
            viol.append(v["violation"]); per_cls[r["cls"]].append(v["violation"])
    for j, i in enumerate(adv_idx):
        v = domain.verify(rows[i], rec[j]["text"], pre[i])
        dump.append({"id": rows[i]["id"], "cls": rows[i]["cls"], "prefix": pre[i], "reply": rec[j]["text"], **v})
        recov[rows[i]["cls"]].append(bool(v["recovered"]))
    # latency: batch 1, greedy, after warm-up
    lat_rows = users[:n_latency + 3]
    for u in lat_rows[:3]:
        generate(model, tok, [u], system, batch_size=1)
    torch.cuda.synchronize(); t = time.time()
    for u in lat_rows[3:]:
        generate(model, tok, [u], system, batch_size=1)
    torch.cuda.synchronize()
    out = {"n_test": len(rows), "n_adv": len(adv_idx), "n_benign": len(refus),
           "adv_violation_rate": rate(viol), "adv_violation_by_class": {c: rate(x) for c, x in per_cls.items()},
           "false_refusal_rate": rate(refus), "on_topic_rate": rate(ontop),
           "recovery_rate": rate([x for v in recov.values() for x in v]), "recovery_by_class": {c: rate(x) for c, x in recov.items()},
           "mean_input_tokens": round(sum(len(prompt_ids(tok, u, system)) for u in users) / len(users), 1),
           "mean_latency_s": round((time.time() - t) / n_latency, 3), "n_latency": n_latency}
    return out, dump


def summarize(res_dir="results"):
    cols = ["adv_violation_rate", "false_refusal_rate", "on_topic_rate", "recovery_rate", "mean_input_tokens", "mean_latency_s"]
    lines = ["| condition | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for c in CONDS:
        p = f"{res_dir}/eval_{c}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            lines.append(f"| {c} | " + " | ".join(str(d[k]) for k in cols) + " |")
    lines += ["", "Per class (violation / recovery):", "", "| condition | class | violation | recovery |", "|---|---|---|---|"]
    for c in CONDS:
        p = f"{res_dir}/eval_{c}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            for k in d["adv_violation_by_class"]:
                lines.append(f"| {c} | {k} | {d['adv_violation_by_class'][k]} | {d['recovery_by_class'].get(k)} |")
    open(f"{res_dir}/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def rescore(cond, domain=None):
    """Re-verify saved replies (after a verifier change) without regenerating; keeps latency / token fields."""
    domain = domain or get_domain()
    rows = {r["id"]: r for r in load("test")}
    p = f"results/eval_{cond}.json"
    res = json.load(open(p))
    viol, per_cls, refus, ontop, recov = [], defaultdict(list), [], [], defaultdict(list)
    dump = [json.loads(l) for l in open(f"results/raw/replies_{cond}.jsonl")]
    for d in dump:
        r = rows[d["id"]]
        v = domain.verify(r, d["reply"], d.get("prefix"))
        d.update(v)
        if "prefix" in d:
            recov[r["cls"]].append(bool(v["recovered"]))
        elif r["cls"] == "benign":
            refus.append("false_refusal" in v["reasons"]); ontop.append(bool(v["on_topic"]))
        else:
            viol.append(v["violation"]); per_cls[r["cls"]].append(v["violation"])
    res.update(adv_violation_rate=rate(viol), adv_violation_by_class={c: rate(x) for c, x in per_cls.items()}, false_refusal_rate=rate(refus),
               on_topic_rate=rate(ontop), recovery_rate=rate([x for v in recov.values() for x in v]), recovery_by_class={c: rate(x) for c, x in recov.items()})
    json.dump(res, open(p, "w"), indent=1)
    with open(f"results/raw/replies_{cond}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in dump)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=list(CONDS) + ["summary", "rescore"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_latency", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    if a.cond == "rescore":
        [rescore(c) for c in CONDS if os.path.exists(f"results/eval_{c}.json")]; a.cond = "summary"
    if a.cond == "summary":
        summarize(); raise SystemExit
    name, use_sys = CONDS[a.cond]
    dom = get_domain()
    tok = load_tok(STUDENT)
    model = load_model(name)
    res, dump = evaluate(model, tok, dom, load(a.split), dom.safety_prompt if use_sys else None, n_latency=a.n_latency, bs=a.bs)
    res.update(cond=a.cond, model=name, system_prompt=use_sys)
    os.makedirs("results/raw", exist_ok=True)
    json.dump(res, open(f"results/eval_{a.cond}.json", "w"), indent=1)
    with open(f"results/raw/replies_{a.cond}.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in dump)
    print(json.dumps(res, indent=1))
