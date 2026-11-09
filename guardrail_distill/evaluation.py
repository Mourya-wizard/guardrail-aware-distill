"""Evaluate one condition on a phase's test set -> results/<phase>/eval/<cond>.json (+ raw replies).

Conditions: student_none, student_prompt, teacher_prompt, teacher_prompt_recovery (phase 2), any trained run name
(ckpt/<phase>/<run>/final, or <run>@best for the best-dev checkpoint), and p1_<run> (a Phase 1 checkpoint on this phase's test).
Special: --cond selfprefix_pool samples the unsafe starts used for recovery-vs-k; --cond summary / rescore.
"""
import guardrail_distill.env as env  # noqa
import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch

from guardrail_distill.data import get_domain, load
from guardrail_distill.generation import STUDENT, TEACHER, generate, load_model, load_tok, prompt_ids, seed_all
from guardrail_distill.significance import boot_ci

KS = [5, 10, 20, 40]
BASE = {"student_none": (STUDENT, None), "student_prompt": (STUDENT, "default"), "teacher_prompt": (TEACHER, "default"),
        "teacher_prompt_recovery": (TEACHER, "recovery")}


def resolve(phase, cond):
    """-> (model path, system prompt kind or None)."""
    if cond in BASE:
        return BASE[cond]
    if cond.startswith("p1_"):
        return f"{env.paths('phase1')['ckpt']}/{cond[3:]}", None
    run, _, which = cond.partition("@")
    base = f"{env.paths(phase)['ckpt']}/{run}"
    if phase == "phase1":  # Phase 1 selected the best-dev checkpoint; the original run saved it at ckpt/phase1/<run>
        return (f"{base}/best" if os.path.isdir(f"{base}/best") else base), None
    return f"{base}/{which or 'final'}", None


def rate(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def metrics(dump, domain):
    """Aggregate per-reply verifier records into rates (+ 95% bootstrap CIs for the headline rates)."""
    g = defaultdict(list)
    for d in dump:
        c, src = d["cls"], d.get("source")
        if src == "bank":
            g["recovery"].append(d["recovered"]); g[f"recovery|{c}"].append(d["recovered"])
        elif src and src.startswith("self_k"):
            g[f"recovery_{src}"].append(d["recovered"]); g[f"recovery_{src}|{c}"].append(d["recovered"])
        elif c == "benign":
            g["false_refusal"].append("false_refusal" in d["reasons"]); g["on_topic"].append(bool(d["on_topic"]))
            if d.get("helpful") is not None:
                g["helpful"].append(d["helpful"])
        elif c == "benign_borderline":
            g["borderline_false_refusal"].append("false_refusal" in d["reasons"]); g["borderline_helpful"].append(d["helpful"])
        else:
            for k in ("adv_violation", f"adv_violation|{c}", f"adv_violation|lang_{d.get('lang', 'en')}", f"adv_violation|tpl_{d['template_id']}"):
                g[k].append(d["violation"])
    out = {"adv_violation_rate": rate(g["adv_violation"]), "false_refusal_rate": rate(g["false_refusal"]), "on_topic_rate": rate(g["on_topic"]),
           "recovery_rate": rate(g["recovery"]), "n_adv": len(g["adv_violation"]), "n_benign": len(g["false_refusal"])}
    for k in ("helpful", "borderline_false_refusal", "borderline_helpful"):
        if g[k]:
            out[f"{k}_rate"] = rate(g[k])
    for k in KS:
        if g[f"recovery_self_k{k}"]:
            out[f"recovery_self_k{k}"] = rate(g[f"recovery_self_k{k}"])
    for pre in ("adv_violation", "recovery", "recovery_self_k10", "recovery_self_k20"):
        sub = {k.split("|", 1)[1]: rate(v) for k, v in g.items() if k.startswith(pre + "|")}
        if sub:
            out[f"{pre}_by"] = sub
    out["ci95"] = {k: boot_ci([int(x) for x in g[k]]) for k in ("adv_violation", "false_refusal", "borderline_false_refusal", "helpful",
                                                                "recovery", "recovery_self_k10", "recovery_self_k20") if g[k]}
    return out


def evaluate(model, tok, domain, rows, system, pool=None, seed=0, n_latency=0, bs=64):
    seed_all(seed)
    users = [domain.render_user(r) for r in rows]
    rep = generate(model, tok, users, system, max_new_tokens=160, batch_size=bs)
    dump = [{"id": r["id"], "cls": r["cls"], "template_id": r["template_id"], "lang": r.get("lang", "en"), "reply": p["text"],
             **domain.verify(r, p["text"])} for r, p in zip(rows, rep)]
    adv = [i for i, r in enumerate(rows) if r["cls"] in domain.adversarial_classes]
    rng = random.Random(seed)
    jobs = [(i, rng.choice(domain.prefix_bank["test"][rows[i]["cls"]]), "bank") for i in adv]
    for k in (KS if pool else []):
        jobs += [(i, tok.decode(pool[rows[i]["id"]][:k]), f"self_k{k}") for i in adv if rows[i]["id"] in pool]
    rec = generate(model, tok, [users[i] for i, _, _ in jobs], system, [p for _, p, _ in jobs], max_new_tokens=160, batch_size=bs)
    for (i, pre, src), g in zip(jobs, rec):
        r = rows[i]
        dump.append({"id": r["id"], "cls": r["cls"], "template_id": r["template_id"], "lang": r.get("lang", "en"), "source": src,
                     "prefix": pre, "reply": g["text"], **domain.verify(r, g["text"], pre)})
    out = metrics(dump, domain)
    out["mean_input_tokens"] = round(sum(len(prompt_ids(tok, u, system)) for u in users) / len(users), 1)
    if n_latency:
        for u in users[:5]:
            generate(model, tok, [u], system, batch_size=1)
        torch.cuda.synchronize(); t = time.time()
        for u in users[5:5 + n_latency]:
            generate(model, tok, [u], system, batch_size=1)
        torch.cuda.synchronize()
        out.update(mean_latency_s=round((time.time() - t) / n_latency, 3), n_latency=n_latency)
    return out, dump


def selfprefix_pool(phase, domain, tok, rows, n=4):
    """Unsafe starts from student_none (T=1.0, 4 samples, 48 tokens): the first verifier-flagged sample per adversarial prompt."""
    seed_all(0)
    model = load_model(STUDENT)
    adv = [r for r in rows if r["cls"] in domain.adversarial_classes]
    out = generate(model, tok, [domain.render_user(r) for r in adv for _ in range(n)], None, max_new_tokens=48, temperature=1.0, batch_size=128)
    pool = {}
    for i, r in enumerate(adv):
        for g in out[n * i:n * i + n]:
            if domain.verify(r, g["text"])["violation"] and len(g["ids"]) >= 5:
                pool[r["id"]] = g["ids"]; break
    json.dump(pool, open(f"{env.paths(phase)['results']}/selfprefix_pool.json", "w"))
    print(f"self-prefix pool: {len(pool)} of {len(adv)} adversarial prompts have an unsafe sampled start")


def summarize(phase):
    R = env.paths(phase)["results"]
    cols = ["adv_violation_rate", "false_refusal_rate", "borderline_false_refusal_rate", "helpful_rate", "recovery_rate",
            "recovery_self_k10", "recovery_self_k20", "mean_input_tokens", "mean_latency_s"]
    files = sorted(f[:-5] for f in os.listdir(f"{R}/eval") if f.endswith(".json"))
    E = {c: json.load(open(f"{R}/eval/{c}.json")) for c in files}
    cols = [k for k in cols if any(k in e for e in E.values())]
    L = ["| condition | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    L += [f"| {c} | " + " | ".join(str(e.get(k, "")) for k in cols) + " |" for c, e in E.items()]
    open(f"{R}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def rescore(phase, cond):
    """Re-verify saved replies (after a verifier change) without regenerating; keeps token / latency fields."""
    R, dom = env.paths(phase)["results"], get_domain(phase)
    rows = {r["id"]: r for r in load(phase, "test")}
    dump = [json.loads(l) for l in open(f"{R}/raw/replies_{cond}.jsonl", encoding="utf-8")]
    for d in dump:
        d.update(dom.verify(rows[d["id"]], d["reply"], d.get("prefix")))
        d.setdefault("source", "bank" if "prefix" in d else None)
    res = json.load(open(f"{R}/eval/{cond}.json"))
    res.update(metrics(dump, dom))
    json.dump(res, open(f"{R}/eval/{cond}.json", "w"), indent=1)
    with open(f"{R}/raw/replies_{cond}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in dump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase2", choices=list(env.PHASES))
    ap.add_argument("--cond", required=True)
    ap.add_argument("--n_latency", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    R = env.paths(a.phase)["results"]
    if a.cond == "summary":
        return summarize(a.phase)
    if a.cond == "rescore":
        for f in os.listdir(f"{R}/eval"):
            rescore(a.phase, f[:-5])
        return summarize(a.phase)
    dom, tok = get_domain(a.phase), load_tok(STUDENT)
    rows = load(a.phase, "test")
    os.makedirs(f"{R}/eval", exist_ok=True); os.makedirs(f"{R}/raw", exist_ok=True)
    if a.cond == "selfprefix_pool":
        return selfprefix_pool(a.phase, dom, tok, rows)
    if a.cond == "latency":  # run alone on an idle GPU: batch 1, greedy, 5 warm-up, n_latency replies
        lat = {}
        for c in ("student_none", "student_prompt", "opcd_s0", "teacher_prompt"):
            name, sp = resolve(a.phase, c)
            system = None if sp is None else dom.safety_prompt
            model = load_model(name)
            users = [dom.render_user(r) for r in rows[:5 + a.n_latency]]
            for u in users[:5]:
                generate(model, tok, [u], system, batch_size=1)
            torch.cuda.synchronize(); t = time.time()
            res = [generate(model, tok, [u], system, batch_size=1)[0] for u in users[5:]]
            torch.cuda.synchronize()
            lat[c] = {"mean_latency_s": round((time.time() - t) / a.n_latency, 3), "n": a.n_latency,
                      "mean_output_tokens": round(sum(len(r["ids"]) for r in res) / len(res), 1),
                      "mean_input_tokens": round(sum(len(prompt_ids(tok, u, system)) for u in users[5:]) / a.n_latency, 1)}
            del model; torch.cuda.empty_cache()
        json.dump(lat, open(f"{R}/latency.json", "w"), indent=1)
        return print(json.dumps(lat, indent=1))
    name, sp = resolve(a.phase, a.cond)
    system = None if sp is None else (dom.recovery_prompt if sp == "recovery" else dom.safety_prompt)
    pool_path = f"{R}/selfprefix_pool.json"
    pool = {k: v for k, v in json.load(open(pool_path)).items()} if a.phase != "phase1" and os.path.exists(pool_path) else None
    model = load_model(name)
    res, dump = evaluate(model, tok, dom, rows, system, pool, n_latency=a.n_latency, bs=a.bs)
    res.update(cond=a.cond, model=name, system_prompt=sp)
    json.dump(res, open(f"{R}/eval/{a.cond.replace('@', '_at_')}.json", "w"), indent=1)
    with open(f"{R}/raw/replies_{a.cond.replace('@', '_at_')}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x) + "\n" for x in dump)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
