"""Phase 3 forgetting check: IFEval and MMLU through EleutherAI lm-evaluation-harness, identical settings for every model.

python -m guardrail_distill.forgetting --model base|<phase2 run> --task ifeval|mmlu [--limit N]
-> results/phase3/forgetting/<task>_<model>/ (harness JSON + per-item samples for paired tests)
"""
import guardrail_distill.env as env  # noqa
import argparse
import os
import sys

from guardrail_distill.generation import STUDENT

SETTINGS = {  # lm-eval settings, identical for every model (MMLU uses mmlu() below)
    "ifeval": ["--tasks", "ifeval", "--apply_chat_template", "--batch_size", "32", "--gen_kwargs", "max_gen_toks=512"],
}


def model_path(name):
    return STUDENT if name in ("base", "base_prompt") else os.path.join(env.paths("phase2")["ckpt"], name, "final")


def mmlu(model_name, out, limit=0, max_tokens=24000):
    """5-shot MMLU (cais/mmlu test, dev shots) in lm-eval's prompt format, scored by the logits of the answer-letter tokens
    at the answer position. Same metric as lm-eval's single-token continuation log-likelihood, but one forward pass per
    question with logits_to_keep=1 (lm-eval peaked at 58 GB for one model here because it keeps full-vocab logits)."""
    import json
    import torch
    from datasets import load_dataset
    from guardrail_distill.generation import load_model, load_tok
    cache = os.path.join(env.ROOT, "data", "external")
    test = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache)
    dev = load_dataset("cais/mmlu", "all", split="dev", cache_dir=cache)
    shots = {}
    for r in dev:
        shots.setdefault(r["subject"], []).append(r)
    L = "ABCD"

    def fmt(r, ans=True):
        q = r["question"].strip() + "\n" + "".join(f"{L[i]}. {c}\n" for i, c in enumerate(r["choices"])) + "Answer:"
        return q + (f" {L[r['answer']]}\n\n" if ans else "")

    tok, model = load_tok(model_name), load_model(model_name)
    tok.padding_side = "left"
    letter_ids = [tok(f" {c}", add_special_tokens=False).input_ids[-1] for c in L]
    rows = list(test)[:limit] if limit else list(test)
    prompts = [f"The following are multiple choice questions (with answers) about {r['subject'].replace('_', ' ')}.\n\n"
               + "".join(fmt(s) for s in shots[r["subject"]][:5]) + fmt(r, ans=False) for r in rows]
    lens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    batches, cur = [], []  # token-budget batches: many short prompts or a few long ones per forward pass
    for i in order:
        if cur and (len(cur) + 1) * lens[i] > max_tokens:
            batches.append(cur); cur = []
        cur.append(i)
    batches.append(cur)
    correct = [0] * len(rows)
    for n, idx in enumerate(batches):
        if n % 50 == 0:
            print(f"mmlu batch {n}/{len(batches)}", flush=True)
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, letter_ids]
        for k, i in enumerate(idx):
            correct[i] = int(int(logits[k].argmax()) == rows[i]["answer"])
    os.makedirs(out, exist_ok=True)
    per_subject = {}
    for r, c in zip(rows, correct):
        per_subject.setdefault(r["subject"], []).append(c)
    res = {"task": "mmlu_5shot_letter_logits", "model": model_name, "n": len(rows), "acc": sum(correct) / len(rows),
           "acc_macro_subject": sum(sum(v) / len(v) for v in per_subject.values()) / len(per_subject), "correct": correct}
    json.dump(res, open(os.path.join(out, "results.json"), "w"))
    print({k: v for k, v in res.items() if k != "correct"})


MODELS = ["base", "base_prompt"] + [f"{m}_s{s}" for m in ("opcd", "opcd_gr", "dpo") for s in range(3)]
THRESH = {"mmlu": 0.03, "ifeval": 0.05}  # pre-registered: significant drop larger than this = measurable forgetting


def per_item(task, model):
    """-> {item key: 0/1} (MMLU: correct; IFEval: prompt-level strict accuracy)."""
    import glob
    import json
    d = os.path.join(env.paths("phase3")["results"], "forgetting", f"{task}_{model}")
    if task == "mmlu":
        return dict(enumerate(json.load(open(os.path.join(d, "results.json")))["correct"]))
    f = sorted(glob.glob(os.path.join(d, "*", "samples_ifeval_*.jsonl")))[-1]
    return {r["doc_id"]: int(r["prompt_level_strict_acc"]) for r in map(json.loads, open(f, encoding="utf-8"))}


def refusal_rate(model):
    """IFEval: share of replies that refuse (recovery-marker detector, F1 0.858 vs XSTest human labels), and
    prompt-level strict accuracy on the replies that do NOT refuse (separates scope refusal from lost ability)."""
    import glob
    import json
    from guardrail_distill.validate_public import recovery_marker
    d = os.path.join(env.paths("phase3")["results"], "forgetting", f"ifeval_{model}")
    f = sorted(glob.glob(os.path.join(d, "*", "samples_ifeval_*.jsonl")))[-1]
    rows = [(recovery_marker(x["resps"][0][0]), int(x["prompt_level_strict_acc"])) for x in map(json.loads, open(f, encoding="utf-8"))]
    answered = [acc for ref, acc in rows if not ref]
    return round(sum(r for r, _ in rows) / len(rows), 4), round(sum(answered) / len(answered), 4) if answered else None, len(answered)


def aggregate():
    import json
    import statistics
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from guardrail_distill.significance import boot_ci, mcnemar
    R = env.paths("phase3")["results"]
    out = {}
    for task in ("mmlu", "ifeval"):
        base = per_item(task, "base")
        rows = {}
        for m in MODELS:
            try:
                it = per_item(task, m)
            except (FileNotFoundError, IndexError):
                continue
            ci = boot_ci(list(it.values()))
            rows[m] = {"acc": ci, "delta_vs_base": round(ci[0] - boot_ci(list(base.values()))[0], 4)}
            if task == "ifeval":
                rows[m]["refusal_rate"], rows[m]["acc_when_answered"], rows[m]["n_answered"] = refusal_rate(m)
            if m != "base":
                rows[m]["mcnemar_vs_base"] = mcnemar(it, base)
        seeds = {}
        for meth in ("opcd", "opcd_gr", "dpo"):
            ds = [rows[f"{meth}_s{s}"]["delta_vs_base"] for s in range(3) if f"{meth}_s{s}" in rows]
            if ds:
                sig = [rows[f"{meth}_s{s}"]["mcnemar_vs_base"]["p"] < 0.05 and rows[f"{meth}_s{s}"]["delta_vs_base"] < -THRESH[task]
                       for s in range(3) if f"{meth}_s{s}" in rows]
                seeds[meth] = {"delta_mean": round(statistics.mean(ds), 4), "delta_std": round(statistics.stdev(ds), 4) if len(ds) > 1 else 0.0,
                               "n_seeds": len(ds), "seeds_with_measurable_forgetting": sum(sig)}
        out[task] = {"per_model": rows, "per_method": seeds, "threshold": THRESH[task]}
    json.dump(out, open(os.path.join(R, "forgetting.json"), "w"), indent=1)
    names = {"mmlu": "MMLU (5-shot, accuracy)", "ifeval": "IFEval (prompt-level strict accuracy)"}
    L = []
    for task, o in out.items():
        rc = task == "ifeval"
        L += [f"### {names[task]}", "", "| model | accuracy [95% CI] | delta vs base | McNemar p (vs base) |" + (" refusal rate | accuracy when not refusing (n) |" if rc else ""),
              "|---|---|---|---|" + ("---|---|" if rc else "")]
        for m, r in o["per_model"].items():
            a = r["acc"]
            L.append(f"| {m} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {r['delta_vs_base']:+.3f} | {r.get('mcnemar_vs_base', {}).get('p', '')} |"
                     + (f" {r['refusal_rate']:.3f} | {r['acc_when_answered']:.3f} ({r['n_answered']}) |" if rc else ""))
        L += ["", "| method | mean delta ± std (3 seeds) | seeds with measurable forgetting |", "|---|---|---|"]
        L += [f"| {m} | {s['delta_mean']:+.3f} ± {s['delta_std']:.3f} | {s['seeds_with_measurable_forgetting']} / {s['n_seeds']} |" for m, s in o["per_method"].items()]
        L.append("")
    open(os.path.join(R, "forgetting.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))
    os.makedirs(os.path.join(R, "figures"), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for a_, task in zip(ax, ("mmlu", "ifeval")):
        rows = out[task]["per_model"]
        ys = [r["acc"][0] for r in rows.values()]
        err = [[r["acc"][0] - r["acc"][1] for r in rows.values()], [r["acc"][2] - r["acc"][0] for r in rows.values()]]
        a_.bar(range(len(rows)), ys, yerr=err, capsize=2, color=[{"base": "grey", "base_prompt": "silver"}.get(m, "tab:blue" if m.startswith("opcd_s") else "tab:orange" if m.startswith("opcd_gr") else "tab:green") for m in rows])
        a_.set_xticks(range(len(rows))); a_.set_xticklabels(rows.keys(), rotation=30, ha="right", fontsize=8); a_.set_title(names[task])
        a_.axhline(rows["base"]["acc"][0], color="grey", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(R, "figures", "forgetting.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--task", choices=["ifeval", "mmlu"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true", help="build results/phase3/forgetting.{json,md} and the figure")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate()
    out = os.path.join(env.paths("phase3")["results"], "forgetting", f"{a.task}_{a.model}")
    if a.task == "mmlu":
        return mmlu(model_path(a.model), out, a.limit)
    args = ["lm_eval", "--model", "hf", "--model_args", f"pretrained={model_path(a.model)},dtype=bfloat16,enable_thinking=False",
            *SETTINGS[a.task], "--output_path", out, "--log_samples", "--seed", "0"]
    if a.model == "base_prompt":  # reference: the untrained student WITH the safety system prompt
        from guardrail_distill.data import get_domain
        args += ["--system_instruction", get_domain("phase2").safety_prompt]
    if a.limit:
        args += ["--limit", str(a.limit)]
    from lm_eval.__main__ import cli_evaluate
    sys.argv = args
    cli_evaluate()


if __name__ == "__main__":
    main()
