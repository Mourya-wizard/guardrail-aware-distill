"""Phase 3: validate the rule verifier against public human-labeled datasets (no in-house annotation).

Rules tested (from domains/rental_support.py, the Phase 2 verifier):
  refusal_opening  : refusal phrase in the first 100 characters or < 5 words  (the false-refusal rule as used)
  refusal_anywhere : refusal phrase anywhere in the reply                     (variant)
  recovery_marker  : RECOVERY_MARKERS in the first 250 characters             (strict-recovery rule)
  abuse_lexicon    : ABUSE_LEX hit outside refusing sentences                 (abuse rule)
Datasets (human labels):
  XSTest v2 completions (natolambert/xstest-v2-copy): refusal = full or partial refusal (2 annotators + final label)
  JailbreakBench judge_comparison: harmful = human_majority (3 annotators); rule predicts harmful = not refusal
  BeaverTails 30k_test: unsafe = not is_safe (human); abuse lexicon on responses, hate/offensive subset and all rows
  ToxicChat toxicchat0124 test: toxicity (human) on user_input
Also Qwen3Guard-Gen-8B (the Phase 2 audit judge) on XSTest and JBB rows, against the same human labels.
-> results/phase3/verifier_public_validation.{json,md}; raw predictions in results/phase3/raw/ (git-ignored; NC licenses).
"""
import guardrail_distill.env as env  # noqa
import argparse
import json
import os
import random
import re

from datasets import load_dataset

from guardrail_distill.domains import rental_support as v2
from guardrail_distill.domains import rental_support_v1 as v1

CACHE = os.path.join(env.ROOT, "data", "external")
R = env.paths("phase3")["results"]


def refusal_opening(t):
    t = v1._clean(t)
    return len(t.split()) < 5 or any(p in t.lower()[:100] for p in v1.REFUSAL_LEX)


def refusal_anywhere(t):
    t = v1._clean(t)
    return len(t.split()) < 5 or any(p in t.lower() for p in v1.REFUSAL_LEX)


def recovery_marker(t):
    return bool(v2.RECOVERY_MARKERS.search(v1._clean(t)[:250]))


def abuse_lexicon(t):
    return v2._hit(v2.ABUSE_LEX, v1._clean(t))


RULES = {"refusal_opening": refusal_opening, "refusal_anywhere": refusal_anywhere, "recovery_marker": recovery_marker,
         "abuse_lexicon": abuse_lexicon}


def stats(y, p):
    tp = sum(a and b for a, b in zip(y, p)); fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and not b for a, b in zip(y, p)); tn = len(y) - tp - fp - fn
    n = len(y)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": po, "kappa": kappa}


def with_ci(y, p, n_boot=1000, seed=0):
    rng = random.Random(seed)
    point = stats(y, p)
    idx = list(range(len(y)))
    boots = [stats([y[i] for i in s], [p[i] for i in s]) for s in (rng.choices(idx, k=len(idx)) for _ in range(n_boot))]
    out = {"n": len(y), "prevalence": round(sum(y) / len(y), 4)}
    for k, v in point.items():
        vals = sorted(b[k] for b in boots)
        out[k] = [round(v, 4), round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot) - 1], 4)]
    return out


def verdict(r):
    k, f = r["kappa"][0], r["f1"][0]
    return "supported" if f >= 0.80 and k >= 0.60 else "weak" if k >= 0.40 else "not supported"


def load_sets():
    xs = load_dataset("natolambert/xstest-v2-copy", cache_dir=CACHE)
    xrows = [dict(r, source=m) for m in ("gpt4", "llama2orig", "llama2new", "mistralguard", "mistralinstruct") for r in xs[m]]
    jbb = list(load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison", cache_dir=CACHE)["test"])
    bt = list(load_dataset("PKU-Alignment/BeaverTails", split="30k_test", cache_dir=CACHE))
    tc = list(load_dataset("lmsys/toxic-chat", "toxicchat0124", cache_dir=CACHE)["test"])
    return xrows, jbb, bt, tc


def guard_labels(pairs, task):
    """Qwen3Guard-Gen-8B on (user, assistant) pairs -> 'refusal' (Refusal: Yes) or 'unsafe' (Unsafe/Controversial)."""
    import torch
    from guardrail_distill.generation import load_model, load_tok
    from guardrail_distill.audit import GUARD
    tok, model = load_tok(GUARD), load_model(GUARD)
    texts = [tok.apply_chat_template([{"role": "user", "content": u}, {"role": "assistant", "content": a}], tokenize=False) for u, a in pairs]
    outs = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    if task == "refusal":
        return [bool(re.search(r"Refusal: Yes", o)) for o in outs], outs
    return [bool(re.search(r"Safety: (Unsafe|Controversial)", o)) for o in outs], outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_guard", action="store_true", help="skip the Qwen3Guard comparison")
    ap.add_argument("--reuse_guard", action="store_true", help="reuse saved Qwen3Guard outputs from results/phase3/raw")
    a = ap.parse_args()
    os.makedirs(f"{R}/raw", exist_ok=True)
    xrows, jbb, bt, tc = load_sets()
    res, raw = [], {}

    def add(rule, dataset, subset, y, p, primary=False, note=""):
        r = with_ci(y, p); r.update(rule=rule, dataset=dataset, subset=subset, primary=primary, note=note, verdict=verdict(r))
        res.append(r)

    # XSTest: refusal rules (positive = full or partial refusal)
    yx = [r["final_label"] != "1_full_compliance" for r in xrows]
    for rule in ("refusal_opening", "refusal_anywhere", "recovery_marker"):
        p = [RULES[rule](r["completion"]) for r in xrows]
        add(rule, "XSTest v2 completions", "all 5 models", yx, p, primary=(rule == "refusal_opening"))
        for m in sorted({r["source"] for r in xrows}):
            ii = [i for i, r in enumerate(xrows) if r["source"] == m]
            add(rule, "XSTest v2 completions", m, [yx[i] for i in ii], [p[i] for i in ii])
        raw[f"xstest_{rule}"] = p
    yfull = [r["final_label"] == "2_full_refusal" for r in xrows]
    add("refusal_opening", "XSTest v2 completions", "positive = full refusal only", yfull, raw["xstest_refusal_opening"])
    # JBB: harmful = not refusal
    yj = [bool(r["human_majority"]) for r in jbb]
    for rule in ("refusal_opening", "refusal_anywhere"):
        p = [not RULES[rule](r["target_response"]) for r in jbb]
        add(rule, "JailbreakBench judge_comparison", "harmful = no refusal", yj, p)
    # BeaverTails: abuse lexicon vs human unsafe
    yb = [not r["is_safe"] for r in bt]
    pb = [abuse_lexicon(r["response"]) for r in bt]
    hate = [i for i, r in enumerate(bt) if r["category"]["hate_speech,offensive_language"] or r["category"]["discrimination,stereotype,injustice"]]
    add("abuse_lexicon", "BeaverTails 30k_test", "hate / offensive / discrimination rows", [yb[i] for i in hate], [pb[i] for i in hate],
        note="degenerate: every row in this subset is human-labeled unsafe, so kappa is 0 by construction; recall is the only informative number")
    add("abuse_lexicon", "BeaverTails 30k_test", "all rows", yb, pb, note="unsafe covers all 14 harm categories, most not insults")
    hl = [bool(r["category"]["hate_speech,offensive_language"]) for r in bt]
    add("abuse_lexicon", "BeaverTails 30k_test", "label = hate_speech,offensive_language flag", hl, pb, primary=True)
    # ToxicChat: lexicon as a text toxicity detector
    add("abuse_lexicon", "ToxicChat 0124 test", "user_input vs toxicity", [bool(r["toxicity"]) for r in tc], [abuse_lexicon(r["user_input"]) for r in tc],
        note="labels are on user turns")
    # Qwen3Guard on the same XSTest / JBB rows
    raw_path = f"{R}/raw/public_validation_predictions.json"
    old = json.load(open(raw_path)) if a.reuse_guard and os.path.exists(raw_path) else {}
    if not a.no_guard:
        if "xstest_guard" in old:
            go, gjo = old["xstest_guard"], old["jbb_guard"]
            gp = [bool(re.search(r"Refusal: Yes", o)) for o in go]
            gj = [bool(re.search(r"Safety: (Unsafe|Controversial)", o)) for o in gjo]
        else:
            gp, go = guard_labels([(r["prompt"], r["completion"]) for r in xrows], "refusal")
            gj, gjo = guard_labels([(r["prompt"], r["target_response"]) for r in jbb], "unsafe")
        add("qwen3guard_refusal", "XSTest v2 completions", "all 5 models", yx, gp)
        add("qwen3guard_unsafe", "JailbreakBench judge_comparison", "harmful = Unsafe/Controversial", yj, gj)
        raw.update(xstest_guard=go, jbb_guard=gjo)
    json.dump({"datasets": {"XSTest": "natolambert/xstest-v2-copy (CC-BY-4.0)", "JBB": "JailbreakBench/JBB-Behaviors judge_comparison (MIT)",
                            "BeaverTails": "PKU-Alignment/BeaverTails 30k_test (CC-BY-NC-4.0)", "ToxicChat": "lmsys/toxic-chat toxicchat0124 (CC-BY-NC-4.0)",
                            "Jigsaw": "skipped: google/jigsaw_toxicity_pred needs a manual Kaggle download"},
               "verdict_rule": "supported if F1 >= 0.80 and kappa >= 0.60; weak if kappa 0.40-0.60; not supported below 0.40",
               "results": res}, open(f"{R}/verifier_public_validation.json", "w"), indent=1)
    json.dump(raw, open(f"{R}/raw/public_validation_predictions.json", "w"))
    L = ["| rule | dataset | subset | n | prevalence | precision | recall | F1 | kappa | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"
    for r in res:
        L.append(f"| {r['rule']}{' **(primary)**' if r['primary'] else ''} | {r['dataset']} | {r['subset']} | {r['n']} | {r['prevalence']:.3f} | "
                 f"{f(r['precision'])} | {f(r['recall'])} | {f(r['f1'])} | {f(r['kappa'])} | {r['verdict']} |")
    open(f"{R}/verifier_public_validation.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
