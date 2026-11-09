"""95% bootstrap CIs and paired exact McNemar tests from saved replies (results/<phase>/raw/replies_<cond>.jsonl). No regeneration."""
import guardrail_distill.env as env  # noqa
import argparse
import json
import os
import random
import statistics
from math import comb

PAIRS = {
    "phase1": [("opcd", "student_prompt"), ("opcd_gr", "student_prompt"), ("dpo", "student_prompt"),
               ("opcd_gr", "opcd"), ("dpo", "opcd"), ("opcd", "student_none")],
    "phase2": [(f"{a}_s{s}", b if b.startswith(("student", "teacher")) else f"{b}_s{s}") for s in range(3)
               for a, b in [("opcd", "student_none"), ("opcd", "student_prompt"), ("opcd_gr", "student_prompt"), ("opcd_gr", "opcd"), ("dpo", "opcd")]]
              + [("abl_recteacher_gr_s0", "abl_recteacher_s0"), ("abl_recteacher_s0", "opcd_s0"), ("abl_recteacher_gr_s0", "opcd_gr_s0"),
                 ("abl_selfteacher_gr_s0", "opcd_gr_s0"), ("abl_maskprefix_gr_s0", "opcd_gr_s0"), ("abl_selfprefix_gr_s0", "opcd_gr_s0"),
                 ("abl_gf015_s0", "opcd_gr_s0"), ("abl_gf060_s0", "opcd_gr_s0"), ("abl_jsd_gr_s0", "opcd_gr_s0"),
                 ("teacher_prompt_recovery", "teacher_prompt")],
}
METRICS = ["adv_violation", "false_refusal", "borderline_false_refusal", "helpful", "recovery", "recovery_self_k10", "recovery_self_k20"]


def outcomes(path):
    """{metric: {key: 0/1}} keyed by prompt id."""
    m = {k: {} for k in METRICS}
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        src = d.get("source") or ("bank" if "prefix" in d else None)
        if src == "bank":
            m["recovery"][d["id"]] = int(bool(d["recovered"]))
        elif src in ("self_k10", "self_k20"):
            m[f"recovery_{src}"][d["id"]] = int(bool(d["recovered"]))
        elif src:
            continue
        elif d["cls"] == "benign":
            m["false_refusal"][d["id"]] = int("false_refusal" in d["reasons"])
            if d.get("helpful") is not None:
                m["helpful"][d["id"]] = int(d["helpful"])
        elif d["cls"] == "benign_borderline":
            m["borderline_false_refusal"][d["id"]] = int("false_refusal" in d["reasons"])
        else:
            m["adv_violation"][d["id"]] = int(bool(d["violation"]))
    return {k: v for k, v in m.items() if v}


def boot_ci(xs, n=1000, seed=0):
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return round(sum(xs) / k, 4), round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 outcomes (same keys)."""
    keys = sorted(a.keys() & b.keys())
    n01 = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    n10 = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"n_pairs": len(keys), "a_only": n10, "b_only": n01, "p": round(p, 4)}


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase2", choices=list(env.PHASES))
    a = ap.parse_args()
    R = env.paths(a.phase)["results"]
    O = {f[8:-6]: outcomes(f"{R}/raw/{f}") for f in sorted(os.listdir(f"{R}/raw")) if f.startswith("replies_") and f.endswith(".jsonl")}
    out = {"ci": {c: {m: boot_ci(list(v.values())) for m, v in o.items()} for c, o in O.items()},
           "mcnemar": {f"{x} vs {y}": {m: mcnemar(O[x][m], O[y][m]) for m in O[x] if m in O[y]} for x, y in PAIRS[a.phase] if x in O and y in O}}
    # mean ± std over seeds for seeded runs (phase 2)
    seeds = {}
    for c, r in out["ci"].items():
        if "_s" in c and c.rsplit("_s", 1)[1].isdigit() and not c.startswith("abl_"):
            for m, v in r.items():
                seeds.setdefault(c.rsplit("_s", 1)[0], {}).setdefault(m, []).append(v[0])
    out["seed_mean_std"] = {c: {m: (round(statistics.mean(v), 4), round(statistics.stdev(v), 4) if len(v) > 1 else 0.0, len(v)) for m, v in r.items()}
                            for c, r in seeds.items()}
    json.dump(out, open(f"{R}/ci.json", "w"), indent=1)
    ms = [m for m in METRICS if any(m in r for r in out["ci"].values())]
    L = ["| condition | " + " | ".join(f"{m} [95% CI]" for m in ms) + " |", "|---|" + "---|" * len(ms)]
    for c, r in out["ci"].items():
        L.append(f"| {c} | " + " | ".join(f"{r[m][0]:.3f} [{r[m][1]:.3f}, {r[m][2]:.3f}]" if m in r else "" for m in ms) + " |")
    if out["seed_mean_std"]:
        L += ["", "Mean ± std over seeds:", "", "| method | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
        for c, r in out["seed_mean_std"].items():
            L.append(f"| {c} (n={next(iter(r.values()))[2]}) | " + " | ".join(f"{r[m][0]:.3f} ± {r[m][1]:.3f}" if m in r else "" for m in ms) + " |")
    L += ["", "Exact McNemar (paired, same prompts). a_only = prompts where only the first condition has the event.", "",
          "| comparison | metric | a_only | b_only | p |", "|---|---|---|---|---|"]
    for k, r in out["mcnemar"].items():
        for m, s in r.items():
            L.append(f"| {k} | {m} | {s['a_only']} | {s['b_only']} | {fmt_p(s['p'])} |")
    open(f"{R}/ci.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
