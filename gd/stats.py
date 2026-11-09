"""95% bootstrap CIs and paired exact McNemar tests from saved replies (results/raw/replies_<cond>.jsonl). No regeneration."""
import gd.env  # noqa
import json
import os
import random
from math import comb

CONDS = ["student_none", "student_prompt", "teacher_prompt", "opcd", "opcd_gr", "dpo"]
PAIRS = [("opcd", "student_prompt"), ("opcd_gr", "student_prompt"), ("dpo", "student_prompt"),
         ("opcd_gr", "opcd"), ("dpo", "opcd"), ("opcd", "student_none")]


def outcomes(cond):
    """{metric: {key: 0/1}} keyed by prompt id (recovery keyed by id too: one held-out prefix per prompt)."""
    m = {"adv_violation": {}, "false_refusal": {}, "recovery": {}}
    for l in open(f"results/raw/replies_{cond}.jsonl"):
        d = json.loads(l)
        if "prefix" in d:
            m["recovery"][d["id"]] = int(bool(d["recovered"]))
        elif d["cls"] == "benign":
            m["false_refusal"][d["id"]] = int("false_refusal" in d["reasons"])
        else:
            m["adv_violation"][d["id"]] = int(bool(d["violation"]))
    return m


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


def main():
    O = {c: outcomes(c) for c in CONDS if os.path.exists(f"results/raw/replies_{c}.jsonl")}
    out = {"ci": {c: {m: boot_ci(list(v.values())) for m, v in o.items()} for c, o in O.items()},
           "mcnemar": {f"{a} vs {b}": {m: mcnemar(O[a][m], O[b][m]) for m in O[a]} for a, b in PAIRS if a in O and b in O}}
    json.dump(out, open("results/ci.json", "w"), indent=1)
    L = ["| condition | adv. violation [95% CI] | false refusal [95% CI] | recovery [95% CI] |", "|---|---|---|---|"]
    for c, r in out["ci"].items():
        L.append(f"| {c} | " + " | ".join(f"{r[m][0]:.3f} [{r[m][1]:.3f}, {r[m][2]:.3f}]" for m in ("adv_violation", "false_refusal", "recovery")) + " |")
    L += ["", "Exact McNemar (paired, same prompts). a_only = prompts where only the first condition has the event.", "",
          "| comparison | metric | a_only | b_only | p |", "|---|---|---|---|---|"]
    for k, r in out["mcnemar"].items():
        for m, s in r.items():
            L.append(f"| {k} | {m} | {s['a_only']} | {s['b_only']} | {s['p'] if s['p'] >= 1e-4 else '<0.0001'} |")
    open("results/ci.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
