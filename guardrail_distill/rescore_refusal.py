"""Phase 3: re-score Phase 2 false-refusal rates with the refusal detectors validated on XSTest (no regeneration;
Phase 2 results are not overwritten). Triggered because the opening-refusal rule was "weak" on XSTest (high precision,
recall 0.44), so Phase 2 false-refusal rates are likely undercounts.
-> results/phase3/phase2_false_refusal_rescored.{json,md}
"""
import guardrail_distill.env as env  # noqa
import json
import os
import statistics

from guardrail_distill.significance import boot_ci
from guardrail_distill.validate_public import recovery_marker, refusal_anywhere, refusal_opening

DETECTORS = {"opening rule (Phase 2)": refusal_opening, "refusal anywhere": refusal_anywhere,
             "recovery marker (best on XSTest)": recovery_marker}
CONDS = ["student_none", "student_prompt", "teacher_prompt"] + [f"{m}_s{s}" for m in ("opcd", "opcd_gr", "dpo") for s in range(3)]


def main():
    P2, R = env.paths("phase2")["results"], env.paths("phase3")["results"]
    out = {}
    for c in CONDS:
        recs = [json.loads(l) for l in open(f"{P2}/raw/replies_{c}.jsonl", encoding="utf-8")]
        recs = [d for d in recs if not d.get("source") and d["cls"] in ("benign", "benign_borderline")]
        out[c] = {cls: {name: boot_ci([int(f(d["reply"])) for d in recs if d["cls"] == cls]) for name, f in DETECTORS.items()}
                  for cls in ("benign", "benign_borderline")}
    seeds = {}
    for m in ("opcd", "opcd_gr", "dpo"):
        seeds[m] = {cls: {name: (round(statistics.mean(out[f"{m}_s{s}"][cls][name][0] for s in range(3)), 4),
                                 round(statistics.stdev(out[f"{m}_s{s}"][cls][name][0] for s in range(3)), 4))
                          for name in DETECTORS} for cls in ("benign", "benign_borderline")}
    json.dump({"per_condition": out, "seed_mean_std": seeds,
               "note": "the recovery marker includes words such as 'private', 'instead', 'official channel' that also occur in helpful "
                       "rental answers, so it can over-count refusals in this domain; read the three columns as a range"},
              open(f"{R}/phase2_false_refusal_rescored.json", "w"), indent=1)
    names = list(DETECTORS)
    L = ["| condition | class | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for c in ("student_none", "student_prompt", "teacher_prompt"):
        for cls in ("benign", "benign_borderline"):
            L.append(f"| {c} | {cls} | " + " | ".join(f"{out[c][cls][n][0]:.3f} [{out[c][cls][n][1]:.3f}, {out[c][cls][n][2]:.3f}]" for n in names) + " |")
    for m in ("opcd", "opcd_gr", "dpo"):
        for cls in ("benign", "benign_borderline"):
            L.append(f"| {m} (3 seeds) | {cls} | " + " | ".join(f"{seeds[m][cls][n][0]:.3f} ± {seeds[m][cls][n][1]:.3f}" for n in names) + " |")
    open(f"{R}/phase2_false_refusal_rescored.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
