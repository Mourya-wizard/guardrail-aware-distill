"""Writes the Phase 2 job list (results/phase2/raw/queue/jobs.json) for scripts/job_queue.py.
Order = priority: seeds first, then the teacher-quality 2x2, then the other ablations, then evaluations."""
import json
import os
import sys

PY = sys.executable
STEPS = "200"  # fixed for every Phase 2 run (Phase 1 used 150; 300 did not fit the time budget)
MEM = {"train_large": 45, "train_self": 17, "dpo": 19, "eval_student": 8, "eval_teacher": 32, "audit": 20, "pool": 6}


def train(run, seed=0, gf=0.0, **flags):
    cmd = [PY, "-m", "guardrail_distill.context_distill", "--phase", "phase2", "--run", run, "--seed", str(seed),
           "--guided_frac", str(gf), "--steps", STEPS]
    for k, v in flags.items():
        cmd += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    return {"name": f"train_{run}", "cmd": cmd, "mem_gb": MEM["train_self" if flags.get("teacher") == "self" else "train_large"], "threads": 6, "deps": []}


def evaluate(cond, deps=(), teacher=False, extra=()):
    return {"name": f"eval_{cond}", "cmd": [PY, "-m", "guardrail_distill.evaluation", "--phase", "phase2", "--cond", cond, *extra],
            "mem_gb": MEM["eval_teacher" if teacher else "eval_student"], "threads": 3, "deps": list(deps)}


jobs = [{"name": "selfprefix_pool", "cmd": [PY, "-m", "guardrail_distill.evaluation", "--phase", "phase2", "--cond", "selfprefix_pool"],
         "mem_gb": MEM["pool"], "threads": 3, "deps": []}]
# untrained baselines and the Phase 1 checkpoints on the Phase 2 test set (N3); short, so they go before training
jobs += [evaluate(c, ["selfprefix_pool"]) for c in ("student_none", "student_prompt", "p1_opcd", "p1_opcd_gr", "p1_dpo")]
jobs.append({"name": "audit_verifier", "cmd": [PY, "-m", "guardrail_distill.audit"], "mem_gb": MEM["audit"], "threads": 3,
             "deps": [f"eval_{c}" for c in ("student_none", "student_prompt", "p1_opcd", "p1_opcd_gr", "p1_dpo")]})
for s in range(3):
    jobs += [train(f"opcd_s{s}", s, 0.0), train(f"opcd_gr_s{s}", s, 0.3)]
jobs += [train("abl_recteacher_s0", 0, 0.0, teacher_prompt="recovery"), train("abl_recteacher_gr_s0", 0, 0.3, teacher_prompt="recovery"),
         train("abl_selfteacher_gr_s0", 0, 0.3, teacher="self"), train("abl_maskprefix_gr_s0", 0, 0.3, mask_prefix=True),
         train("abl_selfprefix_gr_s0", 0, 0.3, prefix_source="self"), train("abl_gf015_s0", 0, 0.15), train("abl_gf060_s0", 0, 0.6),
         train("abl_jsd_gr_s0", 0, 0.3, divergence="jsd")]
for s in range(3):
    jobs.append({"name": f"train_dpo_s{s}", "cmd": [PY, "-m", "guardrail_distill.dpo_baseline", "--phase", "phase2", "--run", f"dpo_s{s}",
                                                   "--seed", str(s), "--match_steps", STEPS], "mem_gb": MEM["dpo"], "threads": 6, "deps": []})

runs = [j["name"][6:] for j in jobs if j["name"].startswith("train_")]
jobs += [evaluate(c, ["selfprefix_pool"], teacher=True) for c in ("teacher_prompt", "teacher_prompt_recovery")]
jobs += [evaluate(r, ["selfprefix_pool", f"train_{r}"]) for r in runs]
jobs += [evaluate(f"{r}@best", ["selfprefix_pool", f"train_{r}"]) for r in runs if r.startswith("opcd")]
# latency needs the whole GPU budget, so the queue only starts it when nothing else is running
jobs.append({"name": "latency", "cmd": [PY, "-m", "guardrail_distill.evaluation", "--phase", "phase2", "--cond", "latency", "--n_latency", "50"],
             "mem_gb": 64, "threads": 6, "deps": ["train_opcd_s0"]})

if __name__ == "__main__":
    qdir = os.path.join("results", "phase2", "raw", "queue")
    os.makedirs(qdir, exist_ok=True)
    path = os.path.join(qdir, "jobs.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    names = {j["name"] for j in old}
    json.dump(old + [j for j in jobs if j["name"] not in names], open(path, "w"), indent=1)
    print(len(old), "existing +", len([j for j in jobs if j["name"] not in names]), "new jobs ->", path)
