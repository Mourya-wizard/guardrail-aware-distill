"""Writes the Phase 3 job list (results/phase3/raw/queue/jobs.json) for scripts/job_queue.py:
verifier validation with Qwen3Guard (one 8B job), then IFEval + MMLU for the base student and the 9 Phase 2 checkpoints."""
import json
import os
import sys

PY = sys.executable
MODELS = ["base"] + [f"{m}_s{s}" for m in ("opcd", "opcd_gr", "dpo") for s in range(3)]

jobs = [{"name": "validate_public", "cmd": [PY, "-m", "guardrail_distill.validate_public"], "mem_gb": 22, "threads": 3, "deps": []}]
for m in MODELS:
    for task, mem in (("mmlu", 9), ("ifeval", 10)):
        jobs.append({"name": f"{task}_{m}", "cmd": [PY, "-m", "guardrail_distill.forgetting", "--model", m, "--task", task],
                     "mem_gb": mem, "threads": 3, "deps": []})
# reference: untrained student WITH the safety system prompt (IFEval only; MMLU is log-likelihood scored without a chat template)
jobs.append({"name": "ifeval_base_prompt", "cmd": [PY, "-m", "guardrail_distill.forgetting", "--model", "base_prompt", "--task", "ifeval"],
             "mem_gb": 10, "threads": 3, "deps": []})

if __name__ == "__main__":
    qdir = os.path.join("results", "phase3", "raw", "queue")
    os.makedirs(qdir, exist_ok=True)
    path = os.path.join(qdir, "jobs.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    names = {j["name"] for j in old}
    json.dump(old + [j for j in jobs if j["name"] not in names], open(path, "w"), indent=1)
    print(len(old), "existing +", len([j for j in jobs if j["name"] not in names]), "new jobs ->", path)
