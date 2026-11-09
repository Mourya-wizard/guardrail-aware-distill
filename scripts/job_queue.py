"""Tiny GPU job queue: runs jobs from a JSON file in parallel under a GPU-memory and CPU-thread budget.

Each job: {"name", "cmd": [...], "mem_gb", "threads", "deps": [names]}. The job file is re-read every poll, so jobs can be
appended while the queue runs. Status is persisted, so a restarted queue skips finished jobs. A failed job is retried once;
jobs that depend on a job that failed twice are skipped.
Usage: python scripts/job_queue.py --jobs results/phase2/raw/queue/jobs.json [--mem_gb 70 --threads 21]
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--mem_gb", type=float, default=70)
    ap.add_argument("--threads", type=int, default=21)
    ap.add_argument("--poll", type=float, default=15)
    a = ap.parse_args()
    qdir = os.path.dirname(a.jobs)
    status_path = os.path.join(qdir, "status.json")
    status = json.load(open(status_path)) if os.path.exists(status_path) else {}
    running = {}  # name -> (Popen, job)
    while True:
        jobs = {j["name"]: j for j in json.load(open(a.jobs))}
        for name, (p, j) in list(running.items()):
            if p.poll() is not None:
                del running[name]
                st = status[name]
                st["end"] = time.strftime("%H:%M:%S"); st["rc"] = p.returncode
                if p.returncode == 0:
                    st["state"] = "done"
                else:
                    st["tries"] = st.get("tries", 1)
                    st["state"] = "retry" if st["tries"] < 2 else "failed"
                print(f"[{st['end']}] {name}: {st['state']} (rc {p.returncode})", flush=True)
        for name, j in jobs.items():
            st = status.get(name, {})
            if st.get("state") in ("done", "failed", "running", "skipped"):
                continue
            if any(status.get(d, {}).get("state") in ("failed", "skipped") for d in j.get("deps", [])):
                status[name] = {"state": "skipped"}; print(f"{name}: skipped (dependency failed)", flush=True); continue
            if not all(status.get(d, {}).get("state") == "done" for d in j.get("deps", [])):
                continue
            mem = sum(r[1]["mem_gb"] for r in running.values())
            thr = sum(r[1]["threads"] for r in running.values())
            if mem + j["mem_gb"] > a.mem_gb or thr + j["threads"] > a.threads:
                continue
            env = dict(os.environ, GD_THREADS=str(j["threads"]), PYTHONIOENCODING="utf-8")
            log = open(os.path.join(qdir, f"{name}.log"), "a", encoding="utf-8")
            p = subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env)
            tries = st.get("tries", 0) + 1
            status[name] = {"state": "running", "start": time.strftime("%H:%M:%S"), "tries": tries}
            running[name] = (p, j)
            print(f"[{status[name]['start']}] start {name} (mem {mem + j['mem_gb']:.0f} GB, threads {thr + j['threads']})", flush=True)
        json.dump(status, open(status_path, "w"), indent=1)
        if not running and all(status.get(n, {}).get("state") in ("done", "failed", "skipped") for n in jobs):
            print("queue empty", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
