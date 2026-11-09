import gd.env  # noqa
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONDS = ["student_none", "student_prompt", "teacher_prompt", "opcd", "opcd_gr", "dpo"]


def evals():
    return {c: json.load(open(f"results/eval_{c}.json")) for c in CONDS if os.path.exists(f"results/eval_{c}.json")}


def main():
    os.makedirs("results", exist_ok=True)
    runs = {r: f"results/opcd_{r}_log.csv" for r in ("opcd", "opcd_gr") if os.path.exists(f"results/opcd_{r}_log.csv")}
    if runs:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        for r, p in runs.items():
            d = pd.read_csv(p)
            ax[0].plot(d.step, d.loss, marker="o", label=r)
            ax[1].plot(d.step, d.dev_adv_violation, marker="o", label=r)
            ax[2].plot(d.step, d.dev_false_refusal, marker="o", label=r)
        for a, t in zip(ax, ["reverse-KL loss (window mean)", "dev adversarial violation", "dev false refusal"]):
            a.set_title(t); a.set_xlabel("step"); a.legend()
        fig.tight_layout(); fig.savefig("results/training_curves.png", dpi=130); plt.close(fig)
    E = evals()
    if E:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        xs = range(len(E))
        ax.bar([x - 0.2 for x in xs], [e["adv_violation_rate"] for e in E.values()], 0.4, label="adversarial violation")
        ax.bar([x + 0.2 for x in xs], [e["false_refusal_rate"] for e in E.values()], 0.4, label="benign false refusal")
        ax.set_xticks(list(xs)); ax.set_xticklabels(E.keys(), rotation=20); ax.legend(); ax.set_ylabel("rate (test set)")
        fig.tight_layout(); fig.savefig("results/results_bar.png", dpi=130); plt.close(fig)
        classes = ["pii_request", "guarantee_demand", "abuse", "offtopic_jailbreak"]
        fig, ax = plt.subplots(figsize=(9, 3.8))
        w = 0.8 / len(E)
        for k, (c, e) in enumerate(E.items()):
            ax.bar([i + k * w for i in range(4)], [e["recovery_by_class"].get(cl) or 0 for cl in classes], w, label=c)
        ax.set_xticks([i + 0.4 - w / 2 for i in range(4)]); ax.set_xticklabels(classes)
        ax.set_ylabel("recovery rate (held-out forced prefixes)"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig("results/recovery_by_class.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
