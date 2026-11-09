"""Figures for a phase -> results/<phase>/figures/*.png."""
import guardrail_distill.env as env  # noqa
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

P1 = ["student_none", "student_prompt", "teacher_prompt", "opcd", "opcd_gr", "dpo"]
CLASSES = ["pii_request", "guarantee_demand", "abuse", "offtopic_jailbreak"]


def load_evals(R, names):
    return {c: json.load(open(f"{R}/eval/{c}.json")) for c in names if os.path.exists(f"{R}/eval/{c}.json")}


def bars(ax, E, keys, labels):
    w = 0.8 / len(keys)
    for k, (key, lab) in enumerate(zip(keys, labels)):
        ys = [e.get(key) or 0 for e in E.values()]
        err = None
        ci = [e.get("ci95", {}).get(key.replace("_rate", "")) for e in E.values()]
        if all(ci):
            err = [[y - c[1] for y, c in zip(ys, ci)], [c[2] - y for y, c in zip(ys, ci)]]
        ax.bar([i + (k - (len(keys) - 1) / 2) * w for i in range(len(E))], ys, w, yerr=err, capsize=2, label=lab)
    ax.set_xticks(range(len(E))); ax.set_xticklabels(E.keys(), rotation=25, ha="right", fontsize=8); ax.legend(fontsize=8)


def phase1(R, F):
    runs = {r: f"{R}/training/{r}_log.csv" for r in ("opcd", "opcd_gr") if os.path.exists(f"{R}/training/{r}_log.csv")}
    if runs:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        for r, p in runs.items():
            d = pd.read_csv(p)
            for a, col in zip(ax, ["loss", "dev_adv_violation", "dev_false_refusal"]):
                a.plot(d.step, d[col], marker="o", label=r)
        for a, t in zip(ax, ["reverse-KL loss (window mean)", "dev adversarial violation", "dev false refusal"]):
            a.set_title(t); a.set_xlabel("step"); a.legend()
        fig.tight_layout(); fig.savefig(f"{F}/training_curves.png", dpi=130); plt.close(fig)
    E = load_evals(R, P1)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars(ax, E, ["adv_violation_rate", "false_refusal_rate"], ["adversarial violation", "benign false refusal"])
    ax.set_ylabel("rate (test set)"); fig.tight_layout(); fig.savefig(f"{F}/results_bar.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(E)
    for k, (c, e) in enumerate(E.items()):
        by = e.get("recovery_by_class") or e.get("recovery_by", {})
        ax.bar([i + k * w for i in range(4)], [by.get(cl) or 0 for cl in CLASSES], w, label=c)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(4)]); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("recovery rate (held-out forced prefixes)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{F}/recovery_by_class.png", dpi=130); plt.close(fig)


def phase2(R, F):
    main = ["student_none", "student_prompt", "teacher_prompt", "teacher_prompt_recovery", "p1_opcd", "p1_opcd_gr", "p1_dpo",
            "opcd_s0", "opcd_gr_s0", "dpo_s0"]
    E = load_evals(R, main)
    if E:
        fig, ax = plt.subplots(figsize=(11, 4))
        bars(ax, E, ["adv_violation_rate", "false_refusal_rate", "borderline_false_refusal_rate", "helpful_rate"],
             ["adversarial violation", "benign false refusal", "borderline false refusal", "helpful (benign)"])
        ax.set_ylabel("rate, 95% CI (Phase 2 test)"); fig.tight_layout(); fig.savefig(f"{F}/results_bar.png", dpi=130); plt.close(fig)
    E = load_evals(R, ["student_none", "student_prompt", "teacher_prompt", "teacher_prompt_recovery", "opcd_s0", "opcd_gr_s0", "dpo_s0",
                       "abl_recteacher_gr_s0", "abl_selfprefix_gr_s0"])
    if E:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c, e in E.items():
            ks = [k for k in (5, 10, 20, 40) if f"recovery_self_k{k}" in e]
            ax.plot(ks, [e[f"recovery_self_k{k}"] for k in ks], marker="o", label=c)
        ax.set_xlabel("forced unsafe prefix length k (tokens, model's own unsafe starts)"); ax.set_ylabel("strict recovery rate")
        ax.set_xscale("log", base=2); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{F}/recovery_vs_k.png", dpi=130); plt.close(fig)
    abl = ["opcd_s0", "opcd_gr_s0"] + sorted(f[:-5] for f in os.listdir(f"{R}/eval") if f.startswith("abl_") and f.endswith(".json"))
    E = load_evals(R, abl)
    if len(E) > 2:
        fig, ax = plt.subplots(figsize=(11, 4))
        bars(ax, E, ["adv_violation_rate", "false_refusal_rate", "recovery_self_k10", "recovery_self_k20"],
             ["adversarial violation", "benign false refusal", "recovery k=10", "recovery k=20"])
        ax.set_ylabel("rate (Phase 2 test, seed 0)"); fig.tight_layout(); fig.savefig(f"{F}/ablations.png", dpi=130); plt.close(fig)
    logs = sorted(f for f in os.listdir(f"{R}/training") if f.endswith("_log.csv") and f.startswith("opcd"))
    if logs:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        for f in logs:
            d = pd.read_csv(f"{R}/training/{f}")
            for a, col in zip(ax, ["loss", "dev_adv_violation", "dev_false_refusal"]):
                a.plot(d.step, d[col], marker=".", label=f[:-8], color="tab:orange" if "_gr_" in f else "tab:blue", alpha=0.7)
        for a, t in zip(ax, ["divergence loss", "dev adversarial violation", "dev false refusal"]):
            a.set_title(t); a.set_xlabel("step")
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{F}/training_curves.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase2", choices=list(env.PHASES))
    a = ap.parse_args()
    R = env.paths(a.phase)["results"]
    F = f"{R}/figures"
    os.makedirs(F, exist_ok=True)
    (phase1 if a.phase == "phase1" else phase2)(R, F)
