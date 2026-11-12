#!/usr/bin/env python3
"""Tables and figures for the extra ablation (run_ablation2.sh).
Usage: python3 ablation2_summary.py results_ablation2"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else "results_ablation2"
os.makedirs(f"{R}/figures", exist_ok=True)
ev = pd.read_csv(f"{R}/eval.csv")
for c in ("acceptance", "unsafe_offload", "rejection"):
    ev[c] = ev[c] * 100
SC = ["No attack", "DDoS on path", "Compromised edge", "Fake tasks", "Mixed"]
LBL = {"Proposed": "Full (mask + security reward)", "Proposed_wo_mask": "No masking",
       "Proposed_wo_secreward": "No security reward"}


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def pval(a, b):
    if len(a) < 2 or len(b) < 2 or (np.std(a) == 0 and np.std(b) == 0):
        return "-"
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


main = ev[np.isclose(ev.mask_th, 0.4)]
doc = ["# Extra ablation: masking vs. security reward, and masking threshold\n",
       "20 UAVs, 0.5 tasks/s. Mean ± 95% CI over n runs (3 training seeds × 3 evaluation seeds). "
       "p = Welch t-test against the full method.\n"]
doc.append("## Table A1 - Masking vs. security reward\n")
doc.append("| Scenario | Variant | n | Acceptance (%) | Unsafe (%) | Latency (s) | p (acc.) | p (unsafe) |")
doc.append("|---|---|---|---|---|---|---|---|")
for sc in SC:
    full = main[(main.method == "Proposed") & (main.scenario == sc)]
    for m in LBL:
        x = main[(main.method == m) & (main.scenario == sc)]
        if x.empty:
            continue
        pa = "-" if m == "Proposed" else pval(full.acceptance, x.acceptance)
        pu = "-" if m == "Proposed" else pval(full.unsafe_offload, x.unsafe_offload)
        doc.append(f"| {sc} | {LBL[m]} | {len(x)} | {x.acceptance.mean():.2f} ± {ci95(x.acceptance):.2f} | "
                   f"{x.unsafe_offload.mean():.2f} ± {ci95(x.unsafe_offload):.2f} | "
                   f"{x.latency.mean():.2f} ± {ci95(x.latency):.2f} | {pa} | {pu} |")

doc.append("\n## Table A2 - Masking threshold tau_min (Proposed)\n")
doc.append("| Scenario | tau_min | n | Acceptance (%) | Unsafe (%) | Latency (s) |")
doc.append("|---|---|---|---|---|---|")
thr = ev[ev.method == "Proposed"]
for sc in ["No attack", "Compromised edge", "Mixed"]:
    for th in sorted(thr.mask_th.unique()):
        x = thr[(thr.scenario == sc) & np.isclose(thr.mask_th, th)]
        if x.empty:
            continue
        doc.append(f"| {sc} | {th:.1f} | {len(x)} | {x.acceptance.mean():.2f} ± {ci95(x.acceptance):.2f} | "
                   f"{x.unsafe_offload.mean():.2f} ± {ci95(x.unsafe_offload):.2f} | {x.latency.mean():.2f} |")
open(f"{R}/ABLATION2.md", "w").write("\n".join(doc) + "\n")
print("\n".join(doc))

# Figure A1: unsafe offloading and acceptance per variant in the insider-attack scenarios
cols = {"Proposed": "#2a78d6", "Proposed_wo_mask": "#eb6834", "Proposed_wo_secreward": "#1baf7a"}
hat = {"Proposed": "", "Proposed_wo_mask": "//", "Proposed_wo_secreward": "\\\\"}
scs = ["Compromised edge", "Mixed"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, met, yl in zip(axs, ["acceptance", "unsafe_offload"], ["Task acceptance (%)", "Unsafe offloading (%)"]):
    w = 0.26
    for i, m in enumerate(LBL):
        mu = [main[(main.method == m) & (main.scenario == sc)][met].mean() for sc in scs]
        er = [ci95(main[(main.method == m) & (main.scenario == sc)][met]) for sc in scs]
        ax.bar(np.arange(len(scs)) + (i - 1) * w, mu, w, yerr=er, capsize=3, color=cols[m],
               hatch=hat[m], edgecolor="white", label=LBL[m])
    ax.set_xticks(range(len(scs)), scs)
    ax.set_ylabel(yl)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(f"{R}/figures/figA1_mask_vs_reward.png", dpi=300)

# Figure A2: masking threshold
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
for sc, mk in zip(["No attack", "Compromised edge", "Mixed"], ["o", "s", "^"]):
    th = sorted(thr.mask_th.unique())
    for ax, met in zip(axs, ["acceptance", "unsafe_offload"]):
        mu = [thr[(thr.scenario == sc) & np.isclose(thr.mask_th, t)][met].mean() for t in th]
        er = [ci95(thr[(thr.scenario == sc) & np.isclose(thr.mask_th, t)][met]) for t in th]
        ax.errorbar(th, mu, yerr=er, marker=mk, capsize=3, label=sc)
for ax, yl in zip(axs, ["Task acceptance (%)", "Unsafe offloading (%)"]):
    ax.set_xlabel("Masking threshold $\\tau_{\\min}$")
    ax.set_ylabel(yl)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(f"{R}/figures/figA2_mask_threshold.png", dpi=300)
print("figures written to", f"{R}/figures")
