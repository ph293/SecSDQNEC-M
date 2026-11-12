"""
Graphs + result tables from the ns-3 output CSV files.
(Only plotting is done in Python; every number comes from the ns-3 simulation.)
Usage: python3 plots_ns3.py results
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = f"{R}/figures"; os.makedirs(OUT, exist_ok=True)
ORDER = ["Proposed", "SDQNEC", "DDQNEC", "DQNEC", "Greedy", "Random"]
RLONLY = ["Proposed", "SDQNEC", "DDQNEC", "DQNEC"]
TABLE_ORDER = ["Random", "Greedy", "DQNEC", "DDQNEC", "SDQNEC", "Proposed"]
LABEL = {"Proposed": "Proposed (SecSDQNEC-M)"}
STYLE = {"Proposed": dict(color="#2a78d6", marker="o", ls="-"),
         "SDQNEC": dict(color="#eb6834", marker="s", ls="--"),
         "DDQNEC": dict(color="#1baf7a", marker="^", ls="-."),
         "DQNEC": dict(color="#eda100", marker="D", ls=":"),
         "Greedy": dict(color="#e87ba4", marker="v", ls="-"),
         "Random": dict(color="#008300", marker="x", ls="--")}
COLS4 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
HATCH = ["", "//", "\\\\", "xx", "..", "--"]
SCEN = ["No attack", "DDoS on path", "Compromised edge", "Fake tasks", "Mixed"]
ABL = ["Proposed", "Proposed_wo_trust", "Proposed_wo_middleware", "Proposed_wo_security"]
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                     "lines.markersize": 6, "savefig.dpi": 300})


def save(fig, n):
    fig.tight_layout(); fig.savefig(f"{OUT}/{n}.png"); plt.close(fig)


ev = pd.read_csv(f"{R}/eval.csv")
METRICS = ["latency", "energy", "cost", "rejection", "acceptance", "utilization", "throughput",
           "optimal_path", "unsafe_offload", "decision_us", "flowmods_per_s"]
agg = ev.groupby(["method", "n_uav", "arrival", "scenario"])[METRICS].agg(["mean", "std"]).reset_index()
agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
agg.to_csv(f"{R}/eval_summary.csv", index=False)


def line(df, x, y, xl, yl, name, pct=False):
    fig, ax = plt.subplots(figsize=(4.6, 3.3)); k = 100 if pct else 1
    for m in ORDER:
        d = df[df.method == m].sort_values(x)
        ax.errorbar(d[x], d[y + "_mean"] * k, yerr=d[y + "_std"].fillna(0) * k, capsize=2.5,
                    elinewidth=0.8, label=LABEL.get(m, m), **STYLE[m])
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.legend(fontsize=8); save(fig, name)


def bars(df, y, yl, name, methods=ORDER, pct=False, scen=SCEN):
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); k = 100 if pct else 1
    w = 0.8 / len(methods); x = np.arange(len(scen))
    for i, m in enumerate(methods):
        d = df[df.method == m].set_index("scenario").reindex(scen)
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, d[y + "_mean"] * k, w * 0.92,
               yerr=d[y + "_std"].fillna(0) * k, capsize=2, color=COLS4[i], hatch=HATCH[i],
               edgecolor="white", linewidth=0.5, error_kw=dict(elinewidth=0.7),
               label=LABEL.get(m, m).replace("_", " "))
    ax.set_xticks(x); ax.set_xticklabels(scen); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=min(len(methods), 3), loc="lower center", bbox_to_anchor=(0.5, 1.0)); save(fig, name)


na = agg[agg.scenario == "No attack"]
e1 = na[(na.arrival == 0.5)]
e2 = na[(na.n_uav == 20)]
sc = agg[(agg.n_uav == 20) & (agg.arrival == 0.5)]

# training convergence
for key, yl, n in [("reward", "Episode reward (moving avg. 10)", "fig03_reward_convergence"),
                   ("rejection", "Rejection rate during training", "fig04_training_rejection")]:
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    import glob
    for m in RLONLY:
        files = sorted(glob.glob(f"{R}/train_{m}_s*.csv")) or glob.glob(f"{R}/train_{m}.csv")
        if not files:
            continue
        t = pd.concat([pd.read_csv(x) for x in files]).groupby("window")[key].agg(["mean", "std"]).reset_index()
        st = dict(STYLE[m]); st.pop("marker")
        mu = t["mean"].rolling(10, min_periods=1).mean()
        sd = t["std"].fillna(0).rolling(10, min_periods=1).mean()
        ax.plot(t.window, mu, label=LABEL.get(m, m), **st)
        ax.fill_between(t.window, mu - sd, mu + sd, color=STYLE[m]["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Episode (15 s of simulated time)"); ax.set_ylabel(yl); ax.legend(fontsize=8)
    ax.set_title(f"mean ± std over {len(files) if files else 1} training seeds", fontsize=8, color=INK2)
    save(fig, n)

line(e1, "n_uav", "latency", "Number of UAVs", "Average latency (s)", "fig05_latency_vs_uavs")
line(e1, "n_uav", "rejection", "Number of UAVs", "Task rejection rate (%)", "fig06_rejection_vs_uavs", True)
line(e1, "n_uav", "cost", "Number of UAVs", "Average system cost per task", "fig07_cost_vs_uavs")
line(e2, "arrival", "energy", "Task arrival rate (tasks/s per UAV)", "Energy per task (J)", "fig08_energy_vs_arrival")
line(e2, "arrival", "rejection", "Task arrival rate (tasks/s per UAV)", "Task rejection rate (%)", "fig09_rejection_vs_arrival", True)
line(e2, "arrival", "throughput", "Task arrival rate (tasks/s per UAV)", "Throughput (tasks/s)", "fig10_throughput_vs_arrival")
line(e1, "n_uav", "utilization", "Number of UAVs", "Resource utilisation (%)", "fig11_utilization_vs_uavs", True)
bars(sc, "acceptance", "Task acceptance rate (%)", "fig13_acceptance_under_attack", pct=True)
bars(sc, "latency", "Average latency (s)", "fig14_latency_under_attack")
bars(sc, "unsafe_offload", "Tasks sent to attacked node/path (%)", "fig15_unsafe_offloading", pct=True)

# detector
det = pd.read_csv(f"{R}/detector_results.csv")
best = det[det.detector.str.contains("Two-tier")].iloc[0]
cm = np.array([[best.tn, best.fp], [best.fn, best.tp]]).astype(int)
fig, ax = plt.subplots(figsize=(3.6, 3.2)); ax.imshow(cm, cmap="Blues"); ax.grid(False)
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "#0b0b0b", fontsize=11)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Normal", "Attack"]); ax.set_yticklabels(["Normal", "Attack"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); save(fig, "fig16_confusion_matrix")

for cols, labels, n in [(["accuracy", "precision", "recall", "f1"], ["Accuracy", "Precision", "Recall", "F1-score"], "fig17_detection_metrics"),
                        (["recall_ddos", "recall_fake", "recall_comp"], ["DDoS on path", "Fake tasks", "Compromised node"], "fig18_detection_per_attack")]:
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(len(cols)); w = 0.8 / len(det)
    for i, (_, r) in enumerate(det.iterrows()):
        ax.bar(x + (i - (len(det) - 1) / 2) * w, [r[c] * 100 for c in cols], w * 0.92, color=COLS4[::-1][i + 1],
               hatch=HATCH[::-1][i + 1], edgecolor="white", linewidth=0.5, label=r.detector)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False); ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); save(fig, n)

scores = pd.read_csv(f"{R}/detector_scores.csv")
from_s = scores.sort_values("score", ascending=False)
P, N = (from_s.label == 1).sum(), (from_s.label == 0).sum()
tpr = np.r_[0, np.cumsum(from_s.label == 1) / P]; fpr = np.r_[0, np.cumsum(from_s.label == 0) / N]
auc = getattr(np, "trapezoid", getattr(np, "trapz", None))(tpr, fpr)
fig, ax = plt.subplots(figsize=(3.8, 3.3))
ax.plot(fpr, tpr, color=COLS4[0], label=f"Two-tier AE (AUC = {auc:.3f})"); ax.plot([0, 1], [0, 1], color=INK2, ls=":", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(fontsize=8, loc="lower right"); save(fig, "fig19_roc_curve")

dt = pd.read_csv(f"{R}/detection_time.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.barh(dt.attack, dt.mean_detection_time_s, color=COLS4[0], height=0.55)
for i, v in enumerate(dt.mean_detection_time_s):
    ax.text(v + 0.05, i, f"{v:.2f} s", va="center", fontsize=9)
ax.set_xlabel("Mean detection time (s)"); ax.grid(axis="y", visible=False); ax.set_xlim(0, dt.mean_detection_time_s.max() * 1.3)
save(fig, "fig20_detection_time")

ab = sc[sc.method.isin(ABL)].groupby("method")[["acceptance_mean", "latency_mean", "unsafe_offload_mean"]].mean().reindex(ABL)
fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2))
for ax, c, yl, k in zip(axs, ["acceptance_mean", "latency_mean", "unsafe_offload_mean"],
                        ["Acceptance (%)", "Latency (s)", "Unsafe offloading (%)"], [100, 1, 100]):
    ax.bar(range(4), ab[c] * k, color=COLS4[:4], hatch=HATCH[:4], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(["Full", "No\ntrust", "No\nMW", "No\nsecurity"], fontsize=8)
    ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
save(fig, "fig21_ablation")

# ---------------- tables ----------------
def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values])

def pm(r, c, k=1, d=2):
    return f"{r[c+'_mean']*k:.{d}f} ± {(0 if pd.isna(r[c+'_std']) else r[c+'_std'])*k:.{d}f}"

doc = ["# ns-3 evaluation results\n", "20 UAVs, 0.5 tasks/s per UAV unless stated; mean ± std over evaluation runs.\n"]
t = sc[sc.scenario == "No attack"].set_index("method").reindex(TABLE_ORDER).reset_index()
rows = [[r.method, pm(r, "latency"), pm(r, "energy"), pm(r, "cost"), pm(r, "rejection", 100), pm(r, "utilization", 100),
         pm(r, "throughput")] for _, r in t.iterrows()]
doc += ["## Table A - Offloading performance (no attack)\n",
        md(pd.DataFrame(rows, columns=["Method", "Latency (s)", "Energy (J)", "Cost", "Rejection (%)", "Utilisation (%)", "Throughput (tasks/s)"])), ""]
for s in SCEN[1:]:
    t = sc[sc.scenario == s].set_index("method").reindex(TABLE_ORDER).reset_index()
    rows = [[r.method, pm(r, "latency"), pm(r, "acceptance", 100), pm(r, "unsafe_offload", 100), pm(r, "cost")] for _, r in t.iterrows()]
    doc += [f"## Table B - Under attack: {s}\n",
            md(pd.DataFrame(rows, columns=["Method", "Latency (s)", "Acceptance (%)", "Unsafe offloading (%)", "Cost"])), ""]
dd = det.copy()
for c in ["accuracy", "precision", "recall", "f1", "fpr", "recall_ddos", "recall_fake", "recall_comp"]:
    dd[c] = (dd[c] * 100).round(2)
doc += ["## Table D - Anomaly detection (ns-3 traffic windows)\n", md(dd), "", f"ROC-AUC (two-tier) = {auc:.4f}\n",
        "Detection time:\n", md(dt.round(3)), ""]
ab2 = sc[sc.method.isin(ABL)].groupby("method")[["latency_mean", "acceptance_mean", "unsafe_offload_mean", "cost_mean"]].mean().reindex(ABL).reset_index()
ab2[["acceptance_mean", "unsafe_offload_mean"]] *= 100
doc += ["## Table E - Ablation (average over 5 scenarios)\n", md(ab2.round(3)), ""]
for s in SCEN:
    e = sc[(sc.method.isin(ABL)) & (sc.scenario == s)].set_index("method").reindex(ABL).reset_index()
    rows = [[r.method, pm(r, "acceptance", 100), pm(r, "unsafe_offload", 100), pm(r, "latency")] for _, r in e.iterrows()]
    doc += [f"### Table E-{s} - Ablation, {s}\n",
            md(pd.DataFrame(rows, columns=["Method", "Acceptance (%)", "Unsafe offloading (%)", "Latency (s)"])), ""]
dec = sc[sc.scenario == "No attack"].set_index("method")
doc += ["## Table F - Overhead\n", md(dec[["decision_us_mean", "flowmods_per_s_mean"]].round(2).reset_index()), ""]
open(f"{R}/RESULTS.md", "w").write("\n".join(doc))
print("\n".join(doc))
