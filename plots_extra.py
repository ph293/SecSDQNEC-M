"""
Statistics, graphs and tables for the extra experiments (run_extra.sh).
Every number comes from the ns-3 simulation; Python only aggregates and plots.
Usage: python3 plots_extra.py results_v3 results_extra
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score

V3 = sys.argv[1] if len(sys.argv) > 1 else "results_v3"
R = sys.argv[2] if len(sys.argv) > 2 else "results_extra"
FIG = f"{R}/figures"; os.makedirs(FIG, exist_ok=True)

ORDER = ["Proposed", "SDQNEC_IDS", "SDQNEC", "DDQNEC", "DQNEC", "Greedy", "Random"]
TABLE_ORDER = ["Random", "Greedy", "DQNEC", "DDQNEC", "SDQNEC", "SDQNEC_IDS", "Proposed"]
ABL = ["Proposed", "Proposed_wo_trust", "Proposed_wo_middleware", "Proposed_wo_security"]
SCEN = ["No attack", "DDoS on path", "Compromised edge", "Fake tasks", "Mixed"]
LABEL = {"Proposed": "Proposed (SecSDQNEC-M)", "SDQNEC_IDS": "SDQNEC + IDS"}
STYLE = {"Proposed": dict(color="#2a78d6", marker="o", ls="-"),
         "SDQNEC_IDS": dict(color="#8a5cd6", marker="P", ls="-"),
         "SDQNEC": dict(color="#eb6834", marker="s", ls="--"),
         "DDQNEC": dict(color="#1baf7a", marker="^", ls="-."),
         "DQNEC": dict(color="#eda100", marker="D", ls=":"),
         "Greedy": dict(color="#e87ba4", marker="v", ls="-"),
         "Random": dict(color="#008300", marker="x", ls="--")}
HATCH = {"Proposed": "", "SDQNEC_IDS": "++", "SDQNEC": "//", "DDQNEC": "\\\\", "DQNEC": "xx",
         "Greedy": "..", "Random": "--"}
INK2, GRID = "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
                     "lines.markersize": 6, "savefig.dpi": 300})


def save(fig, n):
    fig.tight_layout(); fig.savefig(f"{FIG}/{n}.png"); plt.close(fig)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


# ---------------- load: v3 runs (eval seed 100+s) + extra runs (seeds 200+s, 300+s) ----------------
a = pd.read_csv(f"{V3}/eval.csv")
a = a[(a.n_uav == 20) & (a.arrival == 0.5)].copy()
for c, v in (("ddos_mbps", 92.0), ("drop_prob", 0.6)):
    if c not in a.columns:
        a[c] = v
b = pd.read_csv(f"{R}/eval.csv")
ev = pd.concat([a, b], ignore_index=True)
ev = ev.drop_duplicates(subset=["method", "scenario", "seed", "train_seed", "ddos_mbps", "drop_prob"])
main = ev[(np.isclose(ev.ddos_mbps, 92.0)) & (np.isclose(ev.drop_prob, 0.6))]

doc = ["# Extra experiments (ns-3)\n",
       "20 UAVs, 0.5 tasks/s per UAV. Each value: mean ± 95% confidence interval over n runs "
       "(3 training seeds × 3 evaluation seeds). p = Welch t-test of Proposed vs. that method "
       "(two-sided); p < 0.05 means the difference is statistically significant.\n"]

# ---------------- Table S - significance per scenario ----------------
summary = []
for s in SCEN:
    d = main[main.scenario == s]
    prop = d[d.method == "Proposed"]
    rows = []
    for m in TABLE_ORDER:
        x = d[d.method == m]
        if x.empty:
            continue
        r = {"Method": LABEL.get(m, m), "n": len(x)}
        for col, k, nm in (("acceptance", 100, "Acceptance (%)"), ("unsafe_offload", 100, "Unsafe (%)"),
                           ("latency", 1, "Latency (s)"), ("cost", 1, "Cost")):
            r[nm] = f"{x[col].mean() * k:.2f} ± {ci95(x[col]) * k:.2f}"
            summary.append(dict(method=m, scenario=s, metric=col, mean=x[col].mean(), ci95=ci95(x[col]), n=len(x)))
        if m != "Proposed" and len(prop) > 1 and len(x) > 1:
            for col, nm in (("acceptance", "p (acceptance)"), ("unsafe_offload", "p (unsafe)")):
                if x[col].std() == 0 and prop[col].std() == 0:
                    r[nm] = "-"
                    continue
                p = stats.ttest_ind(prop[col], x[col], equal_var=False).pvalue
                r[nm] = "< 0.001" if p < 0.001 else f"{p:.3f}"
        else:
            r["p (acceptance)"] = r["p (unsafe)"] = "-"
        rows.append(r)
    doc += [f"## Table S - {s}\n", md(pd.DataFrame(rows)), ""]
pd.DataFrame(summary).to_csv(f"{R}/extra_summary.csv", index=False)

# ---------------- ablation with CI ----------------
rows = []
for s in SCEN:
    for m in ABL:
        x = main[(main.scenario == s) & (main.method == m)]
        if x.empty:
            continue
        rows.append({"Scenario": s, "Method": m, "n": len(x),
                     "Acceptance (%)": f"{x.acceptance.mean() * 100:.2f} ± {ci95(x.acceptance) * 100:.2f}",
                     "Unsafe (%)": f"{x.unsafe_offload.mean() * 100:.2f} ± {ci95(x.unsafe_offload) * 100:.2f}",
                     "Latency (s)": f"{x.latency.mean():.2f} ± {ci95(x.latency):.2f}"})
doc += ["## Table S-Ablation (95% CI)\n", md(pd.DataFrame(rows)), ""]

# ---------------- fig24: acceptance under attack incl. conventional IDS baseline, 95% CI ----------------
fig, ax = plt.subplots(figsize=(7.6, 3.5))
meth = [m for m in ORDER if m in set(main.method)]
w = 0.84 / len(meth); x = np.arange(len(SCEN))
for i, m in enumerate(meth):
    mu = [main[(main.scenario == s) & (main.method == m)].acceptance.mean() * 100 for s in SCEN]
    er = [ci95(main[(main.scenario == s) & (main.method == m)].acceptance) * 100 for s in SCEN]
    ax.bar(x + (i - (len(meth) - 1) / 2) * w, mu, w * 0.92, yerr=er, capsize=2, color=STYLE[m]["color"],
           hatch=HATCH[m], edgecolor="white", linewidth=0.5, error_kw=dict(elinewidth=0.7), label=LABEL.get(m, m))
ax.set_xticks(x); ax.set_xticklabels(SCEN); ax.set_ylabel("Task acceptance rate (%)"); ax.grid(axis="x", visible=False)
ax.legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig24_acceptance_with_ids_ci95")

# ---------------- fig25/26: sensitivity to attack intensity ----------------
SENS = ["Proposed", "SDQNEC_IDS", "SDQNEC"]


def sens(df, xcol, xs, xl, name, title):
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, col, yl in zip(axs, ["acceptance", "unsafe_offload"], ["Task acceptance (%)", "Unsafe offloading (%)"]):
        for m in SENS:
            mu, er = [], []
            for v in xs:
                q = df[(df.method == m) & np.isclose(df[xcol], v)][col] * 100
                mu.append(q.mean()); er.append(q.std(ddof=1) if len(q) > 1 else 0.0)  # +-1 std (3 runs)
            ax.errorbar(xs, mu, yerr=er, capsize=2.5, elinewidth=0.8, label=LABEL.get(m, m), **STYLE[m])
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_ylim(0, 100 if col == "acceptance" else None)
        if col != "acceptance": ax.set_ylim(bottom=0)
    axs[0].legend(fontsize=8); fig.text(0.99, 0.01, "error bars: ±1 std over 3 training seeds", ha="right", fontsize=7, color=INK2)
    save(fig, name)
    rows = []
    for m in SENS:
        r = {"Method": LABEL.get(m, m)}
        for v in xs:
            q = df[(df.method == m) & np.isclose(df[xcol], v)]
            r[f"{v:g}"] = f"{q.acceptance.mean() * 100:.1f} / {q.unsafe_offload.mean() * 100:.1f}"
        rows.append(r)
    return [f"## {title} (acceptance % / unsafe %)\n", md(pd.DataFrame(rows)), ""]


dd = ev[(ev.scenario == "DDoS on path") & (np.isclose(ev.drop_prob, 0.6)) & (ev.seed == ev.train_seed + 100)]
cc = ev[(ev.scenario == "Compromised edge") & (np.isclose(ev.ddos_mbps, 92.0)) & (ev.seed == ev.train_seed + 100)]
doc += sens(dd, "ddos_mbps", [30, 60, 90, 120], "DDoS flood rate (Mbit/s)", "fig25_sensitivity_ddos_rate",
            "Table T1 - Sensitivity to DDoS rate")
doc += sens(cc, "drop_prob", [0.2, 0.4, 0.6, 0.8], "Drop probability of compromised server",
            "fig26_sensitivity_drop_prob", "Table T2 - Sensitivity to compromised-server drop probability")

# ---------------- detector comparison ----------------
det = pd.read_csv(f"{V3}/detector_results.csv")
sc = pd.read_csv(f"{V3}/detector_scores.csv")
det["auc"] = np.nan
det.loc[det.detector.str.startswith("Two-tier"), "auc"] = roc_auc_score(sc.label, sc.score)
ml = pd.read_csv(f"{R}/detector_ml.csv")
allp = pd.concat([det, ml], ignore_index=True)
order = ["Static threshold", "Local AE (middleware only)", "Isolation Forest", "One-Class SVM", "LOF",
         "Two-tier AE (proposed)"]
allp = allp.set_index("detector").reindex([o for o in order if o in set(allp.detector)]).reset_index()
t = allp[["detector", "accuracy", "precision", "recall", "f1", "fpr", "auc", "recall_ddos", "recall_fake",
          "recall_comp"]].copy()
for c in t.columns[1:]:
    t[c] = (t[c] * (1 if c == "auc" else 100)).round(3 if c == "auc" else 2)
doc += ["## Table D2 - Detector comparison on the same ns-3 test windows\n",
        "All ML detectors use the same two-tier layout and the same 99th-percentile threshold rule.\n",
        md(t), ""]
fig, ax = plt.subplots(figsize=(7.6, 3.3))
mets = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("recall_comp", "Recall (compromised)")]
cols = ["#e87ba4", "#eda100", "#1baf7a", "#8a5cd6", "#008300", "#2a78d6"]
w = 0.84 / len(allp); x = np.arange(len(mets))
for i, (_, r) in enumerate(allp.iterrows()):
    ax.bar(x + (i - (len(allp) - 1) / 2) * w, [r[k] * 100 for k, _ in mets], w * 0.92, color=cols[i % 6],
           edgecolor="white", linewidth=0.5, label=r.detector)
ax.set_xticks(x); ax.set_xticklabels([n for _, n in mets]); ax.set_ylabel("Score (%)"); ax.set_ylim(0, 105)
ax.grid(axis="x", visible=False); ax.legend(fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
save(fig, "fig27_detector_comparison")

open(f"{R}/EXTRA_RESULTS.md", "w").write("\n".join(doc))
print("\n".join(doc))
