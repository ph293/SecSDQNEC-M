#!/usr/bin/env python3
"""Compare the two-tier autoencoder with standard ML anomaly detectors on the SAME ns-3 windows.

Every detector uses the same two-tier layout (path model at the controller, UAV and server
models at the middleware), is trained on normal training windows only, and gets its alarm
threshold from the 99th percentile of held-out normal windows, exactly like the autoencoder.

Usage:  python3 detector_ml.py <results_dir_with_det_csv> <output_dir>
"""
import sys, os
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

R = sys.argv[1] if len(sys.argv) > 1 else "results_v3"
OUT = sys.argv[2] if len(sys.argv) > 2 else R
os.makedirs(OUT, exist_ok=True)
NF = {0: 6, 1: 4, 2: 5}


def read(fn):
    rows = []
    for line in open(fn):
        v = line.strip().split(",")
        if len(v) < 8:
            continue
        e = int(v[3])
        if len(v) < 7 + NF[e]:
            continue
        rows.append((e, int(v[5]), np.array(v[7:7 + NF[e]], dtype=float)))
    return rows


tr, te = read(f"{R}/det_train.csv"), read(f"{R}/det_test.csv")
DET = {
    "Isolation Forest": lambda: IsolationForest(n_estimators=200, random_state=7),
    "One-Class SVM": lambda: OneClassSVM(kernel="rbf", gamma="scale", nu=0.05),
    "LOF": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
}
rng = np.random.default_rng(7)
lines = ["detector,accuracy,precision,recall,f1,fpr,auc,tp,fp,fn,tn,recall_ddos,recall_fake,recall_comp"]
for name, make in DET.items():
    models = {}
    for e in range(3):
        X = np.array([f for (et, lab, f) in tr if et == e and lab == 0])
        idx = rng.permutation(len(X))
        k = len(X) * 7 // 10
        sc = StandardScaler().fit(X[idx[:k]])
        m = make().fit(sc.transform(X[idx[:k]]))
        val = -m.score_samples(sc.transform(X[idx[k:]]))   # higher = more anomalous
        models[e] = (sc, m, np.percentile(val, 99.0), np.median(val))
    tp = fp = fn = tn = 0
    pos = [0, 0, 0]
    hit = [0, 0, 0]
    ys, ss = [], []
    for (e, lab, f) in te:
        sc, m, thr, med = models[e]
        s = -m.score_samples(sc.transform(f.reshape(1, -1)))[0]
        pred = s > thr
        ys.append(lab)
        ss.append((s - med) / max(1e-12, thr - med))   # comparable across the three tiers
        if lab:
            pos[e] += 1
            hit[e] += pred
            tp += pred
            fn += not pred
        else:
            fp += pred
            tn += not pred
    acc = (tp + tn) / (tp + tn + fp + fn)
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    auc = roc_auc_score(ys, ss)
    lines.append(",".join(str(x) for x in [name, acc, p, r, f1, fp / max(1, fp + tn), auc, tp, fp, fn, tn,
                                            hit[0] / max(1, pos[0]), hit[1] / max(1, pos[1]),
                                            hit[2] / max(1, pos[2])]))
    print(f"{name:18s} acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f} AUC={auc:.4f}")
open(f"{OUT}/detector_ml.csv", "w").write("\n".join(lines) + "\n")
