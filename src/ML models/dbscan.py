"""DBSCAN density-based clustering of patients (unsupervised)."""
import importlib.util
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from src.config import MODELS_DIR, RANDOM_STATE
from src.ml_utils import BLUE, RED, _save

N_SAMPLE = 10000   # DBSCAN is O(n^2) in the worst case -> work on a random sample

INFO = {
    "slug": "dbscan", "name": "DBSCAN",
    "type": "Unsupervised - Density-based clustering", "family": "unsupervised",
    "summary": "Finds dense groups of similar patients and flags unusual patients as noise - no k required.",
    "vital_use": "Detects atypical patients (very long stays, extreme medication counts, many prior visits) as noise points.",
    "how": [
        "DBSCAN looks at the neighbourhood of each patient. eps is the radius of that neighbourhood and min_samples is how many patients must lie inside it for the area to count as dense. Patients in dense regions are grouped into clusters; patients that do not belong to any dense region are labelled noise (-1).",
        "eps is chosen from the k-distance plot: every patient's distance to its min_samples-th neighbour is sorted, and the 'knee' of that curve is a sensible radius. Too small an eps labels almost everything noise, too large merges everything into one cluster."],
    "why": [
        "Hospital data contains unusual encounters - extremely long stays, dozens of medications, many emergency visits. DBSCAN does not force these patients into a cluster; it reports them as noise, which makes it a simple outlier detector for clinical review.",
        "Because DBSCAN does not need k, it complements K-Means. We also compare the readmission rate of the noise points with the clustered patients."],
    "strengths": ["No need to choose k", "Finds arbitrary-shaped clusters", "Identifies outliers"],
    "limits": ["Sensitive to eps and min_samples", "Struggles when densities differ", "No predict() for new patients"],
    "show_params": [],
}


def _knee(values):
    x = np.linspace(0, 1, len(values))
    y = (values - values.min()) / (values.max() - values.min() + 1e-12)
    return int(np.argmax(x - y))


def _numeric(data):
    spec = importlib.util.spec_from_file_location("km_elbow", Path(__file__).with_name("kmeans_elbow.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.numeric_matrix(data)[0]


def run(data):
    X_all = _numeric(data)
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_all), min(N_SAMPLE, len(X_all)), replace=False)
    X = X_all[idx]
    y = data["y_train"].to_numpy()[idx]
    min_samples = 2 * X.shape[1]

    dist, _ = NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)
    kdist = np.sort(dist[:, -1])
    eps = float(np.clip(kdist[_knee(kdist)], np.percentile(kdist, 80), np.percentile(kdist, 98)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(kdist, color=BLUE, lw=2)
    ax.axhline(eps, color=RED, ls="--", label=f"eps = {eps:.2f}")
    ax.set(xlabel="Patients (sorted)", ylabel=f"Distance to {min_samples}th nearest neighbour")
    ax.set_title("k-distance plot (choosing eps)", fontweight="bold")
    ax.legend()
    ax.grid(alpha=.3)
    _save(fig, "dbscan_kdistance.png")

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = db.labels_
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise = labels == -1

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    P = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(P[noise, 0], P[noise, 1], c="#b0b8c4", s=9, alpha=.7, label=f"Noise ({noise.sum():,})")
    cmap = plt.get_cmap("tab10")
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], color=cmap(c % 10), s=7, alpha=.65, label=f"Cluster {c} ({m.sum():,})")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.0f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.0f}% variance)")
    ax.set_title("DBSCAN result (PCA projection)", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25)
    _save(fig, "dbscan_clusters.png")

    groups = [{"label": "Noise points", "size": int(noise.sum()),
               "pct": round(100 * float(noise.mean()), 1),
               "readmit_rate": round(100 * float(y[noise].mean()), 1) if noise.any() else None}]
    for c in sorted(set(labels) - {-1}):
        m = labels == c
        groups.append({"label": f"Cluster {c}", "size": int(m.sum()), "pct": round(100 * float(m.mean()), 1),
                       "readmit_rate": round(100 * float(y[m].mean()), 1)})
    joblib.dump(db, MODELS_DIR / "dbscan.pkl", compress=3)
    return {"slug": INFO["slug"], "name": INFO["name"], "kind": "unsupervised", "eps": eps,
            "min_samples": int(min_samples), "n_clusters": n_clusters, "n_noise": int(noise.sum()),
            "noise_pct": round(100 * float(noise.mean()), 1), "n_samples": int(len(X)), "groups": groups,
            "params": {"eps": f"{eps:.3f}", "min_samples": str(min_samples),
                       "features": ", ".join(data["num_cols"])},
            "overall_readmit_rate": round(100 * float(y.mean()), 1)}
