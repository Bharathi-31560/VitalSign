"""K-Means elbow analysis: inertia and silhouette for k = 2..10."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import RANDOM_STATE
from src.ml_utils import BLUE, RED, TEAL, _save

K_RANGE = range(2, 11)


def numeric_matrix(data):
    """Scaled numeric columns only (clustering uses the vital/utilisation numbers)."""
    idx = [i for i, n in enumerate(data["feature_names"]) if n.startswith("num__")]
    return data["X_train_t"][:, idx], idx


def knee_point(xs, ys, decreasing=True):
    """Point farthest from the straight line joining the curve's end points."""
    x = (np.array(xs) - min(xs)) / (max(xs) - min(xs))
    y = (np.array(ys) - min(ys)) / (max(ys) - min(ys))
    dist = ((1 - x) - y) if decreasing else (x - y)
    return int(np.argmax(dist))


def run(data):
    X, _ = numeric_matrix(data)
    rng = np.random.RandomState(RANDOM_STATE)
    sil_idx = rng.choice(len(X), min(6000, len(X)), replace=False)
    inertia, sil = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE).fit(X)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(X[sil_idx], km.labels_[sil_idx])))
    ks = list(K_RANGE)
    best = knee_point(ks, inertia)
    chosen_k = ks[best]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ks, inertia, "o-", color=BLUE, lw=2)
    axes[0].axvline(chosen_k, color=RED, ls="--", label=f"Elbow at k = {chosen_k}")
    axes[0].set(xlabel="Number of clusters (k)", ylabel="Inertia (within-cluster sum of squares)")
    axes[0].set_title("Elbow method", fontweight="bold")
    axes[0].legend()
    axes[1].plot(ks, sil, "o-", color=TEAL, lw=2)
    axes[1].axvline(chosen_k, color=RED, ls="--")
    axes[1].set(xlabel="Number of clusters (k)", ylabel="Silhouette score")
    axes[1].set_title("Silhouette score", fontweight="bold")
    for a in axes:
        a.grid(alpha=.3)
    _save(fig, "kmeans_elbow.png")
    return {"k_values": ks, "inertia": inertia, "silhouette": sil, "chosen_k": chosen_k,
            "chosen_inertia": inertia[best], "chosen_silhouette": sil[best],
            "n_samples": int(len(X)), "n_dims": int(X.shape[1])}
