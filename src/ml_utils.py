"""Shared helpers for all model modules: training, evaluation, charts, saving."""
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CHARTS_DIR, MODELS_DIR, RANDOM_STATE, TARGET_LABELS

BLUE, RED, TEAL, GREY = "#2f80ed", "#eb5757", "#19b5a5", "#8a97a8"


def display_name(name):
    return name[5:] if name.startswith("num__") else name


def evaluate_classifier(estimator, X_test, y_test):
    """Compute every metric from real predictions on the held-out test set."""
    y_pred = estimator.predict(X_test)
    proba = estimator.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    rep = classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0,
                                target_names=[TARGET_LABELS[0], TARGET_LABELS[1]])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr[keep]],
                      "tpr": [round(float(x), 4) for x in tpr[keep]]},
        "per_class": {k: {m: float(v[m]) for m in ("precision", "recall", "f1-score", "support")}
                      for k, v in rep.items() if k in (TARGET_LABELS[0], TARGET_LABELS[1])},
        "n_test": int(len(y_test)),
    }


def top_importances(estimator, feature_names, kind, top=15):
    """Return [(feature, importance)] for tree models ('tree') or linear models ('coef')."""
    if kind == "coef":
        vals = np.asarray(estimator.coef_).ravel()
    elif kind == "tree":
        vals = np.asarray(estimator.feature_importances_)
    else:
        return []
    order = np.argsort(np.abs(vals))[::-1][:top]
    return [{"feature": display_name(feature_names[i]), "importance": float(vals[i])} for i in order]


def train_and_evaluate(info, estimator, data, max_train_samples=None, importance=None):
    """Train `estimator` on the preprocessed training data, evaluate on the test set,
    save a full (preprocessing + model) pipeline to models/<slug>.pkl and return metrics."""
    slug, name = info["slug"], info["name"]
    X, y = data["X_train_t"], data["y_train"].to_numpy()
    if max_train_samples and len(X) > max_train_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_train_samples, stratify=y,
                                      random_state=RANDOM_STATE)
    t0 = time.time()
    estimator.fit(X, y)
    train_time = time.time() - t0

    res = evaluate_classifier(estimator, data["X_test_t"], data["y_test"])
    head = slice(0, min(len(X), 10000))
    res["train_accuracy"] = float(accuracy_score(y[head], estimator.predict(X[head])))
    res.update({
        "slug": slug, "name": name, "kind": "supervised",
        "train_time_sec": round(train_time, 1),
        "n_train_used": int(len(X)), "n_train_total": int(len(data["X_train_t"])),
        "n_features": int(X.shape[1]),
        "params": {k: str(v) for k, v in estimator.get_params().items()
                   if k in info.get("show_params", [])},
        "importances": top_importances(estimator, data["feature_names"], importance),
        "importance_kind": importance,
    })

    MODELS_DIR.mkdir(exist_ok=True)
    pipe = Pipeline([("preprocess", data["preprocessor"]), ("model", estimator)])
    joblib.dump(pipe, MODELS_DIR / f"{slug}.pkl", compress=3)

    confusion_chart(slug, name, res["confusion_matrix"])
    roc_chart(slug, name, res)
    if res["importances"]:
        importance_chart(slug, name, res["importances"], importance)
    return res


# ---------------------------------------------------------------- charts ----
def _save(fig, filename):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / filename, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def confusion_chart(slug, name, cm):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [TARGET_LABELS[0], TARGET_LABELS[1]]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix - {name}", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1b2f4b")
    _save(fig, f"confusion_matrix_{slug}.png")


def roc_chart(slug, name, res):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(res["roc_curve"]["fpr"], res["roc_curve"]["tpr"], color=BLUE, lw=2.2,
            label=f"AUC = {res['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(f"ROC curve - {name}", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=.3)
    _save(fig, f"roc_{slug}.png")


def importance_chart(slug, name, items, kind):
    items = items[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    vals = [i["importance"] for i in items]
    colors = [BLUE if v >= 0 else RED for v in vals]
    ax.barh([i["feature"] for i in items], vals, color=colors)
    ax.set_title(f"{'Coefficients' if kind == 'coef' else 'Feature importance'} - {name}",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=.3)
    _save(fig, f"importance_{slug}.png")


def comparison_charts(results):
    """Bar chart of all metrics + combined ROC curves for the supervised models."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    colors = [BLUE, TEAL, "#f2a93b", RED, "#7b61c4"]
    names = [r["name"] for r in results]
    x = np.arange(len(results))
    w = 0.16
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, lab, col) in enumerate(zip(metrics, labels, colors)):
        ax.bar(x + (i - 2) * w, [r[m] for r in results], w, label=lab, color=col)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the held-out test set", fontsize=13, fontweight="bold")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -.28), frameon=False)
    ax.grid(axis="y", alpha=.3)
    _save(fig, "model_comparison.png")

    fig, ax = plt.subplots(figsize=(7, 5.2))
    for r in results:
        ax.plot(r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], lw=2, label=f"{r['name']} ({r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curves - all models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3)
    _save(fig, "roc_curves.png")
