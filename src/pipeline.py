"""End-to-end training pipeline: EDA -> preprocessing -> models -> comparison -> saved artifacts."""
import json
import shutil
import traceback

import numpy as np

from src import model_registry as registry
from src.config import CHARTS_DIR, MODELS_DIR, TARGET, TARGET_LABELS
from src.data import eda
from src.data.load_data import dataset_summary, load_dataset
from src.data.preprocess import build_form_fields, prepare_data, save_preprocessor
from src.ml_utils import comparison_charts


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _dump(obj, name):
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_Encoder, indent=1)


def run_pipeline(progress=None):
    """Run everything. `progress(pct, message)` is optional (used by the web UI)."""
    def step(pct, msg):
        print(f"[{pct:3d}%] {msg}")
        if progress:
            progress(pct, msg)

    MODELS_DIR.mkdir(exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    step(2, "Loading dataset")
    raw = load_dataset(use_cache=False)

    step(6, "Cleaning data, splitting train/test and fitting preprocessing on the training set")
    data = prepare_data(raw)
    save_preprocessor(data["preprocessor"])
    _dump(data["report"], "preprocess_report.json")

    step(12, "Exploratory data analysis (charts)")
    stats = eda.compute_stats(raw, data["df_clean"], data["num_cols"])
    stats["charts"] = eda.generate_charts(raw, data["df_clean"], data["num_cols"], stats)
    _dump(stats, "eda_report.json")

    summary = dataset_summary(raw)
    meta = {"target": TARGET, "target_labels": TARGET_LABELS, "features": data["report"]["features"],
            "numeric_features": data["num_cols"], "categorical_features": data["cat_cols"],
            "form_fields": build_form_fields(data["X_train"]), "dataset": summary["name"]}

    results = {"supervised": [], "unsupervised": {}, "errors": {}}
    sup = registry.SUPERVISED
    for i, slug in enumerate(sup):
        info = registry.get_info(slug)
        step(20 + int(55 * i / len(sup)), f"Training {info['name']}")
        try:
            results["supervised"].append(registry.load_module(slug).run(data))
        except Exception as exc:  # one failing model must not stop the pipeline
            traceback.print_exc()
            results["errors"][slug] = str(exc)

    if not results["supervised"]:
        raise RuntimeError("No model could be trained - see the console for details.")

    step(78, "Comparing models and saving the best one")
    sup_res = results["supervised"]
    best = max(sup_res, key=lambda r: (round(r["roc_auc"], 4), r["f1"]))
    results["best_model"] = best["slug"]
    results["best_criterion"] = "ROC-AUC (ties broken by F1)"
    shutil.copy(MODELS_DIR / f"{best['slug']}.pkl", MODELS_DIR / "best_model.pkl")
    comparison_charts(sup_res)

    step(84, "K-Means elbow analysis and clustering")
    try:
        elbow = registry.load_module("kmeans_elbow").run(data)
        results["unsupervised"]["kmeans"] = registry.load_module("kmeans").run(data, elbow)
    except Exception as exc:
        traceback.print_exc()
        results["errors"]["kmeans"] = str(exc)
    step(94, "DBSCAN clustering")
    try:
        results["unsupervised"]["dbscan"] = registry.load_module("dbscan").run(data)
    except Exception as exc:
        traceback.print_exc()
        results["errors"]["dbscan"] = str(exc)

    meta["best_model"] = best["slug"]
    _dump(meta, "meta.json")
    _dump(results, "results.json")
    step(100, "Done")
    return results
