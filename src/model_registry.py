"""Loads the model modules from the 'src/ML models' folder (folder name contains a space,
so the modules are loaded by file path instead of a normal import)."""
import importlib.util

from src.config import ML_MODELS_DIR

# (slug, file) in display order; kmeans_elbow is a helper used by kmeans
SUPERVISED = ["logistic_regression", "decision_tree", "random_forest", "gradient_boost", "adaboost", "knn", "svm"]
UNSUPERVISED = ["kmeans", "dbscan"]
ALL = SUPERVISED + UNSUPERVISED
_MODULES = {}


def load_module(slug):
    if slug not in ALL + ["kmeans_elbow"]:
        raise KeyError(slug)
    if slug not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"vs_model_{slug}", ML_MODELS_DIR / f"{slug}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[slug] = mod
    return _MODULES[slug]


def get_info(slug):
    return load_module(slug).INFO


def all_info():
    return [get_info(s) for s in ALL]
