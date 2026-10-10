"""Input validation and prediction (uses the SAME saved pipeline as training)."""
import joblib
import pandas as pd

from src.config import MODELS_DIR, TARGET_LABELS

_CACHE = {}


def load_pipeline(slug):
    path = MODELS_DIR / ("best_model.pkl" if slug == "best" else f"{slug}.pkl")
    if not path.exists():
        raise FileNotFoundError(f"Model file {path.name} not found.")
    key = (str(path), path.stat().st_mtime)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


def defaults(fields):
    return {f["name"]: str(f["default"]) for f in fields}


def validate_form(form, fields):
    """Return (values, errors). `values` keeps the raw strings for re-display."""
    values, errors = {}, {}
    for f in fields:
        raw = (form.get(f["name"]) or "").strip()
        values[f["name"]] = raw
        if raw == "":
            errors[f["name"]] = f"{f['label']} is required."
        elif f["kind"] == "number":
            try:
                v = float(raw)
            except ValueError:
                errors[f["name"]] = f"{f['label']} must be a number."
                continue
            if v != int(v) or v < 0:
                errors[f["name"]] = f"{f['label']} must be a whole number of 0 or more."
            elif v < f["min"] or v > f["max"]:
                errors[f["name"]] = (f"{f['label']} must be between {int(f['min'])} and {int(f['max'])} "
                                     "(the range seen in the training data).")
        elif raw not in [o["value"] for o in f["options"]]:
            errors[f["name"]] = f"Choose a valid option for {f['label']}."
    return values, errors


def predict(pipeline, values, fields, feature_order):
    row = {}
    for f in fields:
        v = values[f["name"]]
        row[f["name"]] = float(v) if f["kind"] == "number" or f.get("numeric") else v
    X = pd.DataFrame([row])[feature_order]
    pred = int(pipeline.predict(X)[0])
    prob = float(pipeline.predict_proba(X)[0][1]) if hasattr(pipeline, "predict_proba") else None
    return {"prediction": pred, "label": TARGET_LABELS[pred], "probability": prob}
