"""Cleaning + leakage-safe preprocessing for the Vital Sign dataset.

Workflow (no data leakage):
    load -> clean (stateless rules only) -> split X / y -> train/test split
         -> FIT imputer/scaler/encoder on TRAIN only -> transform train & test
The fitted ColumnTransformer is saved and re-used by the prediction page.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (AGE_COL, AGE_MAP, AGE_NUM, CODE_COLS, CODE_LABELS, EXCLUDED_DISCHARGE,
                        FEATURE_LABELS, FORM_GROUPS, ID_COLS, MISSING_DROP_THRESHOLD, MODELS_DIR,
                        NON_FEATURE_COLS, RANDOM_STATE, TARGET, TARGET_LABELS, TARGET_RAW,
                        TEST_SIZE)


# ----------------------------------------------------------- stateless cleaning
def icd9_group(code):
    """Map an ICD-9 diagnosis code to a broad clinical group (fixed lookup rules)."""
    if pd.isna(code):
        return "Unknown"
    c = str(code)
    if c.startswith("250"):
        return "Diabetes"
    if c[0] in "VE":
        return "Other"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    if 390 <= v <= 459 or v == 785:
        return "Circulatory"
    if 460 <= v <= 519 or v == 786:
        return "Respiratory"
    if 520 <= v <= 579 or v == 787:
        return "Digestive"
    if 580 <= v <= 629 or v == 788:
        return "Genitourinary"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 800 <= v <= 999:
        return "Injury"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def clean_data(df):
    """Apply rule-based cleaning that does not learn anything from the data.

    Returns (clean_df, report). `clean_df` still contains the age band, raw
    target and binary target for EDA; model features are chosen by
    `get_feature_columns`.
    """
    steps = []

    def log(name, detail, before, after):
        steps.append({"step": name, "detail": detail, "rows_before": int(before),
                      "rows_after": int(after), "removed": int(before - after)})

    d = df.copy()
    raw_shape = df.shape
    raw_missing = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    n = len(d)
    d = d.drop_duplicates()
    log("Remove duplicate rows", "Exact duplicate records", n, len(d))

    if "gender" in d:
        n = len(d)
        d = d[d["gender"] != "Unknown/Invalid"]
        log("Remove invalid gender", "Rows with gender = 'Unknown/Invalid'", n, len(d))

    if "discharge_disposition_id" in d:
        n = len(d)
        d = d[~d["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE)]
        log("Remove expired / hospice discharges",
            "These patients cannot be readmitted, so they would bias the target", n, len(d))

    if "patient_nbr" in d:
        n = len(d)
        d = d.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
        log("Keep first encounter per patient",
            "Prevents the same patient appearing in both train and test sets (leakage)", n, len(d))

    d[TARGET] = (d[TARGET_RAW] != "NO").astype(int)

    dropped = []
    for c in ID_COLS:
        if c in d:
            dropped.append((c, "identifier - not a clinical feature"))
    for c in d.columns:
        if c in ID_COLS or c in (TARGET, TARGET_RAW):
            continue
        frac = d[c].isna().mean()
        if frac > MISSING_DROP_THRESHOLD:
            dropped.append((c, f"{frac:.0%} missing values"))
        elif d[c].nunique(dropna=True) <= 1:
            dropped.append((c, "constant column (single value)"))
    d = d.drop(columns=[c for c, _ in dropped])

    if AGE_COL in d:
        d[AGE_NUM] = d[AGE_COL].map(AGE_MAP)
    for c in ("diag_1", "diag_2", "diag_3"):
        if c in d:
            d[c] = d[c].map(icd9_group)
    for c in CODE_COLS:
        if c in d:
            d[c] = d[c].astype(str)

    feats = get_feature_columns(d)
    cat_missing_before_fill = {c: int(d[c].isna().sum()) for c in feats if d[c].isna().sum() > 0}
    for c in feats:
        if not pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].fillna("Unknown")   # "unknown" is itself informative
    d = d.reset_index(drop=True)

    report = {
        "original_shape": list(raw_shape),
        "cleaned_shape": list(d.shape),
        "steps": steps,
        "dropped_columns": [{"column": c, "reason": r} for c, r in dropped],
        "missing_before": raw_missing,
        "missing_before_total": int(sum(raw_missing.values())),
        "filled_unknown": cat_missing_before_fill,
        "missing_after_cleaning": int(d[feats].isna().sum().sum()),
        "derived": [
            f"{AGE_NUM}: numeric mid-point of the age band (e.g. [50-60) -> 55)",
            "diag_1 / diag_2 / diag_3: ICD-9 codes grouped into broad clinical groups",
            f"{TARGET}: 1 if readmitted (<30 or >30 days), 0 if not readmitted",
            "admission_type_id / discharge_disposition_id / admission_source_id: "
            "treated as categories, not numbers",
        ],
    }
    return d, report


def get_feature_columns(d):
    return [c for c in d.columns if c not in NON_FEATURE_COLS and c not in ID_COLS]


# --------------------------------------------------------------- pipeline objs
def build_preprocessor(num_cols, cat_cols):
    """ColumnTransformer: impute+scale numerics, impute+one-hot categoricals."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        # rare categories (<1 % of training rows) are grouped together
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def pretty_encoded(name, cat_cols):
    base = name.split("__", 1)[-1]
    if name.startswith("cat__"):
        for c in sorted(cat_cols, key=len, reverse=True):
            if base.startswith(c + "_"):
                return f"{c} = {base[len(c) + 1:]}"
    return base


def prepare_data(df_raw):
    """Full leakage-safe preparation. Returns a dict used by all model modules."""
    d, report = clean_data(df_raw)
    feats = get_feature_columns(d)
    X, y = d[feats], d[TARGET]
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in feats if c not in num_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    pre = build_preprocessor(num_cols, cat_cols)
    X_train_t = pre.fit_transform(X_train)      # fit on TRAIN only
    X_test_t = pre.transform(X_test)            # test is only transformed
    names = [pretty_encoded(n, cat_cols) if n.startswith("cat__") else n
             for n in pre.get_feature_names_out()]

    scaled = pd.DataFrame(X_train_t[:, :len(num_cols)], columns=num_cols)
    n_enc = len(names) - len(num_cols)
    report.update({
        "features": feats, "numeric_features": num_cols, "categorical_features": cat_cols,
        "n_encoded_features": int(n_enc), "n_final_features": int(len(names)),
        "scaled_check": [{"feature": c, "mean": round(float(scaled[c].mean()), 4),
                          "std": round(float(scaled[c].std(ddof=0)), 4)} for c in num_cols],
        "train_size": int(len(X_train)), "test_size": int(len(X_test)),
        "test_fraction": TEST_SIZE,
        "train_balance": {TARGET_LABELS[k]: int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_balance": {TARGET_LABELS[k]: int(v) for k, v in y_test.value_counts().sort_index().items()},
        "encoded_cardinality": [
            {"feature": c, "categories_after_encoding": int(len(cats))}
            for c, cats in zip(cat_cols, pre.named_transformers_["cat"]["onehot"].categories_)],
        "sample_before": _records(df_raw[[c for c in ["race", "age", "time_in_hospital", "num_medications",
                                                      "diag_1", "insulin", "readmitted"] if c in df_raw]].head(5)),
        "sample_after": _records(pd.DataFrame(X_train_t[:5, :10], columns=names[:10]).round(3)),
    })
    return {"df_clean": d, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "preprocessor": pre, "X_train_t": X_train_t, "X_test_t": X_test_t,
            "feature_names": names, "num_cols": num_cols, "cat_cols": cat_cols, "report": report}


def _records(frame):
    return {"columns": [str(c) for c in frame.columns],
            "rows": [[("" if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for v in r]
                     for r in frame.itertuples(index=False)]}


# ------------------------------------------------------------ prediction form
def build_form_fields(X_train):
    """Describe every model input so the prediction form is generated from the data."""
    fields = []
    group_of = {c: (g, open_) for g, cols, open_ in FORM_GROUPS for c in cols}
    for c in X_train.columns:
        label = FEATURE_LABELS.get(c, c.replace("_", " ").replace("-", " + ").capitalize())
        group, open_ = group_of.get(c, ("Other", False))
        f = {"name": c, "label": label, "group": group, "open": open_}
        if c == AGE_NUM:
            inv = {v: k for k, v in AGE_MAP.items()}
            vals = sorted(int(v) for v in X_train[c].dropna().unique())
            f.update(kind="select", numeric=True, default=str(int(X_train[c].median() // 10 * 10 + 5)),
                     options=[{"value": str(v), "label": inv.get(v, str(v))} for v in vals])
            if f["default"] not in [o["value"] for o in f["options"]]:
                f["default"] = f["options"][len(f["options"]) // 2]["value"]
        elif pd.api.types.is_numeric_dtype(X_train[c]):
            f.update(kind="number", min=float(X_train[c].min()), max=float(X_train[c].max()),
                     default=int(round(float(X_train[c].median()))))
        else:
            counts = X_train[c].value_counts()
            labels = CODE_LABELS.get(c, {})
            f.update(kind="select", numeric=False, default=str(counts.index[0]),
                     options=[{"value": str(v), "label": f"{v} - {labels[str(v)]}" if str(v) in labels else str(v)}
                              for v in counts.index])
        fields.append(f)
    return fields


def save_preprocessor(pre, path=None):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pre, path or MODELS_DIR / "preprocessing_pipeline.pkl", compress=3)
