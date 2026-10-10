"""VitalSign - Flask application (all routes live here).

Run with:  python app/app.py     ->  http://127.0.0.1:5000
Heavy ML work lives in src/; this file only wires data to templates.
"""
import json
import sys
import threading
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src import model_registry as registry  # noqa: E402
from src import predict as predictor  # noqa: E402
from src.config import APP_TITLE, MODELS_DIR, TARGET_LABELS  # noqa: E402
from src.data.load_data import DatasetError, dataset_summary, load_dataset  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(APP_DIR / "templates"), static_folder=str(APP_DIR / "static"))
app.config["JSON_SORT_KEYS"] = False
ROWS_PER_PAGE = 25
REQUIRED = ["results.json", "meta.json", "eda_report.json", "preprocess_report.json", "best_model.pkl"]

# ------------------------------------------------------------------ helpers
_JSON = {}
_SUMMARY = {}
PIPE = {"state": "idle", "progress": 0, "message": "", "error": None}


def artifacts_ready():
    return all((MODELS_DIR / f).exists() for f in REQUIRED)


def read_json(name):
    path = MODELS_DIR / name
    if not path.exists():
        return None
    key = (name, path.stat().st_mtime)
    if key not in _JSON:
        with open(path, encoding="utf-8") as f:
            _JSON[key] = json.load(f)
    return _JSON[key]


def get_summary():
    df = load_dataset()
    if _SUMMARY.get("id") != id(df):
        _SUMMARY.update(id=id(df), data=dataset_summary(df))
    return df, _SUMMARY["data"]


def results_by_slug():
    res = read_json("results.json") or {}
    out = {r["slug"]: r for r in res.get("supervised", [])}
    out.update(res.get("unsupervised", {}))
    return res, out


def error_page(title, message, code=500, retrain=False):
    return render_template("error.html", title=title, message=message, retrain=retrain, active=""), code


@app.context_processor
def inject_globals():
    return {"app_title": APP_TITLE, "pipeline_ready": artifacts_ready(), "pipe": PIPE,
            "target_labels": TARGET_LABELS}


@app.template_filter("num")
def fmt_num(v):
    try:
        return f"{int(v):,}"
    except (TypeError, ValueError):
        return v


@app.template_filter("pct")
def fmt_pct(v, digits=1):
    try:
        return f"{100 * float(v):.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


@app.template_filter("score")
def fmt_score(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "-"


def need_artifacts():
    return None if artifacts_ready() else error_page(
        "Analysis outputs not found",
        "The models and charts have not been generated yet. Click the button to run the full pipeline "
        "(takes about 2-4 minutes), or run  python main.py --train  in a terminal.", 200, retrain=True)


# ------------------------------------------------------------- pipeline run
def _worker():
    from src.pipeline import run_pipeline
    PIPE.update(state="running", progress=1, message="Starting...", error=None)
    try:
        run_pipeline(lambda p, m: PIPE.update(progress=p, message=m))
        _JSON.clear()
        PIPE.update(state="done", progress=100, message="Finished")
    except Exception as exc:
        traceback.print_exc()
        PIPE.update(state="error", error=f"The pipeline failed: {exc}")


@app.route("/pipeline/run", methods=["POST"])
def pipeline_run():
    if PIPE["state"] != "running":
        threading.Thread(target=_worker, daemon=True).start()
        PIPE.update(state="running", progress=1, message="Starting...", error=None)
    return jsonify(PIPE)


@app.route("/pipeline/status")
def pipeline_status():
    return jsonify({**PIPE, "ready": artifacts_ready()})


# ------------------------------------------------------------------- routes
@app.route("/")
def dashboard():
    try:
        df, s = get_summary()
    except DatasetError as exc:
        return error_page("Dataset problem", str(exc))
    res, _ = results_by_slug()
    eda_rep = read_json("eda_report.json")
    meta = read_json("meta.json")
    models = sorted(res.get("supervised", []), key=lambda r: -r["roc_auc"])
    num_stats = df[s["numeric"]].agg(["mean", "median", "std", "min", "max"]).T.round(2)
    return render_template("home.html", active="dashboard", s=s, res=res, models=models, eda=eda_rep,
                           num_stats=num_stats.reset_index().to_dict("records"), meta=meta)


@app.route("/dataset")
def dataset():
    try:
        df, s = get_summary()
    except DatasetError as exc:
        return error_page("Dataset problem", str(exc))
    view = request.args.get("view", "all")
    data = {"head": df.head(ROWS_PER_PAGE), "tail": df.tail(ROWS_PER_PAGE)}.get(view)
    pages = 1
    page = 1
    if data is None:
        view = "all"
        pages = (len(df) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE
        page = min(max(request.args.get("page", 1, type=int), 1), pages)
        data = df.iloc[(page - 1) * ROWS_PER_PAGE: page * ROWS_PER_PAGE]
    start = (page - 1) * ROWS_PER_PAGE if view == "all" else (0 if view == "head" else len(df) - len(data))
    rows = [["" if v != v else v for v in r] for r in data.itertuples(index=False)]
    return render_template("load_dataset.html", active="dataset", s=s, columns=list(df.columns), rows=rows,
                           view=view, page=page, pages=pages, start=start)


@app.route("/eda")
def eda_page():
    if (r := need_artifacts()):
        return r
    eda_rep = read_json("eda_report.json")
    sections = {}
    for c in eda_rep["charts"]:
        sections.setdefault(c["section"], []).append(c)
    return render_template("eda.html", active="eda", eda=eda_rep, sections=sections)


@app.route("/preprocessing")
def preprocessing():
    if (r := need_artifacts()):
        return r
    return render_template("Preprocessing.html", active="preprocessing", p=read_json("preprocess_report.json"))


@app.route("/models")
def models():
    if (r := need_artifacts()):
        return r
    _, by_slug = results_by_slug()
    cards = [{**registry.get_info(s), "result": by_slug.get(s)} for s in registry.ALL]
    return render_template("models.html", active="models", cards=cards, best=read_json("results.json")["best_model"])


@app.route("/models/<slug>")
def model_detail(slug):
    if slug not in registry.ALL:
        return error_page("Model not found", f"There is no model called '{slug}'.", 404)
    if (r := need_artifacts()):
        return r
    res, by_slug = results_by_slug()
    info = registry.get_info(slug)
    idx = registry.ALL.index(slug)
    return render_template("model_detail.html", active="models", info=info, r=by_slug.get(slug),
                           error=res.get("errors", {}).get(slug), best=res["best_model"],
                           prev=registry.get_info(registry.ALL[idx - 1]) if idx else None,
                           nxt=registry.get_info(registry.ALL[idx + 1]) if idx < len(registry.ALL) - 1 else None,
                           elbow=(by_slug.get("kmeans") or {}).get("elbow"))


@app.route("/evaluation")
def evaluation():
    if (r := need_artifacts()):
        return r
    res, by_slug = results_by_slug()
    sup = res["supervised"]
    slug = request.args.get("model", res["best_model"])
    r = by_slug.get(slug) if slug in [m["slug"] for m in sup] else by_slug[res["best_model"]]
    return render_template("evaluation.html", active="evaluation", models=sup, r=r, best=res["best_model"],
                           errors=res.get("errors", {}))


@app.route("/comparison")
def comparison():
    if (r := need_artifacts()):
        return r
    res, _ = results_by_slug()
    ranked = sorted(res["supervised"], key=lambda r: (-round(r["roc_auc"], 4), -r["f1"]))
    return render_template("comparison.html", active="comparison", ranked=ranked, res=res,
                           km=res["unsupervised"].get("kmeans"), db=res["unsupervised"].get("dbscan"))


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if (r := need_artifacts()):
        return r
    meta = read_json("meta.json")
    res, _ = results_by_slug()
    fields = meta["form_fields"]
    options = [{"slug": "best", "name": f"Best model ({next(m['name'] for m in res['supervised'] if m['slug'] == res['best_model'])})"}]
    options += [{"slug": m["slug"], "name": m["name"]} for m in res["supervised"]]
    values, errors, result, chosen = predictor.defaults(fields), {}, None, "best"
    if request.method == "POST":
        chosen = request.form.get("model", "best")
        if chosen not in [o["slug"] for o in options]:
            chosen = "best"
        values, errors = predictor.validate_form(request.form, fields)
        if not errors:
            try:
                pipe = predictor.load_pipeline(chosen)
                result = predictor.predict(pipe, values, fields, meta["features"])
                used = res["best_model"] if chosen == "best" else chosen
                result["model_name"] = next(m["name"] for m in res["supervised"] if m["slug"] == used)
                result["auc"] = next(m["roc_auc"] for m in res["supervised"] if m["slug"] == used)
                labels = {f["name"]: {o["value"]: o["label"] for o in f.get("options", [])} for f in fields}
                result["summary"] = [(f["label"], labels[f["name"]].get(values[f["name"]], values[f["name"]]))
                                     for f in fields if f["group"] in ("Patient", "Admission", "Hospital utilisation")]
            except Exception:
                traceback.print_exc()
                errors["_model"] = ("The saved model could not be loaded (it may have been created with a "
                                    "different scikit-learn version). Re-run the pipeline: python main.py --train")
    groups = {}
    for f in fields:
        groups.setdefault(f["group"], {"open": f["open"], "fields": []})["fields"].append(f)
    return render_template("prediction.html", active="prediction", groups=groups, values=values, errors=errors,
                           result=result, options=options, chosen=chosen)


@app.errorhandler(404)
def not_found(_):
    return error_page("Page not found", "The page you requested does not exist.", 404)


@app.errorhandler(500)
def server_error(_):
    return error_page("Something went wrong", "An unexpected error occurred. Please try again.", 500)


def run():
    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    run()
