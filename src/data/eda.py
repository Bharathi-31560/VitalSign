"""Exploratory data analysis: statistics + charts generated from the real dataset."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import AGE_COL, CHARTS_DIR, TARGET, TARGET_LABELS, TARGET_RAW

BLUE, RED, TEAL = "#2f80ed", "#eb5757", "#19b5a5"
PALETTE = {0: BLUE, 1: RED}
sns.set_theme(style="whitegrid", font_scale=0.95)


def _save(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / name, dpi=105, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _legend_names(ax):
    leg = ax.get_legend()
    if leg:
        leg.set_title(None)
        for t, k in zip(leg.get_texts(), (0, 1)):
            t.set_text(TARGET_LABELS[k])
            t.set_fontsize(8)


def compute_stats(raw, clean, num_cols):
    """Tabular EDA numbers used by eda.html."""
    desc = clean[num_cols].describe().T
    outliers = []
    for c in num_cols:
        q1, q3 = clean[c].quantile([.25, .75])
        iqr = q3 - q1
        n_out = int(((clean[c] < q1 - 1.5 * iqr) | (clean[c] > q3 + 1.5 * iqr)).sum())
        outliers.append({"feature": c, "outliers": n_out, "pct": round(100 * n_out / len(clean), 1),
                         "skew": round(float(clean[c].skew()), 2)})
    corr = clean[num_cols + [TARGET]].corr()
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            pairs.append({"a": a, "b": b, "r": round(float(corr.loc[a, b]), 3)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    with_target = sorted(({"feature": c, "r": round(float(corr.loc[c, TARGET]), 3)} for c in num_cols),
                         key=lambda p: -abs(p["r"]))
    cats = [c for c in clean.columns if c not in num_cols and c not in (TARGET, TARGET_RAW, AGE_COL)]
    unique_tbl = [{"feature": c, "unique": int(clean[c].nunique()),
                   "top": str(clean[c].value_counts().index[0]),
                   "top_pct": round(100 * float(clean[c].value_counts(normalize=True).iloc[0]), 1)}
                  for c in cats]
    rate = float(clean[TARGET].mean())
    return {
        "describe": [{"feature": i, **{k: round(float(v), 2) for k, v in r.items()}} for i, r in desc.iterrows()],
        "outliers": outliers, "top_pairs": pairs[:6], "target_corr": with_target,
        "categorical_unique": unique_tbl,
        "class_distribution": {TARGET_LABELS[int(k)]: int(v) for k, v in clean[TARGET].value_counts().sort_index().items()},
        "class_distribution_raw": {str(k): int(v) for k, v in raw[TARGET_RAW].value_counts().items()},
        "readmit_rate": round(100 * rate, 1),
        "imbalance_ratio": round(float(clean[TARGET].value_counts().max() / clean[TARGET].value_counts().min()), 2),
        "rows": int(len(clean)),
    }


def generate_charts(raw, clean, num_cols, stats):
    """Create all EDA charts; a failing chart is skipped (never crashes the pipeline)."""
    charts = []
    rate = stats["readmit_rate"]

    def add(fn, file, title, section, desc):
        try:
            fn()
            charts.append({"file": file, "title": title, "section": section, "desc": desc})
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[EDA] chart {file} skipped: {exc}")

    key_num = [c for c in ["time_in_hospital", "num_medications", "num_lab_procedures", "number_diagnoses"] if c in clean]

    def target_dist():
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        raw_counts = raw[TARGET_RAW].value_counts()
        ax[0].bar(raw_counts.index, raw_counts.values, color=[BLUE, "#f2a93b", RED][:len(raw_counts)])
        ax[0].set_title("Original target: readmitted", fontweight="bold")
        for i, v in enumerate(raw_counts.values):
            ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        cnt = clean[TARGET].value_counts().sort_index()
        ax[1].bar([TARGET_LABELS[i] for i in cnt.index], cnt.values, color=[BLUE, RED])
        ax[1].set_title("Binary target used for modelling (cleaned data)", fontweight="bold")
        for i, v in enumerate(cnt.values):
            ax[1].text(i, v, f"{v:,} ({100 * v / cnt.sum():.1f}%)", ha="center", va="bottom", fontsize=9)
        for a in ax:
            a.set_ylabel("Patients")
        _save(fig, "target_distribution.png")
    add(target_dist, "target_distribution.png", "Target / class distribution", "Target",
        f"Left: the three original outcome values. Right: the binary target used for modelling. {rate}% of the cleaned patients were readmitted, so the classes are moderately imbalanced (ratio {stats['imbalance_ratio']}:1) - accuracy alone is not enough, which is why precision, recall, F1 and ROC-AUC are reported.")

    def missing():
        m = (raw.isna().mean() * 100)
        m = m[m > 0].sort_values()
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.barh(m.index, m.values, color=TEAL)
        ax.set_xlabel("% missing")
        ax.set_title("Missing values per column (raw data)", fontweight="bold")
        for i, v in enumerate(m.values):
            ax.text(v + .5, i, f"{v:.1f}%", va="center", fontsize=9)
        _save(fig, "missing_values.png")
    add(missing, "missing_values.png", "Missing values", "Data quality",
        "Only a few columns contain missing values. weight is almost entirely missing and is dropped; payer_code and medical_specialty are partly missing and are kept with an explicit 'Unknown' category.")

    def hist():
        cols = num_cols[:9]
        fig, axes = plt.subplots(3, 3, figsize=(12, 9))
        for ax, c in zip(axes.ravel(), cols):
            sns.histplot(clean[c], bins=min(30, max(clean[c].nunique(), 5)), color=BLUE, ax=ax)
            ax.set_title(c, fontweight="bold")
            ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, "feature_histogram.png")
    add(hist, "feature_histogram.png", "Histograms of numeric features", "Distributions",
        "Shape of every numeric feature. Most utilisation counts (prior outpatient / emergency / inpatient visits, procedures) are heavily right-skewed: most patients have 0 or 1, a few have many. time_in_hospital and num_medications are right-skewed too.")

    def dist():
        fig, axes = plt.subplots(1, len(key_num), figsize=(4.2 * len(key_num), 3.8))
        for ax, c in zip(np.atleast_1d(axes), key_num):
            for k in (0, 1):
                sns.kdeplot(clean.loc[clean[TARGET] == k, c], ax=ax, color=PALETTE[k], fill=True, alpha=.25,
                            label=TARGET_LABELS[k], warn_singular=False)
            ax.set_title(c, fontweight="bold")
            ax.set_xlabel("")
        np.atleast_1d(axes)[0].legend()
        fig.tight_layout()
        _save(fig, "distribution_plots.png")
    add(dist, "distribution_plots.png", "Distribution plots by outcome", "Distributions",
        "Density curves of key clinical numbers split by outcome. Large overlap between the two curves means the single feature separates the groups only weakly, which explains why readmission is a hard prediction problem.")

    def box():
        cols = num_cols[:8]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7))
        for ax, c in zip(axes.ravel(), cols):
            sns.boxplot(data=clean, x=TARGET, y=c, hue=TARGET, palette=PALETTE, legend=False, ax=ax,
                        fliersize=2)
            ax.set_xticks([0, 1], ["No", "Yes"])
            ax.set_xlabel("Readmitted")
            ax.set_title(c, fontweight="bold")
        fig.tight_layout()
        _save(fig, "boxplot.png")
    top_out = max(stats["outliers"], key=lambda o: o["pct"])
    add(box, "boxplot.png", "Box plots by outcome (outliers)", "Outliers",
        f"Box plots show the median, quartiles and outliers (dots beyond 1.5 x IQR). {top_out['feature']} has the largest share of outliers ({top_out['pct']}%). Outliers are real clinical cases (very long stays, many visits), so they are kept and handled by scaling and tree-based models rather than deleted.")

    def corr():
        c = clean[num_cols + [TARGET]].corr()
        fig, ax = plt.subplots(figsize=(8.5, 7))
        sns.heatmap(c, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .8}, ax=ax, annot_kws={"size": 8})
        ax.set_title("Correlation matrix", fontweight="bold")
        _save(fig, "correlation_matrix.png")
    p = stats["top_pairs"][0]
    t = stats["target_corr"][0]
    add(corr, "correlation_matrix.png", "Correlation heatmap", "Relationships",
        f"Pearson correlation between numeric features and the target. The strongest feature-feature correlation is {p['a']} vs {p['b']} (r = {p['r']}). The numeric feature most correlated with readmission is {t['feature']} (r = {t['r']}) - correlations with the target are all weak, so models must combine many features.")

    def scatter():
        s = clean.sample(min(4000, len(clean)), random_state=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for ax, (x, y) in zip(axes, [("time_in_hospital", "num_medications"), ("num_lab_procedures", "num_medications")]):
            sns.scatterplot(data=s, x=x, y=y, hue=TARGET, palette=PALETTE, alpha=.45, s=18, ax=ax)
            _legend_names(ax)
            ax.set_title(f"{y} vs {x}", fontweight="bold")
        fig.tight_layout()
        _save(fig, "scatter_plot.png")
    add(scatter, "scatter_plot.png", "Scatter plots", "Relationships",
        "A random sample of 4,000 patients. Longer stays go together with more medications and lab procedures, but readmitted (red) and not-readmitted (blue) patients are heavily mixed - no simple boundary exists.")

    def age():
        order = sorted(clean[AGE_COL].unique(), key=lambda a: int(a[1:].split("-")[0]))
        g = clean.groupby(AGE_COL)[TARGET].mean().reindex(order) * 100
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.countplot(data=clean, x=AGE_COL, hue=TARGET, order=order, palette=PALETTE, ax=axes[0])
        _legend_names(axes[0])
        axes[0].tick_params(axis="x", rotation=40)
        axes[0].set_title("Patients per age band", fontweight="bold")
        axes[1].plot(order, g.values, "o-", color=RED, lw=2)
        axes[1].set_ylabel("% readmitted")
        axes[1].tick_params(axis="x", rotation=40)
        axes[1].set_title("Readmission rate per age band", fontweight="bold")
        fig.tight_layout()
        _save(fig, "age_readmission.png")
    add(age, "age_readmission.png", "Age and readmission", "Categorical features",
        "Most patients are between 50 and 90 years old. The line shows how the percentage of readmitted patients changes with age band.")

    def counts():
        cols = [c for c in ["gender", "race", "diag_1", "A1Cresult", "insulin", "change", "diabetesMed",
                            "admission_type_id"] if c in clean]
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, c in zip(axes.ravel(), cols):
            order = clean[c].value_counts().index[:8]
            sns.countplot(data=clean[clean[c].isin(order)], x=c, hue=TARGET, order=order, palette=PALETTE, ax=ax)
            _legend_names(ax)
            ax.tick_params(axis="x", rotation=35)
            ax.set_title(c, fontweight="bold")
            ax.set_xlabel("")
        fig.tight_layout()
        _save(fig, "count_plots.png")
    add(counts, "count_plots.png", "Count plots of categorical features", "Categorical features",
        "Count of patients in the most frequent categories of eight categorical features, coloured by outcome. Compare the blue/red ratio between bars: categories where the red bar is relatively taller have a higher share of readmitted patients.")

    def prior():
        d = clean.copy()
        d["prior_inpatient"] = d["number_inpatient"].clip(upper=5)
        g = d.groupby("prior_inpatient")[TARGET].agg(["mean", "size"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(g.index.astype(int).astype(str).tolist()[:-1] + ["5+"], g["mean"] * 100, color=RED)
        for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(i, m * 100 + .8, f"n={n:,}", ha="center", fontsize=8)
        ax.set_xlabel("Inpatient visits in the previous year")
        ax.set_ylabel("% readmitted")
        ax.set_title("Readmission rate vs prior inpatient visits", fontweight="bold")
        _save(fig, "prior_visits.png")
    gm = clean.groupby(clean["number_inpatient"].clip(upper=5))[TARGET].mean() * 100
    add(prior, "prior_visits.png", "Readmission rate vs prior inpatient visits", "Target",
        f"The readmission rate rises from {gm.iloc[0]:.0f}% for patients with no prior inpatient stay to {gm.iloc[-1]:.0f}% for patients with five or more - prior hospital use is one of the clearest risk signals.")

    def pair():
        s = clean[key_num + [TARGET]].sample(min(1500, len(clean)), random_state=2)
        g = sns.pairplot(s, hue=TARGET, palette=PALETTE, corner=True, plot_kws={"alpha": .4, "s": 14},
                         diag_kind="kde")
        g.figure.suptitle("Pairwise relationships (1,500-patient sample)", y=1.02, fontweight="bold")
        for t, k in zip(g._legend.get_texts(), (0, 1)):
            t.set_text(TARGET_LABELS[k])
        g.savefig(CHARTS_DIR / "pairplot.png", dpi=90, bbox_inches="tight")
        plt.close("all")
    add(pair, "pairplot.png", "Pairwise relationships", "Relationships",
        "Every combination of four key numeric features for a 1,500-patient sample. Diagonal: distribution per outcome. Off-diagonal: scatter plots. Considerable overlap again confirms that no pair of features alone separates the outcomes.")
    return charts
