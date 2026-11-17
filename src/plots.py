"""Figures for the report and website. Every plot is drawn from real data
passed in by the pipeline; nothing is hard-coded."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

INK, MUTED, GRID = "#17202E", "#5B6676", "#DDE2E8"
ACCENT, POS, NEG, AMBER, VIOLET = "#B3202A", "#1F7A6D", "#8A94A6", "#C98A16", "#5B4B9A"
MODEL_COLORS = {"Majority baseline": NEG, "Popularity baseline": AMBER,
                "Logistic Regression": ACCENT, "Random Forest": VIOLET,
                "LR (movie features only)": "#7A9CC6"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.png")
    plt.close(fig)


# ------------------------------------------------------------------ EDA
def rating_distribution(r):
    counts = r["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    colors = [NEG if s <= 3 else POS for s in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors)
    for x, v in zip(counts.index.astype(str), counts.values):
        ax.text(x, v, f"{v:,}\n({v / len(r):.1%})", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set(xlabel="Star rating", ylabel="Number of ratings",
           title="Rating distribution (teal = positive class, 4-5 stars)")
    ax.set_ylim(0, counts.max() * 1.22)
    _save(fig, "eda_rating_distribution")


def ratings_per_user_movie(r):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    pu = r.groupby("user_id").size()
    axes[0].hist(pu, bins=np.logspace(np.log10(20), np.log10(pu.max()), 30),
                 color=POS)
    axes[0].set_xscale("log")
    axes[0].set(xlabel="Ratings per user (log scale)", ylabel="Users",
                title=f"Ratings per user (min {pu.min()}, median {int(pu.median())}, max {pu.max()})")
    pm = r.groupby("movie_id").size().sort_values(ascending=False).to_numpy()
    axes[1].plot(np.arange(1, len(pm) + 1), pm, color=ACCENT, lw=1.6)
    axes[1].set_yscale("log")
    top = pm[: int(len(pm) * 0.1)].sum() / pm.sum()
    axes[1].set(xlabel="Movie rank by number of ratings", ylabel="Ratings (log scale)",
                title=f"Long tail: top 10% of movies hold {top:.0%} of ratings")
    _save(fig, "eda_ratings_per_user_movie")


def genre_positive_fraction(df, movies):
    d = df.merge(movies[["movie_id"] + C.GENRES], on="movie_id")
    rows = [(g, int(d[g].sum()), d.loc[d[g] == 1, "target"].mean())
            for g in C.GENRES if d[g].sum() > 0]
    t = pd.DataFrame(rows, columns=["genre", "n", "pos"]).sort_values("pos")
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh(t["genre"], t["pos"], color=[POS if p >= df.target.mean() else NEG for p in t["pos"]])
    ax.axvline(df["target"].mean(), color=INK, ls="--", lw=1)
    ax.text(df["target"].mean(), len(t) - 0.4, " overall", fontsize=8, color=INK)
    for i, (p, n) in enumerate(zip(t["pos"], t["n"])):
        ax.text(0.76, i, f"{p:.0%}  (n={n:,})", va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set(xlabel="Share of ratings that are 4-5 stars",
           title="Positive-rating fraction by genre (all 100K ratings)")
    _save(fig, "eda_genre_positive_fraction")
    return t.sort_values("pos", ascending=False)


def ratings_over_time(split_df, info):
    s = split_df.set_index("datetime").resample("W")["rating"].count()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(s.index, s.values, width=6, color=NEG)
    tv = pd.to_datetime(info["t_val_cutoff"], unit="s", utc=True)
    tt = pd.to_datetime(info["t_test_cutoff"], unit="s", utc=True)
    ymax = s.max() * 1.1
    ax.axvspan(s.index.min() - pd.Timedelta(days=7), tv, color=POS, alpha=0.08)
    ax.axvspan(tv, tt, color=AMBER, alpha=0.15)
    ax.axvspan(tt, s.index.max() + pd.Timedelta(days=7), color=ACCENT, alpha=0.12)
    for x, lab in [(s.index.min(), "train 80%"), (tv, "val 10%"), (tt, "test 10%")]:
        ax.text(x, ymax * 0.95, " " + lab, fontsize=8.5, color=INK, va="top")
    ax.set_ylim(0, ymax)
    ax.set(ylabel="Ratings per week", title="Ratings over time with the chronological split")
    _save(fig, "eda_ratings_over_time")


def history_coverage(meta_fixed, meta_seq):
    labels = ["Validation", "Test"]
    fixed = [(meta_fixed[meta_fixed.split == s].user_hist_count == 0).mean() for s in ["val", "test"]]
    seq = [(meta_seq[meta_seq.split == s].user_hist_count == 0).mean() for s in ["val", "test"]]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(2)
    ax.bar(x - 0.18, fixed, 0.36, color=ACCENT, label="Fixed snapshot (primary)")
    ax.bar(x + 0.18, seq, 0.36, color=POS, label="Sequential (secondary)")
    for xi, v in zip(list(x - 0.18) + list(x + 0.18), fixed + seq):
        ax.text(xi, v + 0.01, f"{v:.1%}", ha="center", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set(ylabel="Share of rows", title="Rows whose user has NO earlier history")
    ax.legend(loc="upper center", fontsize=8, ncol=2)
    _save(fig, "eda_cold_start_coverage")


# ------------------------------------------------------------------ results
def pr_curves(y, preds, name="results_pr_curves", title="Precision-recall curves (test set)"):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for m, p in preds.items():
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=MODEL_COLORS.get(m, INK), lw=1.8,
                label=f"{m} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(y.mean(), color=MUTED, ls=":", lw=1)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0.4, 1.0), title=title)
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, name)


def confusion_matrices(metrics: dict, name="results_confusion_matrices"):
    names = list(metrics)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2))
    for ax, m in zip(np.atleast_1d(axes), names):
        c = metrics[m]["confusion"]
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=mat.sum())
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if v > mat.sum() * 0.45 else INK)
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.grid(False)
        ax.set_title(f"{m}\n(threshold {metrics[m]['threshold']:.3f})", fontsize=9)
    _save(fig, name)


def metric_bars(metrics: dict, name="results_metric_comparison"):
    keys = ["accuracy", "precision", "recall", "f1", "pr_auc"]
    names = list(metrics)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.8 / len(names)
    x = np.arange(len(keys))
    for i, m in enumerate(names):
        vals = [metrics[m][k] for k in keys]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, color=MODEL_COLORS.get(m, INK), label=m)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "PR-AUC"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=len(names), loc="upper center", bbox_to_anchor=(0.5, 1.15))
    ax.set_ylabel("Test score")
    _save(fig, name)


def lr_coefficients(model, columns):
    coef = model.named_steps["clf"].coef_[0]
    s = pd.Series(coef, index=columns).sort_values()
    s = pd.concat([s.head(8), s.tail(10)])
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.barh(s.index, s.values, color=[POS if v > 0 else NEG for v in s.values])
    ax.axvline(0, color=INK, lw=0.8)
    ax.set(xlabel="Coefficient (standardised features)",
           title="Logistic Regression: largest coefficients")
    _save(fig, "results_lr_coefficients")
    return pd.Series(coef, index=columns)


def subgroup_f1(rows, models, name="results_f1_by_user_history",
                title="Test F1 by amount of user history (fixed snapshot)"):
    groups = [r["group"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(models)
    x = np.arange(len(groups))
    for i, m in enumerate(models):
        ax.bar(x + i * w - 0.4 + w / 2, [r[f"{m}|f1"] for r in rows], w,
               color=MODEL_COLORS.get(m, INK), label=m)
    ax.set_xticks(x, [f"{g}\n(n={r['n']:,})" for g, r in zip(groups, rows)])
    ax.set_ylim(0, 1)
    ax.set(ylabel="F1", title=title, xlabel="User's ratings available before prediction")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, name)


def calibration(y, preds):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":")
    for m, p in preds.items():
        fy, fx = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(fx, fy, marker="o", ms=3, color=MODEL_COLORS.get(m, INK), label=m)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate",
           title="Calibration (test, 10 quantile bins)")
    ax.legend(fontsize=8)
    _save(fig, "results_calibration")


def threshold_curve(y_va, p_va, chosen):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(y_va, p_va)
    f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-12, None)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(th, pr[:-1], color=POS, label="Precision")
    ax.plot(th, rc[:-1], color=AMBER, label="Recall")
    ax.plot(th, f1, color=ACCENT, lw=2, label="F1")
    ax.axvline(chosen, color=INK, ls="--", lw=1)
    ax.text(chosen, 0.05, f" chosen {chosen:.3f}", fontsize=8)
    ax.set(xlabel="Probability threshold", ylim=(0, 1.02),
           title="Logistic Regression threshold selection (validation only)")
    ax.legend(fontsize=8, loc="center left")
    _save(fig, "results_threshold_selection_val")
