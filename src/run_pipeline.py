"""Run the full MovieMatch experiment end to end.

    python -m src.run_pipeline

Steps: load + verify data -> quality checks -> target -> chronological split
-> leakage-safe features -> baselines + models (tuned on validation) ->
thresholds (validation) -> ONE evaluation on test -> subgroup, ranking and
error analysis -> secondary analyses -> figures -> JSON for the website.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import evaluate as E
from . import features as F
from . import load_data as L
from . import plots as PL
from . import preprocessing as P
from . import train as T


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 6)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def main():
    np.random.seed(C.RANDOM_SEED)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    R = {"config": {
        "positive_rule": f"rating >= {C.POSITIVE_MIN_RATING}",
        "split": f"chronological {C.TRAIN_FRAC:.0%}/{C.VAL_FRAC:.0%}/{1 - C.TRAIN_FRAC - C.VAL_FRAC:.0%}",
        "smoothing": {"movie": C.MOVIE_PRIOR_STRENGTH, "user": C.USER_PRIOR_STRENGTH,
                      "user_genre": C.USER_GENRE_PRIOR_STRENGTH},
        "random_seed": C.RANDOM_SEED}}

    # ------------------------------------------------------------ 1. data
    log("Loading and verifying data")
    L.download_dataset()
    ratings, movies = L.load_ratings(), L.load_movies()
    R["dataset"] = L.verify_counts(ratings, movies)
    R["quality"] = P.quality_report(ratings, movies)
    df = P.add_target(ratings)
    n_users, n_movies = R["dataset"]["n_users"], R["dataset"]["n_movies_in_catalogue"]
    pu, pm = df.groupby("user_id").size(), df.groupby("movie_id").size()
    R["dataset"].update({
        "first_rating_utc": str(df.datetime.min()), "last_rating_utc": str(df.datetime.max()),
        "positive_rate": round(float(df.target.mean()), 4),
        "rating_counts": {int(k): int(v) for k, v in df.rating.value_counts().sort_index().items()},
        "sparsity": round(1 - len(df) / (n_users * n_movies), 4),
        "ratings_per_user": {"min": int(pu.min()), "median": float(pu.median()), "max": int(pu.max())},
        "ratings_per_movie": {"min": int(pm.min()), "median": float(pm.median()), "max": int(pm.max())},
        "movies_with_lt_5_ratings": int((pm < 5).sum()),
        "top10pct_movies_share": round(float(pm.sort_values(ascending=False).iloc[:len(pm) // 10].sum() / len(df)), 4),
    })

    split_df, split_info = P.chronological_split(df)
    R["split"] = split_info

    log("EDA figures")
    PL.rating_distribution(df)
    PL.ratings_per_user_movie(df)
    genre_tab = PL.genre_positive_fraction(df, movies)
    PL.ratings_over_time(split_df, split_info)
    R["eda_genres"] = genre_tab.round(4).to_dict(orient="records")
    weekly = split_df.set_index("datetime").resample("W")["rating"].count()
    R["eda_weekly"] = [{"week": str(k.date()), "n": int(v)} for k, v in weekly.items()]

    # ------------------------------------------------------------ 2. features
    log("Building leakage-safe features")
    X, y, meta, priors = F.build_features(split_df, movies, train_mode="expanding")
    Xs, ys, meta_s = F.build_sequential_features(split_df, movies)
    R["priors"] = priors
    R["features"] = list(X.columns)
    PL.history_coverage(meta, meta_s)
    tr, va, te = (meta.split == s for s in ["train", "val", "test"])
    R["cold_start"] = {
        proto: {s: {"rows_user_cold": round(float((m[m.split == s].user_hist_count == 0).mean()), 4),
                    "rows_movie_cold": round(float((m[m.split == s].movie_hist_count == 0).mean()), 4),
                    "users_cold": int(m[(m.split == s) & (m.user_hist_count == 0)].user_id.nunique()),
                    "users_total": int(m[m.split == s].user_id.nunique())}
                for s in ["val", "test"]}
        for proto, m in [("fixed_snapshot", meta), ("sequential", meta_s)]}

    # ------------------------------------------------------------ 3. models
    log("Fitting baselines")
    models = {
        "Majority baseline": T.fit_majority(X[tr], y[tr]),
        "Popularity baseline": T.PopularityBaseline().fit(X[tr], y[tr]),
    }
    R["majority_class"] = int(models["Majority baseline"].classes_[
        np.argmax(models["Majority baseline"].class_prior_)])

    log("Tuning Logistic Regression on validation")
    lr, lr_c, lr_log = T.tune(T.make_logreg, T.LOGREG_GRID, X[tr], y[tr], X[va], y[va])
    models["Logistic Regression"] = lr

    log("Tuning Random Forest on validation")
    rf, rf_leaf, rf_log = T.tune(T.make_forest, T.FOREST_GRID, X[tr], y[tr], X[va], y[va])
    models["Random Forest"] = rf

    log("Ablation: LR with movie features only (non-personalised)")
    mcols = T.movie_only_columns(X.columns)
    lr_m, lr_m_c, lr_m_log = T.tune(T.make_logreg, T.LOGREG_GRID,
                                    X.loc[tr, mcols], y[tr], X.loc[va, mcols], y[va])
    R["tuning"] = {"Logistic Regression": {"chosen_C": lr_c, "log": lr_log},
                   "Random Forest": {"chosen_min_samples_leaf": rf_leaf, "log": rf_log,
                                     "n_estimators": 300, "max_features": "sqrt"},
                   "LR (movie features only)": {"chosen_C": lr_m_c, "log": lr_m_log,
                                                "features": mcols}}

    def predict(name, XX):
        if name == "LR (movie features only)":
            return pd.Series(lr_m.predict_proba(XX[mcols])[:, 1], index=XX.index)
        return pd.Series(models[name].predict_proba(XX)[:, 1], index=XX.index)

    names = list(models) + ["LR (movie features only)"]

    # ------------------------------------------------------------ 4. thresholds (validation)
    log("Choosing thresholds on validation")
    thresholds, val_metrics = {}, {}
    for n in names:
        p = predict(n, X[va])
        if n == "Majority baseline":
            thresholds[n] = 0.5        # constant score; predicts the majority class
        else:
            thresholds[n], _ = T.choose_threshold(y[va], p)
        val_metrics[n] = E.classification_metrics(y[va], p, thresholds[n])
    PL.threshold_curve(y[va], predict("Logistic Regression", X[va]), thresholds["Logistic Regression"])
    R["validation"] = val_metrics
    R["thresholds"] = thresholds
    R["model_selection"] = {"criterion": "validation PR-AUC",
                            "best_trained_model": max(["Logistic Regression", "Random Forest"],
                                                      key=lambda n: val_metrics[n]["pr_auc"]),
                            "primary_model_per_guide": "Logistic Regression"}

    # ------------------------------------------------------------ 5. TEST (once)
    log("Final evaluation on the test split (primary: fixed snapshot)")
    test_p = {n: predict(n, X[te]) for n in names}
    yt = y[te]
    test_metrics = {n: E.classification_metrics(yt, test_p[n], thresholds[n]) for n in names}
    R["test"] = test_metrics
    main4 = ["Majority baseline", "Popularity baseline", "Logistic Regression", "Random Forest"]
    PL.pr_curves(yt, {n: test_p[n] for n in names[1:]})
    PL.confusion_matrices({n: test_metrics[n] for n in main4})
    PL.metric_bars({n: test_metrics[n] for n in main4})
    PL.calibration(yt, {n: test_p[n] for n in ["Popularity baseline", "Logistic Regression", "Random Forest"]})
    coefs = PL.lr_coefficients(lr, list(X.columns))
    R["lr_coefficients"] = coefs.sort_values(key=abs, ascending=False).round(4).to_dict()

    mt = meta[te]
    ub = E.bucketize(mt.user_hist_count, E.USER_BUCKETS)
    mb = E.bucketize(mt.movie_hist_count, E.MOVIE_BUCKETS)
    sub_models = ["Popularity baseline", "Logistic Regression", "Random Forest"]
    order_u = [b[2] for b in E.USER_BUCKETS]
    order_m = [b[2] for b in E.MOVIE_BUCKETS]
    su = sorted(E.subgroup_table(mt, yt, {n: test_p[n] for n in sub_models}, thresholds, ub),
                key=lambda r: order_u.index(r["group"]))
    sm = sorted(E.subgroup_table(mt, yt, {n: test_p[n] for n in sub_models}, thresholds, mb),
                key=lambda r: order_m.index(r["group"]))
    R["test_by_user_history"], R["test_by_movie_history"] = su, sm
    PL.subgroup_f1(su, sub_models)
    warm = mt.user_hist_count > 0
    R["test_warm_users_only"] = {n: E.classification_metrics(yt[warm], test_p[n][warm], thresholds[n])
                                 for n in names}
    R["ranking_test"] = E.ranking_metrics(mt, yt, {n: test_p[n] for n in sub_models})

    # genre-level error rates for LR
    tg = mt.merge(movies[["movie_id"] + C.GENRES], on="movie_id", how="left")
    lr_hat = (test_p["Logistic Regression"].to_numpy() >= thresholds["Logistic Regression"]).astype(int)
    tg["err"] = (lr_hat != yt.to_numpy()).astype(int)
    tg["fp"] = ((lr_hat == 1) & (yt.to_numpy() == 0)).astype(int)
    tg["fn"] = ((lr_hat == 0) & (yt.to_numpy() == 1)).astype(int)
    R["test_errors_by_genre"] = sorted([
        {"genre": g, "n": int(tg[g].sum()),
         "error_rate": round(float(tg.loc[tg[g] == 1, "err"].mean()), 4),
         "fp_rate_of_rows": round(float(tg.loc[tg[g] == 1, "fp"].mean()), 4),
         "fn_rate_of_rows": round(float(tg.loc[tg[g] == 1, "fn"].mean()), 4)}
        for g in C.GENRES if tg[g].sum() >= 30], key=lambda r: -r["error_rate"])

    # ------------------------------------------------------------ 6. error examples
    log("Selecting error examples")
    ex = mt.copy()
    ex["p"] = test_p["Logistic Regression"].values
    ex["y"] = yt.values
    ex["title"] = ex.movie_id.map(movies.set_index("movie_id")["title"])
    ex["movie_pos_rate"] = X.loc[te, "movie_pos_rate"].values
    ex["user_genre_affinity"] = X.loc[te, "user_genre_affinity"].values
    ex["user_pos_rate"] = X.loc[te, "user_pos_rate"].values
    thr = thresholds["Logistic Regression"]
    fp = ex[(ex.p >= thr) & (ex.y == 0)]
    fn = ex[(ex.p < thr) & (ex.y == 1)]

    def pick(d, ascending):
        warm_d = d[d.user_hist_count > 0].sort_values("p", ascending=ascending).head(4)
        cold_d = d[d.user_hist_count == 0].sort_values("p", ascending=ascending).head(3)
        return pd.concat([warm_d, cold_d])

    cols = ["user_id", "title", "rating", "p", "user_hist_count", "movie_hist_count",
            "movie_pos_rate", "user_pos_rate", "user_genre_affinity"]
    R["error_examples"] = {
        "false_positives": pick(fp, False)[cols].round(3).to_dict(orient="records"),
        "false_negatives": pick(fn, True)[cols].round(3).to_dict(orient="records"),
        "fp_summary": {"n": int(len(fp)), "share_cold_user": round(float((fp.user_hist_count == 0).mean()), 4),
                       "mean_rating": round(float(fp.rating.mean()), 3),
                       "share_rated_3": round(float((fp.rating == 3).mean()), 4)},
        "fn_summary": {"n": int(len(fn)), "share_cold_user": round(float((fn.user_hist_count == 0).mean()), 4),
                       "mean_rating": round(float(fn.rating.mean()), 3),
                       "share_rated_4": round(float((fn.rating == 4).mean()), 4)},
    }
    R["lr_mean_score_by_true_rating"] = ex.groupby("rating")["p"].mean().round(4).to_dict()

    # ------------------------------------------------------------ 7. SECONDARY: sequential
    log("Secondary protocol: sequential history")
    thr_s, seq_val, seq_test, seq_p = {}, {}, {}, {}
    for n in names:
        pv = predict(n, Xs[va])
        thr_s[n] = 0.5 if n == "Majority baseline" else T.choose_threshold(ys[va], pv)[0]
        seq_val[n] = E.classification_metrics(ys[va], pv, thr_s[n])
        seq_p[n] = predict(n, Xs[te])
        seq_test[n] = E.classification_metrics(ys[te], seq_p[n], thr_s[n])
    mts = meta_s[te]
    ubs = E.bucketize(mts.user_hist_count, E.USER_BUCKETS)
    sus = sorted(E.subgroup_table(mts, ys[te], {n: seq_p[n] for n in sub_models}, thr_s, ubs),
                 key=lambda r: order_u.index(r["group"]))
    PL.subgroup_f1(sus, sub_models, name="results_f1_by_user_history_sequential",
                   title="Test F1 by amount of user history (sequential protocol)")
    PL.pr_curves(ys[te], {n: seq_p[n] for n in names[1:]}, name="results_pr_curves_sequential",
                 title="Precision-recall curves (test, sequential protocol)")
    R["sequential"] = {"thresholds": thr_s, "validation": seq_val, "test": seq_test,
                       "test_by_user_history": sus,
                       "ranking_test": E.ranking_metrics(mts, ys[te], {n: seq_p[n] for n in sub_models})}

    # ------------------------------------------------------------ 7b. uncertainty (cluster bootstrap)
    log("Bootstrap confidence intervals (resampling users; predictions fixed)")
    R["bootstrap_pr_auc_diff"] = {
        "fixed_snapshot": {
            "LR - Popularity": E.paired_bootstrap(mt, yt, test_p["Logistic Regression"], test_p["Popularity baseline"]),
            "LR - LR(movie only)": E.paired_bootstrap(mt, yt, test_p["Logistic Regression"], test_p["LR (movie features only)"]),
            "RF - LR": E.paired_bootstrap(mt, yt, test_p["Random Forest"], test_p["Logistic Regression"])},
        "sequential": {
            "LR - Popularity": E.paired_bootstrap(mts, ys[te], seq_p["Logistic Regression"], seq_p["Popularity baseline"]),
            "LR - LR(movie only)": E.paired_bootstrap(mts, ys[te], seq_p["Logistic Regression"], seq_p["LR (movie features only)"]),
            "RF - LR": E.paired_bootstrap(mts, ys[te], seq_p["Random Forest"], seq_p["Logistic Regression"])}}
    # MovieLens may not be redistributed, so this file holds NO user IDs, movie IDs,
    # ratings, labels or timestamps. Row i is the i-th test rating in the order
    # produced by preprocessing.chronological_split (sorted by timestamp, user_id,
    # movie_id); join it back to the downloaded data to verify (see notebook 03).
    preds_out = mt[["user_hist_count", "movie_hist_count"]].reset_index(drop=True).astype(int)
    preds_out.insert(0, "test_row", range(len(preds_out)))
    for n in names:
        preds_out[f"fixed|{n}"] = test_p[n].values
        preds_out[f"sequential|{n}"] = seq_p[n].values
    preds_out.to_csv(C.OUTPUT_DIR / "test_predictions.csv", index=False, float_format="%.17g")  # exact round-trip

    # ------------------------------------------------------------ 8. leakage demonstration (validation only)
    log("Leakage demonstration (validation only)")
    Xl, yl, _, _ = F.build_features(split_df, movies, train_mode="leaky")
    leaky = T.make_logreg(lr_c).fit(Xl[tr], yl[tr])
    from sklearn.metrics import average_precision_score as AP
    R["leakage_demo"] = {
        "description": "LR trained on training-row features that include each row's own rating "
                       "(naive snapshot) vs. the correct strictly-earlier features. Validation only.",
        "leaky": {"train_pr_auc": round(AP(yl[tr], leaky.predict_proba(Xl[tr])[:, 1]), 4),
                  "val_pr_auc": round(AP(y[va], leaky.predict_proba(X[va])[:, 1]), 4)},
        "correct": {"train_pr_auc": round(AP(y[tr], lr.predict_proba(X[tr])[:, 1]), 4),
                    "val_pr_auc": round(AP(y[va], lr.predict_proba(X[va])[:, 1]), 4)},
    }

    # ------------------------------------------------------------ 9. sensitivity: drop 3-star ratings
    log("Sensitivity analysis: excluding 3-star ratings")
    s3 = split_df[split_df.rating != 3].copy()
    X3, y3, m3, _ = F.build_features(s3, movies)
    tr3, va3, te3 = (m3.split == s for s in ["train", "val", "test"])
    lr3, c3, _ = T.tune(T.make_logreg, T.LOGREG_GRID, X3[tr3], y3[tr3], X3[va3], y3[va3])
    sens = {"rows": {s: int((m3.split == s).sum()) for s in ["train", "val", "test"]},
            "chosen_C": c3, "test": {}}
    for n, mod in [("Majority baseline", T.fit_majority(X3[tr3], y3[tr3])),
                   ("Popularity baseline", T.PopularityBaseline()), ("Logistic Regression", lr3)]:
        pv = mod.predict_proba(X3[va3])[:, 1]
        th = 0.5 if n == "Majority baseline" else T.choose_threshold(y3[va3], pv)[0]
        sens["test"][n] = E.classification_metrics(y3[te3], mod.predict_proba(X3[te3])[:, 1], th)
    R["sensitivity_exclude_3star"] = sens

    # ------------------------------------------------------------ 10. website demo data
    log("Preparing demonstration data for the website")
    train_rows = split_df[split_df.split == "train"]
    titles = movies.set_index("movie_id")["title"]
    test_rows = split_df[split_df.split == "test"]
    tcount = test_rows.groupby("user_id").size()
    trcount = train_rows.groupby("user_id").size()
    eligible = [u for u in tcount[tcount >= 10].index if trcount.get(u, 0) >= 20]
    rng = np.random.default_rng(C.RANDOM_SEED)
    sample = sorted(rng.choice(eligible, size=min(6, len(eligible)), replace=False).tolist())
    cold_users = [u for u in tcount[tcount >= 10].index if trcount.get(u, 0) == 0]
    if cold_users:
        sample.append(int(rng.choice(cold_users)))
    demo = []
    gen = movies.set_index("movie_id")[C.GENRES]
    for u in sample:
        seen_train = set(train_rows.loc[train_rows.user_id == u, "movie_id"])
        cand = pd.DataFrame({"user_id": u, "movie_id": [m for m in movies.movie_id if m not in seen_train]})
        Xc = F.snapshot_features_for_pairs(cand, split_df, movies)
        cand["lr"] = lr.predict_proba(Xc)[:, 1]
        cand["pop"] = Xc["movie_pos_rate"].values
        cand["movie_hist_count"] = np.expm1(Xc["movie_hist_count_log"].values).round().astype(int)
        later = test_rows[test_rows.user_id == u].set_index("movie_id")["target"]
        later_val = split_df[(split_df.split == "val") & (split_df.user_id == u)].set_index("movie_id")["target"]
        later = pd.concat([later_val, later])

        def item(r):
            return {"movie_id": int(r.movie_id), "title": titles[r.movie_id],
                    "genres": [g for g in C.GENRES if gen.loc[r.movie_id, g] == 1],
                    "lr": round(float(r.lr), 4), "pop": round(float(r["pop"]), 4),
                    "movie_hist_count": int(r.movie_hist_count),
                    "later": (int(later[r.movie_id]) if r.movie_id in later.index else None)}
        test_items = cand[cand.movie_id.isin(test_rows.loc[test_rows.user_id == u, "movie_id"])]
        demo.append({
            "user_id": int(u),
            "profile": F.snapshot_user_profile(train_rows, movies, u, priors),
            "n_test_items": int(len(test_items)),
            "n_test_items_positive": int(sum(1 for m in test_items.movie_id if later.get(m) == 1)),
            # only the 25 highest-scored are exported, to keep shared examples small
            "test_items": [item(r) for _, r in test_items.sort_values("lr", ascending=False).head(25).iterrows()],
            "top_lr": [item(r) for _, r in cand.sort_values("lr", ascending=False).head(10).iterrows()],
            "top_pop": [item(r) for _, r in cand.sort_values("pop", ascending=False).head(10).iterrows()],
        })

    # threshold sweep + PR curve points for website interactions (test, primary)
    from sklearn.metrics import precision_recall_curve
    sweep = []
    pl = test_p["Logistic Regression"].to_numpy()
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        m_ = E.classification_metrics(yt, pl, t)
        sweep.append({"t": float(t), "precision": m_["precision"], "recall": m_["recall"],
                      "f1": m_["f1"], "accuracy": m_["accuracy"], "confusion": m_["confusion"]})
    curves = {}
    for n in names[1:]:
        pr, rc, _ = precision_recall_curve(yt, test_p[n])
        idx = np.unique(np.linspace(0, len(pr) - 1, 120).astype(int))
        curves[n] = [[round(float(rc[i]), 4), round(float(pr[i]), 4)] for i in idx]

    web = {"generated": time.strftime("%Y-%m-%d"),
           **{k: R[k] for k in ["dataset", "quality", "split", "cold_start", "validation", "test",
                                "thresholds", "tuning", "test_by_user_history", "test_by_movie_history",
                                "test_warm_users_only", "ranking_test", "sequential", "leakage_demo",
                                "sensitivity_exclude_3star", "error_examples", "eda_genres",
                                "test_errors_by_genre", "lr_coefficients", "priors", "features",
                                "model_selection", "lr_mean_score_by_true_rating", "bootstrap_pr_auc_diff",
                                "eda_weekly", "config"]},
           "threshold_sweep_test_lr": sweep, "pr_curves_test": curves, "demo_users": demo}

    # ------------------------------------------------------------ 11. save
    log("Saving outputs")
    (C.OUTPUT_DIR / "metrics.json").write_text(json.dumps(jsonable(R), indent=2))
    pd.DataFrame(R["test"]).T.drop(columns=["confusion"]).to_csv(C.OUTPUT_DIR / "test_metrics.csv")
    pd.DataFrame(R["validation"]).T.drop(columns=["confusion"]).to_csv(C.OUTPUT_DIR / "validation_metrics.csv")
    C.WEBSITE_DIR.joinpath("data").mkdir(parents=True, exist_ok=True)
    payload = json.dumps(jsonable(web), separators=(",", ":"))
    # data.js lets the page work from file:// AND GitHub Pages (no fetch needed)
    (C.WEBSITE_DIR / "data" / "results.js").write_text(
        "/* Generated by src/run_pipeline.py - do not edit by hand. */\n"
        f"window.MOVIEMATCH = {payload};\n")
    joblib.dump(lr, C.MODEL_DIR / "logistic_regression.joblib", compress=3)
    joblib.dump(lr_m, C.MODEL_DIR / "logistic_regression_movie_only.joblib", compress=3)
    (C.MODEL_DIR / "feature_columns.json").write_text(json.dumps(
        {"all": list(X.columns), "movie_only": mcols}, indent=2))
    # Copy figures so the website shows exactly the figures in outputs/figures.
    import shutil
    web_fig = C.WEBSITE_DIR / "assets" / "figures"
    web_fig.mkdir(parents=True, exist_ok=True)
    for f in C.FIG_DIR.glob("*.png"):
        shutil.copy(f, web_fig / f.name)
    # The forest is large (not saved); it can be rebuilt by re-running this script.
    log(f"Split fingerprint (SHA-256): {split_info['assignment_sha256']}")
    log("Done")
    return R


if __name__ == "__main__":
    main()
