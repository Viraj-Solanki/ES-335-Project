"""Metrics, subgroup analysis and per-user ranking evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def classification_metrics(y, p, threshold):
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    # A constant score has ROC-AUC 0.5 by definition; sklearn handles it.
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y, yhat), 4),
        "precision": round(precision_score(y, yhat, zero_division=0), 4),
        "recall": round(recall_score(y, yhat, zero_division=0), 4),
        "f1": round(f1_score(y, yhat, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y)),
        "positive_rate": round(float(np.mean(y)), 4),
    }


USER_BUCKETS = [(-1, 0, "0 (cold start)"), (0, 49, "1-49"),
                (49, 199, "50-199"), (199, 10**9, "200+")]
MOVIE_BUCKETS = [(-1, 0, "0 (new movie)"), (0, 19, "1-19"),
                 (19, 99, "20-99"), (99, 10**9, "100+")]


def bucketize(counts, buckets):
    out = pd.Series(index=counts.index, dtype=object)
    for lo, hi, name in buckets:
        out[(counts > lo) & (counts <= hi)] = name
    return out


def subgroup_table(meta, y, preds: dict, thresholds: dict, by: pd.Series):
    rows = []
    for group, idx in by.groupby(by).groups.items():
        yy = y.loc[idx]
        row = {"group": group, "n": int(len(idx)),
               "positive_rate": round(float(yy.mean()), 4)}
        for name, p in preds.items():
            pp = p.loc[idx].to_numpy()
            yhat = (pp >= thresholds[name]).astype(int)
            row[f"{name}|f1"] = round(f1_score(yy, yhat, zero_division=0), 4)
            row[f"{name}|accuracy"] = round(accuracy_score(yy, yhat), 4)
            row[f"{name}|pr_auc"] = (round(average_precision_score(yy, pp), 4)
                                     if 0 < yy.sum() < len(yy) else None)
        rows.append(row)
    return rows


def ranking_metrics(meta, y, preds: dict, k_values=(5, 10), min_items=10):
    """Per-user ranking over the movies each user actually rated in the
    evaluation window (the only candidates with known labels).

    For every user with >= min_items evaluation ratings, sort those movies
    by score and measure precision@k (share of positives in the top k).
    Also reports the expected precision of a random ordering (= the user's
    positive rate) so the gain can be read honestly.
    """
    df = meta[["user_id"]].copy()
    df["y"] = y.values
    for name, p in preds.items():
        df[name] = p.values
    counts = df.groupby("user_id").size()
    users = counts[counts >= min_items].index
    res = {"n_users": int(len(users)), "min_items": min_items, "models": {}}
    rnd = df[df.user_id.isin(users)].groupby("user_id")["y"].mean()
    res["random_order"] = {f"p@{k}": round(float(rnd.mean()), 4) for k in k_values}
    for name in preds:
        vals = {k: [] for k in k_values}
        aps = []
        for u in users:
            d = df[df.user_id == u].sort_values(name, ascending=False,
                                                kind="mergesort")
            for k in k_values:
                vals[k].append(d["y"].iloc[:k].mean())
            if 0 < d["y"].sum() < len(d):
                aps.append(average_precision_score(d["y"], d[name]))
        res["models"][name] = {f"p@{k}": round(float(np.mean(v)), 4)
                               for k, v in vals.items()}
        res["models"][name]["mean_user_ap"] = round(float(np.mean(aps)), 4)
    return res


def paired_bootstrap(meta, y, p_a, p_b, n_boot=1000, seed=335):
    """95% CI for PR-AUC(a) - PR-AUC(b) on the same rows.

    Resamples USERS with replacement (cluster bootstrap) because ratings
    from the same user are not independent. Uses saved test predictions
    only - nothing is refitted or re-tuned.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"u": meta["user_id"].values, "y": y.values,
                       "a": np.asarray(p_a), "b": np.asarray(p_b)})
    groups = [g.index.to_numpy() for _, g in df.groupby("u")]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        d = df.loc[idx]
        if 0 < d.y.sum() < len(d):
            diffs.append(average_precision_score(d.y, d.a) - average_precision_score(d.y, d.b))
    diffs = np.array(diffs)
    point = average_precision_score(df.y, df.a) - average_precision_score(df.y, df.b)
    return {"diff": round(float(point), 4),
            "ci95": [round(float(np.percentile(diffs, 2.5)), 4),
                     round(float(np.percentile(diffs, 97.5)), 4)],
            "share_boot_positive": round(float((diffs > 0).mean()), 3),
            "n_boot": int(len(diffs)), "clusters": int(len(groups))}
