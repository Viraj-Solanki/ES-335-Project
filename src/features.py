"""Leakage-safe feature construction for (user, candidate movie) pairs.

Two ways of computing history statistics are used, and the difference is the
heart of the leakage-prevention design:

1. TRAINING rows  -> ``expanding`` history.
   Each training interaction only sees the user's / movie's ratings with a
   STRICTLY EARLIER timestamp. Ratings in the same second as the row are
   excluded too, so the row's own rating (and anything simultaneous) can never
   leak into its own features.

2. VALIDATION / TEST rows -> ``snapshot`` history.
   Statistics are computed once from the whole training period and frozen
   (the "fixed snapshot" design of Section 9 of the guide). Validation and
   test ratings are never used to build any feature.

A third mode, ``leaky``, deliberately computes training-row features from the
full training period *including the row's own rating*. It exists only for
the leakage demonstration in the report and is never used for real results.
"""
import numpy as np
import pandas as pd

from . import config as C

G = C.GENRES
NG = len(G)


# --------------------------------------------------------------- helpers
def _with_genres(df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    cols = ["movie_id", "release_year"] + G
    return df.merge(movies[cols], on="movie_id", how="left")


def _stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row additive contributions that history statistics are built
    from: a count, a rating sum, a positive count, plus per-genre versions."""
    out = pd.DataFrame(index=df.index)
    out["n"] = 1
    out["r"] = df["rating"].astype(float)
    out["p"] = df["target"].astype(float)
    g = df[G].to_numpy(dtype=float)
    for j, name in enumerate(G):
        out[f"n_{name}"] = g[:, j]
        out[f"p_{name}"] = g[:, j] * df["target"].to_numpy()
    return out


def _expanding(train: pd.DataFrame, key: str, cols: list) -> pd.DataFrame:
    """Cumulative statistics of `key` (user or movie) using only ratings with
    a strictly earlier timestamp than each row."""
    contrib = _stat_columns(train)[cols]
    contrib[key] = train[key].values
    contrib["timestamp"] = train["timestamp"].values
    grp = (contrib.groupby([key, "timestamp"], sort=True)[cols].sum()
           .reset_index())
    cum = grp.groupby(key)[cols].cumsum()
    before = cum - grp[cols]          # exclude the same-timestamp group
    before[key] = grp[key]
    before["timestamp"] = grp["timestamp"]
    rows = train[[key, "timestamp"]].merge(before, on=[key, "timestamp"],
                                           how="left")
    rows.index = train.index
    return rows[cols]


def _snapshot(train: pd.DataFrame, key: str, cols: list,
              rows: pd.DataFrame) -> pd.DataFrame:
    """Statistics of `key` computed from the WHOLE training period, looked
    up for arbitrary rows (validation/test, or leaky training rows)."""
    contrib = _stat_columns(train)[cols]
    contrib[key] = train[key].values
    totals = contrib.groupby(key)[cols].sum()
    out = rows[[key]].merge(totals, left_on=key, right_index=True,
                            how="left")[cols].fillna(0.0)
    out.index = rows.index
    return out


USER_COLS = ["n", "r", "p"] + [f"n_{g}" for g in G] + [f"p_{g}" for g in G]
MOVIE_COLS = ["n", "r", "p"]


# --------------------------------------------------------------- priors
def compute_priors(train: pd.DataFrame) -> dict:
    """Global constants used as smoothing priors / cold-start fallbacks.
    Computed from the training period only."""
    return {
        "global_pos_rate": float(train["target"].mean()),
        "global_mean_rating": float(train["rating"].mean()),
    }


# --------------------------------------------------------------- assemble
def _assemble(rows: pd.DataFrame, ust: pd.DataFrame, mst: pd.DataFrame,
              priors: dict) -> pd.DataFrame:
    gp, gm = priors["global_pos_rate"], priors["global_mean_rating"]
    ku, km, kg = (C.USER_PRIOR_STRENGTH, C.MOVIE_PRIOR_STRENGTH,
                  C.USER_GENRE_PRIOR_STRENGTH)
    X = pd.DataFrame(index=rows.index)

    # ---- user history
    un = ust["n"].to_numpy()
    upr = (ust["p"].to_numpy() + ku * gp) / (un + ku)
    X["user_hist_count_log"] = np.log1p(un)
    X["user_mean_rating"] = (ust["r"].to_numpy() + ku * gm) / (un + ku)
    X["user_pos_rate"] = upr
    X["user_is_cold"] = (un == 0).astype(int)

    # ---- user x candidate-genre preference
    cand = rows[G].to_numpy(dtype=float)                       # N x 19
    n_g = ust[[f"n_{g}" for g in G]].to_numpy()
    p_g = ust[[f"p_{g}" for g in G]].to_numpy()
    pref = (p_g + kg * upr[:, None]) / (n_g + kg)               # smoothed
    k = cand.sum(axis=1)
    safe_k = np.where(k == 0, 1, k)
    aff = np.where(k == 0, upr, (pref * cand).sum(axis=1) / safe_k)
    share = np.where(un == 0, 0.0,
                     (n_g * cand).sum(axis=1) / np.where(un == 0, 1, un)
                     / safe_k)
    X["user_genre_affinity"] = aff
    X["user_genre_affinity_rel"] = aff - upr      # taste beyond user's base rate
    X["user_genre_share"] = share                 # familiarity with the genres

    # ---- movie history (training period only)
    mn = mst["n"].to_numpy()
    X["movie_hist_count_log"] = np.log1p(mn)
    X["movie_mean_rating"] = (mst["r"].to_numpy() + km * gm) / (mn + km)
    X["movie_pos_rate"] = (mst["p"].to_numpy() + km * gp) / (mn + km)
    X["movie_is_cold"] = (mn == 0).astype(int)

    # ---- candidate metadata
    X["release_year"] = rows["release_year"].to_numpy()  # NaN imputed later
    for g in G:
        X[f"genre_{g}"] = rows[g].to_numpy()
    return X


def build_features(split_df: pd.DataFrame, movies: pd.DataFrame,
                   train_mode: str = "expanding"):
    """Return (X, y, meta, priors) for every row of the split dataframe.

    train_mode = "expanding" (correct) or "leaky" (demonstration only).
    """
    df = _with_genres(split_df, movies)
    train = df[df["split"] == "train"]
    other = df[df["split"] != "train"]
    priors = compute_priors(train)

    if train_mode == "expanding":
        u_tr = _expanding(train, "user_id", USER_COLS)
        m_tr = _expanding(train, "movie_id", MOVIE_COLS)
    elif train_mode == "leaky":
        u_tr = _snapshot(train, "user_id", USER_COLS, train)
        m_tr = _snapshot(train, "movie_id", MOVIE_COLS, train)
    else:
        raise ValueError(train_mode)
    u_ot = _snapshot(train, "user_id", USER_COLS, other)
    m_ot = _snapshot(train, "movie_id", MOVIE_COLS, other)

    X = pd.concat([_assemble(train, u_tr, m_tr, priors),
                   _assemble(other, u_ot, m_ot, priors)]).loc[df.index]
    meta = df[["user_id", "movie_id", "rating", "timestamp", "split"]].copy()
    meta["user_hist_count"] = np.expm1(X["user_hist_count_log"]).round()
    meta["movie_hist_count"] = np.expm1(X["movie_hist_count_log"]).round()
    return X, df["target"].astype(int), meta, priors


def snapshot_user_profile(train: pd.DataFrame, movies: pd.DataFrame,
                          user_id: int, priors: dict) -> dict:
    """Readable training-period profile of one user (used for the website)."""
    d = _with_genres(train[train["user_id"] == user_id], movies)
    out = {"n_ratings": int(len(d)),
           "mean_rating": round(float(d["rating"].mean()), 2) if len(d) else None,
           "positive_rate": round(float(d["target"].mean()), 3) if len(d) else None,
           "genres": []}
    for g in G:
        sub = d[d[g] == 1]
        if len(sub):
            out["genres"].append({"genre": g, "n": int(len(sub)),
                                  "positive_rate": round(float(sub["target"].mean()), 3)})
    out["genres"].sort(key=lambda x: -x["n"])
    return out


def snapshot_features_for_pairs(pairs: pd.DataFrame, split_df: pd.DataFrame,
                                movies: pd.DataFrame) -> pd.DataFrame:
    """Features for arbitrary (user_id, movie_id) pairs - e.g. movies a user
    has NOT rated - using the frozen training-period snapshot only. Used to
    produce the recommendation lists shown on the website."""
    train = _with_genres(split_df[split_df["split"] == "train"], movies)
    priors = compute_priors(train)
    rows = _with_genres(pairs[["user_id", "movie_id"]].reset_index(drop=True),
                        movies)
    ust = _snapshot(train, "user_id", USER_COLS, rows)
    mst = _snapshot(train, "movie_id", MOVIE_COLS, rows)
    return _assemble(rows, ust, mst, priors)


def build_sequential_features(split_df: pd.DataFrame, movies: pd.DataFrame):
    """SECONDARY protocol (Section 9, 'sequential simulation').

    Every row - in any split - is described using ALL ratings with a
    strictly earlier timestamp, including earlier validation/test ratings.
    Rule: when predicting a rating made at time t, everything rated before t
    is assumed observable; nothing at or after t is. Priors (global rates)
    still come from the training period only. The model is NOT refitted on
    these features: it is the same model trained on the training split.
    """
    df = _with_genres(split_df, movies)
    priors = compute_priors(df[df["split"] == "train"])
    ust = _expanding(df, "user_id", USER_COLS)
    mst = _expanding(df, "movie_id", MOVIE_COLS)
    X = _assemble(df, ust, mst, priors)
    meta = df[["user_id", "movie_id", "rating", "timestamp", "split"]].copy()
    meta["user_hist_count"] = np.expm1(X["user_hist_count_log"]).round()
    meta["movie_hist_count"] = np.expm1(X["movie_hist_count_log"]).round()
    return X, df["target"].astype(int), meta
