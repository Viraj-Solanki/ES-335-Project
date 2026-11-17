"""Cleaning, target construction and the chronological split."""
import hashlib

import pandas as pd

from . import config as C


def quality_report(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Data-quality checks requested in Section 5 of the project guide."""
    merged = ratings.merge(movies[["movie_id"]], on="movie_id", how="left",
                           indicator=True)
    return {
        "missing_values_ratings": int(ratings.isna().sum().sum()),
        "missing_release_date": int(movies["release_date"].isna().sum()),
        "duplicate_user_movie_pairs": int(
            ratings.duplicated(["user_id", "movie_id"]).sum()),
        "fully_duplicate_rows": int(ratings.duplicated().sum()),
        "ratings_outside_1_5": int((~ratings["rating"].between(1, 5)).sum()),
        "unmatched_movie_ids_after_join": int(
            (merged["_merge"] != "both").sum()),
        "movies_with_unknown_genre_flag": int(movies["unknown"].sum()),
        "movies_with_no_genre_flag": int(
            (movies[C.GENRES].sum(axis=1) == 0).sum()),
        "duplicate_titles_in_catalogue": int(
            movies["title"].duplicated().sum()),
    }


def add_target(ratings: pd.DataFrame, min_pos: int = C.POSITIVE_MIN_RATING
               ) -> pd.DataFrame:
    """Binary target fixed BEFORE modelling: 4-5 stars -> 1, 1-3 -> 0."""
    out = ratings.copy()
    out["target"] = (out["rating"] >= min_pos).astype("int8")
    out["datetime"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out


def chronological_split(df: pd.DataFrame,
                        train_frac: float = C.TRAIN_FRAC,
                        val_frac: float = C.VAL_FRAC):
    """Global chronological split by timestamp.

    Rows are sorted by timestamp (ties broken by user_id, movie_id so the
    result is deterministic). Cut points are timestamps: every rating with
    timestamp < t_val goes to train, t_val <= ts < t_test to validation,
    ts >= t_test to test. Using timestamp cut points (instead of raw row
    positions) guarantees that no training rating is later than any
    validation rating, even when several ratings share the same second.
    """
    d = df.sort_values(["timestamp", "user_id", "movie_id"],
                       kind="mergesort").reset_index(drop=True)
    n = len(d)
    t_val = int(d.loc[int(n * train_frac), "timestamp"])
    t_test = int(d.loc[int(n * (train_frac + val_frac)), "timestamp"])
    split = pd.Series("train", index=d.index)
    split[d["timestamp"] >= t_val] = "val"
    split[d["timestamp"] >= t_test] = "test"
    d["split"] = split.values
    info = {"t_val_cutoff": t_val, "t_test_cutoff": t_test}
    for name in ["train", "val", "test"]:
        part = d[d["split"] == name]
        info[name] = {
            "rows": int(len(part)),
            "share": round(len(part) / n, 4),
            "first": str(part["datetime"].min()),
            "last": str(part["datetime"].max()),
            "positive_rate": round(float(part["target"].mean()), 4),
            "users": int(part["user_id"].nunique()),
            "movies": int(part["movie_id"].nunique()),
        }
    # Fingerprint of the exact split, so anyone can confirm they reproduced it.
    key = (d["user_id"].astype(str) + ":" + d["movie_id"].astype(str) + ":" + d["split"]).str.cat(sep="\n")
    info["assignment_sha256"] = hashlib.sha256(key.encode()).hexdigest()
    info["t_val_cutoff_utc"] = str(pd.to_datetime(t_val, unit="s", utc=True))
    info["t_test_cutoff_utc"] = str(pd.to_datetime(t_test, unit="s", utc=True))
    return d, info
