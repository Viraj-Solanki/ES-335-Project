"""Download, verify and load MovieLens 100K.

Usage (from the project root):
    python -m src.load_data          # downloads + verifies the archive

The raw data is NOT shipped with this repository because the MovieLens
licence does not allow redistribution. This script fetches it from the
official GroupLens server and checks the MD5 checksum.
"""
import hashlib
import io
import urllib.request
import zipfile

import pandas as pd

from . import config as C

RATING_COLS = ["user_id", "movie_id", "rating", "timestamp"]
ITEM_COLS = ["movie_id", "title", "release_date", "video_release_date",
             "imdb_url"] + C.GENRES


def download_dataset(force: bool = False) -> None:
    """Download ml-100k.zip from GroupLens, verify MD5 and extract it."""
    if (C.RAW_DIR / "u.data").exists() and not force:
        print(f"Dataset already present at {C.RAW_DIR}")
        return
    print(f"Downloading {C.DATASET_URL} ...")
    try:
        with urllib.request.urlopen(C.DATASET_URL, timeout=60) as resp:
            payload = resp.read()
    except OSError as err:
        raise SystemExit(
            f"Could not download the dataset ({err}).\n"
            "Download ml-100k.zip manually from https://grouplens.org/datasets/movielens/100k/\n"
            f"and extract it so that {C.RAW_DIR / 'u.data'} exists, then re-run.")
    md5 = hashlib.md5(payload).hexdigest()
    if md5 != C.DATASET_MD5:
        raise ValueError(f"MD5 mismatch: got {md5}, expected {C.DATASET_MD5}")
    C.RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(C.RAW_DIR.parent)
    print(f"Verified (MD5 {md5}) and extracted to {C.RAW_DIR}")


def load_ratings() -> pd.DataFrame:
    """u.data: tab separated user | movie | rating | unix timestamp."""
    df = pd.read_csv(C.RAW_DIR / "u.data", sep="\t", names=RATING_COLS,
                     dtype={"user_id": "int32", "movie_id": "int32",
                            "rating": "int8", "timestamp": "int64"})
    return df


def load_movies() -> pd.DataFrame:
    """u.item: the README says 'tab separated', but the file is actually
    pipe ('|') separated and Latin-1 encoded (non-ASCII titles)."""
    df = pd.read_csv(C.RAW_DIR / "u.item", sep="|", names=ITEM_COLS,
                     encoding="latin-1")
    df["release_date"] = pd.to_datetime(df["release_date"],
                                        format="%d-%b-%Y", errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df[C.GENRES] = df[C.GENRES].astype("int8")
    # video_release_date is empty for every row in the official file
    return df.drop(columns=["video_release_date", "imdb_url"])


def verify_counts(ratings: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """Check the loaded tables against the numbers stated in the README."""
    checks = {
        "n_ratings": len(ratings),
        "n_users": ratings["user_id"].nunique(),
        "n_movies_rated": ratings["movie_id"].nunique(),
        "n_movies_in_catalogue": len(movies),
    }
    assert checks["n_ratings"] == C.EXPECTED_RATINGS, checks
    assert checks["n_users"] == C.EXPECTED_USERS, checks
    assert checks["n_movies_in_catalogue"] == C.EXPECTED_MOVIES, checks
    return checks


if __name__ == "__main__":
    download_dataset()
    r, m = load_ratings(), load_movies()
    print(verify_counts(r, m))
