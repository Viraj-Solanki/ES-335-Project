"""Central configuration for the MovieMatch pipeline.

Every constant that affects results lives here so that the report, the
notebooks and the website all use exactly the same settings.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ml-100k"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
WEBSITE_DIR = ROOT / "docs"   # static website (GitHub Pages can serve /docs)

# ---------------------------------------------------------------- dataset
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# MD5 of the official archive as published by GroupLens.
DATASET_MD5 = "0e33842e24a9c977be4e0107933c0723"
EXPECTED_RATINGS = 100_000
EXPECTED_USERS = 943
EXPECTED_MOVIES = 1_682

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

# ---------------------------------------------------------------- target
POSITIVE_MIN_RATING = 4          # rating >= 4  -> 1 (positive)
                                 # rating <= 3  -> 0 (not positive)

# ---------------------------------------------------------------- split
TRAIN_FRAC = 0.80                # earliest 80 % of interactions
VAL_FRAC = 0.10                  # next 10 %
                                 # latest 10 %  -> final test

# ---------------------------------------------------------------- features
# Bayesian smoothing strengths (pseudo-counts) used to shrink noisy
# averages toward a prior. Fixed before modelling, not tuned on test.
MOVIE_PRIOR_STRENGTH = 10
USER_PRIOR_STRENGTH = 5
USER_GENRE_PRIOR_STRENGTH = 3

RANDOM_SEED = 335
