"""Automated leakage and split checks.

Run from the project root:   python -m pytest -q tests
(or)                          python tests/test_leakage.py

The feature code is vectorised; these tests recompute features for random
rows with slow, obvious loops and confirm both versions agree.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                       # noqa: E402
from src import features as F                     # noqa: E402
from src import load_data as L                    # noqa: E402
from src import preprocessing as P                # noqa: E402

RATINGS, MOVIES = L.load_ratings(), L.load_movies()
SPLIT, INFO = P.chronological_split(P.add_target(RATINGS))
X, Y, META, PRIORS = F.build_features(SPLIT, MOVIES)
RNG = np.random.default_rng(0)


def test_split_is_chronological():
    tr = SPLIT[SPLIT.split == "train"].timestamp
    va = SPLIT[SPLIT.split == "val"].timestamp
    te = SPLIT[SPLIT.split == "test"].timestamp
    assert tr.max() < va.min() and va.max() < te.min()
    assert len(tr) + len(va) + len(te) == C.EXPECTED_RATINGS


def test_target_rule():
    assert (SPLIT.loc[SPLIT.rating >= 4, "target"] == 1).all()
    assert (SPLIT.loc[SPLIT.rating <= 3, "target"] == 0).all()


def _manual_user_stats(user, before_ts):
    tr = SPLIT[(SPLIT.split == "train") & (SPLIT.user_id == user)
               & (SPLIT.timestamp < before_ts)]
    return len(tr), tr.rating.sum(), tr.target.sum()


def test_training_features_use_only_strictly_earlier_ratings():
    train_idx = SPLIT.index[SPLIT.split == "train"]
    gm, gp = PRIORS["global_mean_rating"], PRIORS["global_pos_rate"]
    k = C.USER_PRIOR_STRENGTH
    for i in RNG.choice(train_idx, 300, replace=False):
        row = SPLIT.loc[i]
        n, rsum, psum = _manual_user_stats(row.user_id, row.timestamp)
        assert META.loc[i, "user_hist_count"] == n
        assert np.isclose(X.loc[i, "user_mean_rating"], (rsum + k * gm) / (n + k))
        assert np.isclose(X.loc[i, "user_pos_rate"], (psum + k * gp) / (n + k))


def test_eval_features_use_training_snapshot_only():
    """Validation/test features must equal statistics from the training
    period only. If any validation/test rating leaked in, counts would differ."""
    tr = SPLIT[SPLIT.split == "train"]
    eval_idx = SPLIT.index[SPLIT.split != "train"]
    for i in RNG.choice(eval_idx, 300, replace=False):
        row = SPLIT.loc[i]
        assert META.loc[i, "user_hist_count"] == (tr.user_id == row.user_id).sum()
        assert META.loc[i, "movie_hist_count"] == (tr.movie_id == row.movie_id).sum()


def test_sequential_protocol_uses_only_strictly_earlier_ratings():
    _, _, ms = F.build_sequential_features(SPLIT, MOVIES)
    for i in RNG.choice(SPLIT.index[SPLIT.split == "test"], 200, replace=False):
        row = SPLIT.loc[i]
        n = ((SPLIT.user_id == row.user_id) & (SPLIT.timestamp < row.timestamp)).sum()
        assert ms.loc[i, "user_hist_count"] == n


def test_feature_matrix_has_no_rating_or_target():
    forbidden = {"rating", "target", "timestamp", "user_id", "movie_id"}
    assert not forbidden & set(X.columns)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
