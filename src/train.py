"""Baselines and models. All tuning uses the VALIDATION split only."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

# Features that describe the candidate movie only (no information about
# the user). Used for the "non-personalised" Logistic Regression ablation.
MOVIE_ONLY_PREFIXES = ("movie_", "genre_", "release_year")


def movie_only_columns(columns):
    return [c for c in columns if c.startswith(MOVIE_ONLY_PREFIXES)]


# ---------------------------------------------------------------- baselines
def fit_majority(X_tr, y_tr):
    """Predicts the most frequent training class for every pair."""
    return DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)


class PopularityBaseline:
    """Non-personalised: score = smoothed training-period positive-rating
    fraction of the candidate movie (feature `movie_pos_rate`). Every user
    gets the same ordering of movies."""
    name = "Popularity baseline"

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = X["movie_pos_rate"].to_numpy()
        return np.column_stack([1 - s, s])


# ---------------------------------------------------------------- models
def make_logreg(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # release_year has 1 NaN
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=2000)),
    ])


def make_forest(min_leaf: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=min_leaf, max_features="sqrt",
            n_jobs=-1, random_state=C.RANDOM_SEED)),
    ])


LOGREG_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
FOREST_GRID = [5, 20, 50, 100]


def tune(factory, grid, X_tr, y_tr, X_va, y_va):
    """Fit one model per hyper-parameter value on TRAIN, score each by
    validation average precision (PR-AUC), keep the best."""
    log = []
    best = None
    for h in grid:
        model = factory(h).fit(X_tr, y_tr)
        ap = average_precision_score(y_va, model.predict_proba(X_va)[:, 1])
        log.append({"param": h, "val_pr_auc": round(float(ap), 4)})
        if best is None or ap > best[1]:
            best = (h, ap, model)
    return best[2], best[0], log


def choose_threshold(y_va, p_va):
    """Probability threshold that maximises F1 on VALIDATION data."""
    prec, rec, thr = precision_recall_curve(y_va, p_va)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])
