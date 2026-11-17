# MovieMatch: Personalized Movie Recommendation Using User History

**ES 335 machine learning project · Solanki Viraj (24110348) · Individual Project**

Given a user's earlier MovieLens ratings and a candidate movie, predict whether the user will rate it **4–5 stars**, rank movies by that prediction, and test whether this personalisation beats a non-personalised popularity baseline.

* **Website:** https://viraj-solanki.github.io/ES-335-Project/ (source in [`docs/`](docs/))
* **Full report:** [`REPORT.md`](REPORT.md)

## Result in brief

Test set, chronological 80/10/10 split; hyperparameters and thresholds chosen on validation only.

| Model | Fixed: F1 | Fixed: PR-AUC | Fixed: Accuracy | Sequential: F1 | Sequential: PR-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.720 | 0.563 | 0.563 | 0.720 | 0.563 |
| Popularity baseline | 0.737 | 0.743 | 0.634 | 0.735 | 0.745 |
| Logistic Regression | 0.732 | 0.746 | 0.634 | 0.757 | 0.775 |
| Random Forest | 0.731 | 0.740 | 0.638 | 0.755 | 0.777 |
| LR (movie features only) | 0.735 | 0.746 | 0.645 | 0.736 | 0.748 |

* **Primary protocol (fixed snapshot).** Features for validation and test rows use the training period only. 85.8% of test rows come from users with *no* training ratings. Logistic Regression is statistically indistinguishable from popularity: PR-AUC difference +0.004, user-bootstrap 95% CI [-0.012, +0.017].
* **Secondary protocol (sequential).** Every rating made before the prediction moment is visible. Logistic Regression beats popularity by +0.031 PR-AUC (95% CI [+0.010, +0.050]). Removing the user features (movie-only ablation) removes the gain.

## Where things are

| Path | Contents |
|---|---|
| `src/config.py` | Every setting that affects results: target rule, split fractions, smoothing strengths, seed |
| `src/load_data.py` | Downloads MovieLens 100K, checks the MD5, loads and verifies the files |
| `src/preprocessing.py` | Quality checks, binary target, chronological split (with a SHA-256 fingerprint) |
| `src/features.py` | Leakage-safe user and movie features for training, validation/test and the sequential protocol |
| `src/train.py` | Majority and popularity baselines, Logistic Regression, Random Forest, tuning, threshold choice |
| `src/evaluate.py` | Metrics, subgroup tables, per-user ranking, cluster bootstrap |
| `src/plots.py` | All figures |
| `src/run_pipeline.py` | Runs the whole experiment; writes `outputs/` and `docs/data/results.js` |
| `src/make_report.py` | Writes `REPORT.md` from `outputs/metrics.json` |
| `tests/test_leakage.py` | Brute-force checks of the split and feature timing |
| `notebooks/01_data_exploration.ipynb` | Data loading, quality checks, exploratory plots |
| `notebooks/02_features_and_split.ipynb` | Split, features, cold-start coverage, leakage tests |
| `notebooks/03_model_evaluation.ipynb` | Re-fits LR, verifies saved predictions, all result tables |
| `outputs/metrics.json` | Every result, including extra experiments (sequential protocol, ablation, bootstrap, leakage demo, 3-star sensitivity, subgroups, ranking) |
| `outputs/test_metrics.csv`, `validation_metrics.csv` | Main metric tables |
| `outputs/test_predictions.csv` | Per-row test scores for every model and protocol (no MovieLens IDs or ratings, see below) |
| `outputs/figures/` | 14 figures used in the report and website |
| `outputs/models/` | Trained Logistic Regression pipelines and their feature lists |
| `docs/` | Static website: `index.html`, `styles.css`, `script.js`, `data/results.js`, `assets/figures/` |

## Data access

MovieLens 100K, GroupLens Research, University of Minnesota: <https://grouplens.org/datasets/movielens/100k/>. The licence allows research use but **not redistribution**, so this repository contains no MovieLens files and no row-level copy of the ratings.

```bash
python -m src.load_data
```

downloads `ml-100k.zip` from the official server, checks its MD5 (`0e33842e24a9c977be4e0107933c0723`) and extracts it to `data/raw/ml-100k/`. If the download fails, get the zip from the page above and extract it so that `data/raw/ml-100k/u.data` exists. Only `u.data` (ratings) and `u.item` (titles, release dates, genres) are used; the demographic file `u.user` is not.

## Setup and rebuild

Python 3.10 or newer (tested on 3.12).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.load_data              # 1. get and verify the data
python tests/test_leakage.py         # 2. split and leakage tests (or: python -m pytest -q tests)
python -m src.run_pipeline           # 3. full experiment, about 3 minutes
python -m src.make_report            # 4. regenerate REPORT.md
jupyter notebook notebooks/          # 5. open 01 → 02 → 03
```

Step 3 rewrites everything in `outputs/` and the website's `docs/data/results.js` and `docs/assets/figures/`, so the report, notebooks and site always show the same numbers.

**Exact split.** Ratings sorted by (timestamp, user_id, movie_id) and cut at timestamps. Validation starts at Unix time 889237269 (1998-03-07 02:21:09+00:00); test starts at 891382309 (1998-03-31 22:11:49+00:00). Rows: train 79,999, validation 10,001, test 10,000. Target: rating ≥ 4. Seed: 335. Step 3 prints a SHA-256 fingerprint of the split assignment; a correct reproduction prints
`5c82227b4259f25598c390b74bc74cb8edbd9d137bbeee8cc531b73555df887d`.

**Saved predictions.** Row *i* of `outputs/test_predictions.csv` is the *i*-th test rating in split order. Notebook 03 joins the file back to the downloaded data and recomputes every reported test metric from it (read it with `pd.read_csv(..., float_precision="round_trip")`, because some scores tie exactly with a threshold).

**Determinism.** Re-running gives identical metrics. The Random Forest averages trees in parallel, so its stored threshold can differ in the 17th significant digit between runs; no reported number changes.

**Website.** Open `docs/index.html` in a browser; no server or build step is needed. To fill the "Code" links on the site, put the repository URL in the `<meta name="repo-url">` tag near the top of `docs/index.html`. GitHub Pages can publish the site from the `/docs` folder of the main branch.

## How leakage is prevented

1. The split is made by timestamp before any feature, scaler or model is fitted.
2. Training rows only see ratings with a strictly earlier timestamp; their own rating and same-second ratings are excluded.
3. Validation and test rows use a snapshot of the training period frozen at the cutoff (primary protocol).
4. Smoothing priors and the imputation value come from training data; scaling and imputation are inside a scikit-learn `Pipeline`.
5. Hyperparameters and thresholds use validation data only; all choices were fixed before the test split was scored.
6. `tests/test_leakage.py` recomputes features for random rows with explicit loops and checks they match.

A deliberately leaky variant is included as a demonstration (validation only, REPORT.md §6.8).

## AI-use disclosure

AI assistance (Claude, by Anthropic) was used for brainstorming the scope, drafting and debugging code, designing leakage checks, and assisting with documentation. All code was executed and verified locally end to end, dataset integrity was verified against the official GroupLens repository, and every evaluation metric was generated programmatically from the pipeline. I take full responsibility for the methodology, implementation, and conclusions.

## Citations

* F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19. https://doi.org/10.1145/2827872
* GroupLens Research, MovieLens 100K dataset: https://grouplens.org/datasets/movielens/100k/. This project is not endorsed by the University of Minnesota or GroupLens.
* F. Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research* 12, 2825–2830.
* Libraries: pandas, NumPy, Matplotlib, joblib, Jupyter. Fonts: Plus Jakarta Sans and Inter (Google Fonts, SIL Open Font License).
* *MovieMatch: ES 335 End-to-end Machine Learning Project Guide* (course material).
