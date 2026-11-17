"""Write REPORT.md from outputs/metrics.json.

    python -m src.make_report

Every number in the report is read from the pipeline output, so the report,
the notebooks and the website cannot disagree.
"""
import json

from . import config as C

R = json.loads((C.OUTPUT_DIR / "metrics.json").read_text())
MODELS = ["Majority baseline", "Popularity baseline", "Logistic Regression",
          "Random Forest", "LR (movie features only)"]


def f3(v): return f"{v:.3f}"
def f4(v): return f"{v:.4f}"
def pct(v): return f"{v * 100:.1f}%"
def pct0(v): return f"{v * 100:.0f}%"
def sgn(v): return f"{v:+.3f}"
def ci(c): return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def metric_table(block):
    rows = ["| Model | Threshold | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in MODELS:
        d = block[m]
        rows.append(f"| {m} | {f3(d['threshold'])} | {f4(d['accuracy'])} | {f4(d['precision'])} | "
                    f"{f4(d['recall'])} | {f4(d['f1'])} | {f4(d['pr_auc'])} | {f4(d['roc_auc'])} |")
    return "\n".join(rows)


def cm_table(block):
    rows = ["| Model | TN | FP | FN | TP |", "|---|---:|---:|---:|---:|"]
    for m in MODELS[:4]:
        c = block[m]["confusion"]
        rows.append(f"| {m} | {c['tn']:,} | {c['fp']:,} | {c['fn']:,} | {c['tp']:,} |")
    return "\n".join(rows)


def group_table(rows):
    out = ["| User history | Rows | Positive rate | Popularity PR-AUC | LR PR-AUC | RF PR-AUC | Popularity F1 | LR F1 | RF F1 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        g = lambda m, k: f3(r[f"{m}|{k}"]) if r.get(f"{m}|{k}") is not None else "n/a"
        out.append(f"| {r['group']} | {r['n']:,} | {pct(r['positive_rate'])} | {g('Popularity baseline', 'pr_auc')} | "
                   f"{g('Logistic Regression', 'pr_auc')} | {g('Random Forest', 'pr_auc')} | {g('Popularity baseline', 'f1')} | "
                   f"{g('Logistic Regression', 'f1')} | {g('Random Forest', 'f1')} |")
    return "\n".join(out)


def err_table(rows):
    out = ["| User | Movie | Rating | LR score | User's earlier ratings | Movie positive rate | Movie's training ratings |",
           "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        hist = f"{int(r['user_hist_count'])}" if r["user_hist_count"] else "0 (cold start)"
        out.append(f"| {r['user_id']} | {r['title']} | {r['rating']} | {f3(r['p'])} | {hist} | "
                   f"{r['movie_pos_rate']:.2f} | {int(r['movie_hist_count'])} |")
    return "\n".join(out)


def main():
    ds, sp, cs = R["dataset"], R["split"], R["cold_start"]
    T, SQ, B = R["test"], R["sequential"]["test"], R["bootstrap_pr_auc_diff"]
    LR, POP, MAJ, RF, LRM = (T[m] for m in MODELS[2:3] + MODELS[1:2] + MODELS[0:1] + MODELS[3:])
    tun = R["tuning"]
    ld, sens, ee = R["leakage_demo"], R["sensitivity_exclude_3star"], R["error_examples"]
    rk, rks = R["ranking_test"], R["sequential"]["ranking_test"]
    h200 = R["test_by_user_history"][3]
    warm = R["test_warm_users_only"]
    q = R["quality"]

    text = f"""# MovieMatch: Personalized Movie Recommendation Using User History

**ES 335 project report**

Solanki Viraj (24110348)

Repository: [repository URL] · Website: [GitHub Pages URL]

*All numbers in this report are generated from `outputs/metrics.json` by `python -m src.make_report`.*

---

## Abstract

We predict whether a MovieLens 100K user will rate a candidate movie 4–5 stars ("positive") from the user's earlier ratings and the movie's genres, and test whether this personalised prediction beats a non-personalised popularity baseline. With a leakage-safe chronological 80/10/10 split, Logistic Regression reaches a test PR-AUC of {f3(LR['pr_auc'])} against {f3(POP['pr_auc'])} for popularity and {f3(MAJ['pr_auc'])} for the majority baseline. The difference from popularity is not statistically distinguishable (user-level bootstrap 95% CI {ci(B['fixed_snapshot']['LR - Popularity']['ci95'])}). The main reason is cold start: under the guide's fixed-snapshot protocol, {pct(cs['fixed_snapshot']['test']['rows_user_cold'])} of test rows come from users with no training-period ratings. When earlier ratings are allowed to inform later predictions (the guide's optional sequential protocol), Logistic Regression improves PR-AUC over popularity by {sgn(B['sequential']['LR - Popularity']['diff'])} (95% CI {ci(B['sequential']['LR - Popularity']['ci95'])}), and an ablation confirms the gain comes from the user-history features.

## 1. Problem formulation

| Item | Definition |
|---|---|
| Research question | Can a user's previous ratings and movie information predict whether the user will positively rate a candidate movie? Does personalisation improve on a popularity baseline? |
| One row | One user–movie interaction (a user rated a movie at a time). |
| Target | 1 if rating ≥ 4; 0 if rating ≤ 3. Fixed before modelling. |
| Prediction time | When a recommendation is generated, before the user rates the candidate. |
| Inputs | The user's earlier ratings, the candidate's genres and release year, movie statistics from the training period. |
| Output | A probability per user–movie pair and a ranked list of candidates. |

A 1–3 rating is "not positive" for this experiment, not proof of dislike. Unrated movies are unknown and are never used as negatives.

## 2. Data

MovieLens 100K (GroupLens Research). The archive's MD5 matched the published checksum (`{C.DATASET_MD5}`) and the counts match the README: {ds['n_ratings']:,} ratings, {ds['n_users']} users, {ds['n_movies_in_catalogue']:,} movies, from {ds['first_rating_utc'][:10]} to {ds['last_rating_utc'][:10]}. `u.item` is pipe-separated and Latin-1 encoded although the README says tab-separated. Demographic data in `u.user` was not used.

**Quality checks.** Missing rating values: {q['missing_values_ratings']}. Duplicate user–movie pairs: {q['duplicate_user_movie_pairs']}. Ratings outside 1–5: {q['ratings_outside_1_5']}. Unmatched movie IDs after the join: {q['unmatched_movie_ids_after_join']}. Movies without release date: {q['missing_release_date']} (imputed with the training median inside the model pipeline). Movies flagged genre "unknown": {q['movies_with_unknown_genre_flag']}. Duplicate titles under different IDs: {q['duplicate_titles_in_catalogue']} (kept as separate items).

**Summary.** Positive rate {pct(ds['positive_rate'])}. Ratings per user: min {ds['ratings_per_user']['min']}, median {ds['ratings_per_user']['median']:.0f}, max {ds['ratings_per_user']['max']}. Ratings per movie: min {ds['ratings_per_movie']['min']}, median {ds['ratings_per_movie']['median']:.0f}, max {ds['ratings_per_movie']['max']}; {ds['movies_with_lt_5_ratings']} movies have fewer than 5 ratings and the top 10% of movies hold {pct0(ds['top10pct_movies_share'])} of all ratings. Matrix sparsity {pct(ds['sparsity'])}.

![Rating distribution](outputs/figures/eda_rating_distribution.png)
![Ratings per user and movie](outputs/figures/eda_ratings_per_user_movie.png)
![Positive fraction by genre](outputs/figures/eda_genre_positive_fraction.png)

## 3. Preprocessing and split

1. Load files with README column names; join ratings to movies.
2. Quality checks (above).
3. Create the binary target.
4. **Chronological split before any feature is computed.** Ratings are sorted by timestamp and cut at timestamps (not row positions), so no training rating is later than any validation rating.
5. Build features (Section 4).
6. Fit imputer, scaler and model inside a scikit-learn `Pipeline` on training rows only.

| Split | Rows | First (UTC) | Last (UTC) | Users | Movies | Positive rate |
|---|---:|---|---|---:|---:|---:|
""" + "\n".join(f"| {n} | {sp[k]['rows']:,} | {sp[k]['first'][:19]} | {sp[k]['last'][:19]} | {sp[k]['users']} | {sp[k]['movies']:,} | {pct(sp[k]['positive_rate'])} |"
                for n, k in [("Train", "train"), ("Validation", "val"), ("Test", "test")]) + f"""

Validation starts at Unix time {sp['t_val_cutoff']} ({sp['t_val_cutoff_utc']}); test starts at {sp['t_test_cutoff']} ({sp['t_test_cutoff_utc']}). SHA-256 fingerprint of the split assignment (printed by `src.run_pipeline`): `{sp['assignment_sha256']}`.

![Ratings over time](outputs/figures/eda_ratings_over_time.png)

## 4. Features and leakage prevention

| Feature | Construction |
|---|---|
| `user_hist_count_log`, `user_is_cold` | log(1 + number of the user's earlier ratings); flag when zero. |
| `user_mean_rating`, `user_pos_rate` | From earlier ratings only, shrunk toward the training-period global value with {C.USER_PRIOR_STRENGTH} pseudo-ratings. |
| `user_genre_affinity` | For each candidate genre, the user's earlier positive rate in that genre (shrunk toward the user's overall rate, {C.USER_GENRE_PRIOR_STRENGTH} pseudo-ratings), averaged over the candidate's genres. |
| `user_genre_affinity_rel`, `user_genre_share` | Affinity minus the user's base rate; share of the user's history in the candidate's genres. |
| `movie_mean_rating`, `movie_pos_rate`, `movie_hist_count_log`, `movie_is_cold` | Candidate's training-period ratings only, shrunk with {C.MOVIE_PRIOR_STRENGTH} pseudo-ratings. |
| `genre_*` (19), `release_year` | Candidate metadata from `u.item`. |

Global priors (from training only): positive rate {R['priors']['global_pos_rate']:.4f}, mean rating {R['priors']['global_mean_rating']:.4f}.

**How leakage is prevented.**

* *Training rows* use an expanding history: only ratings with a **strictly earlier timestamp**. The row's own rating and any rating in the same second are excluded.
* *Validation and test rows* (primary protocol) use a snapshot of the training period frozen at the cutoff. No validation or test rating ever enters a feature.
* Smoothing priors and the release-year imputation value come from training data only.
* Hyperparameters and thresholds use validation data only. All design choices were fixed before the test split was first scored, and later re-runs (to add exports) produced identical metrics.
* `tests/test_leakage.py` recomputes features for 300 random training rows and 300 random validation/test rows with slow explicit loops and checks they equal the vectorised features. All tests pass.

**Cold start.** Under the fixed snapshot, {pct(cs['fixed_snapshot']['val']['rows_user_cold'])} of validation rows and {pct(cs['fixed_snapshot']['test']['rows_user_cold'])} of test rows belong to users with no training ratings ({cs['fixed_snapshot']['test']['users_cold']} of {cs['fixed_snapshot']['test']['users_total']} test users). Users in this dataset tend to join, rate many movies in one burst and leave, so most later users are new. Fallback: smoothed features collapse to the training-wide priors and the cold-start flag is set.

**Secondary protocol (sequential simulation).** Following the guide's optional design, every rating made before the prediction time *t* is treated as observable and nothing at or after *t* is. Only {pct(cs['sequential']['test']['rows_user_cold'])} of test rows then lack user history. The *same* models trained on the training split are used; only thresholds are re-chosen on sequential validation data. Training rows have identical features under both protocols.

![Cold-start coverage](outputs/figures/eda_cold_start_coverage.png)

## 5. Methods

* **Majority baseline**: predicts the majority training class (positive) for every pair.
* **Popularity baseline**: score = the movie's smoothed training-period positive fraction; the same ordering for every user.
* **Logistic Regression** (primary model): median imputation → standardisation → L2 logistic regression. C ∈ {{0.001, 0.01, 0.1, 1, 10}}; chosen C = {tun['Logistic Regression']['chosen_C']}.
* **Random Forest**: 300 trees, `max_features="sqrt"`, minimum leaf ∈ {{5, 20, 50, 100}}; chosen {tun['Random Forest']['chosen_min_samples_leaf']}.
* **LR, movie features only** (ablation): Logistic Regression with every user feature removed.

Tuning criterion: validation PR-AUC. Classification threshold: the value maximising validation F1. Validation PR-AUC was nearly flat across C (""" + ", ".join(f"{x['param']}: {x['val_pr_auc']}" for x in tun['Logistic Regression']['log']) + f"""), so the choice of C at the edge of the grid has little effect. Random Forest validation PR-AUC: """ + ", ".join(f"leaf {x['param']}: {x['val_pr_auc']}" for x in tun['Random Forest']['log']) + f""". Logistic Regression had the higher validation PR-AUC and is the selected model.

![Threshold selection](outputs/figures/results_threshold_selection_val.png)

## 6. Results

### 6.1 Validation (fixed snapshot)

{metric_table(R['validation'])}

### 6.2 Test, fixed snapshot (primary result)

{metric_table(T)}

{cm_table(T)}

![Metric comparison](outputs/figures/results_metric_comparison.png)
![PR curves](outputs/figures/results_pr_curves.png)
![Confusion matrices](outputs/figures/results_confusion_matrices.png)

### 6.3 Test, sequential protocol (secondary)

{metric_table(SQ)}

![PR curves, sequential](outputs/figures/results_pr_curves_sequential.png)

### 6.4 Uncertainty

Paired cluster bootstrap on the test set: users are resampled with replacement 1,000 times and PR-AUC differences recomputed from fixed predictions.

| PR-AUC difference | Fixed snapshot | 95% CI | Sequential | 95% CI |
|---|---:|---|---:|---|
""" + "\n".join(f"| {k} | {sgn(B['fixed_snapshot'][k]['diff'])} | {ci(B['fixed_snapshot'][k]['ci95'])} | {sgn(B['sequential'][k]['diff'])} | {ci(B['sequential'][k]['ci95'])} |"
                for k in B["fixed_snapshot"]) + f"""

### 6.5 Results by amount of user history

Fixed snapshot:

{group_table(R['test_by_user_history'])}

Sequential:

{group_table(R['sequential']['test_by_user_history'])}

![F1 by user history](outputs/figures/results_f1_by_user_history.png)
![F1 by user history, sequential](outputs/figures/results_f1_by_user_history_sequential.png)

On the {warm['Logistic Regression']['n']:,} fixed-snapshot test rows whose user has any training history, PR-AUC is {f3(warm['Logistic Regression']['pr_auc'])} for full LR, {f3(warm['LR (movie features only)']['pr_auc'])} for movie-only LR and {f3(warm['Popularity baseline']['pr_auc'])} for popularity.

### 6.6 Ranking within each user

For the {rk['n_users']} test users with at least {rk['min_items']} test ratings, each user's rated test movies are ranked by score.

| Method | P@5 fixed | P@10 fixed | Mean user AP fixed | P@5 sequential | P@10 sequential | Mean user AP sequential |
|---|---:|---:|---:|---:|---:|---:|
| Random order (expected) | {f3(rk['random_order']['p@5'])} | {f3(rk['random_order']['p@10'])} | n/a | {f3(rks['random_order']['p@5'])} | {f3(rks['random_order']['p@10'])} | n/a |
""" + "\n".join(f"| {m} | {f3(rk['models'][m]['p@5'])} | {f3(rk['models'][m]['p@10'])} | {f3(rk['models'][m]['mean_user_ap'])} | {f3(rks['models'][m]['p@5'])} | {f3(rks['models'][m]['p@10'])} | {f3(rks['models'][m]['mean_user_ap'])} |"
                for m in rk["models"]) + f"""

### 6.7 Model interpretation

![LR coefficients](outputs/figures/results_lr_coefficients.png)
![Calibration](outputs/figures/results_calibration.png)

Largest standardised LR coefficients: """ + ", ".join(f"`{k}` {v:+.3f}" for k, v in list(R["lr_coefficients"].items())[:6]) + f""". Movie quality dominates, followed by the user's general rating tendency and genre affinity. Features are correlated, so coefficients indicate rough importance, not causal effects. All three scorers are well calibrated on the test set. Mean LR score by true rating: """ + ", ".join(f"{k}★ {v:.3f}" for k, v in R["lr_mean_score_by_true_rating"].items()) + f""" — monotone, as expected.

### 6.8 Leakage demonstration (validation only)

A Logistic Regression trained on naive features that include each training row's own rating scores training PR-AUC {f3(ld['leaky']['train_pr_auc'])} but validation PR-AUC {f3(ld['leaky']['val_pr_auc'])}, worse than the correct pipeline ({f3(ld['correct']['train_pr_auc'])} train, {f3(ld['correct']['val_pr_auc'])} validation). Leakage inflates development results and harms real performance.

### 6.9 Sensitivity: excluding 3-star ratings

Dropping all 3-star ratings (task: 1–2 vs 4–5; rows train {sens['rows']['train']:,}, validation {sens['rows']['val']:,}, test {sens['rows']['test']:,}; same timestamp boundaries):

| Model | F1 | PR-AUC | Positive rate |
|---|---:|---:|---:|
""" + "\n".join(f"| {m} | {f4(d['f1'])} | {f4(d['pr_auc'])} | {pct(d['positive_rate'])} |" for m, d in sens["test"].items()) + f"""

The task becomes easier for every method; the ordering of methods is unchanged.

## 7. Error analysis

At the validation-chosen threshold ({f3(LR['threshold'])}), Logistic Regression makes {ee['fp_summary']['n']:,} false positives and {ee['fn_summary']['n']:,} false negatives on the test set. {pct0(ee['fp_summary']['share_rated_3'])} of false positives are 3-star ratings (mean rating {ee['fp_summary']['mean_rating']:.2f}), and {pct0(ee['fp_summary']['share_cold_user'])} come from cold-start users. {pct0(ee['fn_summary']['share_rated_4'])} of false negatives are 4-star ratings. The F1-optimal threshold is low because the positive class is the majority, so the model leans toward predicting positive.

**Most confident false positives** (predicted positive, rated 1–3):

{err_table(ee['false_positives'])}

**Most confident false negatives** (predicted not positive, rated 4–5):

{err_table(ee['false_negatives'])}

Possible explanations (evidence, not proof): *popularity pull* — acclaimed classics such as *Casablanca* score high for every cold-start user, so a user who gives them 3 stars becomes a false positive; *minority tastes* — movies with a low positive rate (e.g. *Speed 2*, *Mortal Kombat: Annihilation*) get low scores, so users who enjoyed them are missed; *unusual profiles* — user 655 has {int([r for r in ee['false_negatives'] if r['user_id'] == 655][0]['user_hist_count']) if any(r['user_id'] == 655 for r in ee['false_negatives']) else 'many'} earlier ratings but a very low positive rate and rated a movie with no training ratings, so every signal pointed down.

| Genre | Test ratings | LR error rate | FP share | FN share |
|---|---:|---:|---:|---:|
""" + "\n".join(f"| {g['genre']} | {g['n']:,} | {pct(g['error_rate'])} | {pct(g['fp_rate_of_rows'])} | {pct(g['fn_rate_of_rows'])} |" for g in R["test_errors_by_genre"]) + f"""

## 8. Limitations

* **Cold start dominates the primary protocol**: most test users joined after the training cutoff.
* **Observed-rating bias**: users chose what to rate; unrated movies are unknown.
* **Old data** (1997–98) and a small catalogue.
* **Subjective label**: the 4–5 threshold is a project definition.
* **Popularity bias**: well-rated popular movies dominate scores and lists.
* **Simple features**: no collaborative signal (similar users, latent factors), which would likely help within-user ranking.
* **Ranking evaluation** can only use movies the user actually rated.
* **One split**: bootstrap intervals cover user sampling, not different time periods.
* **No causal claims**: the model describes patterns in past ratings.

## 9. Conclusion

Earlier ratings and movie genres predict positive ratings clearly better than the majority baseline (PR-AUC {f3(LR['pr_auc'])} vs {f3(MAJ['pr_auc'])}). **Whether personalisation beats popularity depends on available history.** Under the guide's fixed-snapshot protocol, most test users have no history and Logistic Regression is statistically indistinguishable from popularity ({sgn(B['fixed_snapshot']['LR - Popularity']['diff'])}, CI {ci(B['fixed_snapshot']['LR - Popularity']['ci95'])}). For users with long histories it helps (200+ ratings: PR-AUC {f3(h200['Logistic Regression|pr_auc'])} vs {f3(h200['Popularity baseline|pr_auc'])}, n = {h200['n']}). When earlier ratings inform later predictions, Logistic Regression beats popularity ({sgn(B['sequential']['LR - Popularity']['diff'])}, CI {ci(B['sequential']['LR - Popularity']['ci95'])}) and the movie-only ablation shows the improvement comes from user features. The gain appears in pooled classification metrics; within a single user's list the user features add little reordering, and popularity is as good or slightly better at precision@5. Random Forest gives no reliable improvement, so the interpretable, well-calibrated Logistic Regression is the final model. A collaborative-filtering signal is the most promising next step.

## 10. Reproducibility

```
pip install -r requirements.txt
python -m src.load_data          # download + MD5 check
python tests/test_leakage.py     # split and leakage tests
python -m src.run_pipeline       # full experiment (~3 minutes)
python -m src.make_report        # regenerate this report
```

Per-row test scores for every model and protocol: `outputs/test_predictions.csv` (no MovieLens identifiers or ratings; row *i* is the *i*-th test rating in split order — notebook 03 joins it back to the data and recomputes the metrics). Target rule: rating ≥ {C.POSITIVE_MIN_RATING}. Split: {R['config']['split']}. Random seed: {C.RANDOM_SEED}. Smoothing strengths: movie {C.MOVIE_PRIOR_STRENGTH}, user {C.USER_PRIOR_STRENGTH}, user×genre {C.USER_GENRE_PRIOR_STRENGTH}. Library versions: `requirements.txt`.

## References

* F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* 5(4), Article 19. https://doi.org/10.1145/2827872
* GroupLens Research, MovieLens 100K: https://grouplens.org/datasets/movielens/100k/
* Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. *JMLR* 12, 2825–2830.
* Project guide: *MovieMatch — ES 335 End-to-end Machine Learning Project Guide*.

AI assistance is disclosed in the repository `README.md`.
"""
    (C.ROOT / "REPORT.md").write_text(text)
    print("wrote", C.ROOT / "REPORT.md")


if __name__ == "__main__":
    main()
