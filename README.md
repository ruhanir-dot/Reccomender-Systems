# Recommender Systems

Coursework for **ECS 172 (Recommender Systems)** — Ruhani Rekhi.

This repo is just a display of the projects built for that class. Each folder is a self-contained
assignment with its own data exploration notebook, model notebook(s), written report, and the
`submission.csv` that was turned in. Datasets are **not** committed (storage reasons), so notebooks
expect the assignment CSVs to be dropped into their folder before running.

## Projects

| # | Project | Task | Main approach | Result |
|---|---------|------|---------------|--------|
| 1 | [Content-Based-Filtering/](Content-Based-Filtering/) | Rank 5 candidate items per user by content relevance | TF-IDF item vectors + rating/recency-weighted user profiles, cosine similarity | Val MRR **0.5782** (random baseline 0.4567) |
| 2 | [Collaborative-Filtering/](Collaborative-Filtering/) | Predict movie ratings (1–5) | Six methods of increasing complexity, ending in a genre-personalized hybrid item-item CF | Val RMSE **0.8750** (global mean baseline 1.1201) |
| 3 | [Multi-stage-Recommendation/](Multi-stage-Recommendation/) | Predict the top-10 Steam games each user plays next | Two-stage: 4-channel retrieval (TF-IDF + ItemKNN + ALS + BPR) → LightGBM LambdaRank re-ranker on 22 features | Mean of Recall@10 / NDCG@10 **0.1034** (~60× the random baseline of 0.0017) |

### 1. Content-Based Filtering — next item retrieval

Software-product reviews (2,000 users, 8,681 items, 32,206 interactions). Items are represented with
TF-IDF over title / description / features / categories / store, users as a weighted average of the
items they reviewed, and the 5 test candidates are ranked by cosine similarity to the user profile.

- Submitted model: [final_submission_model.ipynb](Content-Based-Filtering/final_submission_model.ipynb)
- Data exploration: [data_exploration.ipynb](Content-Based-Filtering/data_exploration.ipynb)
- Assignment write-up: [README.md](Content-Based-Filtering/README.md)

### 2. Collaborative Filtering — rating prediction

Movie ratings with a per-user temporal 80/20 split (635,723 train / 162,035 validation). Progression
of methods: global mean → user/item mean → regularized bias model → item-item CF on top of the bias
baseline → genre- and decade-aware hybrid CF.

| Method | Val RMSE |
|---|---|
| Global mean | 1.1201 |
| User mean | 1.0466 |
| Item mean | 0.9853 |
| Bias model (λ_u=5, λ_i=1) | 0.9130 |
| Item-CF (K=30, bias baseline) | 0.8798 |
| Genre-hybrid CF (K=30, λ_genre=5, λ_decade=25) | **0.8750** |

- Report: [report.md](Collaborative-Filtering/report.md) · [report.pdf](Collaborative-Filtering/report.pdf)
- Notebooks: [baselineModel.ipynb](Collaborative-Filtering/baselineModel.ipynb),
  [itemCollaborativeFiltering.ipynb](Collaborative-Filtering/itemCollaborativeFiltering.ipynb),
  [hybridModel.ipynb](Collaborative-Filtering/hybridModel.ipynb)
- Data exploration: [data_exploration.ipynb](Collaborative-Filtering/data_exploration.ipynb)

### 3. Multi-Stage Steam Game Recommender

10,000 users, a 32,132-game catalog, 122,366 interactions, and a 99.85% sparse user-item matrix.
Stage 1 fuses four retrieval channels into ~360 candidates per user; stage 2 re-ranks them with a
LightGBM LambdaRank model over 22 engineered features. The report also covers cold-user / cold-item
handling and retrieval recall.

- Model: [multi_stage_recsys.ipynb](Multi-stage-Recommendation/multi_stage_recsys.ipynb)
  — run with `jupyter nbconvert --execute --to notebook --inplace multi_stage_recsys.ipynb`
- Report: [report.md](Multi-stage-Recommendation/report.md) · [report.pdf](Multi-stage-Recommendation/report.pdf)
- Data exploration: [data_exploration.ipynb](Multi-stage-Recommendation/data_exploration.ipynb)
- Pipeline diagram: [image_two.png](Multi-stage-Recommendation/image_two.png)

## Stack

Python, Jupyter, pandas / NumPy / SciPy sparse, scikit-learn (TF-IDF, cosine similarity),
implicit (ALS, BPR), LightGBM, matplotlib / seaborn.
