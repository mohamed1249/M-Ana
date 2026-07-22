# Recommenders Module Notebook

This notebook documents `MAna.recommend`, the recommendation layer for
popularity baselines, content-based retrieval, item-neighborhood collaborative
filtering, hybrid fusion, and offline top-k evaluation.

It uses real data from an old MovieLens-style recommender project: 100k user
ratings plus a compact TMDB movie catalog with titles, overviews, genres,
keywords, votes, and runtime. The examples keep the interaction data real and
select eligible users from the training set instead of inventing synthetic
profiles.

Open it here:

[MAna.recommend - Popularity, Content, Collaborative, Hybrid, and Evaluation](../notebooks/recommenders_module.ipynb)

The code comments are intentionally detailed because recommender systems are
easy to overfit quietly: random splits leak future behavior, popularity can
hide catalog bias, content profiles need consistent feature preparation, and
offline metrics should be read beside coverage and diversity.
