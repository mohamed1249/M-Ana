# Analysis Module Notebook

This notebook documents `MAna.analysis`, the visualization and dashboard helper
module. It uses multiple real samples from older project work: clothing reviews,
YouTube trending data, YouTube category names, TMDB movies, and electricity load
data.

The notebook focuses on matching each chart family with a suitable data problem:
review distributions, YouTube trends, electricity-load animation, movie
hierarchy, genre flow, financial waterfall analysis, funnels, and dashboard
starters.

Open it here:

[MAna.analysis - Visual Exploration and Communication](../notebooks/analysis_module.ipynb)

The notebook includes heavier code comments because plotting examples are often
less about the function call itself and more about preparing the right table for
the visual question.

Dashboard examples point to `http://127.0.0.1:8050/` and use the new
`run=True` path when running inside Jupyter. The `pairplot()` example also shows
that `size` can be either a column name or a fixed marker size such as `20`.
