# NLP Module Notebook

This notebook documents `MAna.nlp`, the text-processing layer for practical
data-science workflows: cleaning, tokenization, chunking, vectorization,
sentiment analysis, topic modeling, embeddings, semantic search, and supervised
text classification.

It uses real datasets from the documentation set: women's clothing reviews,
TMDB movie overviews, and YouTube trending-video metadata. The examples stay
local by default, while optional transformer-backed cells show how to move into
larger NLP models when those dependencies and model weights are available.

Open it here:

[MAna.nlp - Text Cleaning, Search, Topics, and Classification](../notebooks/nlp_module.ipynb)

The code comments are intentionally detailed because NLP workflows can look
simple while hiding important choices: whether negations survive cleaning,
whether a vocabulary is refit after splitting data, how chunk metadata is
preserved, and when optional model downloads should be kept out of a default
run.
