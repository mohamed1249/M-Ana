# Quick start

## Install for development

```bash
git clone https://github.com/mohamed1249/M-Ana.git
cd M-Ana
python -m pip install -e .
```

Add feature groups only when needed:

```bash
python -m pip install -e ".[stata,database,timeseries]"
python -m pip install -e ".[nlp,rag]"
python -m pip install -e ".[big]"
```

## Use a module

```python
from MAna.stata import descriptive_statistics
from MAna.timeseries import chronological_split_and_scale

summary = descriptive_statistics(frame["revenue"])
split = chronological_split_and_scale(X, y, test_size=0.2, gap=7)
```

Optional integrations raise an installation message when their dependency group
is absent. Importing `MAna` itself does not require Spark, Pinecone, Prophet,
TensorFlow, or PyTorch.
