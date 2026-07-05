# Evaluate a recommender without leakage

Use a temporal holdout so each user's test interaction occurs after their
training interactions.

```python
from MAna.recommend import PopularityRecommender, temporal_train_test_split

train, test = temporal_train_test_split(
    interactions,
    user_col="user_id",
    item_col="item_id",
    timestamp_col="timestamp",
)

model = PopularityRecommender().fit(train)
recommendations = model.recommend(user_id="u-42", top_n=10)
```

Compare sophisticated models with the popularity baseline and report ranking
quality together with coverage or diversity.
