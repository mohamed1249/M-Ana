import unittest

import numpy as np
import pandas as pd
from scipy import sparse


class InteractionDataTests(unittest.TestCase):
    def test_binary_matrix_aggregates_duplicate_interactions(self):
        from MAna.recommend import build_interaction_matrix

        interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2"],
                "item_id": ["a", "a", "b", "b"],
            }
        )
        result = build_interaction_matrix(interactions)

        self.assertEqual(result.matrix.shape, (2, 2))
        self.assertEqual(result.matrix[result.user_index("u1"), result.item_index("a")], 1)
        self.assertEqual(result.matrix.nnz, 3)

    def test_temporal_split_holds_out_each_users_latest_item(self):
        from MAna.recommend import temporal_train_test_split

        interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2", "u2", "u3"],
                "item_id": ["a", "b", "c", "a", "d", "x"],
                "timestamp": [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-04",
                    "2026-01-01",
                ],
            }
        )
        train, test = temporal_train_test_split(
            interactions,
            timestamp_column="timestamp",
        )

        self.assertEqual(set(test["item_id"]), {"c", "d"})
        self.assertIn("x", train["item_id"].tolist())
        for user_id in test["user_id"].unique():
            train_latest = train.loc[train["user_id"] == user_id, "timestamp"].max()
            test_earliest = test.loc[test["user_id"] == user_id, "timestamp"].min()
            self.assertLess(train_latest, test_earliest)


class PopularityRecommenderTests(unittest.TestCase):
    def test_bayesian_popularity_excludes_seen_items(self):
        from MAna.recommend import PopularityRecommender

        interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2", "u3"],
                "item_id": ["a", "b", "a", "c", "c"],
                "rating": [5, 4, 4, 5, 5],
            }
        )
        model = PopularityRecommender(
            score_column="rating",
            score_mode="bayesian",
            shrinkage=2,
        ).fit(interactions)

        personalized = model.recommend("u1", top_n=3)
        cold_start = model.recommend("unknown", top_n=3)

        self.assertEqual(personalized["item_id"].tolist(), ["c"])
        self.assertEqual(cold_start.iloc[0]["item_id"], "c")
        self.assertEqual(personalized.iloc[0]["source"], "popularity")


class ContentRecommenderTests(unittest.TestCase):
    def test_feature_preparation_is_reusable(self):
        from MAna.recommend import combine_text_features, prepare_numeric_features

        items = pd.DataFrame(
            {
                "title": ["Red", "Blue"],
                "genre": ["Drama", None],
                "year": [2020, None],
            }
        )
        text = combine_text_features(
            items,
            ["title", "genre"],
            weights={"genre": 2},
        )
        numeric, transformer = prepare_numeric_features(items, ["year"])
        transformed = transformer.transform(pd.DataFrame({"year": [2021, None]}))

        self.assertEqual(text.iloc[0], "Red Drama Drama")
        self.assertEqual(numeric.shape, (2, 1))
        self.assertEqual(transformed.shape, (2, 1))
        self.assertTrue(np.isfinite(transformed).all())

    def test_content_profiles_and_similar_items_support_sparse_features(self):
        from MAna.recommend import ContentRecommender

        features = sparse.csr_matrix(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        model = ContentRecommender().fit(
            ["a", "b", "c", "d"],
            features,
            user_histories={"u1": ["a"]},
        )

        recommendations = model.recommend("u1", top_n=2)
        neighbors = model.similar_items("c", top_n=1)

        self.assertEqual(recommendations.iloc[0]["item_id"], "b")
        self.assertNotIn("a", recommendations["item_id"].tolist())
        self.assertEqual(neighbors.iloc[0]["item_id"], "d")


class CollaborativeRecommenderTests(unittest.TestCase):
    def test_item_knn_uses_cooccurrence_without_recommending_seen_items(self):
        from MAna.recommend import ItemKNNRecommender

        interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"],
                "item_id": ["a", "b", "a", "c", "a", "c", "b", "d"],
            }
        )
        model = ItemKNNRecommender(n_neighbors=3).fit(interactions)

        recommendations = model.recommend("u1", top_n=2)
        neighbors = model.similar_items("a", top_n=1)

        self.assertEqual(recommendations.iloc[0]["item_id"], "c")
        self.assertTrue({"a", "b"}.isdisjoint(recommendations["item_id"]))
        self.assertEqual(neighbors.iloc[0]["item_id"], "c")


class HybridRecommenderTests(unittest.TestCase):
    def test_weighted_fusion_calibrates_component_scores(self):
        from MAna.recommend import weighted_score_fusion

        frames = {
            "content": pd.DataFrame(
                {"item_id": ["a", "b"], "score": [1.0, 0.5]}
            ),
            "collaborative": pd.DataFrame(
                {"item_id": ["b", "c"], "score": [1.0, 0.2]}
            ),
        }
        result = weighted_score_fusion(
            frames,
            weights={"content": 1.0, "collaborative": 2.0},
            top_n=3,
        )

        self.assertEqual(result.iloc[0]["item_id"], "b")
        self.assertEqual(result.iloc[0]["component_scores"]["collaborative"], 1.0)

    def test_hybrid_recommender_accepts_shared_contract(self):
        from MAna.recommend import HybridRecommender

        class StaticRecommender:
            def __init__(self, items):
                self.items = items

            def recommend(self, user_id, top_n=10, exclude_items=()):
                available = [item for item in self.items if item not in set(exclude_items)]
                return pd.DataFrame(
                    {
                        "item_id": available[:top_n],
                        "score": np.arange(len(available), 0, -1)[:top_n],
                    }
                )

        hybrid = HybridRecommender(
            {
                "one": StaticRecommender(["a", "b", "c"]),
                "two": StaticRecommender(["b", "c", "a"]),
            },
            method="rrf",
            weights={"one": 1.0, "two": 2.0},
        )
        result = hybrid.recommend("u1", top_n=2)

        self.assertEqual(result.iloc[0]["item_id"], "b")
        self.assertEqual(result.iloc[0]["source"], "hybrid_rrf")


class RecommenderEvaluationTests(unittest.TestCase):
    def test_ranking_metrics_coverage_and_diversity(self):
        from MAna.recommend import (
            evaluate_rankings,
            intra_list_diversity,
        )

        recommendations = {"u1": ["a", "x"], "u2": ["x", "b"]}
        relevant = {"u1": ["a"], "u2": ["b"]}
        metrics = evaluate_rankings(
            recommendations,
            relevant,
            k=2,
            catalog_items=["a", "b", "x", "z"],
        )
        diversity = intra_list_diversity(
            ["a", "b"],
            ["a", "b"],
            np.eye(2),
        )

        self.assertAlmostEqual(metrics["precision_at_k"], 0.5)
        self.assertAlmostEqual(metrics["recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["mrr_at_k"], 0.75)
        self.assertAlmostEqual(metrics["catalog_coverage"], 0.75)
        self.assertAlmostEqual(diversity, 1.0)

    def test_new_recommender_helpers_are_public(self):
        import MAna
        import MAna.recommend as recommend

        expected = {
            "ContentRecommender",
            "HybridRecommender",
            "ItemKNNRecommender",
            "PopularityRecommender",
            "evaluate_rankings",
            "temporal_train_test_split",
            "weighted_score_fusion",
        }
        self.assertIn("recommend", MAna.__all__)
        self.assertTrue(expected.issubset(set(recommend.__all__)))


if __name__ == "__main__":
    unittest.main()
