"""Popularity, content, collaborative, hybrid, and evaluation utilities."""

from ..rag.ranking import RankedItem, fuse_rankings, reciprocal_rank_fusion
from .base import BaseRecommender, recommendation_frame
from .collaborative import ItemKNNRecommender
from .content import (
    ContentRecommender,
    NumericFeatureTransformer,
    combine_text_features,
    concatenate_feature_embeddings,
    prepare_numeric_features,
)
from .data import (
    InteractionMatrix,
    build_interaction_matrix,
    build_user_histories,
    temporal_train_test_split,
    validate_interactions,
)
from .evaluation import (
    catalog_coverage,
    evaluate_rankings,
    evaluate_recommender,
    intra_list_diversity,
)
from .hybrid import HybridRecommender, weighted_score_fusion
from .popularity import PopularityRecommender


__all__ = [
    "BaseRecommender",
    "ContentRecommender",
    "HybridRecommender",
    "InteractionMatrix",
    "ItemKNNRecommender",
    "NumericFeatureTransformer",
    "PopularityRecommender",
    "RankedItem",
    "build_interaction_matrix",
    "build_user_histories",
    "catalog_coverage",
    "combine_text_features",
    "concatenate_feature_embeddings",
    "evaluate_rankings",
    "evaluate_recommender",
    "fuse_rankings",
    "intra_list_diversity",
    "prepare_numeric_features",
    "recommendation_frame",
    "reciprocal_rank_fusion",
    "temporal_train_test_split",
    "validate_interactions",
    "weighted_score_fusion",
]
