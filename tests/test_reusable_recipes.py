import copy
import tempfile
import unittest

import numpy as np
import pandas as pd


class NLPRecipeTests(unittest.TestCase):
    def test_mixed_feature_classifier_and_evaluation(self):
        from MAna.nlp import build_text_classifier, evaluate_classifier

        frame = pd.DataFrame(
            {
                "review": [
                    "excellent quality love it",
                    "great product works perfectly",
                    "happy with this purchase",
                    "reliable and very good",
                    "terrible quality broke quickly",
                    "bad product does not work",
                    "awful and disappointing",
                    "unreliable waste of money",
                ],
                "age": [25, 31, 42, 38, 28, 35, 44, 30],
                "department": ["A", "A", "B", "B", "A", "A", "B", "B"],
            }
        )
        target = np.asarray([1, 1, 1, 1, 0, 0, 0, 0])
        model = build_text_classifier(
            "review",
            numeric_columns=["age"],
            categorical_columns=["department"],
        )
        model.fit(frame, target)
        metrics = evaluate_classifier(model, frame, target)

        self.assertGreaterEqual(metrics["accuracy"], 0.75)
        self.assertIn("f1_macro", metrics)
        self.assertEqual(metrics["confusion_matrix"].shape, (2, 2))

    def test_segmented_embeddings_preserve_fixed_sections(self):
        from MAna.nlp import encode_segmented_texts

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([[len(text), len(text.split())] for text in texts])

        vectors = encode_segmented_texts(
            ["one two three four", "short text"],
            n_segments=3,
            encoder=FakeEncoder(),
        )
        self.assertEqual(vectors.shape, (2, 6))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


class RAGRecipeTests(unittest.TestCase):
    def test_semantic_index_search_save_and_load(self):
        from MAna.rag import SemanticIndex

        index = SemanticIndex(backend="tfidf").fit(
            ["python data analysis", "football match goal", "cooking pasta"],
            metadata=[
                {"source": "notes.pdf", "page": 1},
                {"source": "sports.pdf", "page": 4},
                {"source": "food.pdf", "page": 2},
            ],
        )
        hit = index.search("python analysis", top_k=1)[0]
        self.assertEqual(hit.metadata["source"], "notes.pdf")

        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            restored = SemanticIndex.load(directory)
            restored_hit = restored.search("python analysis", top_k=1)[0]
        self.assertEqual(restored_hit.id, hit.id)
        self.assertEqual(restored_hit.text, hit.text)

    def test_rank_fusion_supports_weights_and_stable_ids(self):
        from MAna.rag import reciprocal_rank_fusion

        ranking = reciprocal_rank_fusion(
            {
                "semantic": ["movie-1", "movie-2", "movie-3"],
                "metadata": ["movie-2", "movie-3", "movie-1"],
            },
            weights={"semantic": 1.0, "metadata": 2.0},
        )
        self.assertEqual(ranking[0], "movie-2")
        self.assertEqual(set(ranking), {"movie-1", "movie-2", "movie-3"})

    def test_grounded_messages_include_citations(self):
        from MAna.rag import build_grounded_messages

        messages = build_grounded_messages(
            "What predicts churn?",
            [
                {
                    "text": "Usage decline is a leading signal.",
                    "metadata": {"source": "report.pdf", "page": 7},
                    "score": 0.91,
                }
            ],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("[Source 1: report.pdf, page 7", messages[1]["content"])

    def test_history_trimming_keeps_order_and_does_not_mutate(self):
        from MAna.rag import trim_chat_history

        messages = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "old question with many words"},
            {"role": "assistant", "content": "old answer with many words"},
            {"role": "user", "content": "new question"},
        ]
        original = copy.deepcopy(messages)
        trimmed = trim_chat_history(messages, max_tokens=5)

        self.assertEqual(messages, original)
        self.assertEqual([message["role"] for message in trimmed], ["system", "user"])
        self.assertEqual(trimmed[-1]["content"], "new question")

    def test_vector_upserts_retry_and_use_stable_ids(self):
        from MAna.rag import upsert_embedding_batches

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([[len(text), 1.0] for text in texts])

        class FlakyIndex:
            def __init__(self):
                self.calls = 0
                self.payloads = []

            def upsert(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionError("temporary")
                self.payloads.extend(kwargs["vectors"])

        records = [
            {"text": "first document", "metadata": {"source": "a"}},
            {"text": "second document", "metadata": {"source": "b"}},
        ]
        destination = FlakyIndex()
        written = upsert_embedding_batches(
            destination,
            records,
            FakeEncoder(),
            batch_size=2,
            max_retries=1,
            retry_delay=0,
        )

        self.assertEqual(written, 2)
        self.assertEqual(destination.calls, 2)
        self.assertEqual(len({item["id"] for item in destination.payloads}), 2)
        np.testing.assert_allclose(
            [np.linalg.norm(item["values"]) for item in destination.payloads],
            [1.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
