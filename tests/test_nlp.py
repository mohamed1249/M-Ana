import sys
import unittest


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_preserves_negation_and_removes_noise(self):
        from MAna.nlp import TextCleaner

        cleaner = TextCleaner(stop_words="english", min_token_length=2)
        cleaned = cleaner.clean(
            "<p>This is NOT great!</p> Visit https://example.com "
            "or email me@example.com."
        )

        self.assertIn("not great", cleaned)
        self.assertNotIn("example", cleaned)
        self.assertNotIn("<p>", cleaned)

    def test_word_chunking_has_overlap_and_source_offsets(self):
        from MAna.nlp import chunk_text

        source = "one two three four five six"
        chunks = chunk_text(source, chunk_size=3, overlap=1)

        self.assertEqual([chunk.text for chunk in chunks], [
            "one two three",
            "three four five",
            "five six",
        ])
        self.assertEqual(source[chunks[1].start : chunks[1].end], "three four five")
        self.assertEqual(chunks[1].document_index, 0)


class VectorizationTests(unittest.TestCase):
    def test_vectorization_and_similarity(self):
        from MAna.nlp import TextVectorizer

        result = TextVectorizer(method="tfidf").fit_transform(
            ["python data analysis", "football match goal", "python pandas data"]
        )

        self.assertEqual(result.shape[0], 3)
        self.assertIn("python", result.feature_names)
        similarity = result.similarity_matrix()
        self.assertGreater(similarity[0, 2], similarity[0, 1])

    def test_hashing_vectorizer_accepts_explicit_feature_count(self):
        from MAna.nlp import TextVectorizer

        result = TextVectorizer(method="hashing", n_features=32).fit_transform(
            ["alpha beta", "beta gamma"]
        )
        self.assertEqual(result.shape, (2, 32))


class AnalysisTests(unittest.TestCase):
    def test_sentiment_handles_positive_and_negated_language(self):
        from MAna.nlp import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        positive = analyzer.analyze("This is really great")
        negative = analyzer.analyze("This is not good")

        self.assertEqual(positive.label, "positive")
        self.assertEqual(negative.label, "negative")
        self.assertAlmostEqual(
            positive.positive + positive.neutral + positive.negative, 1.0
        )

    def test_topic_model_separates_two_small_themes(self):
        from MAna.nlp import TopicModeler

        corpus = [
            "python data analysis pandas numpy",
            "machine learning model data features",
            "python model training sklearn data",
            "football team scored match goal",
            "basketball team won game score",
            "football player goal league match",
        ]
        result = TopicModeler(n_topics=2, n_top_words=5).fit_transform(corpus)
        all_terms = {term for topic in result.topics for term in topic.terms}

        self.assertEqual(result.document_topics.shape, (6, 2))
        self.assertTrue({"python", "data"} & all_terms)
        self.assertTrue({"football", "match", "goal"} & all_terms)

    def test_semantic_search_ranks_related_document_first(self):
        from MAna.nlp import semantic_search

        results = semantic_search(
            "python data",
            ["cooking recipe", "python data analysis", "football match"],
            top_k=2,
        )
        self.assertEqual(results.iloc[0]["text"], "python data analysis")
        self.assertGreater(results.iloc[0]["score"], results.iloc[1]["score"])

    def test_import_does_not_eagerly_load_transformers(self):
        import MAna

        _ = MAna.nlp
        self.assertNotIn("transformers", sys.modules)
        self.assertNotIn("sentence_transformers", sys.modules)


if __name__ == "__main__":
    unittest.main()
