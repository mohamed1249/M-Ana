import os
import unittest
from unittest import mock

import numpy as np


class RAGIngestionTests(unittest.TestCase):
    def test_chunk_records_preserve_metadata_offsets_and_stable_ids(self):
        from MAna.rag import chunk_document_records

        documents = [
            {
                "id": "report-1",
                "text": "one two three four five",
                "metadata": {"source": "report.pdf", "page": 2},
            }
        ]
        records = chunk_document_records(documents, chunk_size=3, overlap=1)
        repeated = chunk_document_records(documents, chunk_size=3, overlap=1)

        self.assertEqual([record["text"] for record in records], [
            "one two three",
            "three four five",
        ])
        self.assertEqual(records[0]["metadata"]["source"], "report.pdf")
        self.assertEqual(records[0]["metadata"]["page"], 2)
        self.assertEqual(records[0]["metadata"]["document_id"], "report-1")
        self.assertEqual(records[0]["metadata"]["chunk_index"], 0)
        self.assertLess(records[0]["metadata"]["start"], records[0]["metadata"]["end"])
        self.assertEqual(
            [record["id"] for record in records],
            [record["id"] for record in repeated],
        )


class RAGPromptingTests(unittest.TestCase):
    def test_deduplication_and_pinecone_metadata_text(self):
        from MAna.rag import (
            build_cited_context,
            build_grounded_messages,
            deduplicate_hits,
        )

        hits = [
            {
                "id": "doc-1",
                "score": 0.9,
                "metadata": {
                    "text": "Usage decline predicts churn.",
                    "source": "report.pdf",
                    "page": 7,
                },
            },
            {
                "id": "doc-1",
                "score": 0.8,
                "metadata": {"text": "duplicate", "source": "report.pdf"},
            },
        ]
        unique = deduplicate_hits(hits)
        context = build_cited_context(unique)
        messages = build_grounded_messages("What predicts churn?", unique)

        self.assertEqual(len(unique), 1)
        self.assertIn("Usage decline predicts churn.", context)
        self.assertIn("report.pdf, page 7", context)
        self.assertIn("untrusted reference material", messages[0]["content"])


class GenerationAdapterTests(unittest.TestCase):
    def test_environment_variables_are_required(self):
        from MAna.rag import require_environment_variable

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_environment_variable("MANA_TEST_SECRET")

        with mock.patch.dict(os.environ, {"MANA_TEST_SECRET": "configured"}, clear=True):
            self.assertEqual(
                require_environment_variable("MANA_TEST_SECRET"),
                "configured",
            )

    def test_chat_completion_extracts_structured_content(self):
        from MAna.rag import chat_completion

        class Completions:
            def __init__(self):
                self.payload = None

            def create(self, **kwargs):
                self.payload = kwargs
                return {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": "grounded"},
                                    {"text": " answer"},
                                ]
                            }
                        }
                    ]
                }

        completions = Completions()
        client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": completions})()},
        )()
        answer = chat_completion(
            client,
            [{"role": "user", "content": "question"}],
            model="test-model",
        )

        self.assertEqual(answer, "grounded answer")
        self.assertEqual(completions.payload["model"], "test-model")
        self.assertEqual(completions.payload["max_tokens"], 800)

    def test_openai_embedding_adapter_batches_orders_and_normalizes(self):
        from MAna.rag import OpenAIEmbeddingModel

        class Embeddings:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                data = [
                    {"index": index, "embedding": [float(len(text)), 1.0]}
                    for index, text in enumerate(kwargs["input"])
                ]
                return {"data": list(reversed(data))}

        embeddings = Embeddings()
        client = type("Client", (), {"embeddings": embeddings})()
        adapter = OpenAIEmbeddingModel(client, batch_size=2)
        vectors = adapter.encode(["a", "four", "seven77"])

        self.assertEqual(len(embeddings.calls), 2)
        self.assertEqual(vectors.shape, (3, 2))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0)
        self.assertLess(vectors[0, 0], vectors[1, 0])

    def test_provider_neutral_document_task(self):
        from MAna.rag import run_document_task

        captured = []

        def generate(messages, **kwargs):
            captured.extend(messages)
            return {"content": "short summary"}

        answer = run_document_task(
            "A long document",
            task="summarize",
            generate=generate,
        )

        self.assertEqual(answer, "short summary")
        self.assertIn("Summarize", captured[0]["content"])


class RAGPipelineTests(unittest.TestCase):
    def test_pipeline_runs_with_history_kwargs_and_sources(self):
        from MAna.rag import RAGPipeline

        calls = {}

        def retrieve(query, **kwargs):
            calls["query"] = query
            calls["retrieve_kwargs"] = kwargs
            hit = {
                "id": "doc-1",
                "text": "Evidence text",
                "score": 0.95,
                "metadata": {"source": "guide.pdf", "page": 3},
            }
            return [hit, dict(hit)]

        def generate(messages, **kwargs):
            calls["messages"] = messages
            calls["generation_kwargs"] = kwargs
            return {"answer": "Evidence-based answer"}

        pipeline = RAGPipeline(retrieve, generate, history_max_tokens=20)
        result = pipeline.run(
            "How does it work?",
            history=[
                {"role": "system", "content": "untrusted old system"},
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
            retrieve_kwargs={"top_k": 2},
            generation_kwargs={"temperature": 0},
        )

        self.assertEqual(result.answer, "Evidence-based answer")
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.sources[0]["source"], "guide.pdf")
        self.assertEqual(result.sources[0]["page"], 3)
        self.assertEqual(calls["retrieve_kwargs"], {"top_k": 2})
        self.assertEqual(calls["generation_kwargs"], {"temperature": 0})
        self.assertNotIn("untrusted old system", str(calls["messages"]))
        self.assertEqual(calls["messages"][-1]["role"], "user")

    def test_retrieval_metrics_cover_rank_quality(self):
        from MAna.rag import retrieval_metrics

        metrics = retrieval_metrics(["a", "x", "b", "b"], ["a", "b"], k=3)
        empty = retrieval_metrics([], ["a"])

        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["reciprocal_rank"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertGreater(metrics["ndcg"], 0.9)
        self.assertEqual(empty["average_precision"], 0.0)
        self.assertEqual(empty["ndcg"], 0.0)

    def test_new_rag_helpers_are_public(self):
        import MAna.rag as rag

        expected = {
            "OpenAIEmbeddingModel",
            "RAGPipeline",
            "RAGResult",
            "chat_completion",
            "chunk_document_records",
            "deduplicate_hits",
            "openai_embeddings",
            "rag_answer",
            "retrieval_metrics",
            "run_document_task",
        }
        self.assertTrue(expected.issubset(set(rag.__all__)))


if __name__ == "__main__":
    unittest.main()
