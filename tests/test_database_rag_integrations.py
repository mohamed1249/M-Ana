import unittest

import numpy as np


class DatabaseReusableBlockTests(unittest.TestCase):
    def test_sqlite_schema_and_insert_or_ignore(self):
        from MAna.database import SQLiteConnector

        db = SQLiteConnector(":memory:")
        try:
            executed = db.initialize_schema(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );
                """
            )
            inserted = db.insert_or_ignore(
                "items",
                [{"name": "fatigue"}, {"name": "cough"}, {"name": "fatigue"}],
            )
            result = db.query("SELECT name FROM items ORDER BY name")

            self.assertEqual(executed, 1)
            self.assertEqual(inserted, 2)
            self.assertEqual(result["name"].tolist(), ["cough", "fatigue"])
        finally:
            db.close()

    def test_mongodb_from_uri_uses_ping_and_preserves_uri(self):
        import MAna.database.connectors as connectors
        from MAna.database.connectors import MongoDBConnector

        class FakeAdmin:
            def __init__(self):
                self.commands = []

            def command(self, command_name):
                self.commands.append(command_name)
                return {"ok": 1}

        class FakeClient:
            instances = []

            def __init__(self, uri, **kwargs):
                self.uri = uri
                self.kwargs = kwargs
                self.admin = FakeAdmin()
                FakeClient.instances.append(self)

            def __getitem__(self, database_name):
                return {"docs": f"{database_name}.docs"}

        original_available = connectors.PYMONGO_AVAILABLE
        had_pymongo = hasattr(connectors, "pymongo")
        original_pymongo = getattr(connectors, "pymongo", None)
        connectors.PYMONGO_AVAILABLE = True
        connectors.pymongo = type("FakePymongo", (), {"MongoClient": FakeClient})
        try:
            connector = MongoDBConnector.from_uri(
                "mongodb://user:pass@example.com/mydb?authSource=admin"
            )
            collection = connector.get_collection("docs")
        finally:
            connectors.PYMONGO_AVAILABLE = original_available
            if had_pymongo:
                connectors.pymongo = original_pymongo
            else:
                delattr(connectors, "pymongo")

        self.assertEqual(collection, "mydb.docs")
        self.assertEqual(FakeClient.instances[0].admin.commands, ["ping"])
        self.assertEqual(FakeClient.instances[0].kwargs["serverSelectionTimeoutMS"], 5000)

    def test_mongodb_save_document_uses_atomic_upsert(self):
        from MAna.database import save_mongo_document, upsert_mongo_document

        class FakeCollection:
            def __init__(self):
                self.calls = []

            def update_one(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return {"matched_count": 1}

        collection = FakeCollection()
        result = save_mongo_document(
            collection,
            "report.md",
            "Important content",
            extra_fields={"source": "local"},
        )
        upsert_mongo_document(collection, {"file_name": "report.md"}, {"seen": True})

        self.assertEqual(result, {"matched_count": 1})
        first_args, first_kwargs = collection.calls[0]
        self.assertEqual(first_args[0], {"file_name": "report.md"})
        self.assertEqual(first_args[1]["$set"]["content"], "Important content")
        self.assertEqual(first_args[1]["$set"]["source"], "local")
        self.assertTrue(first_kwargs["upsert"])
        self.assertEqual(collection.calls[1][1]["upsert"], True)


class PineconeAdapterTests(unittest.TestCase):
    def test_pinecone_adapter_creates_index_upserts_and_searches(self):
        from MAna.rag import PineconeVectorStore, search_pinecone

        class FakeIndexList:
            def __init__(self, names):
                self._names = names

            def names(self):
                return list(self._names)

        class FakeIndex:
            def __init__(self):
                self.upserts = []
                self.queries = []

            def upsert(self, **kwargs):
                self.upserts.append(kwargs)

            def query(self, **kwargs):
                self.queries.append(kwargs)
                return {"matches": [{"id": "doc-1", "score": 0.99}]}

        class FakeClient:
            def __init__(self):
                self.created = []
                self.index = FakeIndex()

            def list_indexes(self):
                return FakeIndexList([item["name"] for item in self.created])

            def create_index(self, **kwargs):
                self.created.append(kwargs)

            def describe_index(self, index_name):
                return {"status": {"ready": True}, "name": index_name}

            def Index(self, index_name, **kwargs):
                self.index_name = index_name
                return self.index

        class FakeEncoder:
            def encode(self, texts, **kwargs):
                return np.asarray([[len(text), 1.0] for text in texts], dtype=float)

        client = FakeClient()
        store = PineconeVectorStore(
            "mana-docs",
            dimension=2,
            namespace="arabic-medical",
            client=client,
        ).ensure_index(spec={"serverless": {"cloud": "aws", "region": "us-east-1"}})

        written = store.upsert_records(
            [
                {"id": "doc-1", "text": "first reusable block", "metadata": {"kind": "note"}},
                {"text": "second reusable block", "metadata": {"kind": "note"}},
            ],
            FakeEncoder(),
            batch_size=2,
        )
        matches = store.search([3.0, 4.0], top_k=1, metadata_filter={"kind": "note"})
        helper_matches = search_pinecone(client.index, [0.0, 2.0], top_k=1)

        self.assertEqual(client.created[0]["name"], "mana-docs")
        self.assertEqual(client.created[0]["dimension"], 2)
        self.assertEqual(written, 2)
        self.assertEqual(client.index.upserts[0]["namespace"], "arabic-medical")
        self.assertEqual(len(client.index.upserts[0]["vectors"]), 2)
        self.assertEqual(client.index.upserts[0]["vectors"][0]["id"], "doc-1")
        self.assertIn("text", client.index.upserts[0]["vectors"][1]["metadata"])
        self.assertEqual(matches[0]["id"], "doc-1")
        self.assertEqual(helper_matches[0]["id"], "doc-1")
        np.testing.assert_allclose(client.index.queries[0]["vector"], [0.6, 0.8])
        self.assertEqual(client.index.queries[0]["filter"], {"kind": "note"})


if __name__ == "__main__":
    unittest.main()
