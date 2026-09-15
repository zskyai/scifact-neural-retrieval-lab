"""Pluggable vector store for local Qdrant or Milvus deployments."""
from __future__ import annotations
from typing import Any, Sequence

class MilvusStore:
    def __init__(self, uri: str, collection: str, dimension: int):
        try:
            from pymilvus import MilvusClient, DataType
        except ImportError as exc:
            raise RuntimeError("Milvus backend requires pymilvus; install pymilvus>=2.4") from exc
        self.client = MilvusClient(uri=uri); self.collection = collection; self.dimension = dimension
        if not self.client.has_collection(collection_name=collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
            self.client.create_collection(collection_name=collection, schema=schema)
        if not self.client.list_indexes(collection_name=collection):
            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
            self.client.create_index(collection_name=collection, index_params=index_params)
    def upsert(self, rows: Sequence[dict[str, Any]]): self.client.insert(collection_name=self.collection, data=list(rows))
    def search(self, vector: Sequence[float], limit: int = 10):
        hits=self.client.search(collection_name=self.collection, data=[list(vector)], anns_field="vector", limit=limit, output_fields=["*"])[0]
        return [{"id": str(x["id"]), "score": float(x["distance"]), "payload": x.get("entity", {})} for x in hits]
