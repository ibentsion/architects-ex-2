"""Index phase: dense (vector) + sparse (keyword) backends and the index
manifest (rag_plan.md §5 stages 6-7).

Registries are re-exported here so ``rag.config`` can resolve the
``dense_index`` / ``sparse_index`` phase blocks uniformly.
"""
from rag.index.dense import DENSE_REGISTRY, VectorIndex
from rag.index.sparse import SPARSE_REGISTRY, KeywordIndex

__all__ = ["DENSE_REGISTRY", "SPARSE_REGISTRY", "VectorIndex", "KeywordIndex"]
