import os
import uuid
from pathlib import Path
from typing import List
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
)
from fastembed import TextEmbedding, SparseTextEmbedding

# Model and vector configuration constants
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "prithivida/Splade_PP_en_v1"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
VECTOR_SIZE = 384

# Initialize local FastEmbed models
_dense_embedding_model = TextEmbedding(model_name=DENSE_MODEL)
_sparse_embedding_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: dict = {}

def initialize_collection(client: QdrantClient, collection_name: str) -> None:
    # If the collection exists with mismatched vector configurations, recreate it
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        # Check if 'sparse' is missing from sparse vectors configuration
        sparse_config = getattr(info.config, "sparse_vectors_config", {}) or {}
        if SPARSE_VECTOR_NAME not in sparse_config:
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            },
        )

def get_chunks(file_path: Path | str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[DocumentChunk]:
    path = Path(file_path)
    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")
    
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = splitter.split_documents(docs)
    
    chunks = []
    for doc in split_docs:
        chunks.append(
            DocumentChunk(
                id=str(uuid.uuid4()),
                text=doc.page_content,
                metadata=doc.metadata
            )
        )
    return chunks

def ingest_text(client: QdrantClient, collection_name: str, raw_text: str, source: str = "upload") -> List[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    split_texts = splitter.split_text(raw_text)
    
    chunks = [
        DocumentChunk(
            id=str(uuid.uuid4()),
            text=t,
            metadata={"source": source}
        )
        for t in split_texts
    ]
    _index_chunks(client, collection_name, chunks)
    return chunks

def ingest_file(client: QdrantClient, collection_name: str, file_path: Path | str) -> List[DocumentChunk]:
    chunks = get_chunks(file_path)
    _index_chunks(client, collection_name, chunks)
    return chunks

def _index_chunks(client: QdrantClient, collection_name: str, chunks: List[DocumentChunk]):
    texts = [c.text for c in chunks]
    
    # 1. Compute dense embeddings locally
    dense_embeddings = list(_dense_embedding_model.embed(texts))
    
    # 2. Compute sparse SPLADE/BM25 embeddings locally
    sparse_embeddings = list(_sparse_embedding_model.embed(texts))

    points = []
    for chunk, dense_emb, sparse_emb in zip(chunks, dense_embeddings, sparse_embeddings):
        payload = dict(chunk.metadata)
        payload["document"] = chunk.text
        payload["text"] = chunk.text

        points.append(
            PointStruct(
                id=chunk.id,
                vector={
                    DENSE_VECTOR_NAME: dense_emb.tolist(),
                    SPARSE_VECTOR_NAME: SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist(),
                    ),
                },
                payload=payload,
            )
        )

    # 3. Upsert points containing both dense and sparse vectors
    client.upsert(
        collection_name=collection_name,
        points=points,
    )