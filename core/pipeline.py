import os
import cohere
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient

load_dotenv()

class RAGPipeline:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str, graph_engine):
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.graph = graph_engine
        self.cohere = cohere.Client(os.getenv("COHERE_API_KEY"))
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def answer_query(self, query: str):
        # 1. Hybrid / Dense retrieval from Qdrant
        results = self.qdrant.query(
            collection_name=self.collection_name,
            query_text=query,
            limit=10
        )
        candidate_texts = [r.metadata["document"] if "document" in r.metadata else str(r) for r in results]

        # 2. Cohere Rerank
        reranked_docs = []
        if candidate_texts and os.getenv("COHERE_API_KEY"):
            try:
                co_res = self.cohere.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=candidate_texts,
                    top_n=3
                )
                for item in co_res.results:
                    reranked_docs.append({
                        "text": candidate_texts[item.index],
                        "score": item.relevance_score
                    })
            except Exception:
                reranked_docs = [{"text": t, "score": 0.5} for t in candidate_texts[:3]]
        else:
            reranked_docs = [{"text": t, "score": 0.5} for t in candidate_texts[:3]]

        # 3. Retrieve Graph Context
        graph_context = self.graph.query_related_entities(query)

        # 4. Generate Final Answer with Groq
        vector_context_str = "\n\n".join([f"[Score: {d['score']:.2f}] {d['text']}" for d in reranked_docs])
        
        system_prompt = (
            "You are a strict clinical compliance AI assistant. Answer using ONLY the provided "
            "vector chunks and entity knowledge graph. If an answer cannot be deduced, state clearly "
            "that the protocol does not contain that information.\n\n"
            f"=== KNOWLEDGE GRAPH RELATIONSHIPS ===\n{graph_context}\n\n"
            f"=== RETRIEVED DOCUMENT CONTEXT ===\n{vector_context_str}"
        )

        response = self.llm.invoke([
            ("system", system_prompt),
            ("human", query)
        ])

        return {
            "answer": response.content,
            "reranked_docs": reranked_docs,
            "graph_context": graph_context
        }