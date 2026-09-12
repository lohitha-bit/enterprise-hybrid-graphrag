import os
import networkx as nx
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

class EntityRelation(BaseModel):
    subject: str = Field(description="The source entity or role")
    predicate: str = Field(description="The relationship or action taken")
    object: str = Field(description="The target entity, department, or drug")

# Backward compatibility alias
Triplet = EntityRelation

class KnowledgeGraphExtraction(BaseModel):
    triplets: list[EntityRelation]

class GraphEngine:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.llm = ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY")
        )

    def add_triplets(self, triplets: list[EntityRelation], source: str = "seed"):
        for t in triplets:
            if self.graph.has_edge(t.subject, t.object):
                preds = self.graph[t.subject][t.object].get("predicates", [])
                if t.predicate not in preds:
                    preds.append(t.predicate)
                self.graph[t.subject][t.object]["predicates"] = preds
            else:
                self.graph.add_edge(t.subject, t.object, predicate=t.predicate, predicates=[t.predicate], source=source)

    def extract_and_add(self, text: str):
        try:
            structured_llm = self.llm.with_structured_output(KnowledgeGraphExtraction)
            prompt = f"Extract key clinical, compliance, or operational entity relationships (subject, predicate, object) from the text:\n\n{text}"
            result = structured_llm.invoke(prompt)
            if result and result.triplets:
                self.add_triplets(result.triplets, source="extracted")
        except Exception as e:
            print(f"Graph extraction info: {e}")

    def ingest_chunks(self, texts: list[str]):
        for text in texts:
            self.extract_and_add(text)

    def get_graph(self) -> nx.DiGraph:
        return self.graph

    def query_related_entities(self, query: str, depth: int = 2) -> str:
        if self.graph.number_of_nodes() == 0:
            return "Knowledge graph is empty."

        words = set(query.lower().split())
        matched_edges = []
        for u, v, data in self.graph.edges(data=True):
            preds = data.get("predicates") or [data.get("predicate", "connected_to")]
            pred_str = ", ".join(preds)
            if any(w in str(u).lower() or w in str(v).lower() for w in words if len(w) > 2):
                matched_edges.append(f"- ({u}) --[{pred_str}]--> ({v})")
        
        if matched_edges:
            return "\n".join(matched_edges[:12])
        return "No direct graph relationships extracted for this query."