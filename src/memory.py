import networkx as nx
import chromadb
from schemas import ClinicalEvent

class PatientDigitalTwin:
    """
    Hybrid Memory Store combining:
    1. NetworkX Temporal State Graph (strict chronological entity tracking)
    2. ChromaDB Vector Store (semantic search across raw clinical notes)
    """
    def __init__(self, collection_name: str = "medusa_patient_notes"):
        self.graph = nx.DiGraph()
        self.chroma_client = chromadb.Client()
        # Reset collection for each fresh patient run
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def ingest_raw_documents(self, records: list[dict]):
        """Indexes raw clinical documents into ChromaDB for semantic retrieval."""
        if not records:
            return
        self.collection.add(
            documents=[doc["text"] for doc in records],
            metadatas=[{"doc_id": doc["doc_id"], "doc_type": doc["doc_type"], "doc_date": doc["doc_date"]} for doc in records],
            ids=[doc["doc_id"] for doc in records],
        )

    def build_temporal_graph(self, events: list[ClinicalEvent]):
        """Builds a directed chronological graph linking Patient -> Events -> Source Documents."""
        self.graph.add_node("PATIENT", type="Patient")
        sorted_events = sorted(events, key=lambda e: e.event_date)

        for idx, ev in enumerate(sorted_events):
            event_node = f"EV_{idx}_{ev.entity_name}"
            doc_node = ev.doc_id

            self.graph.add_node(
                event_node,
                type=ev.category,
                entity=ev.entity_name,
                status=ev.status,
                value=ev.value_or_dose,
                date=ev.event_date,
                quote=ev.exact_quote,
            )
            self.graph.add_node(doc_node, type="SourceDocument")
            self.graph.add_edge("PATIENT", event_node, relation=ev.status, date=ev.event_date)
            self.graph.add_edge(event_node, doc_node, relation="PROVEN_BY")

    def semantic_search(self, query: str, n_results: int = 2) -> list[dict]:
        """Autonomously retrieves the most semantically relevant raw notes for a missing clinical clue."""
        results = self.collection.query(query_texts=[query], n_results=n_results)
        hits = []
        if results and results["documents"]:
            for doc_text, meta in zip(results["documents"][0], results["metadatas"][0]):
                hits.append({
                    "doc_id": meta["doc_id"],
                    "doc_date": meta["doc_date"],
                    "text": doc_text,
                })
        return hits