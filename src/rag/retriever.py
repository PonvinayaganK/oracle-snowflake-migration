# src/rag/retriever.py
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi  # pip install rank_bm25

from src.rag.vector_store import ChromaDBManager
from src.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH, RAG_TOP_K_SEMANTIC, RAG_TOP_K_KEYWORD
from src.utils.exceptions import RAGError

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Implements a hybrid retrieval system combining semantic (vector) search with keyword (BM25) search.
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME, db_path: str = CHROMA_DB_PATH):
        self.chroma_manager = ChromaDBManager(collection_name=collection_name, db_path=db_path)
        self.bm25_retriever = None
        self._all_documents_for_bm25 = []  # Cache for BM25

    def _initialize_bm25(self):
        """Initializes the BM25 retriever with all documents from ChromaDB."""
        logger.info("Initializing BM25 retriever...")
        try:
            # Retrieve all documents to build the BM25 index
            # Note: This loads all document content into memory, might need optimization for very large datasets
            collection = self.chroma_manager.get_collection()

            # Fetching all documents directly from ChromaDB for BM25 indexing
            # This can be slow for millions of documents. For practical purposes,
            # this assumes a reasonable number of RAG documents.
            all_chroma_docs = collection.get(include=['documents', 'metadatas'])

            self._all_documents_for_bm25 = [
                Document(page_content=doc_content, metadata=metadata)
                for doc_content, metadata in zip(all_chroma_docs['documents'], all_chroma_docs['metadatas'])
            ]

            if self._all_documents_for_bm25:
                corpus = [doc.page_content.split(" ") for doc in self._all_documents_for_bm25]
                self.bm25_retriever = BM25Okapi(corpus)
                logger.info(f"BM25 retriever initialized with {len(corpus)} documents.")
            else:
                logger.warning("No documents found in ChromaDB collection for BM25 initialization.")
        except Exception as e:
            logger.error(f"Failed to initialize BM25 retriever: {e}", exc_info=True)
            self.bm25_retriever = None  # Ensure it's null if init fails
            raise RAGError(f"BM25 initialization failed: {e}")

    def semantic_search(self, query: str, k: int = RAG_TOP_K_SEMANTIC, metadata_filter: Dict = None) -> List[Document]:
        """Performs semantic search using ChromaDB."""
        logger.info(f"Performing semantic search for query: '{query[:50]}...' (k={k})")
        return self.chroma_manager.similarity_search(query, k=k, where=metadata_filter)

    def keyword_search(self, query: str, k: int = RAG_TOP_K_KEYWORD) -> List[Document]:
        """Performs keyword search using BM25."""
        logger.info(f"Performing keyword search for query: '{query[:50]}...' (k={k})")
        if self.bm25_retriever is None:
            self._initialize_bm25()
            if self.bm25_retriever is None:  # If initialization failed
                logger.warning("BM25 retriever not initialized. Skipping keyword search.")
                return []

        tokenized_query = query.split(" ")
        scores = self.bm25_retriever.get_scores(tokenized_query)

        # Get top k documents based on scores
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        retrieved_docs = [self._all_documents_for_bm25[i] for i in ranked_indices]

        logger.info(f"Retrieved {len(retrieved_docs)} documents via keyword search.")
        return retrieved_docs

    def _reciprocal_rank_fusion(self, results: List[List[Document]], k: int = 60) -> List[Document]:
        """
        Applies Reciprocal Rank Fusion (RRF) to combine ranked lists of documents.
        https://plg.uwaterloo.ca/~gvcormac/rrf.pdf
        """
        fused_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}  # Store actual Document objects keyed by unique ID

        for ranks in results:
            for rank, doc in enumerate(ranks):
                # Use a unique identifier for the document.
                # A simple hash of content might not be unique for identical snippets from different sources.
                # If your documents have unique IDs from ingestion, use those. Otherwise, generate robust IDs.
                # For this example, let's use a combination of content hash and source to be safer.
                doc_unique_id = f"{hash(doc.page_content)}_{doc.metadata.get('source', '')}_{doc.metadata.get('file_name', '')}"

                if doc_unique_id not in doc_map:
                    doc_map[doc_unique_id] = doc  # Store the original Document object

                # RRF formula: 1 / (k + rank)
                fused_scores[doc_unique_id] = fused_scores.get(doc_unique_id, 0.0) + 1.0 / (
                            k + rank + 1)  # rank is 0-indexed

        # Sort documents by their fused scores
        sorted_doc_ids = sorted(fused_scores.keys(), key=lambda doc_id: fused_scores[doc_id], reverse=True)

        # Reconstruct documents with their original content and updated metadata (fused score)
        fused_docs = []
        for doc_id in sorted_doc_ids:
            original_doc = doc_map[doc_id]
            # Create a new Document object or update existing one to add fused_score
            original_doc.metadata['fused_score'] = fused_scores[doc_id]
            fused_docs.append(original_doc)

        logger.info(f"Applied RRF to fuse results. Total fused: {len(fused_docs)}")
        return fused_docs

    def retrieve(self, query: str, object_type: str, k_semantic: int = RAG_TOP_K_SEMANTIC,
                 k_keyword: int = RAG_TOP_K_KEYWORD) -> List[Document]:
        """
        Performs hybrid retrieval, combining semantic and keyword search.
        Filters semantic search by object_type metadata.
        """
        # Filter for object_type in semantic search
        semantic_filter = {"object_type": object_type}

        semantic_results = self.semantic_search(query, k=k_semantic, metadata_filter=semantic_filter)
        keyword_results = self.keyword_search(query, k=k_keyword)

        # Combine results using RRF
        combined_results = self._reciprocal_rank_fusion([semantic_results, keyword_results])

        logger.info(f"Hybrid retrieval completed, returning {len(combined_results)} documents.")
        return combined_results