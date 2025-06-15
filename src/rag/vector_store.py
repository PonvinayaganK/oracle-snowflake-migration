# src/rag/vector_store.py
import logging
import chromadb
from chromadb.utils import embedding_functions  # Keep this, but might not be directly used for adding
from langchain_core.documents import Document
from typing import List, Dict, Any
import os  # Import os for path checking

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, OPENAI_API_KEY
from src.llm_config.llm_manager import EmbeddingFactory  # Ensure this import is correct and used
from src.utils.exceptions import RAGError, ConfigurationError

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Manages interactions with the ChromaDB vector store.
    """

    def __init__(self, collection_name: str = CHROMA_COLLECTION_NAME, db_path: str = CHROMA_DB_PATH):
        self.db_path = db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_model = None  # Will store the actual LangChain embedding model instance

    def _initialize_client(self):
        """Initializes the ChromaDB client."""
        try:
            # Ensure the directory for ChromaDB exists
            os.makedirs(self.db_path, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            logger.info(f"Initialized ChromaDB client at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
            raise RAGError(f"ChromaDB initialization failed: {e}")

    def _get_embedding_model_instance(self):
        """Gets the embedding model instance from the EmbeddingFactory."""
        if self._embedding_model is None:
            # This calls the static method from EmbeddingFactory to get the configured model
            self._embedding_model = EmbeddingFactory.get_embedding_model(EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def get_collection(self):
        """Gets or creates the ChromaDB collection.
        NOTE: When creating a collection via Chroma's client.get_or_create_collection,
        if you provide `embedding_function`, Chroma will try to manage embeddings.
        However, for strict local control (and retries), we pre-embed using our `_get_embedding_model_instance`
        and then pass the `embeddings` directly to `collection.add`.
        So, typically, the `embedding_function` parameter here might be omitted or set to None
        if you are always providing embeddings via `add` method.
        """
        if self._client is None:
            self._initialize_client()
        if self._collection is None:
            try:
                # We'll create the collection without an explicit embedding_function here,
                # as we provide embeddings directly in `add_documents`.
                self._collection = self._client.get_or_create_collection(name=self.collection_name)

                # If you *must* provide an embedding_function to get_or_create_collection
                # for some reason (e.g., specific ChromaDB version requirement or features),
                # you'd need to adapt. For local models, it's often:
                # from chromadb.utils import embedding_functions
                # self._collection = self._client.get_or_or_create_collection(
                #     name=self.collection_name,
                #     embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                #         model_name=os.path.join(LOCAL_EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME.replace("local-", ""))
                #     )
                # )
                # But this can conflict with our custom `EmbeddingFactory` that handles retries etc.
                # Direct embedding in `add_documents` is generally more robust for custom EFs.

                logger.info(f"Accessed/Created ChromaDB collection: {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to get or create ChromaDB collection: {e}", exc_info=True)
                raise RAGError(f"ChromaDB collection error: {e}")
        return self._collection

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of LangChain Document objects to the ChromaDB collection.
        Documents are embedded first using our configured embedding model.
        """
        collection = self.get_collection()
        embedding_model = self._get_embedding_model_instance()

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate unique IDs based on content hash and metadata for robustness
        # Using a more robust ID generation that minimizes conflicts
        ids = []
        for i, doc in enumerate(documents):
            # Attempt to create a unique ID from metadata and content hash
            base_id = f"{doc.metadata.get('object_type', 'gen')}_{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('file_name', str(abs(hash(doc.page_content)) % 10 ** 8))}_{i}"
            # Clean up base_id for ChromaDB compatibility (e.g., remove invalid characters)
            clean_id = ''.join(e for e in base_id if e.isalnum() or e in ['-', '_', '.']).replace('__', '_')
            if not clean_id:  # Fallback for very simple documents
                clean_id = f"doc_{abs(hash(doc.page_content))}_{i}"

            # Ensure uniqueness if there are many docs with similar names/content
            final_id = clean_id
            counter = 0
            while final_id in collection.get()['ids']:  # Check against current IDs in DB
                final_id = f"{clean_id}_{counter}"
                counter += 1
            ids.append(final_id)

        try:
            # Embed documents before adding them to ChromaDB
            embeddings = embedding_model.embed_documents(texts)

            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids,  # Use unique IDs
                embeddings=embeddings  # Provide pre-calculated embeddings
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB collection '{self.collection_name}'.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}", exc_info=True)
            raise RAGError(f"Failed to add documents to ChromaDB: {e}")

    def similarity_search(self, query: str, k: int = 5, where: Dict[str, Any] = None) -> List[Document]:
        """
        Performs a similarity search in the ChromaDB collection.
        Returns a list of LangChain Document objects.
        """
        collection = self.get_collection()
        embedding_model = self._get_embedding_model_instance()

        try:
            # Embed the query using the same model
            query_embedding = embedding_model.embed_query(query)

            results = collection.query(
                query_embeddings=[query_embedding],  # Provide query embedding
                n_results=k,
                where=where  # Optional metadata filtering
            )

            retrieved_docs = []
            if results and results['documents']:
                for i in range(len(results['documents'][0])):
                    doc_content = results['documents'][0][i]
                    doc_metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    if results['distances'] and len(results['distances'][0]) > i:
                        doc_metadata['relevance_score'] = results['distances'][0][i]
                    retrieved_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB for query.")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Failed to perform similarity search in ChromaDB: {e}", exc_info=True)
            raise RAGError(f"ChromaDB search failed: {e}")