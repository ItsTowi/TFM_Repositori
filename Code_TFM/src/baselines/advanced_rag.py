"""
advanced_rag.py
===============
RAG++ — Traditional RAG mejorado con:
  1. Chunking semántico (por párrafos)
  2. Búsqueda híbrida (vectorial + BM25)
  3. Reranking con cross-encoder

Uso:
    from src.baselines.advanced_rag import AdvancedRAG
    rag = AdvancedRAG(persist_directory="./chroma_db_advanced")
    splits = load_and_split_text(archivo)
    rag.index_documents(splits)
    respuesta = rag.ask("¿Qué es X?")
"""


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever

class AdvancedRAG:
    """
    RAG++ con búsqueda híbrida y reranking.

    Mejoras sobre Traditional RAG:
    - EnsembleRetriever: combina ChromaDB (vectorial) + BM25 (léxico)
    - CrossEncoder reranker: reordena los candidatos por relevancia real
    - Recupera más candidatos (k=20) y luego filtra a los mejores (top_n=4)
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_advanced",
        vector_k: int = 10,       # chunks del retriever vectorial
        bm25_k: int = 10,         # chunks del retriever BM25
        rerank_top_n: int = 4,    # chunks finales tras reranking
        vector_weight: float = 0.6,  # peso del retriever vectorial en el ensemble
        bm25_weight: float = 0.4,    # peso del BM25
    ):
        self.persist_directory = persist_directory
        self.vector_k    = vector_k
        self.bm25_k      = bm25_k
        self.rerank_top_n = rerank_top_n
        self.vector_weight = vector_weight
        self.bm25_weight   = bm25_weight

        # Embeddings vectoriales (mismo modelo que Traditional RAG para comparativa justa)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # LLM generativo
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

        # Modelo de reranking (cross-encoder — más preciso que coseno pero más lento)
        self.reranker_model = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        self.vector_store = None
        self._all_docs = []  # guardamos todos los docs para BM25

    def index_documents(self, splits):
        """Indexa los documentos en ChromaDB y guarda los splits para BM25."""
        self._all_docs = splits
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def load_existing_index(self):
        """Carga un índice vectorial ya existente."""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        results = self.vector_store.get(include=["documents", "metadatas"])
        
        # Debug: verificar qué devuelve Chroma
        print(f"   Docs recuperados de Chroma: {len(results.get('documents', []))}")
        
        from langchain_core.documents import Document
        self._all_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(results.get("documents", []), results.get("metadatas", []))
            if text  # filtrar documentos vacíos
        ]
        print(f"   _all_docs construidos: {len(self._all_docs)}")

    def _build_retriever(self):
        """
        Construye el retriever híbrido con reranking.

        Pipeline:
        1. Vector retriever (ChromaDB, similitud coseno) → top k vectorial
        2. BM25 retriever (léxico, TF-IDF) → top k BM25
        3. EnsembleRetriever → combina ambos con pesos
        4. CrossEncoderReranker → reordena y filtra a top_n
        """
        if not self.vector_store:
            raise ValueError("Primero debes indexar los documentos.")

        # 1. Retriever vectorial
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.vector_k}
        )

        # 2. Retriever BM25 (léxico)
        bm25_retriever = BM25Retriever.from_documents(self._all_docs)
        bm25_retriever.k = self.bm25_k

        # 3. Ensemble híbrido
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight]
        )

        # 4. Reranker cross-encoder
        compressor = CrossEncoderReranker(
            model=self.reranker_model,
            top_n=self.rerank_top_n
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )

        return retriever

    def ask(self, question: str) -> str:
        """Responde una pregunta usando el pipeline RAG++."""
        retriever = self._build_retriever()

        system_prompt = (
            "You are an expert assistant. Use ONLY the following retrieved context "
            "to answer the question. If you don't know, say you don't know.\n\n"
            "CONTEXT:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": question})
        return response["answer"]
    
    async def query(self, question: str) -> tuple[str, list[str]]:
        retriever = self._build_retriever()
        system_prompt = (
            "You are an expert assistant. Use ONLY the following retrieved context "
            "to answer the question. If you don't know, say you don't know.\n\n"
            "CONTEXT:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = await rag_chain.ainvoke({"input": question})
        answer = response["answer"]
        contexts = [doc.page_content for doc in response["context"]]
        return answer, contexts  # ← solo 2, el try/except del evaluador gestiona errores
