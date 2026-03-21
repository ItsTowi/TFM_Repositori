"""
query_adapters.py
=================
Wrappers que normalizan la interfaz de query de cada RAG.
Todos devuelven (respuesta: str, contextos: list[str]).
"""


async def query_traditional(rag, question: str) -> tuple[str, list[str]]:
    """
    Traditional RAG (ChromaDB + LangChain).
    Recupera los chunks del vector_store para pasarlos a RAGAS.
    """
    retriever = rag.vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    answer = rag.ask(question)
    return answer, contexts


async def query_lightrag(rag, question: str, mode: str = "hybrid") -> tuple[str, list[str]]:
    """
    LightRAG.
    mode: "local" | "global" | "hybrid" | "naive"
    Nota: LightRAG no expone chunks directamente, se usa la respuesta como contexto.
    """
    from lightrag import QueryParam
    answer = await rag.aquery(question, param=QueryParam(mode=mode))
    contexts = [answer]
    return answer, contexts


async def query_llamaindex(query_engine, question: str) -> tuple[str, list[str]]:
    """
    LlamaIndex GraphRAG.
    Extrae los source_nodes como contextos para RAGAS.
    """
    response = await query_engine.aquery(question)
    answer = response.response
    contexts = []
    if hasattr(response, "source_nodes"):
        contexts = [n.text for n in response.source_nodes if hasattr(n, "text")]
    if not contexts:
        contexts = [answer]
    return answer, contexts