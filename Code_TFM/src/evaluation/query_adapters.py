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

async def query_advanced(rag, question: str) -> tuple[str, list[str]]:
    """
    RAG++ — usa el pipeline híbrido completo de AdvancedRAG.
    """
    answer, contexts = await rag.query(question)
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
    response = await query_engine.aquery(question)
    answer = response.response
    contexts = []
    if hasattr(response, "source_nodes"):
        contexts = [
            n.node.text for n in response.source_nodes 
            if hasattr(n, "node") and hasattr(n.node, "text")
            and not n.node.text.startswith("Here are")  # filtrar triplets
        ]
    if not contexts:
        contexts = [answer]
    return answer, contexts



async def query_msgraphrag_local(rag, question: str) -> tuple[str, list[str]]:
    return await rag.local_search(question)

async def query_msgraphrag_global(rag, question: str) -> tuple[str, list[str]]:
    return await rag.global_search(question)