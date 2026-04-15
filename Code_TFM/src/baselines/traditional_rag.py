import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class TraditionalRAG:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print(f"Cargando índice existente desde {self.persist_directory}...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("No se encontró un índice previo. El vector_store está vacío.")
            self.vector_store = None

    def index_documents(self, splits):
        """
        Si ya existe el vector_store, añade documentos. 
        Si no, lo crea por primera vez.
        """
        if self.vector_store is not None:
            print("Añadiendo nuevos documentos al índice existente...")
            self.vector_store.add_documents(documents=splits)
        else:
            print("Creando nuevo índice de vectores...")
            self.vector_store = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )

    def ask(self, question: str) -> str:
        if not self.vector_store:
            raise ValueError("La base de datos no existe. Debes indexar documentos primero.")
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
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