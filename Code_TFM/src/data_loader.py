# src/data_loader.py
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_split_text(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    return splits