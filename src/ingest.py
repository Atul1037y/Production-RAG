from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import API_KEY

def ingest_documents(text_data: list[str], persist_directory: str = "chroma_db"):
    """
    Chunks text data and ingests it into a local Chroma vector database.
    """
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    docs = [Document(page_content=text) for text in text_data]
    splits = text_splitter.split_documents(docs)
    
    # 2. Embedding & Indexing (Updated Class Name & Model)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=API_KEY)
    
    # Updated Chroma import
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    return vectorstore

if __name__ == "__main__":
    # Example usage
    sample_text = ["Retrieval-Augmented Generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources."]
    print("Ingesting sample documents...")
    ingest_documents(sample_text)
    print("Ingestion complete. Chroma DB saved locally.")