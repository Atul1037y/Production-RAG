from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import API_KEY

# Configure LLM and Embeddings (Updated Class Names & Models)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2", 
    google_api_key=API_KEY
)

# Defensive prompt template
PROMPT_TEMPLATE = """
You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer based strictly on the context, just say that you don't know. Do not hallucinate or make up information.

Context:
{context}

Question:
{question}

Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain(persist_directory: str = "chroma_db"):
    # Load existing vector database using the updated Chroma import
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Construct the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain