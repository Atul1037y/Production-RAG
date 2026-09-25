from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.rag_pipeline import get_rag_chain

app = FastAPI(title="Production RAG API")

# Initialize the RAG chain
try:
    rag_chain = get_rag_chain()
except Exception as e:
    rag_chain = None
    print(f"Warning: Failed to load RAG chain. Ensure chroma_db exists. Error: {e}")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG chain is not properly initialized. Have you ingested data?")
    
    try:
        response = rag_chain.invoke(request.query)
        return QueryResponse(answer=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)