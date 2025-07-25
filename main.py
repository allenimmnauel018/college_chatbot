import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from ingest import load_and_embed_all
from chatbot import build_chain
from fastapi.openapi.utils import get_openapi

# Setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
app = FastAPI()

qa_chain = build_chain()
ingestion_status = {"status": "Idle", "message": "No ingestion started."}

# --- Request Schemas ---
class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    mode: Literal["web"] = "web"
    db_path: Optional[str] = "vector_db"
    max_chunks: Optional[int] = 1000  # More chunks for Gemini parallelism
    max_pages: Optional[int] = 30
    max_lines_per_page: Optional[int] = 100

# --- Chat Endpoint ---
@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="Chatbot is not ready. Please trigger ingestion via /ingest."
        )
    try:
        result = qa_chain.invoke({"query": request.query})
        return {
            "response": result["result"].strip(),
            "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        }
    except Exception as e:
        logging.error(f"[CHAT ERROR] {e}")
        return {"response": "An error occurred while processing your query."}

# --- Ingestion Trigger ---
@app.post("/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    if ingestion_status["status"] == "Running":
        raise HTTPException(status_code=409, detail="Ingestion already running.")

    ingestion_status.update({
        "status": "Running",
        "message": f"Ingesting from {request.mode}..."
    })

    if request.mode == "web":
        background_tasks.add_task(
            run_ingestion,
            base_url="https://www.aucet.in/",
            db_path=request.db_path,
            max_chunks=request.max_chunks,
            max_pages=request.max_pages,
            max_lines=request.max_lines_per_page
        )

    return {"message": ingestion_status["message"]}

# --- Ingestion Logic ---
def run_ingestion(base_url, db_path, max_chunks, max_pages, max_lines):
    global qa_chain
    try:
        logging.info(f"🔁 Starting ingestion with max_chunks={max_chunks}, max_pages={max_pages}, max_lines={max_lines}")
        load_and_embed_all(
            base_url=base_url,
            db_path=db_path,
            max_chunks=max_chunks,
            max_pages=max_pages,
            max_lines_per_page=max_lines
        )
        qa_chain = build_chain()  # ✅ Refresh the QA chain after new ingestion
        ingestion_status.update({
            "status": "Completed",
            "message": "Ingestion completed and chatbot updated."
        })
        logging.info("✅ Ingestion complete. Chatbot updated with new vector DB.")
    except Exception as e:
        logging.error(f"[INGEST ERROR] {e}")
        ingestion_status.update({
            "status": "Failed",
            "message": f"Ingestion failed: {e}"
        })

# --- Status Endpoint ---
@app.get("/status")
async def get_status():
    return ingestion_status

# --- Health Check ---
@app.get("/health")
async def health_check():
    return {"status": "ok", "chatbot_ready": qa_chain is not None}

# --- Custom OpenAPI Schema ---
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="College Helpdesk API",
        version="1.0.0",
        description="Endpoints for chat, ingestion, and vector-based QA using Gemini and Neo4j.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
