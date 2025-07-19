# main.py

import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from ingest import load_and_embed_all,fetch_site_data_all
from chatbot import build_chain
import networkx as nx
from fastapi.openapi.utils import get_openapi
import pickle

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
    max_chunks: Optional[int] = 300
    max_pages: Optional[int] = 20
    max_lines_per_page: Optional[int] = 100

# --- Chat Endpoint ---
@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="Chatbot is not ready.")
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
        "message": f"Ingesting site from: {request.mode}"
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

    logging.info(f"[INGEST] Triggered mode: {request.mode}")
    return {"message": ingestion_status["message"]}

def run_ingestion(base_url, db_path, max_chunks, max_pages, max_lines):
    try:
        load_and_embed_all(
            base_url=base_url,
            db_path=db_path,
            max_chunks=max_chunks,
            max_pages=max_pages,
            max_lines_per_page=max_lines
        )
        ingestion_status.update({
            "status": "Completed",
            "message": "Ingestion completed successfully."
        })
    except Exception as e:
        logging.error(f"[INGEST ERROR] {e}")
        ingestion_status.update({
            "status": "Failed",
            "message": f"Ingestion failed: {e}"
        })

# --- Ingestion Status ---
@app.get("/status")
async def get_status():
    return ingestion_status

# --- Quick Preview Snippet ---
@app.get("/web-preview")
async def preview_snippet():
    content = fetch_site_data_all()
    return {"snippet": content[:1000]}

# --- Knowledge Graph APIs (Optional) ---
@app.get("/graph-inspect")
async def inspect_graph():
    try:
        with open("graph/kg_auto.gpickle", "rb") as f:
            G = pickle.load(f)
        edges = [
            {"source": u, "target": v, "relation": data.get("relation", "")}
            for u, v, data in G.edges(data=True)
        ]
        return {"edges": edges}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph error: {e}")

@app.get("/graph-nodes")
async def graph_nodes():
    try:
        with open("graph/kg_auto.gpickle", "rb") as f:
            G = pickle.load(f)
        return {"nodes": len(G.nodes), "edges": len(G.edges)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph stats error: {e}")

    
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="College Helpdesk API",
        version="1.0.0",
        description="Endpoints for chat, ingestion, uploads, and graph.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
