import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import Optional

from ingest import run_ingestion  # This is the real ingestion function
from chatbot import build_chain

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# -------------------- App Setup --------------------
app = FastAPI()
qa_chain = build_chain()
ingestion_status = {"status": "Idle", "message": "No ingestion started."}

# -------------------- Request Models --------------------
class QueryRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    base_url: Optional[str] = "https://www.aucet.in/"
    db_path: Optional[str] = "chroma_db"
    max_chunks: Optional[int] = 1000
    max_pages: Optional[int] = 1
    max_lines_per_page: Optional[int] = 60

# -------------------- Chat Endpoint --------------------
@app.post("/chat")
async def chat_with_bot(request: QueryRequest):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="❌ Chatbot not ready. Please run ingestion first.")
    try:
        result = qa_chain.invoke({"query": request.query})
        return {
            "response": result["result"].strip(),
            "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
        }
    except Exception as e:
        logging.error(f"[CHAT ERROR] {e}")
        return {"response": "⚠️ An error occurred while processing your query."}

# -------------------- Ingestion Endpoint --------------------
@app.post("/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    if ingestion_status["status"] == "Running":
        raise HTTPException(status_code=409, detail="🚧 Ingestion already in progress.")

    ingestion_status.update({
        "status": "Running",
        "message": f"Ingesting from: {request.base_url or 'default'}"
    })

    background_tasks.add_task(
        run_ingestion_task,  # 🔁 renamed internal function
        base_url=request.base_url,
        db_path=request.db_path,
        max_chunks=request.max_chunks
    )

    logging.info(f"[INGEST] Started ingestion for: {request.base_url}")
    return {"message": ingestion_status["message"]}

# ✅ Renamed this to avoid conflict
def run_ingestion_task(base_url, db_path, max_chunks):
    global qa_chain
    try:
        run_ingestion(
            base_url=base_url,
            db_path=db_path,
            max_chunks=max_chunks
        )
        qa_chain = build_chain(db_path=db_path)
        ingestion_status.update({
            "status": "Completed",
            "message": "✅ Ingestion completed successfully."
        })
    except Exception as e:
        logging.error(f"[INGEST ERROR] {e}")
        ingestion_status.update({
            "status": "Failed",
            "message": f"Ingestion failed: {e}"
        })

# -------------------- Ingestion Status --------------------
@app.get("/status")
async def get_status():
    return ingestion_status

# -------------------- OpenAPI Customization --------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="🎓 College Helpdesk API",
        version="1.0.0",
        description=(
            "This API powers the college helpdesk chatbot.\n\n"
            "Endpoints:\n"
            "- 🤖 Ask questions via `/chat`\n"
            "- 🌐 Web ingestion via `/ingest`\n"
            "- 🟢 Check status via `/status`\n"
            "👉 Use `/docs` for Swagger UI."
        ),
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return openapi_schema

app.openapi = custom_openapi
