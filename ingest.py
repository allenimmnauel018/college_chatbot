import os
import re
import logging
import html2text
import httpx
import torch
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from chromadb import PersistentClient

from kg_builder import build_auto_kg_with_fallback  # 👈 Add KG builder import

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Markdown Converter ---
md_converter = html2text.HTML2Text()
md_converter.ignore_links = True
md_converter.ignore_images = True
md_converter.ignore_emphasis = True

# --- Clean text ---
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

# --- Fetch homepage text only ---
def fetch_homepage_text(base_url: str) -> str:
    try:
        response = httpx.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove headers, footers, and unnecessary tags
        for tag in soup(["header", "footer", "script", "style", "nav"]):
            tag.decompose()

        markdown = md_converter.handle(str(soup))
        markdown = clean_text(markdown)
        lines = [line for line in markdown.splitlines() if len(line.strip()) > 30]

        return "\n".join(lines[:150])  # Limit to top 150 lines
    except Exception as e:
        logging.error(f"❌ Failed to fetch homepage: {e}")
        return ""

# --- Main ingestion function ---
def run_ingestion(
    base_url: str = "https://www.aucet.in/",
    db_path: str = "chroma_db",
    max_chunks: int = 1000
):
    logging.info("🚀 Starting homepage-only ingestion (text only)...")

    # --- Step 1: Fetch homepage text ---
    raw_text = fetch_homepage_text(base_url)
    if not raw_text:
        logging.warning("🚫 No homepage content extracted.")
        return

    # --- Step 2: Split into chunks ---
    splitter = SpacyTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = splitter.create_documents([raw_text])

    for doc in documents:
        doc.metadata["source"] = "/"  # all from homepage

    if len(documents) > max_chunks:
        logging.warning(f"⚠ Trimming chunks: {len(documents)} → {max_chunks}")
        documents = documents[:max_chunks]

    # --- Step 3: Build Neo4j Knowledge Graph ---
    try:
        logging.info("🧠 Extracting triplets and inserting into Neo4j...")
        inserted = build_auto_kg_with_fallback(documents)
        logging.info(f"✅ Inserted {inserted} triplets into Neo4j.")
    except Exception as e:
        logging.error(f"❌ Failed to build Neo4j KG: {e}")

    # --- Step 4: Save embeddings to ChromaDB ---
    try:
        logging.info("💾 Embedding and saving to ChromaDB...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large",
            model_kwargs={"device": device}
        )

        chroma_client = PersistentClient(path=db_path)
        db = Chroma(
            client=chroma_client,
            collection_name="college_helpdesk",
            embedding_function=embeddings
        )
        db.add_documents(documents)

        logging.info(f"✅ Vector DB saved at: {db_path}")
        logging.info(f"📄 Chunks saved: {len(documents)}")
    except Exception as e:
        logging.error(f"❌ Failed to save to ChromaDB: {e}")


# --- Run if main ---
if __name__ == "__main__":
    run_ingestion()
