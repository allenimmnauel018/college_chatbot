import re
import logging
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from urllib.parse import urljoin, urlparse, urldefrag
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import SpacyTextSplitter
from kg_builder import build_auto_kg, extract_triplets_with_llm
import torch
from typing import List

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOCUMENT_EXTENSIONS = [".pdf", ".doc", ".docx"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"]

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.encode("utf-8", "replace").decode("utf-8")).strip()

def fetch_site_data_all(
    base_url: str = "https://www.aucet.in/",
    max_pages: int = 30,
    max_lines_per_page: int = 100
) -> List:
    visited = set()
    to_visit = [base_url]
    all_docs = []

    logging.info(f"🌐 Starting crawl at: {base_url}")
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop()
        normalized_url = urldefrag(current_url)[0].rstrip("/")

        if normalized_url in visited or not normalized_url.startswith(base_url.rstrip("/")):
            continue

        try:
            response = requests.get(normalized_url, timeout=10)
            response.encoding = response.apparent_encoding
            response.raise_for_status()
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch {normalized_url}: {e}")
            continue

        visited.add(normalized_url)
        soup = BeautifulSoup(response.content, "html.parser")
        text_elements, doc_links = [], []

        # Meta description
        meta = soup.find("meta", attrs={"name": "description"})
        if isinstance(meta, Tag):
            content = meta.get("content", "")
            if isinstance(content, str) and content.strip():
                text_elements.append("Meta Description: " + clean_text(content))

        # Visible text (excluding images)
        for tag in soup.find_all(["header", "footer", "p", "div", "span", "td", "a", "h1", "h2", "h3", "ul", "li", "section"]):
            if isinstance(tag, Tag):
                if tag.find("img"):
                    continue
                raw = tag.get_text(separator=" ", strip=True)
                if raw:
                    cleaned = clean_text(raw)
                    if len(cleaned) > 30:
                        text_elements.append(cleaned)

        # Document links and internal crawling
        for link in soup.find_all("a", href=True):
            if isinstance(link, Tag):
                href = link.get("href")
                if not isinstance(href, str):
                    continue
                href = href.strip()
                if not href or href.startswith("#") or href.lower().startswith("mailto:"):
                    continue

                try:
                    full_url = urldefrag(urljoin(normalized_url + "/", href))[0].rstrip("/")
                except Exception as e:
                    logging.warning(f"Skipping bad href '{href}': {e}")
                    continue

                if any(full_url.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    continue
                elif any(full_url.lower().endswith(ext) for ext in DOCUMENT_EXTENSIONS):
                    doc_links.append(full_url)
                elif full_url.startswith(base_url) and full_url not in visited and full_url not in to_visit:
                    to_visit.append(full_url)

        if doc_links:
            text_elements.append("📎 Documents:\n" + "\n".join([f"[Download Document]({url})" for url in doc_links]))

        text_elements = list(dict.fromkeys(text_elements))[:max_lines_per_page]

        if text_elements:
            path = urlparse(normalized_url).path or "/"
            all_docs.append({
                "source": path,
                "text": "\n".join(text_elements),
                "documents": doc_links
            })
            logging.info(f"✅ {path} → {len(text_elements)} lines, {len(doc_links)} docs")

    logging.info(f"✅ Crawled {len(visited)} pages. Collected {len(all_docs)} text entries.")
    return all_docs


def load_and_embed_all(
    base_url: str = "https://www.aucet.in/",
    db_path: str = "vector_db",
    max_chunks: int = 1000,
    max_pages: int = 30,
    max_lines_per_page: int = 100
) -> None:
    logging.info("🚀 Starting ingestion with Spacy sentence splitting...")

    splitter = SpacyTextSplitter(chunk_size=300, chunk_overlap=50)
    all_chunks = []

    web_pages = fetch_site_data_all(base_url, max_pages, max_lines_per_page)

    for page in web_pages:
        chunks = splitter.create_documents([page["text"]])
        for chunk in chunks:
            chunk.metadata["source"] = page["source"]
            if page.get("documents"):
                chunk.metadata["documents"] = page["documents"]
        all_chunks.extend(chunks)
        logging.info(f"📄 {page['source']} → {len(chunks)} chunks")

    logging.info(f"🧠 Total chunks before trimming: {len(all_chunks)}")
    if len(all_chunks) > max_chunks:
        logging.warning(f"⚠ Chunk limit exceeded: {len(all_chunks)} → trimming to {max_chunks}")
        all_chunks = all_chunks[:max_chunks]

    if not all_chunks:
        logging.error("🚫 No chunks available to embed.")
        return

    # === Extract KG entities and triplets ===
    logging.info("🔍 Extracting entity metadata and triplets for chunks...")
    enriched_count = 0
    for i, chunk in enumerate(all_chunks):
        text = chunk.page_content.strip()[:1000]
        try:
            triplets = extract_triplets_with_llm(text)
            if triplets:
                chunk.metadata["triplets"] = triplets
                entities = set()
                for subj, rel, obj in triplets:
                    entities.update([subj.strip(), obj.strip()])
                if entities:
                    chunk.metadata["entities"] = list(entities)
                enriched_count += 1
                logging.info(f"🧠 Chunk {i+1} enriched with {len(entities)} entities")
        except Exception as e:
            logging.warning(f"⚠️ Entity extraction failed for chunk {i}: {e}")

    logging.info(f"📦 Final chunk count: {len(all_chunks)}")
    logging.info(f"📊 Enriched chunks: {enriched_count} / {len(all_chunks)}")

    # === Build Knowledge Graph ===
    logging.info("🧩 Building Knowledge Graph from extracted triplets...")
    build_auto_kg(all_chunks)

    # === Embed & Save Vector DB ===
    logging.info("🔗 Generating embeddings and saving to FAISS...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": device}
    )

    try:
        db = FAISS.from_documents(all_chunks, embeddings)
        db.save_local(db_path)
        logging.info(f"📦 Vector DB saved at: {db_path}")
    except Exception as e:
        logging.error(f"❌ Error saving vector DB: {e}")
