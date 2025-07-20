import os
import time
import logging
from typing import List, Tuple
from neo4j import GraphDatabase
from gemini_llm import GeminiLLM

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Neo4j Config ---
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "your-password")

# --- Neo4j Handler ---
class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        try:
            with self.driver.session() as session:
                test = session.run("RETURN 1 AS ok").single()
                if test:
                    logging.info("✅ Connected to Neo4j.")
        except Exception as e:
            logging.error(f"❌ Neo4j connection failed: {e}")
            raise

    def close(self):
        self.driver.close()

    def insert_triplet(self, subj: str, rel: str, obj: str, source: str = "/"):
        query = """
        MERGE (s:Entity {name: $subj})
        MERGE (o:Entity {name: $obj})
        MERGE (s)-[r:RELATION {type: $rel, source: $source}]->(o)
        """
        try:
            with self.driver.session() as session:
                session.run(query, subj=subj, rel=rel, obj=obj, source=source)
        except Exception as e:
            logging.warning(f"⚠️ Failed to insert ({subj}, {rel}, {obj}): {e}")

# --- Triplet Extraction ---
def extract_triplets_from_text(text: str) -> List[Tuple[str, str, str]]:
    prompt = f"""
Extract clear factual (subject, relation, object) triplets from this college-related text.

Text:
\"\"\"{text}\"\"\"

Return only valid relationships in this format:
(subject, relation, object)
Avoid general greetings or vague information.
"""

    retries = 0
    result = ""
    while retries < 2:
        try:
            model = "gemini-2.5-flash" if retries == 0 else "gemini-2.5-pro"
            llm = GeminiLLM(model=model)
            logging.info(f"🔍 Using Gemini model: {model}")
            result = llm._call(prompt).strip()
            break
        except Exception as e:
            logging.warning(f"⚠️ Gemini {model} failed: {e}")
            retries += 1

    if not result:
        logging.error("❌ Failed to extract triplets.")
        return []

    triplets = []
    for line in result.splitlines():
        line = line.strip("() ")
        if line.count(",") != 2:
            continue
        parts = [p.strip(" '\"") for p in line.split(",", 2)]
        if len(parts) == 3:
            triplets.append(tuple(parts))

    return triplets

# --- KG Builder ---
def build_auto_kg_with_fallback(chunks) -> int:
    kg = KnowledgeGraph()
    total_inserted = 0

    logging.info("📡 Starting Neo4j KG insertion...")
    for i, doc in enumerate(chunks):
        text = doc.page_content.strip()
        source = doc.metadata.get("source", "/")

        if len(text) < 100:
            continue

        logging.info(f"📄 Chunk {i+1}/{len(chunks)} | Source: {source}")
        try:
            triplets = extract_triplets_from_text(text[:1000])
            for subj, rel, obj in triplets:
                logging.info(f"➕ {subj} --[{rel}]--> {obj}")
                kg.insert_triplet(subj, rel, obj, source)
                total_inserted += 1
        except Exception as e:
            logging.warning(f"⚠️ Failed to process chunk {i+1}: {e}")
        time.sleep(0.2)

    kg.close()
    logging.info(f"✅ Finished. Total triplets inserted: {total_inserted}")
    return total_inserted
