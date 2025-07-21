import os
import time
import spacy
import logging
from neo4j import GraphDatabase
from gemini_llm import GeminiLLM

# Load spaCy model (download with: python -m spacy download en_core_web_md)
try:
    nlp = spacy.load("en_core_web_md")
except Exception as e:
    logging.warning(f"spaCy model not loaded: {e}")
    nlp = None

# Neo4j connection details from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_triplets_with_llm(text: str, model_order=["gemini-2.5-flash", "gemini-2.5-pro"]) -> list[tuple[str, str, str]]:
    prompt = f"""
Extract factual relationships from the following college-related text. Return the information as (subject, relation, object) triplets.

Text:
\"\"\"{text}\"\"\"

Output format:
(subject, relation, object)
"""

    for model_name in model_order:
        try:
            logging.info(f"🔍 Trying with model: {model_name}")
            llm = GeminiLLM(model=model_name)
            result = llm._call(prompt)
            triplets = []
            for line in result.strip().splitlines():
                line = line.strip().strip("()")
                if line and "," in line:
                    parts = [p.strip() for p in line.split(",", 2)]
                    if len(parts) == 3 and all(parts):
                        triplets.append(tuple(parts))
            if triplets:
                return triplets
        except Exception as e:
            logging.warning(f"⚠ Error using {model_name}: {e}")
    return []

def insert_triplet(tx, subj, rel, obj, source):
    subj = subj.strip()
    rel = rel.strip()
    obj = obj.strip()

    if not subj or not rel or not obj:
        logging.warning(f"❌ Skipping invalid triplet: ({subj}, {rel}, {obj})")
        return

    logging.info(f"📥 Inserting: ({subj})-[:{rel}]->({obj}) from {source}")

    query = """
    MERGE (s:Entity {name: $subj})
    MERGE (o:Entity {name: $obj})
    MERGE (s)-[r:RELATION {type: $rel}]->(o)
    SET r.source = $source
    """
    tx.run(query, subj=subj, rel=rel, obj=obj, source=source)

def build_auto_kg(chunks):
    logging.info("⚙ Building Knowledge Graph using Neo4j...")

    total_chunks = len(chunks)
    logging.info(f"📊 Total chunks to process for KG: {total_chunks}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        total_triplets = 0
        for i, doc in enumerate(chunks):
            triplets = doc.metadata.get("triplets", [])
            source = doc.metadata.get("source", "unknown")

            if not triplets:
                logging.info(f"⏭️ Skipping chunk {i+1}/{total_chunks} (no triplets)")
                continue

            logging.info(f"📄 Chunk {i+1}/{total_chunks} | Source: {source} | Triplets: {len(triplets)}")
            for subj, rel, obj in triplets:
                logging.info(f"➕ Triplet: {subj} --[{rel}]--> {obj}")
                session.execute_write(insert_triplet, subj, rel, obj, source)

            total_triplets += len(triplets)

    driver.close()
    logging.info(f"✅ Knowledge Graph built and saved to Neo4j.")
    logging.info(f"📦 Total triplets inserted into Neo4j: {total_triplets}")
