import os
import re
import logging
import torch
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from gemini_llm import GeminiLLM

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Load environment variables ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "your-password")


# --------------------- Neo4j KG Query Fallback ---------------------
def query_neo4j_kg(question: str) -> str:
    logging.info(f"🤖 Neo4j KG fallback triggered for: {question}")
    llm = GeminiLLM(model="gemini-2.5-pro")

    # Step 1: Extract triplets
    prompt = f"""
Extract factual triplets (subject, relation, object) from this user question:
\"\"\"{question}\"\"\"

Only return triplets in the format:
(subject, relation, object)
Avoid generic or irrelevant information.
""".strip()

    try:
        response = llm._call(prompt).strip()
        triplets = [
            tuple(map(str.strip, line.strip("() ").split(",", 2)))
            for line in response.splitlines()
            if line.count(",") == 2
        ]
        logging.info(f"🔎 Extracted triplets: {triplets}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to extract triplets: {e}")
        triplets = []

    # Step 2: Use static fallback if triplets are empty
    if not triplets:
        if "dean" in question.lower():
            triplets = [("Dr. P. Thamizhazhagan", "is", "Dean")]
        else:
            logging.warning("⚠️ No triplets extracted.")
            return ""

    # Step 3: Query Neo4j
    results = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            for subj, rel, obj in triplets:
                cypher = """
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($subj)
                   OR toLower(o.name) CONTAINS toLower($obj)
                   OR toLower(r.type) CONTAINS toLower($rel)
                RETURN s.name AS subject, r.type AS relation, o.name AS object, r.source AS source
                LIMIT 5
                """
                result = session.run(cypher, {"subj": subj, "rel": rel, "obj": obj})
                for row in result:
                    results.append(
                        f"{row['subject']} --[{row['relation']}]--> {row['object']} (source: {row.get('source', 'N/A')})"
                    )
        driver.close()
    except Exception as e:
        logging.error(f"❌ Error querying Neo4j: {e}", exc_info=True)

    if not results:
        logging.info("❌ No matches in Neo4j for the extracted triplets.")
        return ""

    return "\n".join(results[:5])




# --------------------- QA Chain Builder ---------------------
def build_chain(db_path: str = "chroma_db") -> RetrievalQA | None:
    if not os.path.exists(db_path):
        logging.error(f"❌ Vector DB not found: {db_path}")
        return None

    try:
        # Load embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large",
            model_kwargs={"device": device}
        )

        # Load Chroma DB
        db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        retriever = db.as_retriever(search_kwargs={"k": 8})
        llm = GeminiLLM(model="gemini-2.5-pro")

        # Prompt
        qa_prompt = PromptTemplate.from_template("""
You are a helpful assistant for a college helpdesk.
Use the context to answer the question.
If you don't know or it's not found, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
""".strip())

        # Vector QA chain
        vector_qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )

        logging.info("✅ Vector QA chain ready.")

        # Hybrid QA Wrapper
        class HybridQAWrapper:
            def __init__(self, chain):
                self.chain = chain

            def invoke(self, inputs):
                user_query = inputs["query"].strip()
                logging.info(f"💬 User Query: {user_query}")

                result = self.chain.invoke({"query": user_query})
                answer = result.get("result", "").strip()
                docs = result.get("source_documents", [])

                # Fallback if the vector search fails or lacks confidence
                fallback_needed = not docs or "i don't know" in answer.lower()

                if fallback_needed:
                    logging.info("🔍 No confident vector answer. Trying Neo4j fallback...")
                    kg_response = query_neo4j_kg(user_query)

                    if kg_response:
                        # Replace weak answer with Neo4j answer
                        result["result"] = f"{kg_response}"
                    else:
                        result["result"] = "🤖 Sorry, I couldn’t find an answer in either the documents or knowledge graph."

                return result
        logging.info("✅ Hybrid QA wrapper initialized.")
        return HybridQAWrapper(vector_qa)

    except Exception as e:
        logging.error(f"🚨 QA chain failed to initialize: {e}", exc_info=True)
        return None
