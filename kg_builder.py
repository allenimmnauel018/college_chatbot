import spacy
import networkx as nx
import os
import time
from gemini_llm import GeminiLLM

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

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
            print(f"🔍 Trying with model: {model_name}")
            llm = GeminiLLM(model=model_name)
            result = llm._call(prompt)
            triplets = []
            for line in result.strip().splitlines():
                line = line.strip().strip("()")
                if line and "," in line:
                    parts = [p.strip() for p in line.split(",", 2)]
                    if len(parts) == 3:
                        triplets.append(tuple(parts))
            if triplets:
                return triplets
        except Exception as e:
            print(f"⚠ Error using {model_name}: {e}")
    return []

def build_auto_kg(chunks):
    G = nx.MultiDiGraph()
    print("⚙ Building Knowledge Graph...")

    for i, doc in enumerate(chunks):
        text = doc.page_content.strip()
        source = doc.metadata.get("source", "unknown")

        if len(text) < 100:
            continue

        print(f"\n📄 Chunk {i+1}/{len(chunks)} | Source: {source} | Length: {len(text)}")
        short_text = text[:1000]

        triplets = extract_triplets_with_llm(short_text)
        time.sleep(0.5)  # Delay to respect rate limit

        for subj, rel, obj in triplets:
            print(f"➕ {subj} --[{rel}]--> {obj} (source: {source})")
            G.add_edge(subj, obj, relation=rel, source=source)

    os.makedirs("graph", exist_ok=True)
    import pickle
    with open("graph/kg_auto.gpickle", "wb") as f:
        pickle.dump(G, f)
    print("✅ Knowledge Graph saved at: graph/kg_auto.gpickle")

    return G
