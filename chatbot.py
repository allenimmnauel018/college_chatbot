import os
import torch
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gemini_llm import GeminiLLM
from langchain_core.retrievers import BaseRetriever
from typing import List, Optional, Any
from langchain_core.documents import Document
from pydantic import PrivateAttr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_entities_from_question(question: str) -> List[str]:
    """
    Extracts relevant KG-style entities from user query using Gemini.
    """
    prompt = f"""
Extract important named entities or concepts from the question for knowledge graph retrieval.

Question:
\"\"\"{question}\"\"\"

Return a comma-separated list of keywords or entity names only.
""".strip()

    try:
        llm = GeminiLLM(model="gemini-2.5-pro")
        response = llm._call(prompt)
        # Parse response
        entities = [e.strip() for e in response.split(",") if e.strip()]
        logging.info(f"🔍 Inferred entities from question: {entities}")
        return entities
    except Exception as e:
        logging.warning(f"⚠️ Entity extraction failed: {e}")
        return []


def build_chain(db_path: str = "vector_db") -> Optional[RetrievalQA]:
    """
    Builds a RetrievalQA chain using FAISS vector store, HuggingFace embeddings, and Gemini LLM.
    Integrates KG entity filtering into the vector search.
    """
    try:
        if not os.path.exists(db_path):
            logging.warning(f"📂 Vector DB not found at '{db_path}'. Please run ingestion first.")
            return None

        # Setup embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"🔌 Using device: {device}")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            model_kwargs={"device": device}
        )

        # Load vector DB
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Build dynamic retriever with entity-based filtering
        class KGFilteredRetriever(BaseRetriever):
            _base_retriever: Any = PrivateAttr()
            _entities: list[str] = PrivateAttr()

            def __init__(self, base_retriever, question: str):
                super().__init__()  # required by pydantic BaseModel
                self._base_retriever = base_retriever
                self._entities = extract_entities_from_question(question)

            def get_relevant_documents(self, query: str) -> List[Document]:
                all_docs = self._base_retriever.get_relevant_documents(query)

                if not self._entities:
                    return all_docs[:8]

                filtered = [
                    doc for doc in all_docs
                    if any(ent in doc.metadata.get("entities", []) for ent in self._entities)
                ]

                if not filtered:
                    logging.info("⚠ No entity-matched chunks, falling back to top-k.")
                    return all_docs[:8]

                return filtered[:8]

            def _get_relevant_documents(self, query: str) -> List[Document]:
                # Required by BaseRetriever abstract class
                return self.get_relevant_documents(query)


        # Prompt template
        prompt = PromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on provided college data and website context.
Rewrite the question below by expanding any abbreviations or acronyms to their full college names.  
If the answer is not clearly stated in the context, say "I don't know".
But if relevant information is partially present, try to answer concisely.

Context:
{context}

User question:
{question}

Answer:
""".strip())

        # Dynamic Retriever to be initialized at call-time
        def build_dynamic_chain(user_question: str):
            base_retriever = db.as_retriever(search_kwargs={"k": 20})
            retriever = KGFilteredRetriever(base_retriever, user_question)
            llm = GeminiLLM(model="gemini-2.5-pro")

            return RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )


        # Return a wrapper chain function that re-builds on each query
        class OnDemandKGQA:
            def invoke(self, inputs: dict):
                query = inputs["query"]
                chain = build_dynamic_chain(query)
                return chain.invoke(inputs)

        logging.info("✅ KG-enhanced chatbot QA chain ready.")
        return OnDemandKGQA()

    except Exception as e:
        logging.error(f"❌ Error building chatbot chain: {e}", exc_info=True)
        return None
                    