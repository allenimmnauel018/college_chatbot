# chatbot.py
import os
import torch
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gemini_llm import GeminiLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_chain(db_path: str = "vector_db") -> RetrievalQA | None:
    """
    Builds a RetrievalQA chain using FAISS vector store, HuggingFace embeddings, and Gemini LLM.

    Args:
        db_path (str): Path to the FAISS vector database.

    Returns:
        RetrievalQA | None: The QA chain if successful, else None.
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
        retriever = db.as_retriever(search_kwargs={"k": 8})

        # Setup Gemini model
        llm = GeminiLLM(model="gemini-2.5-pro")

        # Prompt template
        prompt = PromptTemplate.from_template("""
You are a helpful AI assistant answering questions based on provided college data and website context.
If the answer is not clearly stated in the context, say "I don't know".
But if relevant information is partially present, try to answer concisely.


Context:
{context}

User question:
{question}

Answer:""".strip())

        # Build RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

        logging.info("✅ Chatbot QA chain built successfully.")
        return chain

    except Exception as e:
        logging.error(f"❌ Error building chatbot chain: {e}", exc_info=True)
        return None
