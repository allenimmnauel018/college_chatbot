# chatbot.py
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gemini_llm import GeminiLLM
import logging # Import logging module for better debugging output

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_chain(db_path="vector_db"):
    logging.info("Building RetrievalQA chain...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logging.info(f"Loaded FAISS DB from {db_path}")

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = GeminiLLM()
    logging.info("Initialized GeminiLLM.")

    prompt = PromptTemplate.from_template(
        """You are a college helpdesk assistant.
Use the following extracted document content to answer the user question.
If the answer is not in the documents or you don't know about it correctly, just say: "I don't know." Do NOT make up an answer.

Documents:
{context}

Question: {question}
Answer:"""
    )
    logging.info("Prompt template set up.")

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# --- FUNCTION FOR SUGGESTION GENERATION ---
def generate_suggestions(llm_model: GeminiLLM, retriever) -> list[str]:
    """
    Generates suggested questions based on the general context of the documents.
    """
    logging.info("Attempting to generate suggestions...")
    suggestion_prompt = """You are a college helpdesk assistant. Based on common inquiries a student might have regarding college, suggest 3 concise and distinct questions a user might ask. Focus on general topics related to college life, admissions, academics, and administration. Do not include any introductory or concluding remarks. Just provide the numbered list.

1.
2.
3.
"""
    try:
        logging.info("Calling LLM for suggestions...")
        response = llm_model._call(suggestion_prompt)
        logging.info(f"Raw LLM response for suggestions:\n{response}")

        # Parse the response to extract questions (assuming numbered list)
        suggestions = []
        for i in range(1, 4): # Expecting 1., 2., 3.
            line_prefix = f"{i}."
            found_line = None
            for line in response.split('\n'):
                if line.strip().startswith(line_prefix):
                    found_line = line.strip().replace(line_prefix, "").strip()
                    if found_line: # Ensure it's not just the number
                        suggestions.append(found_line)
                    break # Move to the next number

        # Ensure we return at most 3 distinct suggestions and clean them
        cleaned_suggestions = list(set([s for s in suggestions if s]))[:3]
        logging.info(f"Parsed suggestions: {cleaned_suggestions}")
        return cleaned_suggestions
    except Exception as e:
        logging.error(f"Error generating suggestions: {e}")
        # Fallback suggestions if generation fails
        return [
            "How do I apply for admission?",
            "What financial aid options are available?",
            "Where can I find information about student services?"
        ]