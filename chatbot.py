# chatbot.py
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gemini_llm import GeminiLLM
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_chain(db_path="vector_db"):
    logger.info("Building ConversationalRetrievalChain...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    logger.info(f"Loaded FAISS DB from {db_path}")

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = GeminiLLM()
    logger.info("Initialized GeminiLLM.")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    logger.info("ConversationBufferMemory initialized.")

    # Define the prompt template for the final QA step of the chain.
    # It expects 'context' (retrieved documents) and 'question' (the user's query,
    # potentially rephrased by the question_generator based on chat history).
    qa_prompt_template = PromptTemplate.from_template(
        """You are a college helpdesk assistant.
Use the following extracted document content to answer the user question.
If the answer is not in the documents or you don't know about it correctly, just say: "I don't know." Do NOT make up an answer.

Documents:
{context}

Question: {question}
Answer:"""
    )
    logger.info("QA Prompt template set up for combine_docs_chain.")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        # IMPORTANT: Pass the custom prompt to the combine_docs_chain.
        # This ensures the final answer generation step uses your specified prompt
        # and correctly interprets 'question' as the input variable.
        combine_docs_chain_kwargs={"prompt": qa_prompt_template}
    )
    logger.info("ConversationalRetrievalChain set up.")
    return qa_chain

def generate_suggestions(llm_model: GeminiLLM, retriever, last_query: str = None) -> list[str]:
    """
    Generates suggested questions based on the general context of the documents
    or, if provided, based on the last user query.
    """
    logger.info("Attempting to generate suggestions...")

    if last_query:
        suggestion_prompt = f"""You are a college helpdesk assistant. The user just asked: "{last_query}".
        Based on this, suggest 3 concise and distinct follow-up questions a user might ask next, related to the previous topic or common related inquiries. Do not include any introductory or concluding remarks. Just provide the numbered list.

        1.
        2.
        3.
        """
        logger.info(f"Generating follow-up suggestions for: '{last_query}'")
    else:
        suggestion_prompt = """You are a college helpdesk assistant. Based on common inquiries a student might have regarding college, suggest 3 concise and distinct questions a user might ask. Focus on general topics related to college life, admissions, academics, and administration. Do not include any introductory or concluding remarks. Just provide the numbered list.

        1.
        2.
        3.
        """
        logger.info("Generating initial general suggestions.")

    try:
        logger.info("Calling LLM for suggestions...")
        response = llm_model._call(suggestion_prompt)
        logger.info(f"Raw LLM response for suggestions:\n{response}")

        suggestions = []
        for i in range(1, 4):
            line_prefix = f"{i}."
            found_line = None
            for line in response.split('\n'):
                if line.strip().startswith(line_prefix):
                    found_line = line.strip().replace(line_prefix, "").strip()
                    if found_line:
                        suggestions.append(found_line)
                    break

        cleaned_suggestions = list(set([s for s in suggestions if s]))[:3]
        logger.info(f"Parsed suggestions: {cleaned_suggestions}")
        return cleaned_suggestions
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        if last_query:
            return [
                "Can you elaborate on that?",
                "What else should I know about this?",
                "Are there related policies?"
            ]
        else:
            return [
                "How do I apply for admission?",
                "What financial aid options are available?",
                "Where can I find information about student services?"
            ]