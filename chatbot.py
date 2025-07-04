from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from gemini_llm import GeminiLLM  # Custom Gemini LLM

# Removed 'model_name="gemini"' from the signature as it was unused
def build_chain(db_path="vector_db"):
    # Load embeddings model
    # Removed model_kwargs={"device": "cuda"} for broader portability.
    # It will default to CPU if CUDA is not available or explicitly specified.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS vector store
    db = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    # Initialize custom Gemini model
    llm = GeminiLLM()

    # Prompt template
    prompt = PromptTemplate.from_template(
        """You are a college helpdesk assistant.
Use the following extracted document content to answer the user question.
If the answer is not in the documents or you don't know about it correctly, just say: "I don't know." Do NOT make up an answer.

Documents:
{context}

Question: {question}
Answer:"""
    )

    # Build RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )