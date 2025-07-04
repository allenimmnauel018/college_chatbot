import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_embed_all(pdf_dir="data", db_path="vector_db"):
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"📄 Loading: {file_path}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_docs.extend(chunks)

    print(f"✅ Total chunks: {len(all_docs)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(all_docs, embeddings)
    db.save_local(db_path)
    print(f"📦 Vector DB saved at: {db_path}")