# college_chatbot
An AI-powered chatbot that helps students get accurate, document-based answers to questions related to academics, curriculum, fees, credits, and more. This version is powered by **Gemini Pro**, **LangChain**, **FAISS**, and **Streamlit** — enabling fast, scalable inference using Google’s cloud-hosted LLMs.

## ✅ Features

- Ask academic questions in natural language

- Retrieves answers from college documents (PDFs)

- Uses FAISS for fast similarity-based document search

- Gemini Pro API as the backend LLM (via LangChain wrapper)

- Source tracking – shows which file/website the answer came from

- Graceful fallback – responds “I don’t know” when unsure

- Simple, clean Streamlit UI

## 🧰 Tech Stack

| Component    | Technology                            |
| ------------ | ------------------------------------- |
| LLM          | `Gemini Pro 2.5` via Google GenAI API |
| Embeddings   | `all-MiniLM-L6-v2` (Hugging Face)     |
| Vector Store | FAISS                                 |
| Frameworks   | LangChain, Streamlit                  |
| Tools        | Python, Sentence Transformers         |



 ## 📂 Project Structure
```
college_chatbot/
├── data/                # PDF documents for retrieval
├── app.py               # Streamlit interface
├── chatbot.py           # Gemini LLM + LangChain RetrievalQA
├── gemini_llm.py        # Custom wrapper for Google Gemini
├── ingest.py            # PDF to vector DB embedding using FAISS
├── vector_db/           # Stored FAISS index
├── requirements.txt
└── README.md

```
## 🧪 How to Run
### 🔧 Prerequisites

- Python 3.10+
- Basic Python environment setup
- A valid Google Gemini API key in .env file:
```
GEMINI_API_KEY=your_api_key_here
```

### 📦 Installation

```bash
pip install -r requirements.txt
``` 
### 📚 Ingest College Documents

```bash
python -c "from ingest import load_and_embed_all; load_and_embed_all()"
```

### 🧠 Launch Chatbot

```bash
streamlit run app.py
```

## ✍️ Sample Questions

- "What is the hostel fee structure for 2020-21?"
- "How many credits are given for project work?"
- "What is the B.Tech IT curriculum?"
- "How to apply for admission?"

## 🙏 Acknowledgments

- Academic materials and guidance from mentors
- Reference books used: LangChain Essentials, Mastering LLMs with LangChain, etc.
- Built during a 6-week InternPro Internship

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
