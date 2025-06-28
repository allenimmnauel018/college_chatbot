# college_chatbot
An AI-powered chatbot that helps students get accurate, document-based answers to questions related to academics, curriculum, fees, credits, and more. Built using **LangChain**, **FAISS**, **Ollama**, and **Streamlit**.

## ✅ Features

- 💬 Ask questions in natural language
- 📄 Retrieves answers from college documents (PDFs) using LangChain RetrievalQA
- ⚡ Runs **locally** using lightweight `phi` model via [Ollama](https://ollama.com/) (`phi` model)
- 🔎 FAISS-powered document search
- 🔤 Spell-check support
- 🤖 “I don’t know” fallback for unknown queries
- 📁 Shows document sources with every answer
- 🧠 (Upcoming) Suggested/auto-generated question interface

## 🧰 Tech Stack
 --------------------------------------------------------------
| Component       | Technology                                 |
|-----------------|--------------------------------------------|
| LLM             | `phi` via Ollama                           |
| Embeddings      | `all-MiniLM-L6-v2` (Hugging Face)          |
| Vector Store    | FAISS                                      |
| Frameworks      | LangChain, Streamlit                       |
| Tools           | PyTorch, Sentence Transformers             |
 --------------------------------------------------------------

 ## 📂 Project Structure
```
college_chatbot/
├── data/ # College documents (PDF)
├── app.py # Streamlit interface
├── chatbot.py # LLM + RetrievalQA chain
├── ingest.py # PDF to vector DB embedding
├── vector_db/ # Saved FAISS index
├── requirements.txt
├── LICENSE
└── README.md
```
## 🧪 How to Run
### 🔧 Prerequisites

- Python 3.10+
- GPU-enabled system (e.g., RTX 3050 or better)
- [Ollama installed](https://ollama.com) and running
- Basic Python environment setup

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
ollama run phi
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
