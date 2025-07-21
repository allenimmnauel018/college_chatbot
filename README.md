# 📘 College Chatbot 

This project is a full-stack college helpdesk chatbot that uses:

* **LLMs (Gemini 2.5)** for semantic reasoning and entity extraction
* **Vector Search (FAISS)** for semantic retrieval
* **Knowledge Graph (Neo4j)** for factual filtering and structured querying
* **FastAPI** for API endpoints
* **Streamlit** for a simple web UI

---

## ✅ Features

* Crawl a university website and extract relevant text
* Automatically generate knowledge graph triplets
* Enrich vector chunks with KG entities
* Query using a combination of LLM, semantic vector search, and KG metadata
* Web UI to chat with the bot
* REST API for ingestion, health, status, and chat

---

## ⚙️ Setup Instructions

### 1. 📦 Create and Activate a Python Environment

```bash
conda create -n college_chatbot python=3.10 -y
conda activate college_chatbot
```

Or use `venv`:

```bash
python -m venv chatbot_env
source chatbot_env/bin/activate  # or chatbot_env\Scripts\activate
```

### 2. 🧩 Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 3. 🔐 Set Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY_1=your_google_gemini_key_1
GEMINI_API_KEY_2=your_google_gemini_key_2
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

---

## 🚀 Running the App

### Step 1: 🧠 Start Neo4j

* Open **Neo4j Desktop**
* Create a project and start a database
* Open Neo4j **Browser** (`http://localhost:7474`) to visualize your KG

### Step 2: 🌐 Run FastAPI Server

```bash
uvicorn main:app --reload
```

* Visit: `http://127.0.0.1:8000/docs` for API explorer

API Endpoints:

* `POST /ingest` — trigger website crawl + embedding + KG creation
* `POST /chat` — ask chatbot questions
* `GET /status` — get ingestion status
* `GET /health` — basic health check

### Step 3: 💬 Run Streamlit UI

```bash
streamlit run app.py
```

* Visit: `http://localhost:8501`

---

## 🧪 Example Workflow

1. Start Neo4j Desktop and run a DB
2. Run: `uvicorn main:app --reload`
3. Open `http://127.0.0.1:8000/docs`
4. `POST /ingest` → starts crawling and populates vector DB + KG
5. `POST /chat` → ask: *"Who is the dean?"* or *"TNEA code for Tindivanam college?"*
6. `GET /status` → monitor background ingestion
7. Open Streamlit UI (`http://localhost:8501`) for chat frontend

---

## 📁 Key Files

| File            | Description                                  |
| --------------- | -------------------------------------------- |
| `ingest.py`     | Web crawler + chunk splitter + vector embed  |
| `kg_builder.py` | Triplet extractor + Neo4j insertion          |
| `chatbot.py`    | Hybrid retriever (KG filter + vector search) |
| `main.py`       | FastAPI backend                              |
| `app.py`        | Streamlit UI                                 |
| `gemini_llm.py` | Gemini model wrapper + key rotation          |

---

## 🧠 Architecture

                      +-------------------------+
                      |     User Query Input    |
                      +-------------------------+
                                   |
                          +------------------+
                          |  Streamlit Front |
                          +------------------+
                                   |
                          +----------------+
                          |  FastAPI App   |
                          +----------------+
                                   |
                          +------------------+
                          | Entity Extractor |
                          |  (Gemini LLM)    |
                          +------------------+
                                   |
                   +-------------------------------+
                   | KG-Aware Retriever            |
                   | (Semantic + Entity Filtering) |
                   +-------------------------------+
                      |                       | 
            +-------------------+     +------------------+
            |  KG Entity Match  |     |  FAISS Fallback  |
            |     (Neo4j)       |     | (Semantic Search)|
            +-------------------+     +------------------+
                          \\             //
                           +-------------+
                           | Gemini LLM  |
                           | Final Answer|
                           +-------------+


## 🧠 Future Enhancements

* Add direct Cypher query support
* Show KG context as graphs in Streamlit
* Enable PDF/Doc file parsing
* Improve fallback ranking logic

---

## 🧑‍💻 Authors

* Built with ❤️ using LangChain, Gemini, FAISS, Neo4j, and Streamlit

## 🙏 Acknowledgments

- Academic materials and guidance from mentors
- Reference books used: LangChain Essentials, Mastering LLMs with LangChain, etc.
- Built during a 6-week InternPro Internship

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
