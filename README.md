# RAGChatbot (RAG + Tool Calling)

A minimal **RAG** chatbot that recommends **exactly one book** based on a natural-language prompt.
After recommending, it **calls a local tool** `get_summary_by_title(title)` to include the **full summary**
for that exact title. The UI is built with **Streamlit**.


---

## ✨ Features
- **Local Vector Store (ChromaDB):** persistent index under `data/chroma/`
- **OpenAI Embeddings:** `text-embedding-3-small` (cost-effective, good quality)
- **OpenAI Chat + Tool Calling:** chooses one title from retrieved context, then calls `get_summary_by_title`
- **Robust Tool Matching:** title lookup is case-insensitive and fuzzy-tolerant
- **Streamlit UI:** simple chat, shows recommendation, final answer, and top-k retrieval context

---

## 🗂 Project Layout
```
RAGChatbot/
├─ app/
│ ├─ ui/
│ │ └─ streamlit_app.py
│ ├─ rag/
│ │ ├─ ingest.py
│ │ ├─ retriever.py
│ │ └─ vectorstore.py
│ └─ llm/
│ ├─ chain.py
│ ├─ openai_client.py
│ ├─ prompts.py
│ └─ tools.py
├─ data/
│ ├─ book_summaries.md
│ ├─ book_summaries_full.json
│ └─ chroma/ # persistent Chroma index (created by ingest)
├─ configs/
│ └─ settings.example.yaml
├─ tests/
│ └─ test_tools.py # unit tests for the tool (no network)
├─ .streamlit/
│ └─ config.toml # optional theme
├─ .env.example
├─ requirements.txt
├─ README.md
└─ .gitignore
```

---

## 🚀 Quickstart

### 0) Prerequisites
- Python **3.11+**
- An OpenAI API key (`OPENAI_API_KEY` in `.env`)

### 1) Setup
```bash
git clone <your-repo-url> smart-librarian
cd smart-librarian

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\\.venv\\Scripts\\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Environment
```bash
cp .env.example .env        # (Windows: copy .env.example .env)
# Open .env and set OPENAI_API_KEY
```

### 3) Dataset
We provide example data:
- **data/book_summaries.md (short summaries + themes) – used for ingestion**
- **data/book_summaries_full.json (full summaries) – used by tool**

### 4) Ingest (build the vector index)
```bash
python -m app.rag.ingest \
  --data-file data/book_summaries.md \
  --persist-dir data/chroma \
  --collection books_v1
```

### 5) Run the UI
```bash
# Option A (recommended):
python -m streamlit run app/ui/streamlit_app.py

# Option B:
# set PYTHONPATH to project root, then:
# streamlit run app/ui/streamlit_app.py

```
