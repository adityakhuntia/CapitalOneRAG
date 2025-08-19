# 🌱 KhetiGyaan

KhetiGyaan is an AI-powered **Retrieval-Augmented Generation (RAG)** system built with **FastAPI**, **LangChain**, **FAISS**, **Google Gemini**, and **PostgreSQL**.  
It enables conversational AI with context-aware, intent-classified, and knowledge-grounded responses.

---

## 🚀 Features

- **Hybrid RAG Pipeline** using FAISS, Google Gemini, Tavily Search, and Wikipedia.  
- **Intent Classification** with `facebook/bart-large-mnli`.  
- **Persistent Chat History** with PostgreSQL (in-memory fallback if DB unavailable).  
- **Season Awareness** for agriculture use cases (Kharif, Rabi, Zaid).  
- **REST API Endpoints** for chat, history, and health check.

---

## ⚙️ Tech Stack

- **Framework**: FastAPI  
- **LLM**: Google Gemini (`gemini-1.5-flash`)  
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`  
- **Vector Store**: FAISS  
- **Database**: PostgreSQL (optional, for chat history)  
- **Classifier**: HuggingFace `facebook/bart-large-mnli`  

---

## 📂 Repository Structure
├── ingest.py # Build and ingest data into FAISS vector store
├── khetigyaan.py # Core bot logic (RAG + intent classification)
├── raggem.py # RAG pipeline with Gemini LLM
├── temp.py # Auxiliary/test scripts
├── vector_store/ # FAISS vector store (generated)
├── .env # Environment variables
├── requirements.txt # Python dependencies
└── README.md

## 🔑 Environment Variables

Create a `.env` file in the project root and place these inside it:
GEMINI_KEY=your_google_gemini_api_key
DATABASE_URL=postgresql://user:password@localhost:5432/agribot
VECTOR_STORE_PATH=vector_store
PORT=8000

## Installtion:
# 1. Clone the repository
git clone https://github.com/adityakhuntia/CapitalOneRAG.git
cd CapitalOneRAG

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 3. Install dependencies
pip install -r requirements.txt


## Running:
# 1. Step 1:
run temp.py using uvicorn on your terminal 
# 2. Step 2:
run khetigyaan.py on another terminal 
