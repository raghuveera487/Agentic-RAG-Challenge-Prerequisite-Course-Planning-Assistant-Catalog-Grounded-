# Academic Course Planning Assistant (RAG System)

## 📌 Overview
This project is a Retrieval-Augmented Generation (RAG) system that helps students plan courses based on prerequisites and university catalog data.

## 🚀 Features
- Course catalog ingestion
- Vector database using Chroma
- Semantic search with embeddings
- LLM-based reasoning (HuggingFace)
- Automated evaluation system

## 🛠️ Tech Stack
- Python
- LangChain
- HuggingFace Transformers
- ChromaDB

## ⚙️ Setup Instructions

1. Install dependencies:
   pip install -r requirements.txt

2. Generate catalog:
   python src/catalog_generator.py

3. Build vector database:
   python src/ingestion_script.py

4. Run chatbot:
   python src/langchain_agent.py

5. Run evaluation:
   python src/evaluation_suite.py

## 📊 Output
- evaluation_report.txt contains system performance

## 👤 Author
Boya Raghuveera
