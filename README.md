# Team Knowledge Assistant

## Overview

Team Knowledge Assistant is a Retrieval-Augmented Generation (RAG) application that allows teams to upload internal documents and ask questions in natural language. The system retrieves relevant information from uploaded documents and generates grounded answers using Groq LLM.

## Features

* PDF document ingestion
* Automatic chunking and embedding generation
* ChromaDB vector storage
* Semantic document retrieval
* Groq-powered answer generation
* Source citations
* Feedback logging
* FastAPI backend

## Tech Stack

* Python
* FastAPI
* LangChain
* ChromaDB
* Sentence Transformers
* Groq API

## Project Structure

TeamKnowledgeAssistant/

* backend/

  * api.py
  * ingest.py
  * query.py

* data/

  * chroma_db/
  * PDFs

* .env

* requirements.txt

* README.md

## Running Locally

1. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run API

```bash
uvicorn backend.api:app --reload
```

4. Open API Docs

```text
http://127.0.0.1:8000/docs
```

## Future Improvements

* Frontend integration
* File upload UI
* Confidence scoring
* Evaluation framework
* Deployment on Render
