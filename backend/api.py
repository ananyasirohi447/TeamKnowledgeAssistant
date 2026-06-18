import os
import json
import shutil
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

app = FastAPI()

# Allows frontend/Lovable to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

CHROMA_PATH = "data/chroma_db"
UPLOAD_DIR = "data/uploads"

os.makedirs(CHROMA_PATH, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

class QuestionRequest(BaseModel):
    question: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str

@app.get("/")
def home():
    return {
        "message": "Team Knowledge Assistant API is running"
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        return {
            "error": "Only PDF and TXT files are supported right now."
        }

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file.filename

    vector_db.add_documents(chunks)
    vector_db.persist()

    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename,
        "chunks_created": len(chunks)
    }

@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    results = vector_db.similarity_search(question, k=3)

    context = ""
    sources = []

    for i, result in enumerate(results):
        context += f"\n--- Context Chunk {i + 1} ---\n"
        context += result.page_content
        context += "\n"

        sources.append({
            "document": result.metadata.get("source"),
            "page": result.metadata.get("page")
        })

    prompt = f"""
Answer ONLY from the context below.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }

@app.post("/feedback")
def save_feedback(request: FeedbackRequest):
    feedback_entry = {
        "question": request.question,
        "answer": request.answer,
        "feedback": request.feedback,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open("feedback_log.json", "r") as file:
            logs = json.load(file)
    except:
        logs = []

    logs.append(feedback_entry)

    with open("feedback_log.json", "w") as file:
        json.dump(logs, file, indent=4)

    return {
        "message": "Feedback saved successfully"
    }