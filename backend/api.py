import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

app = FastAPI()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="data/chroma_db",
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
    return {"message": "Team Knowledge Assistant API is running"}

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
from datetime import datetime
import json

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