import os
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

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

while True:
    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    results = vector_db.similarity_search(question, k=3)

    context = ""

    for i, result in enumerate(results):
        context += f"\n--- Context Chunk {i + 1} ---\n"
        context += result.page_content
        context += "\n"

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

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")

    for i, result in enumerate(results):
        print(f"Source {i + 1}:")
        print("Document:", result.metadata.get("source"))
        print("Page:", result.metadata.get("page"))
        print()