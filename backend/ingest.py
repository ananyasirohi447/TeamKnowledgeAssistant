import os
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Start timer
start_time = time.time()

# Folder paths
DATA_FOLDER = "data"
CHROMA_FOLDER = "data/chroma_db"

# 1. Load all PDFs from data folder
all_docs = []

for file_name in os.listdir(DATA_FOLDER):
    if file_name.endswith(".pdf"):
        file_path = os.path.join(DATA_FOLDER, file_name)
        print(f"Loading: {file_name}")

        loader = PyPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_name

        all_docs.extend(docs)

print(f"\nLoaded {len(all_docs)} docs")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(all_docs)

print(f"Loaded {len(all_docs)} docs → {len(chunks)} chunks")

# 3. Create embeddings model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# 4. Store chunks in ChromaDB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_FOLDER
)

print(f"\nChromaDB saved at: {CHROMA_FOLDER}")

# 5. Verify with top-3 search
query = "what is the refund policy?"

results = vector_db.similarity_search(query, k=3)

print("\nTop 3 Retrieved Chunks:\n")

for i, result in enumerate(results):
    print(f"----- Result {i + 1} -----")
    print(result.page_content)
    print("Source:", result.metadata.get("source"))
    print("Page:", result.metadata.get("page"))
    print()

# 6. Time indexing
end_time = time.time()
print(f"Indexing completed in {round(end_time - start_time, 2)} seconds")
