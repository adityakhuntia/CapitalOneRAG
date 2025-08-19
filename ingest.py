from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import os

# ================================
# 1) Load docs dynamically
# ================================
def load_docs(folder="docs"):
    pages = []
    for filepath in Path(folder).rglob("*.*"):
        if filepath.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(filepath))
        elif filepath.suffix.lower() in [".txt", ".md"]:
            loader = TextLoader(str(filepath))
        elif filepath.suffix.lower() in [".csv"]:
            loader = CSVLoader(str(filepath))
        else:
            continue
        pages.extend(loader.load())
    return pages

# ================================
# 2) Split into chunks
# ================================
def split_docs(pages, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(pages)

# ================================
# 3) Intent classifier
# ================================
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

labels = [
    "weather advisory",
    "crop health",
    "price",
    "government support",
    "fertilizer and pesticides",
    "irrigation",
    "miscellaneous"
]

def classify_intent(text):
    pred = classifier(text, labels, multi_label=False)
    return pred["labels"][0]

def tag_docs(docs):
    for doc in docs:
        snippet = doc.page_content[:500] if len(doc.page_content) > 50 else doc.page_content
        intent = classify_intent(snippet)
        doc.metadata["intent"] = intent
    return docs

# ================================
# 4) Embed + Store (cached)
# ================================
def build_vectorstore(docs, path="vector_store"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./hf_cache"   # <--- cache here
    )

    if os.path.exists(path):
        print("📂 Loading existing vector store...")
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(docs)  # update with new docs
    else:
        print("🆕 Creating new vector store...")
        vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(path)
    return vectorstore

# ================================
# Main
# ================================
if __name__ == "__main__":
    pages = load_docs()
    chunks = split_docs(pages)
    tagged = tag_docs(chunks)
    build_vectorstore(tagged)
    print("✅ Vectorstore updated with intent-tagged embeddings!")
