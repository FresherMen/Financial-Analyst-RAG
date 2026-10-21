import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

def process_pdf(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source PDF document not found at: {file_path}")

    # 1. Load PDF
    print(f"Loading PDF from {file_path}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunking (small chunks better for specific financial metrics)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks.")

    # 3. Free HuggingFace embeddings — no API key needed
    # NOTE: First run will download the model (~90MB), subsequent runs use cache
    print("Loading embedding model (downloads once ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store in Pinecone (free tier)
    # IMPORTANT: Your Pinecone index dimension must be 384 (MiniLM output size)
    # Create index at app.pinecone.io with dimension=384, metric=cosine
    index_name = os.getenv("PINECONE_INDEX_NAME", "fin-analyst-index")
    vectorstore = PineconeVectorStore.from_documents(
        splits, embeddings, index_name=index_name
    )
    print("Ingestion Complete!")
    return vectorstore

if __name__ == "__main__":
    process_pdf("data/nvidia_10k.pdf")