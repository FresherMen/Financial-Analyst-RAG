import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

_CACHED_EMBEDDINGS = None

def get_embeddings():
    """Singleton getter for sentence-transformers embedding model."""
    global _CACHED_EMBEDDINGS
    if _CACHED_EMBEDDINGS is None:
        _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _CACHED_EMBEDDINGS

def get_answer(query: str):
    """
    Retrieves relevant 10-K document chunks from Pinecone
    and synthesizes an analytical answer using Groq Llama 3.3 70B.
    """
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty or whitespace.")

    index_name = os.getenv("PINECONE_INDEX_NAME", "fin-analyst-index")
    embeddings = get_embeddings()

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return qa_chain.invoke({"query": query.strip()})
