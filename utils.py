import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA

def get_answer(query: str):
    """
    Retrieves relevant 10-K document chunks from Pinecone
    and synthesizes an analytical answer using Groq Llama 3.3 70B.
    """
    index_name = os.getenv("PINECONE_INDEX_NAME", "fin-analyst-index")

    # Local embedding model — dimension 384
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )

    # Groq-hosted high-speed LLaMA 3.3 70B
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0  # Zero temperature for strictly factual financial extraction
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True
    )

    return qa_chain.invoke({"query": query})