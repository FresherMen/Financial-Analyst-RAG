import os
import streamlit as st
from utils import get_answer

st.set_page_config(
    page_title="Financial Analyst RAG — SEC 10-K Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom header styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2026;
        border: 1px solid #2a2d35;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📈 SEC 10-K Intelligence")
    st.markdown("""
    **Financial Analyst Assistant** leverages LangChain, Pinecone vector search, and Groq's high-speed Llama 3.3 70B to analyze complex financial disclosures.
    """)
    st.divider()
    st.markdown("### ⚙️ Engine Spec")
    st.markdown("- **Document:** NVIDIA FY2024 Form 10-K")
    st.markdown("- **Embedding:** `all-MiniLM-L6-v2` (384d)")
    st.markdown("- **Vector Store:** Pinecone Serverless")
    st.markdown("- **Inference LLM:** Groq `llama-3.3-70b-versatile`")
    st.divider()
    st.markdown("### 💡 Sample Prompts")
    examples = [
        "What was NVIDIA's total revenue in FY2024 vs FY2023?",
        "What are the primary drivers of Data Center revenue growth?",
        "What are the major supply chain and manufacturing risk factors?",
        "What was the gross margin percentage in fiscal 2024?"
    ]
    for ex in examples:
        if st.button(ex, key=f"btn_{ex[:15]}", use_container_width=True):
            st.session_state["query_input"] = ex

# Main UI
st.title("📊 Financial Analyst Assistant")
st.caption("Grounded SEC 10-K Quantitative & Qualitative Retrieval-Augmented Generation")

# Query Input
default_query = st.session_state.get("query_input", "")
query = st.text_input(
    "Enter your financial question:",
    value=default_query,
    placeholder="e.g. What was NVIDIA's revenue and gross margin in FY2024?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Analyze Filing", type="primary", use_container_width=True)

if (ask_button or default_query) and query:
    if not os.getenv("GROQ_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        st.error("⚠️ Missing API keys! Please set GROQ_API_KEY and PINECONE_API_KEY in your .env file.")
    else:
        with st.spinner("Analyzing SEC 10-K filings and synthesizing metrics..."):
            try:
                response = get_answer(query)
                
                st.subheader("📋 Financial Analysis")
                st.markdown(response["result"])

                # Source Evidence Display
                if "source_documents" in response and response["source_documents"]:
                    with st.expander(f"🔍 Inspect Retrieved 10-K Context ({len(response['source_documents'])} passages)"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            page_num = doc.metadata.get("page", "N/A")
                            source_file = doc.metadata.get("source", "nvidia_10k.pdf")
                            st.markdown(f"**Passage #{i}** — Source: `{source_file}` | Page: `{page_num}`")
                            st.caption(doc.page_content)
                            st.divider()
            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")