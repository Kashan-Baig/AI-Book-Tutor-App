import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ingest import ingest_pdf_to_vectordb
from langchain.prompts import ChatPromptTemplate

# ---------- Load environment ----------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------- Initialize LLM ----------
groq = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key, temperature=0.3)

# ---------- Define Prompt ----------
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert tutor who must answer questions using ONLY the information provided in the context excerpts from the book. 
    Do not use any external knowledge or make assumptions beyond what is explicitly stated in the context.

    **CRITICAL INSTRUCTIONS:**
    1. If the context does not contain sufficient information to answer the question completely and accurately, respond with: "❌ Insufficient evidence in the provided book excerpts."
    2. Your answer must be comprehensive, well-structured, and directly supported by the context.
    3. For every key point in your answer, include an inline citation using this exact format: [Chapter: X | Section: Y | Page: Z]
    4. If multiple sources support the same point, include all relevant citations.
    5. Ensure your response is educational and helpful, maintaining a professional tutoring tone.

    **Context excerpts from the book (with metadata tags):**
    {context}

    **Question:** {input}

    **Answer:**
    """
)

# ---------- Helper Functions ----------
def build_hybrid_retriever(pdf_path):
    retriever, vectorstore, chunks = ingest_pdf_to_vectordb(pdf_path)
    return retriever

def format_docs_with_metadata(docs):
    formatted = []
    for d in docs:
        meta = d.metadata
        tag = f"[Chapter: {meta.get('chapter','?')} | Section: {meta.get('section','?')} | Page: {meta.get('page_number','?')}]"
        formatted.append(f"{tag} {d.page_content}")
    return "\n\n".join(formatted)


# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="📚 AI Book Tutor", layout="centered", page_icon="📘")

# ---------- Custom CSS ----------
# ---------- Custom CSS ----------
st.markdown("""
    <style>
        .main {
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            max-width: 850px;
            padding-top: 2rem;
        }
        .title-card {
            background: var(--background-alt);
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .title-card h1 {
            color: var(--text-color);
            font-weight: 800;
            font-size: 2.3rem;
        }
        .title-card p {
            color: var(--text-muted);
            font-size: 1rem;
        }
        .stFileUploader {
            background: var(--background-alt);
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .answer-box {
            background: var(--background-alt);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-top: 1rem;
            color: var(--text-color);
        }
        .context-box {
            background: var(--context-bg);
            border-radius: 10px;
            padding: 1rem;
            font-size: 0.9rem;
            color: var(--text-color);
        }

        /* ---- Light & Dark Theme Variables ---- */
        @media (prefers-color-scheme: light) {
            :root {
                --background-alt: #ffffff;
                --context-bg: #f3f6fa;
                --text-color: #1e1e1e;
                --text-muted: #555;
            }
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --background-alt: #1e1f25;
                --context-bg: #2a2c34;
                --text-color: #f5f5f5;
                --text-muted: #ccc;
            }
        }
    </style>
""", unsafe_allow_html=True)


# ---------- Title Section ----------
st.markdown("""
<div class="title-card">
    <h1>📚 AI Book Tutor</h1>
    <p>Upload your textbook and ask any question — powered by Hybrid Retrieval (Semantic + TF-IDF)</p>
</div>
""", unsafe_allow_html=True)

# ---------- Upload PDF ----------
uploaded_file = st.file_uploader("📂 Upload your textbook (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("⚙️ Processing PDF and building hybrid retriever..."):
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        retriever = build_hybrid_retriever(temp_path)

    st.success("✅ Retriever built successfully! You can now ask questions below.")

    # ---------- Question Input ----------
    st.markdown("### 💬 Ask a question about your uploaded book:")
    user_query = st.text_input("Type your question here...")

    if user_query:
        with st.spinner("🧠 Thinking..."):
            retrieved_docs = retriever.invoke(user_query)
            formatted_context = format_docs_with_metadata(retrieved_docs)
            result = groq.invoke(prompt.format(context=formatted_context, input=user_query))

        # ---------- Display Answer ----------
        st.markdown("### 🧾 **Answer**")
        st.markdown(f"<div class='answer-box'>{result.content if hasattr(result, 'content') else result}</div>", unsafe_allow_html=True)

        # ---------- Context Section ----------
        with st.expander("📖 View Retrieved Excerpts"):
            for i, doc in enumerate(retrieved_docs, 1):
                meta = doc.metadata
                st.markdown(f"**{i}. [Chapter: {meta.get('chapter','?')} | Section: {meta.get('section','?')} | Page: {meta.get('page_number','?')}]**")
                st.markdown(f"<div class='context-box'>{doc.page_content}</div>", unsafe_allow_html=True)
else:
    st.info("👆 Please upload a PDF file to begin.")
