import os
import re
import time
from uuid import uuid4
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core import embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever

from dotenv import load_dotenv
load_dotenv()

HEADING_PATTERNS = [
    re.compile(r'^(Chapter|CHAPTER)\b.*', flags=re.IGNORECASE | re.MULTILINE),
    re.compile(r'^(Section|SECTION)\b.*', flags=re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\d+(?:\.\d+)*\s+.+', flags=re.MULTILINE),
    re.compile(r'^\d+(?:\.\d+)*\s+[A-Za-z].*', flags=re.MULTILINE),
    re.compile(r'^chapter\.\s+Here\'s\s+the\s+link:', flags=re.IGNORECASE | re.MULTILINE),
]

def extract_heading(text):
    for pattern in HEADING_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None

def ingest_pdf_to_vectordb(pdf_path: str, collection_name: str = None):
    """
    Ingests a PDF and returns a hybrid retriever (FAISS + TF-IDF).

    Steps:
    1️⃣ Loads the PDF into LangChain Documents
    2️⃣ Splits into overlapping text chunks
    3️⃣ Creates HuggingFace embeddings for semantic search (FAISS)
    4️⃣ Builds a TF-IDF retriever for keyword-based search
    5️⃣ Combines both retrievers into a weighted hybrid retriever
    """

    # ✅ Fix: define collection_name properly
    if collection_name is None:
        collection_name = os.path.splitext(os.path.basename(pdf_path))[0]

    start = time.process_time()
    print(f"📄 Starting PDF ingestion for {collection_name}...")

    # Step 1: Load PDF
    loader = PyPDFLoader(pdf_path)
    doc = loader.load()

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(doc)

    chunk_docs = []
    current_chapter = None

    for page_doc in doc:
        page_text = page_doc.page_content or ""
        page_num = page_doc.metadata.get("page", None)
        heading = extract_heading(page_text)  # only one heading per page
        fragments = [(heading, page_text)]  # ✅ Fix: wrap heading + text together

        for frag_idx, (heading, frag_text) in enumerate(fragments):
            chapter = None
            section = None
            if heading:
                low = heading.lower()
                if "chapter" in low:
                    chapter = heading
                    current_chapter = chapter
                elif "section" in low:
                    section = heading
                elif re.match(r'^\d+(?:\.\d+)*\s+', heading):
                    section = heading

            if chapter is None:
                chapter = current_chapter

            small_chunks = (
                text_splitter.split_text(frag_text)
                if len(frag_text) > 1000
                else [frag_text]
            )

            for sidx, chunk_text in enumerate(small_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue

                chunk_meta = {
                    "book_title": collection_name,
                    "chapter": chapter or "",
                    "section": section or "",
                    "page_number": page_num,
                    "source": os.path.basename(pdf_path),
                    "chunk_id": f"{os.path.basename(pdf_path)}_p{page_num}_f{frag_idx}_s{sidx}_{uuid4().hex[:8]}",
                }
                chunk_docs.append(Document(page_content=chunk_text, metadata=chunk_meta))

    # Step 3: Create embeddings and FAISS store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_index_path = f"vector_stores/{collection_name}_faiss"
    os.makedirs("vector_stores", exist_ok=True)
    vectorstore.save_local(faiss_index_path)
    print(f"[ingest] Persisted FAISS store at: {faiss_index_path}")

    retriever1 = vectorstore.as_retriever()

    # Step 4: Build TF-IDF retriever
    retriever2 = TFIDFRetriever.from_documents(chunks)

    # Step 5: Combine into hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2],
        weights=[0.6, 0.4]
    )

    print("✅ Hybrid retriever (FAISS + TF-IDF) created successfully!")
    return hybrid_retriever, vectorstore, chunk_docs





