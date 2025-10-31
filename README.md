# 📚 AI Book Tutor

AI Book Tutor is a **Streamlit-based Retrieval-Augmented Generation (RAG)** application that allows users to upload a **PDF book** and ask questions about its content.  
All answers are generated **only from the uploaded document**, ensuring factual accuracy and relevance — perfect for students, researchers, and lifelong learners.

---

## ✨ Features

- 📄 **PDF Upload & Parsing:** Upload any book in PDF format and automatically extract its contents.  
- 🧠 **Smart Text Chunking:** Intelligently splits text by chapters, sections, and paragraphs for better retrieval accuracy.  
- 🔍 **Hybrid Search (BM25 + Semantic):** Combines **keyword-based** and **embedding-based** retrieval to ensure the most relevant context.  
- 🤖 **Context-Aware LLM Answers:** Uses Groq’s **Llama-3.1-8B-Instant** model (fast and reliable) to generate context-grounded answers.  
- 📑 **Inline Citations:** Answers include **chapter/section/page references** to verify where information came from.  
- 🌐 **Streamlit Interface:** Clean, interactive interface for uploading PDFs and chatting with your book tutor.

---

## 🚀 Setup and Installation

Follow these steps to run the project locally or on Streamlit Cloud.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/AI-Book-Tutor.git
cd AI-Book-Tutor
```

### 2. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- **Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 5. Configure API Keys

The application requires API keys for HuggingFace and Groq. Create a `.env` file in the root directory of the project with the following content:

```
GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

- **Groq API Key:** Obtain from [Groq Console](https://console.groq.com/keys).

## 🏃 How to Run

After completing the setup, run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

This will open the application in your web browser.

## 🌐 Deployment

This application is deployed and can be accessed at: [https://ai-tutor-app-kashan-baig.streamlit.app/](https://ai-tutor-app-kashan-baig.streamlit.app/)

## 📁 Project Structure

```
.
├── streamlit_app.py                  # Main Streamlit application & Defines the RAG chain and hybrid retrieval logic
├── temp/                   # Directory for storing uploaded PDF books
│   └── your_book.pdf    # Contains uploaded PDFs
├── ingest.py               # Handles PDF loading, text splitting, and vector store creation
├── requirements.txt        # Project dependencies
├── vector_stores/          # Stores ChromaDB vector databases
│   └── (auto-generated)
└── .env                    # Environment variables (API keys)
```

## 🛠️ How it Works

1.  **PDF Upload & Ingestion (`app.py` -> `ingest.py`):**
    *   A user uploads a PDF file via the Streamlit interface.
    *   `ingest.py` uses `PyPDFLoader` to load the PDF pages.
    *   Pages are semantically split based on detected headings (chapters, sections) to maintain context.
    *   Larger fragments are further split using `RecursiveCharacterTextSplitter`.
    *   Each chunk is converted into an embedding using `HuggingFaceEmbeddings` and stored in a FAISS vector database. Metadata (book title, chapter, section, page, chunk ID) is enriched for better retrieval and citation.

2.  **RAG Chain Construction (`streamlit_app.py`):**
    *   **Hybrid Retrieval:** It employs an `EnsembleRetriever` combining `TF-ID` (keyword search) and a vector-based retriever (semantic search) from FAISS. These are merged using Reciprocal Rank Fusion (RRF) to get the most relevant documents.
    *   **LLM Integration:** A `ChatGroq` model (e.g., Llama-3.1-8b-instant) is used as the language model.
    *   **Prompt Engineering:** A `ChatPromptTemplate` is used to instruct the LLM to answer *only* from the provided context and to include inline citations.

3.  **Question Answering (`streamlit_app.py`):**
    *   When a user asks a question, `streamlit_app.py` invokes the RAG chain.
    *   The `retriever` fetches the most relevant document chunks from the vector store based on the question.
    *   These chunks, along with their metadata, are formatted and passed to the LLM via the prompt.
    *   The LLM generates an answer, adhering to the prompt's instructions for context-only responses and inline citations.
    *   The answer and source documents are displayed in the Streamlit interface.

## 👨‍💻 Developer

This project was developed by **Kashan Baig**.

- **LinkedIn**: [https://www.linkedin.com/in/kashan-baig-565034339/](https://www.linkedin.com/in/kashan-baig-565034339/)
- **Email**: [baigkashan74@gmail.com](mailto:baigkashan74@gmail.com)
- **GitHub**: [Kashan-Baig](https://github.com/Kashan-Baig)

