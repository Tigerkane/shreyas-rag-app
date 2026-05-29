# 🧠 RAG Chatbot — Local AI Document Q&A

A fully local, privacy-first **Retrieval-Augmented Generation (RAG)** chatbot built with **Streamlit** and **Ollama**. Upload any PDF or text document, ask questions, and get answers grounded in your document — all processing happens on your machine.

---

## 📄 What Document Did I Use and Why?

This RAG app is designed to work with **any PDF, TXT, or Markdown document** you provide. There is no hardcoded document — the app is general-purpose.

**Recommended starting documents:**
- A research paper (e.g. "Attention Is All You Need" — tests dense technical Q&A)
- A company policy PDF (e.g. HR handbook — tests factual retrieval)
- A textbook chapter (tests long-document chunking)

**Why this approach?** Rather than building around one fixed document, the app lets you drop any file into `data/` and index it instantly from the sidebar. This makes it reusable across any domain — legal, medical, academic, or personal notes.

---

## ✂️ How Does Your Chunking Work?

Chunking is handled in `utils/loader.py` using a **character-based sliding window with sentence-boundary awareness**.

**Steps:**
1. The document is loaded as raw text (PyMuPDF for PDFs, plain read for TXT/MD)
2. Text is split into chunks of **500 characters** (configurable via sidebar)
3. Each chunk has a **50-character overlap** with the next — this prevents answers from being cut off at chunk boundaries
4. Instead of hard-cutting at exactly 500 chars, the chunker looks back up to 100 characters for a natural break point: paragraph (`\n\n`), newline (`\n`), or sentence ending (`. `, `! `, `? `)
5. Each chunk is stored with metadata: `source filename`, `chunk_id`, and `start_char` position

**Example:**

```
"...the model uses attention mechanisms.\n\nThe encoder maps..."
                                         ↑ break here (paragraph)
```

This produces cleaner, more semantically coherent chunks compared to a hard character split.

| Parameter | Default | Range |
|---|---|---|
| Chunk size | 500 chars | 200–1000 |
| Chunk overlap | 50 chars | 0–200 |

---

## 🔢 Which Embedding Model Did I Use?

**Model:** `nomic-embed-text:latest` (via Ollama)

**Why nomic-embed-text?**
- Specifically designed for retrieval tasks — outperforms many general-purpose embedding models on semantic search benchmarks
- Produces **768-dimensional vectors** — high enough for precise similarity matching, efficient enough for local hardware
- Runs fully locally via Ollama — no API key, no internet required
- Already available in your Ollama installation (`274 MB`, lightweight)

**How embeddings are used:**
- At index time: each text chunk → `nomic-embed-text` → 768-dim vector → stored in ChromaDB
- At query time: user question → `nomic-embed-text` → query vector → cosine similarity search → top-K chunks retrieved

---

## 🚀 How to Run Locally

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai) installed

### Step 1 — Start Ollama

```bash
ollama serve
```

Make sure you have the required models (you already do):
```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.1:8b        # chat LLM
```

### Step 2 — Install Dependencies

```bash
cd shreya-rag-app
pip install -r requirements.txt
```

### Step 3 — Add Your Document

Drop your PDF or TXT file into the `data/` folder:
```
data/
└── your_document.pdf
```

### Step 4 — Run the App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Step 5 — Index & Chat

1. In the sidebar, click **Index** next to your document (or upload a new one)
2. Wait for embedding to complete
3. Type your question in the chat box
4. The app retrieves the most relevant chunks and generates an answer

---

## 📸 Screenshot

> _Run the app and add a screenshot here as `screenshots/demo.png`_

To take a screenshot after running:
```
screenshots/
└── demo.png    ← paste your app screenshot here
```

---

## 🔮 What Would I Improve With More Time?

### 1. 🔁 Better Chunking Strategy
Replace character-based chunking with **semantic chunking** — splitting at topic boundaries rather than fixed sizes. Libraries like `semantic-text-splitter` or LangChain's `SemanticChunker` would improve retrieval accuracy significantly.

### 2. 🗃️ Multi-Document Support with Filtering
Allow users to upload multiple documents and **filter retrieval by source** — e.g. "only search the HR policy PDF". Currently all documents are searched together.

### 3. 🔄 Hybrid Search (BM25 + Vector)
Combine **dense vector search** (semantic) with **sparse BM25 keyword search**. Hybrid retrieval outperforms either method alone, especially for exact term lookups like names, dates, or code.

### 4. 💬 Conversation Memory with Context Window Management
The current app passes the last 3 turns of history. A proper **memory module** using a sliding context window or summarisation of older turns would improve multi-turn conversations.

### 5. 📊 Retrieval Evaluation & Scoring
Add a feedback button (👍 / 👎) per answer and log retrieval quality metrics like **MRR** and **Hit Rate** to measure and improve the pipeline over time.

### 6. 🌐 Web URL Ingestion
Add support for ingesting content directly from a URL — scrape, clean, chunk, and embed web pages alongside local documents.

---

## 🗂️ Folder Structure

```
shreya-rag-app/
├── app.py                  # Main Streamlit app
├── requirements.txt        # All dependencies
├── README.md               # This file
├── data/
│   └── your_document.pdf   # Place documents here
├── utils/
│   ├── __init__.py
│   ├── loader.py           # Document loading & chunking logic
│   ├── embedder.py         # Embedding generation (Ollama)
│   └── retriever.py        # Vector store & retrieval (ChromaDB)
└── screenshots/
    └── demo.png            # App screenshot
```

---

## 🔒 Privacy

Everything runs locally — no data ever leaves your machine:
- Ollama runs LLMs and embeddings on your hardware
- ChromaDB stores all vectors in `./chroma_db/` on disk
- No telemetry, no cloud calls, no API keys needed
