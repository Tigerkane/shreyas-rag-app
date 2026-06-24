---

## Folder Structure

Your repository **must** follow this structure:


your-name-rag-app/
├── app.py                  # Main Streamlit app
├── requirements.txt        # All dependencies
├── README.md               # Your project README (see template below)
├── data/
│   └── your_document.pdf   # The document your RAG is built on
├── utils/
│   ├── loader.py           # Document loading / chunking logic
│   ├── embedder.py         # Embedding generation
│   └── retriever.py        # Retrieval logic
└── screenshots/
    └── demo.png            # At least one screenshot of working app


> You can add more files/folders — this is the **minimum**.

---

## Technical Requirements

Your app must include all of the following:

### 1. Document Loading
- Accept at least one PDF or text file as the knowledge source
- Chunk the document into meaningful segments (not just raw dump into context)

### 2. Embedding + Retrieval
- Generate embeddings for the chunks (use any: `sentence-transformers`, `Ollama`, Sarvam, etc.)
- Store them in a vector store — `ChromaDB`, `FAISS`, or `LanceDB` are all fine
- Retrieve top-k relevant chunks per user query

### 3. Chat Interface (Streamlit)
- A chat-style UI with message history (`st.chat_message`)
- Show which document chunks were used as context (even a small expander is enough)
- Clear conversation / reset button

### 4. LLM Integration
- Use any open model via Ollama (`llama3.2`, `gemma3`, `qwen2.5`) **or** a free-tier API (Groq, Gemini free)
- Do **not** hardcode API keys in the code — use `.env` or `st.secrets`

### 5. README (required)
Your `README.md` must include:

```markdown
## What document did you use and why?
## How does your chunking work?
## Which embedding model did you use?
## How to run locally
## Screenshot
## What would you improve with more time?
```

---

## What Will Be Evaluated

| Criteria | Weight |
|---|---|
| App runs without errors | 25% |
| RAG pipeline is actually working (not just stuffing full doc) | 25% |
| Code is readable and organized | 20% |
| README is clear and complete | 15% |
| Anything extra (Telugu support, streaming, UI polish) | 15% |

---

## How to Submit

1. **Fork** the batch repository on `code.swecha.org` (link shared in your cohort channel)
2. Push your code to your fork
3. Open a **Merge Request** to the main repo
   - Title: `[Submission] Your Name — RAG App`
   - Description: paste your README content directly
4. Add the label `submission` to your MR

> One MR.

---

## Common Mistakes to Avoid

- ❌ Stuffing the entire document into the prompt — that's not RAG
- ❌ Hardcoding API keys or model paths
- ❌ Missing `requirements.txt` (others should be able to run your app)
- ❌ No chunking — just splitting by page is not enough, think by paragraph or semantic unit
- ❌ No README or one-line README

---

## Resources

- [LangChain RAG Quickstart](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Chat Elements](https://docs.streamlit.io/develop/api-reference/chat)
---