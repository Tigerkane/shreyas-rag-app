# 🧠 Streamlit RAG Chatbot

A powerful, ephemeral Retrieval-Augmented Generation (RAG) application built with Python and Streamlit. This app allows you to converse with an AI, extract knowledge from your personal documents, and summarize entire webpages—all from a sleek, dark-mode UI.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🚀 Live Demo
Try the app right now on Streamlit Community Cloud:  
👉 **[Live App: Shreyas RAG Chatbot](https://shreyas-rag-app-mogalapalli.streamlit.app)**

---

## ✨ Features

* **3 Distinct Chat Modes:**
  * 💬 **Normal Chat:** Pure LLM conversation without extra document context.
  * 📄 **Document RAG:** Upload PDF, TXT, or MD files. The app chunks, embeds, and grounds its answers based strictly on your document.
  * 🌐 **URL RAG:** Paste any URL. The app scrapes the text, indexes it, and allows you to chat with the website's content.
* **Dual AI Interface (Cloud vs Local):**
  * **☁️ Cloud (Groq):** Blazingly fast inference using models like `llama-3.3-70b-versatile` and `qwen-2.5-coder-32b`. Bring your own key (BYOK) right in the UI.
  * **🖥️ Local (Ollama):** 100% offline inference. Automatically detects your local `localhost:11434` instance and fetches your installed models.
* **Ephemeral In-Memory Vector Store:** Completely database-free! It uses a highly optimized NumPy cosine-similarity search. Documents stay securely in RAM and are wiped clean on restart.
* **Save Your Chats:** One-click button to download your chat history (including source citations) as a markdown file!

---

## 🛠️ Local Installation

Want to run this on your own machine? It takes less than 2 minutes!

**1. Clone the repository**
```bash
git clone https://github.com/Tigerkane/shreyas-rag-app.git
cd shreyas-rag-app
```

**2. Install requirements**
Make sure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

## 📁 Architecture

* `app.py`: The main Streamlit user interface and routing logic.
* `utils/embedder.py`: Handles vector embeddings (using `sentence-transformers`) and checks connections to Groq/Ollama.
* `utils/loader.py`: Handles loading PDFs (via PyMuPDF), Text, Markdown, and text-chunking logic.
* `utils/retriever.py`: The custom in-memory NumPy vector store that performs the high-speed cosine similarity search.

## 📝 License
This project is open-source and available for anyone to use, modify, and distribute.
