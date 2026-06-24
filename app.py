"""
app.py - RAG Chatbot with 3 modes: 
  💬 Normal Chat  — plain LLM conversation, no docs needed
  📄 Document     — upload PDF/TXT/MD, chat grounded in file
  🌐 URL          — paste a URL, chat grounded in that page
"""

import os
import re
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from utils.loader import load_and_chunk, chunk_text
from utils.embedder import (
    embed_chunks, embed_query, 
    check_groq_api_key, list_available_chat_models,
    check_ollama_connection, list_available_ollama_models
)
from utils.retriever import VectorStore, build_context

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0d0d0d; color: #e8e8e8; }

.main-title {
    font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
    font-weight: 600; color: #00ff9d; letter-spacing: -0.5px; margin-bottom: 0.2rem;
}
.sub-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; color: #555; margin-bottom: 1.2rem;
}
.mode-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; padding: 3px 10px; border-radius: 20px; margin-bottom: 1rem;
}
.badge-chat { background: #1e1b4b; color: #818cf8; border: 1px solid #312e81; }
.badge-doc  { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
.badge-url  { background: #0c1a3d; color: #60a5fa; border: 1px solid #1e3a8a; }

.chat-user {
    background: #1a1a2e; border-left: 3px solid #00ff9d;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}
.chat-assistant {
    background: #111827; border-left: 3px solid #6366f1;
    padding: 0.75rem 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0;
}
.source-badge {
    display: inline-block; background: #1e293b; color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    padding: 2px 8px; border-radius: 4px; margin: 2px 3px; border: 1px solid #334155;
}
.status-ok  { color: #00ff9d; font-weight: 600; margin-bottom: 10px; display: inline-block;}
.status-err { color: #ff4d6d; font-weight: 600; margin-bottom: 10px; display: inline-block;}

.metric-box {
    background: #111827; border: 1px solid #1e293b;
    border-radius: 8px; padding: 0.75rem 1rem; text-align: center;
}
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; color: #00ff9d; font-weight: 600; }
.metric-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

div[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #1e293b; }

.stButton > button {
    background: #00ff9d; color: #0d0d0d;
    font-family: 'IBM Plex Mono', monospace; font-weight: 600;
    border: none; border-radius: 6px; padding: 0.5rem 1.2rem; transition: all 0.2s;
}
.stButton > button:hover {
    background: #00cc7d; transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0,255,157,0.3);
}
.stSelectbox label, .stSlider label, .stFileUploader label, .stTextInput label, .stRadio label {
    color: #94a3b8 !important; font-size: 0.85rem !important;
}
.stTextInput > div > div > input {
    background: #111827 !important; border: 1px solid #1e293b !important;
    color: #e8e8e8 !important; border-radius: 8px !important;
}
.stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #64748b;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem;
    border-radius: 8px; padding: 8px 20px;
}
.stTabs [aria-selected="true"] { background: #1e293b !important; color: #e8e8e8 !important; }
hr { border-color: #1e293b; margin: 1.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("doc_store",      None),
    ("url_store",      None),
    ("chat_history",   []),     # normal chat
    ("doc_history",    []),     # document chat
    ("url_history",    []),     # url chat
    ("indexed_files",  []),
    ("indexed_urls",   []),
    ("total_chunks",   0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.doc_store is None:
    st.session_state.doc_store = VectorStore(collection_name="doc_collection")
if st.session_state.url_store is None:
    st.session_state.url_store = VectorStore(collection_name="url_collection")

# ── Helper functions ──────────────────────────────────────────────────────────
def scrape_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (RAG-Bot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def call_groq(prompt: str, model: str, temperature: float, api_key: str) -> str:
    from groq import Groq
    if not api_key:
        return "⚠️ Please enter your Groq API key in the sidebar."
    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"

def call_ollama(prompt: str, model: str, temperature: float) -> str:
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. Try a smaller model."
    except Exception as e:
        return f"⚠️ Ollama error: {e}"

def plain_prompt(question: str, history: list) -> str:
    hist = "".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n" for m in history[-6:])
    return f"""You are a helpful, friendly AI assistant. Answer clearly and concisely.

Conversation History:
{hist}
User: {question}
Assistant:"""


def rag_prompt(question: str, context: str, history: list) -> str:
    hist = "".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}\n" for m in history[-6:])
    return f"""You are a helpful AI assistant. Use the context below to answer the question accurately.
If the context doesn't contain the answer, say so honestly.

Context:
{context}

Conversation History:
{hist}
User: {question}
Assistant:"""


def render_history(history: list):
    for msg in history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 <b>Assistant:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                badges = "".join(f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"])
                st.markdown(f"<div style='margin-left:1rem;margin-top:4px'>{badges}</div>", unsafe_allow_html=True)


def empty_state(icon: str, line1: str, line2: str = ""):
    st.markdown(f"""
    <div style='text-align:center;padding:2.5rem;'>
        <div style='font-size:3rem'>{icon}</div>
        <div style='font-family:IBM Plex Mono,monospace;font-size:1rem;color:#475569;margin-top:1rem'>{line1}</div>
        <div style='font-size:0.82rem;color:#334155;margin-top:0.5rem'>{line2}</div>
    </div>""", unsafe_allow_html=True)


def format_chat_history(history: list) -> str:
    out = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        out += f"**{role}**: {msg['content']}\n\n"
        if msg.get("sources"):
            out += f"_Sources: {', '.join(msg['sources'])}_\n\n"
        out += "---\n\n"
    return out


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🧠 RAG Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dual Interface</div>', unsafe_allow_html=True)
    
    st.markdown("**⚙️ AI Backend**")
    backend = st.radio("Provider", ["☁️ Groq (Cloud)", "🖥️ Ollama (Local)"], label_visibility="collapsed")
    is_groq = "Groq" in backend
    
    st.divider()
    
    if is_groq:
        st.markdown("**🔑 Groq API Key (BYOK)**")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        api_key_input = st.text_input("Groq API Key", value=groq_api_key, type="password", placeholder="gsk_...", label_visibility="collapsed")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            groq_api_key = api_key_input
        
        api_ok = check_groq_api_key(groq_api_key)
        st.markdown(f'<span class="{"status-ok" if api_ok else "status-err"}">{"● API Connected" if api_ok else "● Missing/Invalid Key"}</span>', unsafe_allow_html=True)
        
        st.markdown("**Groq Model**")
        chat_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "qwen-2.5-coder-32b"
        ]
        sel_chat = st.radio("Groq Model", chat_models, label_visibility="collapsed")
        
    else:
        st.markdown("**🖥️ Local Status**")
        api_ok = check_ollama_connection()
        st.markdown(f'<span class="{"status-ok" if api_ok else "status-err"}">{"● Ollama Connected" if api_ok else "● Ollama Offline"}</span>', unsafe_allow_html=True)
        if not api_ok:
            st.error("Run `ollama serve` in your terminal to connect.")
            
        st.markdown("**Ollama Model**")
        chat_models = list_available_ollama_models() or ["llama3.1:8b", "qwen2.5:7b"]
        sel_chat = st.radio("Ollama Model", chat_models, label_visibility="collapsed")

    st.divider()

    st.markdown("**📐 Embeddings & RAG Settings**")
    sel_emb  = st.selectbox("Embedding Model", ["all-MiniLM-L6-v2 (Local)"], index=0)
    top_k         = st.slider("Top-K chunks",        1,    10,   5)
    chunk_size    = st.slider("Chunk size (chars)", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Overlap (chars)",      0,  200,  50, step=10)
    temperature   = st.slider("Temperature",        0.0,  1.0, 0.3, step=0.05)

    st.divider()

    # Stats
    st.markdown("**📊 Stats**")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="metric-box"><div class="metric-val">{st.session_state.total_chunks}</div><div class="metric-lbl">Chunks</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-box"><div class="metric-val">{len(st.session_state.indexed_files) + len(st.session_state.indexed_urls)}</div><div class="metric-lbl">Sources</div></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Clear All"):
        st.session_state.doc_store.clear()
        st.session_state.url_store.clear()
        st.session_state.chat_history  = []
        st.session_state.doc_history   = []
        st.session_state.url_history   = []
        st.session_state.indexed_files = []
        st.session_state.indexed_urls  = []
        st.session_state.total_chunks  = 0
        st.rerun()


# ── MAIN — 3 TABS ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Dual Interface (Cloud/Local) · 3 modes</div>', unsafe_allow_html=True)

tab_chat, tab_doc, tab_url = st.tabs(["💬  Normal Chat", "📄  Document", "🌐  URL"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NORMAL CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown('<span class="mode-badge badge-chat">💬 Plain LLM Chat — no documents needed</span>', unsafe_allow_html=True)

    render_history(st.session_state.chat_history)

    if not st.session_state.chat_history:
        empty_state("💬", "Just start typing below", "No setup needed — pure LLM conversation")
    else:
        st.download_button(
            label="💾 Download Chat",
            data=format_chat_history(st.session_state.chat_history),
            file_name="chat_history.md",
            mime="text/markdown",
            key="dl_chat"
        )

    st.divider()
    user_input_chat = st.chat_input("Ask me anything…", key="input_chat")

    if user_input_chat:
        if not api_ok:
            st.error("Please configure your AI backend in the sidebar first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input_chat})
            with st.spinner(f"🤖 Thinking with {sel_chat}…"):
                prompt = plain_prompt(user_input_chat, st.session_state.chat_history[:-1])
                if is_groq:
                    answer = call_groq(prompt, sel_chat, temperature, groq_api_key)
                else:
                    answer = call_ollama(prompt, sel_chat, temperature)
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": []})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════
with tab_doc:
    st.markdown('<span class="mode-badge badge-doc">📄 Document RAG — answers grounded in your file</span>', unsafe_allow_html=True)

    # Upload + index section
    with st.expander("📂 Upload & Index a Document", expanded=not st.session_state.doc_store.is_ready()):
        uploaded = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], key="doc_uploader")
        if uploaded:
            data_dir  = Path("data"); data_dir.mkdir(exist_ok=True)
            save_path = data_dir / uploaded.name
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())
            if st.button("⚡ Index Document", key="btn_index_doc"):
                with st.spinner("Chunking…"):
                    chunks = load_and_chunk(str(save_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    st.error("No text found in file.")
                else:
                    prog = st.progress(0, text="Embedding…")
                    embedded = embed_chunks(chunks, model=sel_emb,
                                            progress_callback=lambda c, t: prog.progress(c/t, text=f"{c}/{t}"))
                    added = st.session_state.doc_store.add_chunks(embedded)
                    prog.empty()
                    st.session_state.indexed_files.append(uploaded.name)
                    st.session_state.total_chunks += added
                    st.success(f"✅ {added} chunks indexed from **{uploaded.name}**")
                    st.rerun()

        # Show already-indexed files
        if st.session_state.indexed_files:
            st.markdown("**Indexed files:**")
            for f in st.session_state.indexed_files:
                st.caption(f"✅ {f}")

        # Index files already in /data
        data_path = Path("data")
        if data_path.exists():
            existing = list(data_path.glob("*.pdf")) + list(data_path.glob("*.txt")) + list(data_path.glob("*.md"))
            unindexed = [d for d in existing if d.name not in st.session_state.indexed_files]
            if unindexed:
                st.markdown("**Files in /data (not yet indexed):**")
                for doc in unindexed:
                    c1, c2 = st.columns([3, 1])
                    c1.caption(f"📄 {doc.name}")
                    if c2.button("Index", key=f"idx_{doc.name}"):
                        with st.spinner(f"Indexing {doc.name}…"):
                            chunks   = load_and_chunk(str(doc), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                            embedded = embed_chunks(chunks, model=sel_emb)
                            added    = st.session_state.doc_store.add_chunks(embedded)
                            st.session_state.indexed_files.append(doc.name)
                            st.session_state.total_chunks += added
                            st.rerun()

    # Chat area
    render_history(st.session_state.doc_history)

    if not st.session_state.doc_history:
        if st.session_state.doc_store.is_ready():
            empty_state("📄", "Document indexed! Ask a question below", "Answers will be sourced from your file")
        else:
            empty_state("📂", "Upload and index a document first", "Use the panel above ↑")
    else:
        st.download_button(
            label="💾 Download Chat",
            data=format_chat_history(st.session_state.doc_history),
            file_name="doc_chat_history.md",
            mime="text/markdown",
            key="dl_doc"
        )

    st.divider()
    user_input_doc = st.chat_input("Ask about your document…", key="input_doc")

    if user_input_doc:
        if not api_ok:
            st.error("Please configure your AI backend in the sidebar first.")
        elif not st.session_state.doc_store.is_ready():
            st.warning("⚠️ Upload and index a document first (expand the panel above).")
        else:
            st.session_state.doc_history.append({"role": "user", "content": user_input_doc})
            with st.spinner("🔍 Searching document…"):
                q_emb   = embed_query(user_input_doc, model=sel_emb)
                results = st.session_state.doc_store.search(q_emb, top_k=top_k)
                context = build_context(results)
                sources = list(dict.fromkeys(r["source"] for r in results))
            with st.spinner(f"🤖 Generating with {sel_chat}…"):
                prompt = rag_prompt(user_input_doc, context, st.session_state.doc_history[:-1])
                if is_groq:
                    answer = call_groq(prompt, sel_chat, temperature, groq_api_key)
                else:
                    answer = call_ollama(prompt, sel_chat, temperature)
            st.session_state.doc_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — URL
# ══════════════════════════════════════════════════════════════════════════════
with tab_url:
    st.markdown('<span class="mode-badge badge-url">🌐 URL RAG — answers grounded in a webpage</span>', unsafe_allow_html=True)

    # URL input + index section
    with st.expander("🌐 Fetch & Index a URL", expanded=not st.session_state.url_store.is_ready()):
        url_input = st.text_input("Paste a URL", placeholder="https://en.wikipedia.org/wiki/...", key="url_field")
        if st.button("🌐 Fetch & Index", key="btn_index_url"):
            if not url_input.strip():
                st.warning("Enter a URL first.")
            else:
                with st.spinner(f"Fetching {url_input}…"):
                    try:
                        text = scrape_url(url_input.strip())
                        if len(text) < 100:
                            st.error("Too little text extracted. The page may require JavaScript.")
                        else:
                            source_name = url_input.strip().split("//")[-1][:60]
                            chunks      = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, source=source_name)
                            embedded    = embed_chunks(chunks, model=sel_emb)
                            added       = st.session_state.url_store.add_chunks(embedded)
                            st.session_state.indexed_urls.append(source_name)
                            st.session_state.total_chunks += added
                            st.success(f"✅ {added} chunks indexed from **{source_name}**")
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))

        if st.session_state.indexed_urls:
            st.markdown("**Indexed URLs:**")
            for u in st.session_state.indexed_urls:
                st.caption(f"🌐 {u}")

    # Chat area
    render_history(st.session_state.url_history)

    if not st.session_state.url_history:
        if st.session_state.url_store.is_ready():
            empty_state("🌐", "URL indexed! Ask about it below", "Answers will be sourced from the webpage")
        else:
            empty_state("🔗", "Paste and fetch a URL first", "Use the panel above ↑")
    else:
        st.download_button(
            label="💾 Download Chat",
            data=format_chat_history(st.session_state.url_history),
            file_name="url_chat_history.md",
            mime="text/markdown",
            key="dl_url"
        )

    st.divider()
    user_input_url = st.chat_input("Ask about the webpage…", key="input_url")

    if user_input_url:
        if not api_ok:
            st.error("Please configure your AI backend in the sidebar first.")
        elif not st.session_state.url_store.is_ready():
            st.warning("⚠️ Fetch and index a URL first (expand the panel above).")
        else:
            st.session_state.url_history.append({"role": "user", "content": user_input_url})
            with st.spinner("🔍 Searching webpage content…"):
                q_emb   = embed_query(user_input_url, model=sel_emb)
                results = st.session_state.url_store.search(q_emb, top_k=top_k)
                context = build_context(results)
                sources = list(dict.fromkeys(r["source"] for r in results))
            with st.spinner(f"🤖 Generating with {sel_chat}…"):
                prompt = rag_prompt(user_input_url, context, st.session_state.url_history[:-1])
                if is_groq:
                    answer = call_groq(prompt, sel_chat, temperature, groq_api_key)
                else:
                    answer = call_ollama(prompt, sel_chat, temperature)
            st.session_state.url_history.append({"role": "assistant", "content": answer, "sources": sources})
            st.rerun()
