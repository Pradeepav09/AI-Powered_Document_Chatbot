import hashlib
import os
import sqlite3
import time
from pathlib import Path
from datetime import datetime

import chromadb
import streamlit as st
from google import genai
from google.genai import errors, types
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Optional OCR imports
# ---------------------------------------------------------------------------
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
APP_TITLE = "DocuMind"
BASE_DIR = Path(__file__).parent
CHROMA_DB_PATH = BASE_DIR / ".chroma_db"
SQLITE_DB_PATH = BASE_DIR / "chat_history.db"
COLLECTION_NAME = "document_chatbot"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RETRIEVAL_COUNT = 5
OCR_DPI = 250

AVAILABLE_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]

SYSTEM_PROMPT = """You are a precise document assistant.
Answer the user's question using only the provided document context.
Rules:
- If the context does not contain enough information, say: "I cannot find the answer in the uploaded documents."
- Do not use outside knowledge to fill gaps.
- Be concise, clear, and well structured.
- Cite page numbers and document names when relevant, using the format: [Document Name, Page 3].
"""

# ---------------------------------------------------------------------------
# Page setup & Theme Colors
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DocuMind", page_icon="🧠", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    bg_color, text_color, sidebar_bg = "#0D0B14", "#F3EEFF", "#171321"
    accent_color, user_bubble, ai_bubble = "#B69CFF", "#2B1E40", "#1F1A2E"
    border_color, btn_text = "#332A4A", "#0D0B14"
    input_bg = "#1F1A2E"
else:
    bg_color, text_color, sidebar_bg = "#FFF8F0", "#43232D", "#F4E9DF"
    accent_color, user_bubble, ai_bubble = "#722F40", "#DCF8C6", "#FFFFFF"
    border_color, btn_text = "#EADCD2", "#FFFFFF"
    input_bg = "#FFFFFF"

# ---------------------------------------------------------------------------
# CSS & UI Styling
# ---------------------------------------------------------------------------
st.markdown(f"""
    <style>
        /* Global Background & Text */
        .stApp, [data-testid="stAppViewContainer"] {{ background-color: {bg_color} !important; color: {text_color} !important; }}
        h1, h2, h3, h4, h5, h6, p, span, div, label, li {{ color: {text_color} !important; }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; border-right: 1px solid {border_color}; }}
        
        /* Buttons */
        .stButton > button {{
            background-color: {accent_color} !important; color: {btn_text} !important;
            border: none !important; border-radius: 8px !important; width: 100%; font-weight: 600;
        }}
        .stButton > button * {{ color: {btn_text} !important; }}
        .stButton > button:hover {{ opacity: 0.8; }}

        /* --- FIX FOR CHAT INPUT AND TEXT AREAS --- */
        [data-testid="stChatInput"] {{ background-color: #FFFFFF !important; border: 1px solid {border_color} !important; }}
        [data-testid="stChatInput"] textarea, [data-testid="stChatInput"] input {{
            color: #000000 !important; -webkit-text-fill-color: #000000 !important; background-color: transparent !important;
        }}
        ::placeholder {{ color: #666666 !important; opacity: 1 !important; }}
        
        /* --- FIX FOR FILE UPLOADER --- */
        [data-testid="stFileUploader"] {{ background-color: {input_bg} !important; border: 1px solid {border_color} !important; border-radius: 10px; padding: 10px; }}
        [data-testid="stFileUploader"] section {{ background-color: #FFFFFF !important; border-radius: 8px; }}
        [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] button {{
            color: #000000 !important; -webkit-text-fill-color: #000000 !important;
        }}
        
        /* --- FIX FOR SELECTBOX DROPDOWN --- */
        div[data-baseweb="select"] > div {{ background-color: {input_bg} !important; color: {text_color} !important; border-color: {border_color} !important; }}
        div[data-baseweb="select"] * {{ color: {text_color} !important; }}
        ul[data-testid="stSelectboxVirtualDropdown"] {{ background-color: {input_bg} !important; }}
        ul[data-testid="stSelectboxVirtualDropdown"] li {{ color: {text_color} !important; }}
        
        /* --- FIX FOR TEXT INPUT & EYE ICON --- */
        /* Force the input background to be white and text to be black */
        [data-testid="stTextInput"] input {{ color: #000000 !important; -webkit-text-fill-color: #000000 !important; background-color: #FFFFFF !important; }}
        
        /* Force the eye icon inside the button to be black */
        [data-testid="stTextInput"] button svg {{
            fill: #000000 !important;
            color: #000000 !important;
            stroke: #000000 !important;
        }}
        
        /* --- FIX FOR TOOLTIPS (Hover text) --- */
        /* Ensure tooltip has a dark background and white text */
        div[data-baseweb="tooltip"] {{ background-color: #333333 !important; }}
        div[data-baseweb="tooltip"] * {{ color: #333333 !important; -webkit-text-fill-color: #FFFFFF !important; }}

        /* --- FIX FOR DOWNLOAD BUTTON --- */
        [data-testid="stDownloadButton"] > button {{
            background-color: {accent_color} !important; color: {btn_text} !important; border-radius: 8px !important;
        }}
        [data-testid="stDownloadButton"] > button * {{ color: {btn_text} !important; }}

        /* Chat Bubbles */
        [data-testid="stChatMessage"] {{ background-color: transparent !important; border: none !important; padding: 0 !important; margin-bottom: 15px; }}
        
        /* User Bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{ display: flex !important; flex-direction: row-reverse !important; }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2) {{
            background-color: {user_bubble} !important; padding: 12px 16px !important;
            border-radius: 15px 15px 0px 15px !important; max-width: 75% !important; margin-left: auto !important;
            border: 1px solid {border_color} !important;
        }}
        
        /* AI Bubble */
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{ display: flex !important; flex-direction: row !important; }}
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div:nth-child(2) {{
            background-color: {ai_bubble} !important; padding: 12px 16px !important;
            border-radius: 15px 15px 15px 0px !important; max-width: 75% !important;
            border: 1px solid {border_color} !important;
        }}

        /* Custom Emojis for Avatars */
        [data-testid="chatAvatarIcon-user"] svg, [data-testid="chatAvatarIcon-assistant"] svg,
        [data-testid="chatAvatarIcon-user"] img, [data-testid="chatAvatarIcon-assistant"] img {{ display: none !important; }}
        
        [data-testid="chatAvatarIcon-user"] {{ background-color: {accent_color} !important; border-radius: 50% !important; display: flex !important; justify-content: center !important; align-items: center !important; width: 35px !important; height: 35px !important; }}
        [data-testid="chatAvatarIcon-user"]::after {{ content: "👤"; font-size: 20px; }}
        [data-testid="chatAvatarIcon-assistant"] {{ background-color: #444 !important; border-radius: 50% !important; display: flex !important; justify-content: center !important; align-items: center !important; width: 35px !important; height: 35px !important; }}
        [data-testid="chatAvatarIcon-assistant"]::after {{ content: "🤖"; font-size: 20px; }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached Resources & Database Migration
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(): return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Preparing document database...")
def get_chroma_client():
    CHROMA_DB_PATH.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

def get_collection():
    return get_chroma_client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

@st.cache_resource
def get_sqlite_connection():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY, api_key_hash TEXT, title TEXT, created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, sources TEXT)""")
    
    c.execute("PRAGMA table_info(chats)")
    columns = [info[1] for info in c.fetchall()]
    if "updated_at" not in columns:
        c.execute("ALTER TABLE chats ADD COLUMN updated_at TEXT")
    
    c.execute("PRAGMA table_info(messages)")
    m_columns = [info[1] for info in c.fetchall()]
    if "timestamp" not in m_columns:
        c.execute("ALTER TABLE messages ADD COLUMN timestamp TEXT")
        
    conn.commit()
    return conn

embed_model = load_embedding_model()
conn = get_sqlite_connection()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
for k, v in {"processed_docs": [], "selected_model": AVAILABLE_MODELS[0], "suggested_questions": [], "current_key_hash": None, "current_chat_id": None, "messages": []}.items():
    st.session_state.setdefault(k, v)

def get_api_key(): return st.session_state.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
def get_key_hash(key): return hashlib.sha256(key.encode()).hexdigest()[:16] if key else None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    col1, col2 = st.columns([0.85, 0.15])
    with col1: st.header(APP_TITLE)
    with col2:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        # Standard tooltip behavior
        if st.button(icon, help="Toggle Theme", key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = None
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.rerun()

    st.markdown("---")
    
    api_key_input = st.text_input("Gemini API Key", type="password", value=st.session_state.get("GEMINI_API_KEY", ""))
    if api_key_input and api_key_input != st.session_state.get("GEMINI_API_KEY"):
        st.session_state["GEMINI_API_KEY"] = api_key_input.strip()
        st.session_state.current_key_hash = get_key_hash(api_key_input.strip())
        st.rerun()

    st.session_state.selected_model = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.selected_model))

    if st.session_state.current_key_hash:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM chats WHERE api_key_hash = ? ORDER BY created_at DESC", (st.session_state.current_key_hash,))
        chats = cursor.fetchall()

        if chats:
            st.markdown("### Chat History")
            grouped = {}
            for chat_id, title, created_at_str in chats:
                try: dt = datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                except: dt = datetime.now()
                diff = (datetime.now() - dt).days
                group = "Today" if diff == 0 else ("30 Days" if diff <= 30 else dt.strftime("%Y-%m"))
                if group not in grouped: grouped[group] = []
                grouped[group].append((chat_id, title))

            for group_name, items in grouped.items():
                st.caption(f"**{group_name}**")
                for cid, title in items:
                    c1, c2 = st.columns([0.85, 0.15])
                    with c1:
                        if st.button(f"💬 {title[:20]}...", key=f"load_{cid}", use_container_width=True):
                            st.session_state.current_chat_id = cid
                            m_cursor = conn.cursor()
                            m_cursor.execute("SELECT role, content, sources FROM messages WHERE chat_id = ? ORDER BY id ASC", (cid,))
                            msgs = []
                            for r, c, s in m_cursor.fetchall():
                                if s:
                                    try: sources = s.split("\n")
                                    except: sources = [s]
                                else: sources = []
                                msgs.append({"role": r, "content": c, "sources": sources})
                            st.session_state.messages = msgs
                            st.session_state.suggested_questions = []
                            st.rerun()
                    with c2:
                        if st.button("🗑️", key=f"del_{cid}"):
                            conn.execute("DELETE FROM messages WHERE chat_id = ?", (cid,))
                            conn.execute("DELETE FROM chats WHERE id = ?", (cid,))
                            conn.commit()
                            if st.session_state.current_chat_id == cid:
                                st.session_state.current_chat_id = None
                                st.session_state.messages = []
                            st.rerun()

    st.markdown("---")
    
    st.header("📄 Knowledge Base")
    uploaded_files = st.file_uploader("Upload PDF, TXT, or MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    use_ocr = st.toggle("Auto-OCR for scanned pages", value=True)

    if uploaded_files and st.button("Process Documents", use_container_width=True):
        st.session_state.process_clicked = True
        st.rerun()

    if st.session_state.processed_docs:
        st.markdown("### Active Documents")
        for i, doc in enumerate(st.session_state.processed_docs):
            c1, c2 = st.columns([0.8, 0.2])
            with c1: st.caption(f"📄 {doc['name']}")
            with c2:
                if st.button("🗑️", key=f"del_doc_{i}"):
                    get_collection().delete(where={"session_id": st.session_state.current_key_hash, "source": doc["name"]})
                    st.session_state.processed_docs.pop(i)
                    st.rerun()

# ---------------------------------------------------------------------------
# Core Logic Functions
# ---------------------------------------------------------------------------
def clean_text(text): return " ".join(text.replace("\x00", " ").split())

def extract_pdf_pages(pdf_file, use_ocr):
    pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    pages = []
    
    pytesseract.pytesseract.tesseract_cmd = r'ADD_YOUR_TESSERACT_PATH'
    poppler_bin_path = r'ADD_YOUR_POPPLER_BIN_PATH'

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not clean_text(text) and use_ocr and OCR_AVAILABLE:
            try:
                pdf_file.seek(0)
                images = convert_from_bytes(pdf_file.getvalue(), dpi=OCR_DPI, first_page=page_number, last_page=page_number, poppler_path=poppler_bin_path)
                if images: text = pytesseract.image_to_string(images[0], lang="eng")
            except Exception as e: st.warning(f"OCR failed on page {page_number}: {e}")
        if clean_text(text): pages.append({"page": page_number, "text": text})
    return pages, len(reader.pages)

def extract_text_file(uploaded_file):
    uploaded_file.seek(0)
    try: text = uploaded_file.getvalue().decode("utf-8")
    except UnicodeDecodeError: text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [{"page": 1, "text": text}], 1

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = clean_text(text)
    if len(text) <= chunk_size: return [text] if text else []
    chunks, step = [], chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk.strip(): chunks.append(chunk)
        if start + chunk_size >= len(text): break
    return chunks

def file_hash(uploaded_file):
    uploaded_file.seek(0)
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    uploaded_file.seek(0)
    return digest

def process_documents(uploaded_files, use_ocr, client):
    collection = get_collection()
    total_chunks, processed_names, first_doc_text = 0, [], ""

    for uploaded_file in uploaded_files:
        f_hash = file_hash(uploaded_file)
        if any(doc["hash"] == f_hash for doc in st.session_state.processed_docs):
            st.info(f"“{uploaded_file.name}” is already in your knowledge base.")
            continue

        if uploaded_file.type == "application/pdf": pages, _ = extract_pdf_pages(uploaded_file, use_ocr)
        else: pages, _ = extract_text_file(uploaded_file)

        if not pages:
            st.warning(f"No readable text found in {uploaded_file.name}.")
            continue

        if not first_doc_text: first_doc_text = pages[0]["text"]

        documents, metadatas, ids = [], [], []
        for page in pages:
            for chunk_number, chunk in enumerate(chunk_text(page["text"])):
                documents.append(chunk)
                metadatas.append({"source": uploaded_file.name, "page": page["page"], "chunk": chunk_number, "session_id": st.session_state.current_key_hash})
                ids.append(f"{st.session_state.current_key_hash}_{f_hash[:8]}_{page['page']}_{chunk_number}")

        if documents:
            embeddings = embed_model.encode(documents, batch_size=32, show_progress_bar=False).tolist()
            collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            st.session_state.processed_docs.append({"name": uploaded_file.name, "hash": f_hash})
            total_chunks += len(documents)
            processed_names.append(uploaded_file.name)

    if processed_names and first_doc_text and client:
        with st.spinner("Generating summary..."):
            try:
                prompt = f"Summarize the following document in 3-4 bullet points, then provide 3 short questions a user could ask. Format exactly like this:\nSUMMARY:\n- point 1\n- point 2\n- point 3\nQUESTIONS:\n- Q1\n- Q2\n- Q3\n\nDocument text:\n{first_doc_text[:4000]}"
                res = client.models.generate_content(model=st.session_state.selected_model, contents=prompt)
                if "SUMMARY:" in res.text and "QUESTIONS:" in res.text:
                    parts = res.text.split("QUESTIONS:")
                    summary = parts[0].replace("SUMMARY:", "").strip()
                    qs = [q.strip("- ").strip() for q in parts[1].strip().split("\n") if q.strip()]
                    st.session_state.suggested_questions = qs[:3]
                    st.session_state.messages.append({"role": "assistant", "content": f"📄 **Document Summary:**\n\n{summary}\n\n*Click a suggested question below!*", "sources": []})
            except Exception as e: st.warning(f"Could not generate summary: {e}")
    return total_chunks, len(processed_names)

def retrieve_context(question, client):
    collection = get_collection()
    if not st.session_state.processed_docs: return [], []

    search_query = question
    if len(st.session_state.messages) > 1:
        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-4:] if m['role'] in ['user', 'assistant'] and not m['content'].startswith('📄')])
        try: search_query = client.models.generate_content(model=st.session_state.selected_model, contents=f"History:\n{history}\n\nFollow-up: {question}\n\nStandalone search query:").text.strip()
        except: pass

    query_embedding = embed_model.encode([search_query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=min(RETRIEVAL_COUNT, collection.count()), where={"session_id": st.session_state.current_key_hash}, include=["documents", "metadatas", "distances"])

    documents, metadatas, distances = results.get("documents", [[]])[0], results.get("metadatas", [[]])[0], results.get("distances", [[]])[0]
    context_blocks, source_records = [], []
    for document, metadata, distance in zip(documents, metadatas, distances):
        page, source = metadata.get("page", "unknown"), metadata.get("source", "uploaded document")
        context_blocks.append(f"[Source: {source} | Page {page}]\n{document}")
        source_records.append({"source": source, "page": page, "distance": round(float(distance), 4), "preview": document[:350].strip() + ("..." if len(document) > 350 else "")})
    return context_blocks, source_records

def query_gemini(question, context_blocks, client):
    context = "\n\n---\n\n".join(context_blocks)
    try:
        response = client.models.generate_content(
            model=st.session_state.selected_model, 
            contents=f"Context:\n{context}\n\nQuestion: {question}", 
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2, max_output_tokens=2048)
        )
        return (response.text or "").strip()
    except Exception as exc: return f"I could not generate an answer. Details: `{exc}`"

# ---------------------------------------------------------------------------
# Execution & Main UI
# ---------------------------------------------------------------------------
api_key = get_api_key()
if not api_key:
    st.info("👈 Enter your Gemini API key in the sidebar to start.")
    st.stop()

if not st.session_state.current_key_hash: st.session_state.current_key_hash = get_key_hash(api_key)
try: client = genai.Client(api_key=api_key)
except Exception as exc: st.error(f"Client error: {exc}"); st.stop()

if st.session_state.get("process_clicked") and uploaded_files:
    chunks, num_docs = process_documents(uploaded_files, use_ocr, client)
    st.session_state.process_clicked = False
    if num_docs > 0:
        st.success(f"Added {num_docs} document(s).")
        st.rerun()

# Header and Export Button on the Right
head_col1, head_col2 = st.columns([0.7, 0.3])
with head_col1:
    st.title("🧠 " + APP_TITLE)
    st.write("Upload documents, ask questions, and receive grounded answers with page references.")

with head_col2:
    st.write("") 
    if st.session_state.messages:
        chat_text = f"# DocuMind Export - {time.strftime('%Y-%m-%d')}\n\n"
        for msg in st.session_state.messages:
            role = "👤 User" if msg["role"] == "user" else "🧠 DocuMind"
            chat_text += f"**{role}:**\n{msg['content']}\n\n"
        st.download_button("📥 Export Chat", data=chat_text, file_name="chat.md", mime="text/markdown", use_container_width=True)

# Render Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Show sources"):
                for source in message["sources"]: st.markdown(source)

# Suggested Questions
if st.session_state.suggested_questions:
    st.write("💡 **Suggested Questions:**")
    cols = st.columns(len(st.session_state.suggested_questions))
    for i, q in enumerate(st.session_state.suggested_questions):
        if cols[i].button(q, key=f"sq_{i}"):
            st.session_state.suggested_questions = []
            st.session_state.clicked_question = q
            st.rerun()

# Chat Input
question = st.chat_input("Ask DocuMind...")
if getattr(st.session_state, "clicked_question", None):
    question = st.session_state.clicked_question
    st.session_state.clicked_question = None

if question:
    if not st.session_state.current_chat_id:
        new_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        title = question[:30] + ("..." if len(question) > 30 else "")
        conn.execute("INSERT INTO chats (id, api_key_hash, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (new_id, st.session_state.current_key_hash, title, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        st.session_state.current_chat_id = new_id

    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    conn.execute("INSERT INTO messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)", (st.session_state.current_chat_id, "user", question, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_blocks, source_records = retrieve_context(question, client)
            if not context_blocks:
                response, sources = "I cannot find the answer in the uploaded documents.", []
            else:
                response = query_gemini(question, context_blocks, client)
                sources = [f"**{r['source']} — Page {r['page']}**  \n> {r['preview']}" for r in source_records]
        
        st.markdown(response)
        if sources:
            with st.expander("📚 Show sources"):
                for source in sources: st.markdown(source)

        st.session_state.messages.append({"role": "assistant", "content": response, "sources": sources})
        conn.execute("INSERT INTO messages (chat_id, role, content, sources, timestamp) VALUES (?, ?, ?, ?, ?)", (st.session_state.current_chat_id, "assistant", response, "\n".join(sources) if sources else None, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
    st.rerun()