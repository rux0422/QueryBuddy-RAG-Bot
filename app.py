"""
QueryBuddy — Multi-Document RAG PDF Q&A
========================================
Verified API usage (Cohere SDK v5 / ClientV2):
  Embed : co.embed(texts, model, input_type, embedding_types=["float"])
          → resp.embeddings.float  (list of float vectors)
  Chat  : co.chat(model, messages=[{"role":..., "content":...}])
          → resp.message.content[0].text

Pinecone: one namespace per document → perfect isolation, unlimited docs.
          Pinecone index must be: dimension=1024, metric=cosine
"""

import os
import re
import streamlit as st
import PyPDF2
from pinecone import Pinecone
import cohere
from config import PINECONE_API_KEY, COHERE_API_KEY

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
EMBED_MODEL   = "embed-english-v3.0"      # 1024-dim
CHAT_MODEL    = "command-r-plus-08-2024"  # latest stable
INDEX_NAME    = "qa-bot-index"
CHUNK_SIZE    = 600    # chars — larger chunks = more context per chunk
CHUNK_OVERLAP = 100    # chars overlap between chunks
COHERE_BATCH  = 90     # stay safely under Cohere's 96-text limit
PINE_BATCH    = 100
TOP_K         = 10     # how many chunks to retrieve per question

# ─────────────────────────────────────────────────────────────
# PAGE
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QueryBuddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
_D = {
    "doc_library":          {},     # {ns: {"name": str, "chunks": int}}
    "active_ns":            None,   # currently selected namespace
    "file_uploader_key":    0,
    "current_question":     "",
    "current_answer":       "",
    "answer_ready":         False,
    "processing":           False,
    "matches":              [],
    "qa_history":           [],
    "show_success":         False,
    "success_msg":          "",
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.answer-box {
    background: #f7f9fc;
    border-left: 5px solid #1976D2;
    padding: 1.2em 1.5em;
    border-radius: 6px;
    font-size: 1.02rem;
    line-height: 1.75;
    margin: 0.5em 0 1em 0;
    white-space: pre-wrap;
}
.success-box {
    background: #e8f5e9;
    border-left: 4px solid #4CAF50;
    padding: 0.8em 1.2em;
    border-radius: 4px;
    margin: 0.6em 0;
}
.seg-box {
    background: #fafafa;
    border-left: 3px solid #90CAF9;
    padding: 0.6em 1em;
    border-radius: 4px;
    font-size: 0.87em;
    line-height: 1.55;
    margin-bottom: 0.5em;
}
.doc-pill {
    display: inline-block;
    background: #e3f2fd;
    border-radius: 12px;
    padding: 0.15em 0.7em;
    font-size: 0.82em;
    color: #1565C0;
    margin-bottom: 0.3em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CLIENTS — initialised once and cached
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to Cohere & Pinecone…")
def _init():
    """
    Verified ClientV2 initialisation (Cohere SDK v5).
    Source: https://github.com/cohere-ai/cohere-python

      import cohere
      co = cohere.ClientV2()   # or ClientV2(api_key="…")

    Pinecone: standard SDK init, no changes needed.
    """
    pc  = Pinecone(api_key=PINECONE_API_KEY)
    idx = pc.Index(INDEX_NAME)
    idx.describe_index_stats()          # fast-fail if wrong key/name
    co  = cohere.ClientV2(api_key=COHERE_API_KEY)
    return idx, co


try:
    _idx, _co = _init()
except Exception as _err:
    st.error(f"❌ Could not connect: {_err}")
    st.info(
        "**Check:**\n"
        "- `PINECONE_API_KEY` and `COHERE_API_KEY` in `config.py`\n"
        f"- Pinecone index **`{INDEX_NAME}`** exists with "
        "`dimension=1024`, `metric=cosine`"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
def _embed(texts: list[str], input_type: str) -> list[list[float]] | None:
    """
    Verified Cohere ClientV2 embed call.

    When embedding_types=["float"] is passed, embeddings come back
    under resp.embeddings.float — this is the documented ClientV2 behaviour.

    input_type:
      "search_document" → for chunks stored in Pinecone
      "search_query"    → for the user's question
    """
    try:
        resp = _co.embed(
            texts=texts,
            model=EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
        )
        return resp.embeddings.float     # ClientV2 path
    except AttributeError:
        # Safety fallback: some builds return list directly
        try:
            return list(resp.embeddings)
        except Exception:
            return None
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None


def _embed_one(text: str, input_type: str) -> list[float] | None:
    vecs = _embed([text], input_type)
    return vecs[0] if vecs else None


# ─────────────────────────────────────────────────────────────
# PDF UTILITIES
# ─────────────────────────────────────────────────────────────
def _extract(file) -> str:
    """Extract all text from a PDF, page by page."""
    try:
        rdr   = PyPDF2.PdfReader(file)
        pages = []
        for page in rdr.pages:
            t = page.extract_text()
            if t and t.strip():
                pages.append(t.strip())
        return "\n\n".join(pages)
    except Exception as e:
        st.error(f"PDF error: {e}")
        return ""


def _chunk(text: str) -> list[str]:
    """
    Sliding-window chunking.
    600-char chunks give ~120-150 tokens — plenty of context per chunk,
    still well within Cohere's 512-token embed limit.
    100-char overlap prevents missing answers that span a boundary.
    """
    out, pos = [], 0
    while pos < len(text):
        chunk = text[pos : pos + CHUNK_SIZE].strip()
        if chunk:
            out.append(chunk)
        pos += CHUNK_SIZE - CHUNK_OVERLAP
    return out


def _ns(filename: str) -> str:
    """Sanitize filename → safe Pinecone namespace string."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"[^a-zA-Z0-9\-_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name[:60] or "doc")


# ─────────────────────────────────────────────────────────────
# INDEX MANAGEMENT
# ─────────────────────────────────────────────────────────────
def _delete_ns(namespace: str):
    """Wipe all vectors for one document from Pinecone."""
    try:
        _idx.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        st.warning(f"Namespace delete warning (non-fatal): {e}")


def _index_pdf(file, namespace: str) -> int:
    """
    Extract → chunk → embed → upsert.
    Returns chunk count on success, 0 on failure.

    Each document lives in its own Pinecone namespace so:
      • documents never interfere with each other
      • queries are perfectly isolated
      • deleting a doc = deleting one namespace
      • no per-doc metadata filtering needed
    """
    text = _extract(file)
    if not text.strip():
        st.error("No text found — PDF may be image/scanned.")
        return 0

    chunks = _chunk(text)
    if not chunks:
        st.error("Chunking failed.")
        return 0

    vectors  = []
    n_embed  = (len(chunks) + COHERE_BATCH - 1) // COHERE_BATCH
    prog1    = st.progress(0.0, text="Embedding… (step 1/2)")

    for b, i in enumerate(range(0, len(chunks), COHERE_BATCH)):
        batch = chunks[i : i + COHERE_BATCH]
        vecs  = _embed(batch, "search_document")
        if vecs is None:
            st.error(f"Embedding failed at batch {b+1}.")
            return 0
        for j, (chunk, vec) in enumerate(zip(batch, vecs)):
            vectors.append({
                "id":       f"c{i+j}",
                "values":   vec,
                "metadata": {"text": chunk},
            })
        prog1.progress((b + 1) / n_embed, text=f"Embedding {b+1}/{n_embed}…")

    prog1.empty()

    n_up  = (len(vectors) + PINE_BATCH - 1) // PINE_BATCH
    prog2 = st.progress(0.0, text="Uploading… (step 2/2)")

    for b, i in enumerate(range(0, len(vectors), PINE_BATCH)):
        try:
            _idx.upsert(vectors=vectors[i : i + PINE_BATCH], namespace=namespace)
        except Exception as e:
            st.error(f"Pinecone upsert error: {e}")
            return 0
        prog2.progress((b + 1) / n_up, text=f"Uploading {b+1}/{n_up}…")

    prog2.empty()
    return len(chunks)


# ─────────────────────────────────────────────────────────────
# ANSWER GENERATION
# ─────────────────────────────────────────────────────────────
def _answer(question: str, context: str, doc_name: str) -> str:
    """
    Verified Cohere ClientV2 chat call.
    Source: https://github.com/cohere-ai/cohere-python

      co.chat(model=..., messages=[{"role": "system"|"user", "content": "..."}])
      → resp.message.content[0].text

    The system prompt is used to inject document context.
    The user message is the plain question — no special formatting needed.
    """
    ctx = context[:15000] if len(context) > 15000 else context

    system = (
        "You are QueryBuddy, an expert document assistant. "
        "A user has uploaded a document and is asking questions about it. "
        "Answer naturally, clearly, and completely using ONLY the context provided. "
        "Write your answer as flowing natural language — not as bullet points unless "
        "the question specifically asks for a list. "
        "If the answer spans multiple points, connect them with natural transitions. "
        "If the document does not contain the answer, say: "
        "'The document does not mention this topic.' "
        "Never say you cannot help. Never make up information.\n\n"
        f"Document: {doc_name}\n\n"
        f"Relevant excerpts:\n{ctx}"
    )

    try:
        resp = _co.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": question},
            ],
        )
        # ClientV2: resp.message.content is a list of content blocks
        return resp.message.content[0].text.strip()
    except Exception as e:
        return f"Error generating answer: {e}"


# ─────────────────────────────────────────────────────────────
# QUESTION PIPELINE
# ─────────────────────────────────────────────────────────────
def _run_qa(question: str, namespace: str, doc_name: str):
    """
    Full retrieval + generation pipeline.
    1. Embed question (input_type='search_query')
    2. Query Pinecone — scoped to document namespace
    3. Build context from top-K chunks
    4. Generate natural language answer via Cohere chat
    """
    # Step 1 — embed question
    qvec = _embed_one(question, "search_query")
    if qvec is None:
        st.session_state.current_answer = "Could not embed your question. Please try again."
        st.session_state.processing     = False
        st.session_state.answer_ready   = True
        return

    # Step 2 — retrieve
    try:
        res = _idx.query(
            vector=qvec,
            top_k=TOP_K,
            include_metadata=True,
            namespace=namespace,
        )
    except Exception as e:
        st.session_state.current_answer = f"Retrieval error: {e}"
        st.session_state.processing     = False
        st.session_state.answer_ready   = True
        return

    hits = res.get("matches", [])
    if not hits:
        st.session_state.current_answer = (
            "I could not find relevant content in this document for your question. "
            "Try rephrasing or ask something else."
        )
        st.session_state.matches      = []
        st.session_state.processing   = False
        st.session_state.answer_ready = True
        return

    # Step 3 — build context (best matches first)
    hits.sort(key=lambda m: m.get("score", 0), reverse=True)
    context = "\n\n---\n\n".join(m["metadata"]["text"] for m in hits)

    # Step 4 — generate answer
    answer = _answer(question, context, doc_name)

    st.session_state.current_answer = answer
    st.session_state.matches        = hits
    st.session_state.processing     = False
    st.session_state.answer_ready   = True


# ─────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────
def cb_index():
    """Index the uploaded file and add it to the library."""
    raw  = st.session_state.get("_uploaded")
    if not raw:
        return
    ns   = _ns(raw.name)
    name = raw.name

    if ns in st.session_state.doc_library:
        st.session_state.show_success = True
        st.session_state.success_msg  = (
            f"ℹ️ <b>{name}</b> is already indexed. Select it from the sidebar."
        )
        st.session_state.file_uploader_key += 1
        return

    _delete_ns(ns)   # clean slate in case of partial previous upload
    n = _index_pdf(raw, ns)
    if n:
        st.session_state.doc_library[ns] = {"name": name, "chunks": n}
        st.session_state.active_ns        = ns
        st.session_state.current_answer   = ""
        st.session_state.answer_ready     = False
        st.session_state.matches          = []
        st.session_state.show_success     = True
        st.session_state.success_msg      = (
            f"✅ <b>{name}</b> indexed — {n} chunks. "
            f"Ask any question below!"
        )
    st.session_state.file_uploader_key += 1


def cb_ask():
    q = st.session_state.get("q_input", "").strip()
    if not q:
        st.warning("Please type a question.")
        return
    if not st.session_state.active_ns:
        st.warning("Select a document first.")
        return
    st.session_state.current_question = q
    st.session_state.current_answer   = ""
    st.session_state.answer_ready     = False
    st.session_state.matches          = []
    st.session_state.processing       = True


def cb_save_next():
    """Save current Q&A to history and clear for next question."""
    q = st.session_state.current_question
    a = st.session_state.current_answer
    if q and a:
        ns   = st.session_state.active_ns or ""
        dname = st.session_state.doc_library.get(ns, {}).get("name", "Unknown")
        st.session_state.qa_history.append({"q": q, "a": a, "doc": dname})
    st.session_state.current_question = ""
    st.session_state.current_answer   = ""
    st.session_state.answer_ready     = False
    st.session_state.matches          = []
    st.session_state.processing       = False


def cb_select_doc(ns: str):
    st.session_state.active_ns        = ns
    st.session_state.current_question = ""
    st.session_state.current_answer   = ""
    st.session_state.answer_ready     = False
    st.session_state.matches          = []


def cb_delete(ns: str):
    _delete_ns(ns)
    st.session_state.doc_library.pop(ns, None)
    if st.session_state.active_ns == ns:
        rest = list(st.session_state.doc_library)
        st.session_state.active_ns      = rest[0] if rest else None
        st.session_state.current_answer = ""
        st.session_state.answer_ready   = False
        st.session_state.matches        = []
    st.rerun()


def cb_reset():
    for ns in list(st.session_state.doc_library):
        _delete_ns(ns)
    for k, v in _D.items():
        if isinstance(v, dict):
            st.session_state[k] = {}
        elif isinstance(v, list):
            st.session_state[k] = []
        elif k == "file_uploader_key":
            st.session_state[k] = st.session_state.get(k, 0) + 1
        else:
            st.session_state[k] = v
    st.rerun()


# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
st.title("📚 QueryBuddy")
st.caption("Upload any number of PDFs of any size — each is indexed separately and answers your questions in natural language.")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Documents")

    lib = st.session_state.doc_library
    if lib:
        for ns, info in lib.items():
            active = ns == st.session_state.active_ns
            ca, cb_ = st.columns([4, 1])
            with ca:
                label = ("🟢 " if active else "⚪ ") + info["name"][:28]
                if st.button(label, key=f"sel_{ns}", use_container_width=True):
                    cb_select_doc(ns)
                    st.rerun()
                st.caption(f"  {info['chunks']} chunks")
            with cb_:
                if st.button("🗑", key=f"del_{ns}"):
                    cb_delete(ns)
        st.markdown("---")
    else:
        st.info("No documents yet.")

    if st.session_state.qa_history:
        st.markdown("### 📝 History")
        for i, item in enumerate(reversed(st.session_state.qa_history)):
            idx_ = len(st.session_state.qa_history) - i
            short = item["q"][:42] + ("…" if len(item["q"]) > 42 else "")
            with st.expander(f"Q{idx_}: {short}"):
                st.caption(f"📄 {item['doc']}")
                st.markdown(
                    f'<div class="answer-box">{item["a"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.button("⚠️ Reset Everything", on_click=cb_reset, use_container_width=True)


# ── UPLOAD SECTION ────────────────────────────────────────────
with st.expander("➕ Upload & Index a PDF", expanded=not bool(st.session_state.doc_library)):
    uploaded = st.file_uploader(
        "Choose PDF",
        type="pdf",
        key=f"fu_{st.session_state.file_uploader_key}",
        label_visibility="collapsed",
    )
    if uploaded:
        st.session_state["_uploaded"] = uploaded
        st.info(f"📄 **{uploaded.name}** — ready to index")
        st.button("🚀 Index This PDF", type="primary", on_click=cb_index)

    if st.session_state.show_success:
        st.markdown(
            f'<div class="success-box">{st.session_state.success_msg}</div>',
            unsafe_allow_html=True,
        )
        st.session_state.show_success = False


# ── Q&A SECTION ───────────────────────────────────────────────
st.markdown("---")

active_ns = st.session_state.active_ns
active_info = st.session_state.doc_library.get(active_ns) if active_ns else None

if active_info:
    st.subheader(f"💬 Ask About: {active_info['name']}")

    st.text_input(
        "Your question:",
        key="q_input",
        placeholder="Ask anything about this document in natural language…",
        label_visibility="collapsed",
    )

    c1, c2, _ = st.columns([1.5, 2, 6])
    with c1:
        st.button("🔎 Ask", type="primary", on_click=cb_ask, use_container_width=True)
    with c2:
        st.button("➡️ Save & Next", on_click=cb_save_next, use_container_width=True)

    # ── Run pipeline ─────────────────────────────────────────
    if st.session_state.processing:
        with st.spinner("Searching document and generating answer…"):
            _run_qa(
                question  = st.session_state.current_question,
                namespace = active_ns,
                doc_name  = active_info["name"],
            )
        st.rerun()

    # ── Display answer ────────────────────────────────────────
    if st.session_state.answer_ready and st.session_state.current_answer:
        st.markdown(f"**Q: {st.session_state.current_question}**")
        st.markdown(
            f'<div class="answer-box">{st.session_state.current_answer}</div>',
            unsafe_allow_html=True,
        )

        # Source segments (collapsible)
        if st.session_state.matches:
            with st.expander("📑 Source Segments", expanded=False):
                for i, m in enumerate(st.session_state.matches):
                    score = m.get("score", 0)
                    st.markdown(
                        f'<div class="seg-box">'
                        f'<b>Segment {i+1}</b> &nbsp;·&nbsp; '
                        f'Score: <code>{score:.4f}</code><br><br>'
                        f'{m["metadata"]["text"]}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

elif st.session_state.doc_library:
    st.info("👈 Select a document from the sidebar to start asking questions.")
else:
    st.info("👆 Upload and index a PDF to get started.")
