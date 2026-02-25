import os
import re
import streamlit as st
import PyPDF2
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import cohere
from config import PINECONE_API_KEY, COHERE_API_KEY

# Page configuration
st.set_page_config(
    page_title="QueryBuddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ALL session state variables at the start
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'current_pdf_safe_name' not in st.session_state:
    st.session_state.current_pdf_safe_name = None
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing_question' not in st.session_state:
    st.session_state.processing_question = False
if 'show_success_message' not in st.session_state:
    st.session_state.show_success_message = False
if 'success_message' not in st.session_state:
    st.session_state.success_message = ""
if 'question_text' not in st.session_state:
    st.session_state.question_text = ""
if 'vector_ids' not in st.session_state:
    st.session_state.vector_ids = []
if 'current_pdf_vectors' not in st.session_state:
    st.session_state.current_pdf_vectors = set()

# CSS styles
st.markdown("""
    <style>
    .stButton > button {
        width: auto !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        height: 2.5rem !important;
    }
    .answer {
        background-color: #f0f2f6;
        padding: 1em;
        border-radius: 5px;
        margin-bottom: 1em;
    }
    .history-item {
        margin-bottom: 1em;
        padding: 0.5em;
        border-left: 3px solid #4CAF50;
    }
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: #4CAF50 !important;
        color: white !important;
        border: none !important;
    }
    .success-message-container {
        padding: 1em;
        background-color: #e8f5e9;
        border-radius: 5px;
        margin: 1em 0;
    }
    </style>
""", unsafe_allow_html=True)


def sanitize_filename(name):
    """Convert a filename into a safe string usable in Pinecone vector IDs."""
    name = os.path.splitext(name)[0]
    name = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name


@st.cache_resource
def initialize_components():
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index("qa-bot-index")
        index.describe_index_stats()
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        co = cohere.Client(COHERE_API_KEY)
        return pc, index, tokenizer, model, co
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return None, None, None, None, None


pc, index, tokenizer, model, co = initialize_components()


def clear_index():
    try:
        if st.session_state.current_pdf_vectors:
            batch_size = 100
            vector_list = list(st.session_state.current_pdf_vectors)
            for i in range(0, len(vector_list), batch_size):
                batch = vector_list[i:i + batch_size]
                index.delete(ids=batch)
            st.session_state.current_pdf_vectors.clear()
        return True
    except Exception as e:
        st.warning(f"Note: Index clearing encountered an issue: {str(e)}")
        return False


def reset_session_state():
    clear_index()
    st.session_state.pdf_processed = False
    st.session_state.num_chunks = 0
    st.session_state.qa_history = []
    st.session_state.current_pdf_name = None
    st.session_state.current_pdf_safe_name = None
    st.session_state.file_uploader_key += 1
    st.session_state.current_question = ""
    st.session_state.current_answer = ""
    st.session_state.results = None
    st.session_state.processing_question = False
    st.session_state.show_success_message = False
    st.session_state.success_message = ""
    st.session_state.question_text = ""
    st.session_state.vector_ids = []
    st.session_state.current_pdf_vectors = set()
    st.rerun()


def embed_text(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().tolist()
    except Exception as e:
        st.error(f"Error embedding text: {str(e)}")
        return None


def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""


def split_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def process_pdf(file):
    try:
        clear_index()

        text = extract_text_from_pdf(file)
        if not text.strip():
            st.error("No text could be extracted from the PDF. It may be scanned/image-based.")
            return 0

        chunks = split_text(text)
        current_pdf_vectors = set()
        safe_name = st.session_state.current_pdf_safe_name

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            vectors = []
            for j, chunk in enumerate(batch_chunks):
                if not chunk.strip():
                    continue
                embedding = embed_text(chunk)
                if embedding is None:
                    continue
                # Clean, predictable vector ID using sanitized filename
                doc_id = f"doc_{safe_name}_{i + j}"
                current_pdf_vectors.add(doc_id)
                vectors.append((doc_id, embedding, {"text": chunk, "pdf_name": safe_name}))

            if vectors:
                try:
                    index.upsert(vectors=vectors)
                except Exception as e:
                    st.error(f"Error uploading batch {i // batch_size + 1}: {str(e)}")
                    continue

        st.session_state.current_pdf_vectors = current_pdf_vectors
        return len(chunks)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return 0


def process_pdf_callback():
    if st.session_state.uploaded_file is not None:
        with st.spinner("Processing PDF... Please wait"):
            num_chunks = process_pdf(st.session_state.uploaded_file)
            if num_chunks > 0:
                st.session_state.num_chunks = num_chunks
                st.session_state.pdf_processed = True
                st.session_state.show_success_message = True
                st.session_state.success_message = (
                    f"✅ Successfully processed {num_chunks} chunks from "
                    f"{st.session_state.uploaded_file.name}"
                )
            else:
                st.error("Failed to process PDF file.")


def generate_answer(query, context):
    """Generate an answer using Cohere Chat API (Generate API was removed Sep 2025)."""
    try:
        max_context_length = 12000
        if len(context) > max_context_length:
            context = context[:max_context_length]

        response = co.chat(
            model="command-r-plus",
            message=query,
            preamble=(
                "You are a helpful assistant. Answer the user's question based only on the "
                "following context from the uploaded document. If the answer cannot be found "
                "in the context, say so clearly.\n\nContext:\n" + context
            ),
            temperature=0.7
        )
        return response.text.strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def ask_question():
    if st.session_state.question_text:
        st.session_state.current_question = st.session_state.question_text
        st.session_state.processing_question = True


def next_question():
    if st.session_state.current_question and st.session_state.current_answer:
        context = []
        if st.session_state.results and 'matches' in st.session_state.results:
            context = [match['metadata']['text'] for match in st.session_state.results['matches']]

        st.session_state.qa_history.append({
            "question": st.session_state.current_question,
            "answer": st.session_state.current_answer,
            "context": context
        })

        st.session_state.current_question = ""
        st.session_state.current_answer = ""
        st.session_state.results = None
        st.session_state.processing_question = False
        st.session_state.question_text = ""


def process_question(query):
    try:
        query_embedding = embed_text(query)
        if query_embedding is None:
            st.session_state.current_answer = "Error processing question. Please try again."
            st.session_state.processing_question = False
            return

        # Cap top_k at 10 — returning all chunks is wasteful; top 10 most relevant is plenty
        TOP_K = min(10, max(1, st.session_state.num_chunks))

        results = index.query(
            vector=query_embedding,
            top_k=TOP_K,
            include_metadata=True
        )

        if not results or 'matches' not in results or len(results['matches']) == 0:
            st.session_state.current_answer = (
                "Could not find anything relevant in the current PDF. "
                "Please try rephrasing your question."
            )
            st.session_state.results = {'matches': []}
            st.session_state.processing_question = False
            return

        safe_name = st.session_state.current_pdf_safe_name

        # PRIMARY filter: exact set membership (reliable, no substring bugs)
        filtered_matches = [
            match for match in results['matches']
            if match['id'] in st.session_state.current_pdf_vectors
        ]

        # FALLBACK filter: match by pdf_name stored in metadata
        # Handles edge cases where session state was partially lost
        if not filtered_matches:
            filtered_matches = [
                match for match in results['matches']
                if match.get('metadata', {}).get('pdf_name') == safe_name
            ]

        # LAST RESORT: if still empty, trust Pinecone results as-is
        # (only one PDF ever in the index at a time after clear_index())
        if not filtered_matches:
            filtered_matches = results['matches']

        if filtered_matches:
            context = " ".join([match['metadata']['text'] for match in filtered_matches])
            answer = generate_answer(query, context)
            st.session_state.current_answer = answer
            st.session_state.results = {'matches': filtered_matches}
        else:
            st.session_state.current_answer = (
                "Could not find anything relevant in the current PDF. "
                "Please try rephrasing your question."
            )
            st.session_state.results = {'matches': []}

        st.session_state.processing_question = False

    except Exception as e:
        st.session_state.current_answer = f"Error processing question: {str(e)}"
        st.session_state.processing_question = False


# ── Main app layout ───────────────────────────────────────────────────────────

st.title("📚 QueryBuddy")

# Sidebar
with st.sidebar:
    st.header("📊 Statistics")
    st.metric("Processed Chunks", st.session_state.num_chunks)
    if st.session_state.current_pdf_name:
        st.metric("Current Document", st.session_state.current_pdf_name)

    if st.button("🗑️ Clear All & Reset", use_container_width=False):
        reset_session_state()

    if st.session_state.qa_history:
        st.markdown("### 📝 Questions & Answers History")
        for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
            with st.expander(
                f"Q{len(st.session_state.qa_history) - i + 1}: {qa['question']}",
                expanded=False
            ):
                st.markdown(
                    f'<div class="answer">{qa["answer"]}</div>',
                    unsafe_allow_html=True
                )

# Main content
uploaded_file = st.file_uploader(
    "📄 Choose a PDF file",
    type="pdf",
    key=f"file_uploader_{st.session_state.file_uploader_key}"
)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

    if st.session_state.current_pdf_name != uploaded_file.name:
        st.session_state.current_pdf_name = uploaded_file.name
        # Generate sanitized safe name once when a new PDF is loaded
        st.session_state.current_pdf_safe_name = sanitize_filename(uploaded_file.name)
        st.session_state.pdf_processed = False
        st.session_state.qa_history = []
        st.session_state.num_chunks = 0
        st.session_state.show_success_message = False
        st.session_state.current_pdf_vectors = set()

    if not st.session_state.pdf_processed:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button("🚀 Process PDF", on_click=process_pdf_callback, use_container_width=False)

    if st.session_state.show_success_message:
        st.markdown(
            f'<div class="success-message-container">{st.session_state.success_message}</div>',
            unsafe_allow_html=True
        )

if st.session_state.pdf_processed:
    st.text_input(
        "🔍 Ask a question about the document:",
        key="question_input",
        value=st.session_state.question_text,
        on_change=lambda: setattr(st.session_state, 'question_text', st.session_state.question_input)
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        if st.button("🔎 Ask Question", type="primary", on_click=ask_question, use_container_width=False):
            pass
    with col3:
        if st.button("🌟 Next Question", on_click=next_question, use_container_width=False):
            pass

    if st.session_state.processing_question:
        with st.spinner("🤔 Analyzing..."):
            process_question(st.session_state.current_question)

    if st.session_state.current_answer:
        st.markdown("### 💡 Latest Answer:")
        st.markdown(
            f'<div class="answer">{st.session_state.current_answer}</div>',
            unsafe_allow_html=True
        )

        if st.session_state.current_answer not in (
            "Could not find anything relevant in the current PDF. Please try rephrasing your question.",
        ):
            with st.expander("📑 View Source Segments"):
                if st.session_state.results and 'matches' in st.session_state.results:
                    for i, match in enumerate(st.session_state.results['matches']):
                        st.markdown(f"**Segment {i + 1}**")
                        st.markdown(match['metadata']['text'])
                        st.markdown("---")
                else:
                    st.write("No source segments available.")
else:
    st.info("👆 Please upload and process a PDF to start asking questions.")
