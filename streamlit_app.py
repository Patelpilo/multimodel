"""
Streamlit UI for Multi-Modal RAG System
"""
import streamlit as st
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import MultiModalRAGSystem
from config.settings import config

# Page config
st.set_page_config(
    page_title="Multi-Modal RAG Document Intelligence",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_system():
    """Initialize the RAG system"""
    try:
        st.session_state.rag_system = MultiModalRAGSystem()
        st.session_state.rag_system.setup()
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return False

def ingest_document(uploaded_file):
    """Ingest uploaded document"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            success = st.session_state.rag_system.ingest_document(tmp_path)

        if success:
            st.session_state.documents_loaded = True
            st.success(f"Successfully ingested {uploaded_file.name}")
        else:
            st.error(f"Failed to ingest {uploaded_file.name}")

    finally:
        os.unlink(tmp_path)

    return success

def display_answer(response):
    """Display answer with formatting"""
    if 'error' in response:
        st.error(f"Error: {response['error']}")
        return

    # Display answer
    st.markdown("### Answer")
    st.markdown(response['answer'])

    # Display confidence
    if response.get('confidence'):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{response['confidence']:.2%}")

        with col2:
            if response.get('citations'):
                st.metric("Citations", len(response['citations']))

    # Display sources
    if response.get('context_chunks'):
        with st.expander("View Sources"):
            for i, chunk in enumerate(response['context_chunks']):
                # Handle both SearchResult objects and dict formats
                if hasattr(chunk, 'chunk'):
                    # SearchResult object
                    page = chunk.chunk.page
                    modality = chunk.chunk.modality
                    score = chunk.score
                    content = chunk.chunk.content
                else:
                    # Dict format
                    page = chunk.get('page', 'N/A')
                    modality = chunk.get('modality', 'text')
                    score = chunk.get('score', 0.0)
                    content = chunk.get('content', '')

                st.markdown(f"**Source {i+1}** (Page {page}, {modality}) - Score: {score:.3f}")
                st.text(content[:500] + "..." if len(content) > 500 else content)
                st.divider()

# Main app
st.title("📚 Multi-Modal RAG Document Intelligence")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("System Configuration")

    # Initialize system
    if st.button("Initialize System", type="primary"):
        if initialize_system():
            st.success("System initialized!")

    st.divider()

    # Document upload
    st.header("Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )

    if uploaded_files and st.session_state.rag_system:
        for uploaded_file in uploaded_files:
            if st.button(f"Ingest {uploaded_file.name}"):
                ingest_document(uploaded_file)

    st.divider()

    # System info
    st.header("System Status")
    if st.session_state.rag_system:
        st.success("✅ System ready")
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded")
        else:
            st.warning("⚠️ No documents loaded")
    else:
        st.error("❌ System not initialized")

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])

with tab1:
    # Chat interface
    if not st.session_state.rag_system:
        st.warning("Please initialize the system first from the sidebar.")
    elif not st.session_state.documents_loaded:
        st.warning("Please upload and ingest documents first.")
    else:
        # Chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("View Details"):
                        st.json(message["metadata"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": prompt
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_system.answer_question(prompt)

                    # Display answer
                    st.markdown(response.get('answer', 'No answer generated.'))

                    # Display metadata
                    with st.expander("Answer Details"):
                        if response.get('citations'):
                            st.markdown(f"**Citations:** {', '.join([f'Page {p}' for p in response['citations']])}")

                        if response.get('confidence'):
                            st.markdown(f"**Confidence:** {response.get('confidence'):.2%}")

                        if response.get('context_chunks'):
                            st.markdown(f"**Sources Used:** {len(response['context_chunks'])}")

                # Add to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.get('answer', ''),
                    "metadata": response
                })

with tab2:
    st.header("System Evaluation")

    if st.session_state.rag_system and st.session_state.documents_loaded:
        # Sample evaluation questions
        sample_questions = [
            {
                "question": "What is the main topic of the document?",
                "type": "text",
                "expected": "General document understanding"
            },
            {
                "question": "Extract data from any table in the document",
                "type": "table",
                "expected": "Table data extraction"
            },
            {
                "question": "Describe any charts or images in the document",
                "type": "image",
                "expected": "Image content description"
            },
            {
                "question": "What are the key findings or conclusions?",
                "type": "text",
                "expected": "Conclusion extraction"
            },
            {
                "question": "Provide numerical data from the document",
                "type": "mixed",
                "expected": "Numerical information retrieval"
            }
        ]

        # Run evaluation
        if st.button("Run Evaluation", type="primary"):
            results = []

            for q in sample_questions:
                with st.spinner(f"Evaluating: {q['question']}"):
                    response = st.session_state.rag_system.answer_question(q['question'])

                    result = {
                        "question": q['question'],
                        "type": q['type'],
                        "answer_length": len(response.get('answer', '')),
                        "citations_count": len(response.get('citations', [])),
                        "confidence": response.get('confidence', 0),
                        "has_answer": bool(response.get('answer') and
                                         "cannot answer" not in response.get('answer', '').lower()),
                        "sources_used": len(response.get('context_chunks', []))
                    }
                    results.append(result)

            # Display results
            st.subheader("Evaluation Results")

            for result in results:
                with st.expander(f"{result['question']} ({result['type']})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Confidence", f"{result['confidence']:.2%}")

                    with col2:
                        st.metric("Citations", result['citations_count'])

                    with col3:
                        st.metric("Sources", result['sources_used'])

                    st.metric("Has Answer", "✅" if result['has_answer'] else "❌")
    else:
        st.warning("Please initialize the system and load documents first.")

with tab3:
    st.header("About This System")

    st.markdown("""
    ### Multi-Modal RAG Document Intelligence

    This system enables intelligent question-answering from complex PDF documents containing:

    - **Text** (including scanned documents via OCR)
    - **Tables** (with structure preservation)
    - **Images** (with text extraction from charts, diagrams)
    - **Mixed content** (documents with all modalities)

    ### Key Features:

    1. **100% Offline** - No cloud API dependencies
    2. **Multi-Modal** - Handles text, tables, and images
    3. **Production-Ready** - Modular, scalable architecture
    4. **Faithful Answers** - With page-level citations
    5. **Local LLMs** - Using Ollama with models like Llama 3.1, Mistral, DeepSeek

    ### Technical Stack:

    - **Embeddings**: Sentence Transformers / BGE models
    - **Vector DB**: FAISS (Facebook AI Similarity Search)
    - **OCR**: PaddleOCR / Tesseract
    - **PDF Processing**: pdfplumber, PyMuPDF, camelot
    - **LLM**: Ollama with local models
    - **UI**: Streamlit / CLI interface

    ### Hardware Requirements:

    - **Minimum**: CPU with 8GB RAM
    - **Recommended**: GPU (RTX 3050 6GB or better)
    - **Storage**: 10GB free space for models
    """)

# Footer
st.markdown("---")
st.caption("Multi-Modal RAG System v1.0 | Built with ❤️ for Document Intelligence")
