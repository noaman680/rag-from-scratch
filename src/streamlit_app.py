"""
streamlit_app.py - Streamlit Web Interface for RAG System
Run with: streamlit run src/streamlit_app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from retrieval import RAGRetriever

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def init_rag_system():
    """Initialize RAG system (cached for performance)"""
    try:
        vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        top_k = int(os.getenv("TOP_K_RESULTS", "3"))
        
        retriever = RAGRetriever(
            vectorstore_path=vectorstore_path,
            embedding_model=embedding_model,
            llm_model=llm_model,
            temperature=temperature,
            top_k=top_k
        )
        
        retriever.initialize()
        return retriever
    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        st.info("Make sure you've run 'python src/indexing.py' first!")
        return None

# Main UI
st.markdown('<div class="main-header">🤖 RAG Q&A System</div>', unsafe_allow_html=True)
st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("🔄 Reload System"):
        st.cache_resource.clear()
        st.session_state.retriever = None
        st.rerun()
    
    st.divider()
    
    st.header("📊 Statistics")
    retriever = init_rag_system()
    if retriever:
        st.metric("Embedding Model", retriever.embedding_model)
        st.metric("LLM Model", retriever.llm_model)
        st.metric("Top K Results", retriever.top_k)
        st.metric("Temperature", retriever.temperature)
    
    st.divider()
    
    st.header("📚 Query History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"{i}. {item['question'][:50]}..."):
                st.write(item['answer'][:200] + "...")
    else:
        st.info("No queries yet")

# Main content
retriever = init_rag_system()

if retriever:
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        question = st.text_input(
            "Your Question:",
            placeholder="What would you like to know?",
            key="question_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_button = st.button("🔍 Ask", type="primary")
    
    # Example questions
    st.markdown("#### 💡 Example Questions:")
    example_cols = st.columns(3)
    
    examples = [
        "What is the main topic of the documents?",
        "Summarize the key points",
        "What are the important dates mentioned?"
    ]
    
    for col, example in zip(example_cols, examples):
        if col.button(example, key=f"example_{example}"):
            question = example
            ask_button = True
    
    # Process query
    if ask_button and question:
        with st.spinner("🔍 Searching for relevant information..."):
            try:
                answer, sources = retriever.query(question)
                
                # Store in history
                st.session_state.history.append({
                    'question': question,
                    'answer': answer
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(answer)
                
                # Display sources
                st.markdown("### 📚 Sources")
                
                for i, doc in enumerate(sources, 1):
                    with st.expander(f"📄 Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                        st.markdown(f"**Page:** {doc.metadata.get('page', 'N/A')}")
                        st.markdown("**Content:**")
                        st.text(doc.page_content)
                        
            except Exception as e:
                st.error(f"Error processing query: {e}")
    
    # Tips section
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - **Be specific**: Ask clear, focused questions
        - **Context matters**: Mention specific topics or documents if relevant
        - **Multiple questions**: Break complex questions into simpler parts
        - **Check sources**: Always verify the source documents
        """)

else:
    st.error("❌ RAG system not initialized")
    st.info("""
    Please ensure:
    1. You've created a `.env` file with your OPENAI_API_KEY
    2. You've run `python src/indexing.py` to create the vector store
    3. Your documents are in the `data/` directory
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with ❤️ using LangChain, OpenAI, and Streamlit
</div>
""", unsafe_allow_html=True)
