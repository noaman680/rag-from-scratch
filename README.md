# 🤖 RAG from Scratch

<<<<<<< HEAD
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-blue)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-purple)
![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
=======
A production-ready **Retrieval Augmented Generation (RAG)** system built from scratch using LangChain, OpenAI, and FAISS. This project demonstrates how to build an AI-powered Q&A system that can answer questions based on your own documents.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)

## 🌟 Features

- 📚 **Multi-format Support**: Load PDF, TXT, and Markdown documents
- 🔍 **Semantic Search**: FAISS vector store for fast similarity search
- 🤖 **GPT-Powered**: Uses OpenAI's GPT models for intelligent answers
- 📝 **Source Attribution**: Always cites source documents
- 🎯 **Multiple Interfaces**: CLI, Web UI (Streamlit), and REST API
- ⚙️ **Configurable**: Environment-based configuration
- 🚀 **Production-Ready**: Error handling, logging, and optimization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE INDEXING                         │
│  (Run once or when documents change)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  1. Load Documents (PDF, TXT)        │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  2. Split into Chunks (1000 chars)   │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  3. Generate Embeddings (OpenAI)     │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  4. Store in FAISS Vector DB         │
        └──────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   ONLINE RETRIEVAL                          │
│  (Happens for every query)                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  5. User Query                       │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  6. Embed Query → Search Vector DB   │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  7. Retrieve Top K Documents         │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  8. Augment Prompt with Context      │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  9. Generate Answer (GPT)            │
        └──────────────────────────────────────┘
```

## 📁 Project Structure

```
rag-project/
├── README.md                  # This file
├── .env.example              # Example environment variables
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
│
├── data/                   # Your documents (add PDFs, TXTs here)
│   └── sample.txt         # Example document
│
├── vectorstore/           # Vector store (generated)
│   └── index.faiss       # FAISS index
│
├── src/
│   ├── indexing.py       # Document indexing pipeline
│   ├── retrieval.py      # Query & retrieval logic
│   ├── streamlit_app.py  # Streamlit web interface
│   └── api.py            # FastAPI REST API
│
├── tests/                # Unit tests
│   └── test_rag.py      # Test cases
│
├── notebooks/            # Jupyter notebooks
│   └── experiments.ipynb # Experimentation
│
└── docs/                 # Documentation
    └── architecture.md  # Architecture details
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/rag-from-scratch.git
cd rag-from-scratch
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your OpenAI API key
# On Windows: notepad .env
# On macOS: open .env
# On Linux: nano .env
```

Your `.env` should look like:
```env
OPENAI_API_KEY=sk-your-actual-api-key-here
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=3
```

5. **Add your documents**
```bash
# Place your documents in the data/ folder
# Supported formats: PDF, TXT
cp your-documents.pdf data/
```

### Usage

#### Step 1: Index Your Documents

```bash
python src/indexing.py
```

This will:
- Load all documents from `data/` folder
- Split them into chunks
- Generate embeddings
- Create a FAISS vector store

**Expected Output:**
```
📚 Loading documents from ./data...
  ✓ Loaded 5 PDF pages
  ✓ Loaded 2 text files
✅ Total documents loaded: 7

✂️  Splitting documents into chunks...
✅ Created 45 chunks

🔢 Creating embeddings...
✅ Vector store saved to ./vectorstore
```

#### Step 2: Query the System

**Option A: Command Line Interface**
```bash
python src/retrieval.py
```

**Option B: Web Interface (Recommended)**
```bash
streamlit run src/streamlit_app.py
```
Then open http://localhost:8501 in your browser

**Option C: REST API**
```bash
uvicorn src.api:app --reload
```
API will be available at http://localhost:8000

Test with curl:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic?"}'
```

## 💡 Example Questions

Try asking:
- "What is the main topic of these documents?"
- "Summarize the key points"
- "What are the important dates mentioned?"
- "Who are the main people discussed?"
- "What recommendations are provided?"

## 🔧 Configuration

Edit `.env` file to customize:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-your-key-here

# Chunking Configuration
CHUNK_SIZE=1000          # Characters per chunk
CHUNK_OVERLAP=200        # Overlap between chunks

# Retrieval Configuration
TOP_K_RESULTS=3          # Number of chunks to retrieve
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0        # 0 = deterministic, 1 = creative

# Paths
DATA_PATH=./data
VECTORSTORE_PATH=./vectorstore
```

## 📊 How It Works

### 1. Document Indexing

```python
# Load documents
documents = load_documents("./data")

# Split into chunks
chunks = split_documents(documents, 
                        chunk_size=1000,
                        overlap=200)

# Generate embeddings
embeddings = OpenAIEmbeddings()
vectors = embeddings.embed_documents(chunks)

# Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### 2. Query Processing

```python
# 1. Embed the query
query_vector = embeddings.embed_query("What is AI?")

# 2. Search for similar chunks
similar_chunks = vectorstore.similarity_search(
    query_vector, 
    k=3
)

# 3. Create prompt with context
prompt = f"""
Context: {similar_chunks}
Question: {query}
Answer:
"""

# 4. Generate answer
answer = llm(prompt)
```

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src
```

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t rag-system .

# Run container
docker run -p 8000:8000 --env-file .env rag-system
```

Or use Docker Compose:
```bash
docker-compose up
```

## 📈 Performance Optimization

### Tips for Better Results

1. **Chunk Size**: Experiment with different sizes
   - Smaller (500): More precise but may lose context
   - Larger (1500): More context but may be less precise

2. **Overlap**: Use 10-20% of chunk size
   - Prevents cutting sentences in half

3. **Top K**: Retrieve 3-5 chunks
   - More chunks = more context but more noise

4. **Embedding Model**: 
   - `text-embedding-3-small`: Faster, good quality
   - `text-embedding-3-large`: Slower, better quality

### Scaling Considerations

- **Caching**: Cache frequent queries
- **Async**: Use async operations for parallel processing
- **Vector DB**: For >100k documents, use Pinecone/Weaviate
- **Batching**: Batch embedding generation

## 🎯 Common Issues & Solutions

### Issue: "Vector store not found"
**Solution**: Run `python src/indexing.py` first

### Issue: "OpenAI API error"
**Solution**: Check your API key in `.env`

### Issue: "No documents loaded"
**Solution**: Add documents to `data/` folder

### Issue: "Poor quality answers"
**Solution**: 
- Adjust chunk size
- Increase top_k
- Try different prompt

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [OpenAI](https://openai.com/) - GPT models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search

## 📚 Learn More

- [RAG Concepts](docs/rag-concepts.md)
- [Architecture Details](docs/architecture.md)
- [API Documentation](http://localhost:8000/docs) (when running API)

## 📧 Contact

For questions or feedback:
- Create an issue
- Email: noaman.sae.comp@gmail.com
- LinkedIn: [My Profile]([https://linkedin.com/in/noaman680](https://www.linkedin.com/in/noaman680/))

---

**Built with ❤️ using LangChain, OpenAI, and FAISS**

⭐ Star this repo if you find it helpful!
>>>>>>> bf0805d (Add complete RAG system with indexing, retrieval, Streamlit UI, and FastAPI)
