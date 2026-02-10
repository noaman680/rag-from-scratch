"""
indexing.py - Document Indexing Module
This module handles loading, chunking, embedding, and storing documents.
Run this ONCE or whenever you update your documents.
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()


class DocumentIndexer:
    """Handles document indexing for RAG system"""
    
    def __init__(
        self,
        data_path: str = "./data",
        vectorstore_path: str = "./vectorstore",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
    
    def load_documents(self) -> List[Document]:
        """
        Load all documents from the data directory
        
        Returns:
            List of Document objects
        """
        print(f"📚 Loading documents from {self.data_path}...")
        
        all_docs = []
        
        # Load PDF files
        try:
            pdf_loader = DirectoryLoader(
                self.data_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            pdf_docs = pdf_loader.load()
            all_docs.extend(pdf_docs)
            print(f"  ✓ Loaded {len(pdf_docs)} PDF pages")
        except Exception as e:
            print(f"  ⚠ PDF loading error: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                self.data_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            txt_docs = txt_loader.load()
            all_docs.extend(txt_docs)
            print(f"  ✓ Loaded {len(txt_docs)} text files")
        except Exception as e:
            print(f"  ⚠ Text loading error: {e}")
        
        print(f"✅ Total documents loaded: {len(all_docs)}")
        return all_docs
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document chunks
        """
        print(f"✂️  Splitting documents into chunks...")
        print(f"   Chunk size: {self.chunk_size} characters")
        print(f"   Chunk overlap: {self.chunk_overlap} characters")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} characters")
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Document]) -> FAISS:
        """
        Create embeddings and store in vector database
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            FAISS vectorstore object
        """
        print(f"🔢 Creating embeddings using {self.embedding_model}...")
        print(f"   This may take a while for large document sets...")
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Create directory if it doesn't exist
        os.makedirs(self.vectorstore_path, exist_ok=True)
        
        # Save to disk
        vectorstore.save_local(self.vectorstore_path)
        print(f"✅ Vector store saved to {self.vectorstore_path}")
        print(f"   Index contains {len(chunks)} vectors")
        
        return vectorstore
    
    def run_indexing_pipeline(self):
        """Execute the complete indexing pipeline"""
        print("="*60)
        print("🚀 Starting RAG Indexing Pipeline")
        print("="*60 + "\n")
        
        # Step 1: Load documents
        documents = self.load_documents()
        
        if not documents:
            print("❌ No documents found! Please add documents to the data directory.")
            return
        
        print()
        
        # Step 2: Split into chunks
        chunks = self.split_documents(documents)
        print()
        
        # Step 3: Create and save vector store
        vectorstore = self.create_vectorstore(chunks)
        
        print("\n" + "="*60)
        print("✨ Indexing Complete!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Documents processed: {len(documents)}")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Vector store location: {self.vectorstore_path}")
        print(f"\n💡 You can now run queries using retrieval.py")
        

def main():
    """Main function to run indexing"""
    
    # Get configuration from environment
    data_path = os.getenv("DATA_PATH", "./data")
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Create indexer and run pipeline
    indexer = DocumentIndexer(
        data_path=data_path,
        vectorstore_path=vectorstore_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model
    )
    
    indexer.run_indexing_pipeline()


if __name__ == "__main__":
    main()
