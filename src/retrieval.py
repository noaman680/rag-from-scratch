"""
retrieval.py - Query and Retrieval Module
This module handles loading the vector store and querying the RAG system.
Run this EVERY TIME you want to ask questions.
"""

import os
from typing import List, Tuple
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()


class RAGRetriever:
    """Handles retrieval and generation for RAG system"""
    
    def __init__(
        self,
        vectorstore_path: str = "./vectorstore",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        top_k: int = 3
    ):
        self.vectorstore_path = vectorstore_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
    
    def load_vectorstore(self):
        """Load the existing vector store"""
        print(f"📂 Loading vector store from {self.vectorstore_path}...")
        
        try:
            self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
            
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            print("✅ Vector store loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading vector store: {e}")
            print(f"   Make sure you've run indexing.py first!")
            raise
    
    def create_rag_chain(self):
        """Create the RAG chain combining retrieval + generation"""
        print(f"🔗 Creating RAG chain...")
        print(f"   LLM: {self.llm_model}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top K results: {self.top_k}")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature
        )
        
        # Create custom prompt template
        prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.

If you don't know the answer based on the context provided, just say "I don't have enough information in the provided documents to answer that question." Don't try to make up an answer.

When providing an answer, always mention which source document(s) you're using.

Context:
{context}

Question: {question}

Helpful Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever from vectorstore
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        # Create RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("✅ RAG chain created successfully")
    
    def query(self, question: str) -> Tuple[str, List[Document]]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (answer, source_documents)
        """
        if not self.qa_chain:
            raise ValueError("RAG chain not initialized. Call create_rag_chain() first.")
        
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        print("🔍 Searching for relevant information...\n")
        
        # Get answer
        result = self.qa_chain({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]
        
        print(f"💡 Answer:\n{answer}\n")
        
        # Print sources
        print(f"📚 Sources ({len(sources)} documents):")
        for i, doc in enumerate(sources, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            print(f"\n{i}. Source: {source} | Page: {page}")
            print(f"   Content preview: {doc.page_content[:200]}...")
        
        print(f"\n{'='*60}\n")
        
        return answer, sources
    
    def initialize(self):
        """Initialize the RAG system (load vectorstore and create chain)"""
        self.load_vectorstore()
        self.create_rag_chain()
    
    def interactive_query_loop(self):
        """Run interactive query loop"""
        print("\n" + "="*60)
        print("🤖 RAG System Ready!")
        print("="*60)
        print("\nCommands:")
        print("  - Type your question to get an answer")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'help' for more commands\n")
        
        while True:
            try:
                question = input("Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit/exit - Exit the program")
                    print("  help - Show this help message")
                    print("  stats - Show system statistics")
                    continue
                
                if question.lower() == 'stats':
                    self.show_stats()
                    continue
                
                # Process the query
                self.query(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
    
    def show_stats(self):
        """Show system statistics"""
        print("\n📊 System Statistics:")
        print(f"   Vector Store: {self.vectorstore_path}")
        print(f"   Embedding Model: {self.embedding_model}")
        print(f"   LLM Model: {self.llm_model}")
        print(f"   Top K Results: {self.top_k}")
        print(f"   Temperature: {self.temperature}\n")


def main():
    """Main function to run retrieval"""
    
    # Get configuration from environment
    vectorstore_path = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    top_k = int(os.getenv("TOP_K_RESULTS", "3"))
    
    # Create retriever
    retriever = RAGRetriever(
        vectorstore_path=vectorstore_path,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        top_k=top_k
    )
    
    # Initialize and run interactive loop
    retriever.initialize()
    retriever.interactive_query_loop()


if __name__ == "__main__":
    main()
