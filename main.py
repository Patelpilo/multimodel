"""
Main entry point for the Multi-Modal RAG system
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.ingestion import DocumentIngestionPipeline
from models.embedding_model import MultiModalEmbedder
from database.vector_store import FAISSVectorStore
from core.generation import AnswerGenerator
from core.retrieval import MultiModalRetrievalEngine
from config.settings import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModalRAGSystem:
    """Complete Multi-Modal RAG System"""
    
    def __init__(self, vector_store_path: str = None):
        self.vector_store_path = vector_store_path or config.VECTOR_STORE_PATH
        self.vector_store = None
        self.ingestion_pipeline = None
        self.answer_generator = None
        
    def setup(self):
        """Setup the system components"""
        logger.info("Setting up Multi-Modal RAG System")
        
        # Initialize components
        self.ingestion_pipeline = DocumentIngestionPipeline()
        self.vector_store = FAISSVectorStore()
        
        # Try to load existing vector store
        if os.path.exists(f"{self.vector_store_path}.index"):
            logger.info("Loading existing vector store...")
            if self.vector_store.load(self.vector_store_path):
                logger.info("Vector store loaded successfully")
            else:
                logger.warning("Failed to load vector store, will create new one")
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator(vector_store=self.vector_store)
        
        logger.info("System setup complete")
    
    def ingest_document(self, pdf_path: str, save_to_store: bool = True):
        """
        Ingest a PDF document
        
        Args:
            pdf_path: Path to PDF file
            save_to_store: Whether to save to vector store
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        logger.info(f"Ingesting document: {pdf_path}")
        
        try:
            # Process document
            chunks = self.ingestion_pipeline.process_document(pdf_path)
            
            # Generate embeddings
            embedder = MultiModalEmbedder()
            embeddings = embedder.embed_batch(chunks)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Add to vector store
            self.vector_store.add_chunks(chunks, embeddings)
            
            if save_to_store:
                # Save vector store
                self.vector_store.save(self.vector_store_path)
                logger.info(f"Document ingested and saved to {self.vector_store_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            return False
    
    def answer_question(self, question: str) -> dict:
        """
        Answer a question based on ingested documents
        
        Args:
            question: User question
            
        Returns:
            Answer with metadata
        """
        if self.vector_store is None or len(self.vector_store.chunks) == 0:
            return {
                "error": "No documents ingested. Please ingest documents first.",
                "answer": "",
                "citations": []
            }
        
        logger.info(f"Answering question: {question}")
        
        try:
            response = self.answer_generator.answer_question(question)
            return response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "error": str(e),
                "answer": "An error occurred while processing your question.",
                "citations": []
            }
    
    def interactive_mode(self):
        """Run interactive Q&A mode"""
        print("\n" + "="*60)
        print("Multi-Modal RAG System - Interactive Mode")
        print("="*60)
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'sources' to see recent sources")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if question.lower() == 'sources':
                    if hasattr(self, 'last_response'):
                        print("\nRecent Sources:")
                        for i, chunk in enumerate(self.last_response.get('context_chunks', [])):
                            print(f"{i+1}. Page {chunk.chunk.page} ({chunk.chunk.modality}) - Score: {chunk.score:.3f}")
                            print(f"   {chunk.chunk.content[:200]}...")
                            print()
                    continue
                
                if not question:
                    continue
                
                # Get answer
                response = self.answer_question(question)
                self.last_response = response
                
                # Display answer
                print("\n" + "-"*60)
                print("ANSWER:")
                print("-"*60)
                print(response.get('answer', 'No answer generated.'))
                print()
                
                # Display citations
                if response.get('citations'):
                    print(f"Citations: {', '.join([f'Page {p}' for p in response['citations']])}")
                
                # Display confidence
                print(f"Confidence: {response.get('confidence', 0):.2f}")
                print("-"*60)
                
            except KeyboardInterrupt:
                print("\n\nSession ended.")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-Modal RAG Document Intelligence System")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest a PDF document')
    ingest_parser.add_argument('pdf_path', help='Path to PDF file')
    ingest_parser.add_argument('--store-path', help='Path to vector store', default=config.VECTOR_STORE_PATH)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('question', help='Question to ask')
    query_parser.add_argument('--store-path', help='Path to vector store', default=config.VECTOR_STORE_PATH)
    
    # Interactive command
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive mode')
    interactive_parser.add_argument('--store-path', help='Path to vector store', default=config.VECTOR_STORE_PATH)
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup the system')
    setup_parser.add_argument('--store-path', help='Path to vector store', default=config.VECTOR_STORE_PATH)
    
    args = parser.parse_args()
    
    # Create system instance
    system = MultiModalRAGSystem(vector_store_path=args.store_path if hasattr(args, 'store_path') else config.VECTOR_STORE_PATH)
    
    if args.command == 'ingest':
        system.setup()
        success = system.ingest_document(args.pdf_path)
        sys.exit(0 if success else 1)
    
    elif args.command == 'query':
        system.setup()
        response = system.answer_question(args.question)
        
        print("\n" + "="*60)
        print("QUESTION:", args.question)
        print("="*60)
        print("\nANSWER:")
        print("-"*60)
        print(response.get('answer', 'No answer generated.'))
        
        if response.get('citations'):
            print(f"\nCitations: {', '.join([f'Page {p}' for p in response['citations']])}")
        
        if response.get('confidence'):
            print(f"Confidence: {response.get('confidence'):.2f}")
        
        print("="*60)
    
    elif args.command == 'interactive':
        system.setup()
        system.interactive_mode()
    
    elif args.command == 'setup':
        system.setup()
        print("System setup complete.")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()