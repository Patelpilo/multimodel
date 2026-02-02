"""
Answer generation pipeline
"""
import os
import sys
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval import MultiModalRetrievalEngine
from models.llm_model import LocalLLM
from database.vector_store import SearchResult
from config.settings import config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """Complete answer generation pipeline"""

    def __init__(self, retrieval_engine: MultiModalRetrievalEngine = None, vector_store=None):
        if retrieval_engine:
            self.retrieval_engine = retrieval_engine
        elif vector_store:
            self.retrieval_engine = MultiModalRetrievalEngine(vector_store)
        else:
            # Default initialization - will need vector_store to be set later
            self.retrieval_engine = None
        self.llm = LocalLLM()
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Complete pipeline: retrieve context and generate answer
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: '{question}'")
        
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieval_engine.retrieve(question)
        
        if not context_chunks:
            return {
                "answer": "No documents have been ingested into the system yet. Please ingest some PDF documents first using the 'ingest' command.",
                "citations": [],
                "confidence": 0.0,
                "context_chunks": [],
                "error": "No documents ingested"
            }
        
        # Step 2: Generate answer using LLM
        # Extract context texts from chunks
        context_texts = [chunk.chunk.content for chunk in context_chunks]
        generated_answer = self.llm.generate(
            prompt=question,
            context=context_texts
        )
        
        # Step 3: Calculate confidence and prepare response
        confidence = self._calculate_confidence(context_chunks)
        citations = sorted(list(set(chunk.chunk.page for chunk in context_chunks)))

        response = {
            "answer": generated_answer,
            "citations": citations,
            "confidence": confidence,
            "context_chunks": [
                {
                    "content": chunk.chunk.content[:500] + "..." if len(chunk.chunk.content) > 500 else chunk.chunk.content,
                    "page": chunk.chunk.page,
                    "modality": chunk.chunk.modality,
                    "score": float(chunk.score),
                    "source_id": chunk.chunk.metadata.get("source_id", "")
                }
                for chunk in context_chunks
            ],
            "total_chunks_retrieved": len(context_chunks),
            "question": question
        }

        logger.info(f"Generated answer with confidence: {confidence:.2f}")
        
        return response
    
    def answer_with_followup(self, 
                            question: str, 
                            conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Generate answer with conversation context
        
        Args:
            question: Current question
            conversation_history: Previous Q&A pairs
            
        Returns:
            Dictionary containing answer and metadata
        """
        # For now, simple implementation without complex conversation
        # In production, you might want to include history in context

        return self.answer_question(question)

    def _calculate_confidence(self, context_chunks: List) -> float:
        """
        Calculate confidence score based on retrieval results

        Args:
            context_chunks: Retrieved context chunks

        Returns:
            Confidence score between 0 and 1
        """
        if not context_chunks:
            return 0.0

        # Simple confidence calculation based on average similarity score
        scores = [chunk.score for chunk in context_chunks]
        avg_score = sum(scores) / len(scores)

        # Normalize to 0-1 range (assuming scores are between 0 and 2)
        confidence = min(avg_score / 2.0, 1.0)

        return confidence
