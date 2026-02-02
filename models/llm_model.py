"""
LLM model interface for local models using Ollama
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import List, Dict, Any, Optional
import requests
from config.settings import config

logger = logging.getLogger(__name__)

class LocalLLM:
    """Interface for local LLM models via Ollama"""

    def __init__(self, model_name: str = None, base_url: str = "http://localhost:11434"):
        self.model_name = model_name or config.LLM_MODEL
        self.base_url = base_url
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE

    def generate(self, prompt: str, context: List[str] = None,
                max_tokens: int = None, temperature: float = None) -> str:
        """
        Generate text response from the LLM

        Args:
            prompt: The main prompt/question
            context: List of context strings to include
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text response
        """
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature

        # Build the full prompt with context
        full_prompt = self._build_prompt(prompt, context)

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return "I apologize, but I encountered an error while generating a response."

        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to LLM: {e}")
            return "I apologize, but I'm unable to connect to the language model service."

    def _build_prompt(self, question: str, context: List[str] = None) -> str:
        """
        Build a complete prompt with context and question

        Args:
            question: User question
            context: List of context strings

        Returns:
            Complete prompt string
        """
        if not context:
            return question

        # Combine context
        context_text = "\n\n".join(context)

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context_text}

Question: {question}

Please provide a comprehensive and accurate answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""

        return prompt

    def is_available(self) -> bool:
        """
        Check if the LLM service is available

        Returns:
            True if service is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model["name"] for model in models]
                return self.model_name in model_names
            return False
        except:
            return False

class AnswerGenerator:
    """Generate answers using retrieval-augmented generation"""

    def __init__(self, retrieval_engine=None):
        self.retrieval_engine = retrieval_engine
        self.llm = LocalLLM()

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using retrieved context

        Args:
            question: User question

        Returns:
            Answer with metadata
        """
        try:
            # Retrieve relevant context
            if self.retrieval_engine:
                context_chunks = self.retrieval_engine.retrieve(question, top_k=config.TOP_K_RETRIEVAL)
                context_texts = [chunk.chunk.content for chunk in context_chunks]
                citations = list(set(chunk.chunk.page for chunk in context_chunks))
            else:
                context_texts = []
                citations = []

            # Generate answer
            if context_texts:
                answer = self.llm.generate(question, context_texts)
                confidence = self._calculate_confidence(context_chunks)
            else:
                answer = "I don't have enough context to answer this question. Please provide more documents or try rephrasing your question."
                confidence = 0.0

            return {
                "answer": answer,
                "citations": sorted(citations),
                "confidence": confidence,
                "context_chunks": context_chunks if context_chunks else []
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "An error occurred while processing your question.",
                "citations": [],
                "confidence": 0.0,
                "context_chunks": []
            }

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
