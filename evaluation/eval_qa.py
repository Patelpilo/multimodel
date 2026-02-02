"""
Evaluation module for the RAG system
"""
import os
import sys
import json
from typing import List, Dict, Any
import pandas as pd
from dataclasses import dataclass
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.generation import AnswerGenerator

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Evaluation result for a single question"""
    question: str
    question_type: str
    answer: str
    citations: List[int]
    confidence: float
    has_answer: bool
    sources_used: int
    retrieval_scores: List[float]
    modality_distribution: Dict[str, int]

class QAEvaluator:
    """Evaluate QA performance"""
    
    def __init__(self, answer_generator: AnswerGenerator):
        self.answer_generator = answer_generator
        
    def evaluate_question(self, question: str, question_type: str = "text") -> EvaluationResult:
        """
        Evaluate a single question
        
        Args:
            question: Question to evaluate
            question_type: Type of question (text/table/image/mixed)
            
        Returns:
            EvaluationResult object
        """
        # Get answer
        response = self.answer_generator.answer_question(question)
        
        # Analyze context chunks
        context_chunks = response.get('context_chunks', [])
        retrieval_scores = [chunk['score'] for chunk in context_chunks]
        
        # Count modalities
        modality_count = {'text': 0, 'table': 0, 'image': 0}
        for chunk in context_chunks:
            modality = chunk.get('modality', 'text')
            modality_count[modality] = modality_count.get(modality, 0) + 1
        
        # Check if answer was generated
        answer_text = response.get('answer', '')
        has_answer = bool(answer_text and 
                         "cannot answer" not in answer_text.lower() and
                         "i couldn't find" not in answer_text.lower())
        
        result = EvaluationResult(
            question=question,
            question_type=question_type,
            answer=answer_text,
            citations=response.get('citations', []),
            confidence=response.get('confidence', 0.0),
            has_answer=has_answer,
            sources_used=len(context_chunks),
            retrieval_scores=retrieval_scores,
            modality_distribution=modality_count
        )
        
        return result
    
    def evaluate_dataset(self, questions: List[Dict]) -> pd.DataFrame:
        """
        Evaluate a dataset of questions
        
        Args:
            questions: List of question dictionaries with 'question' and 'type' keys
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for q in questions:
            logger.info(f"Evaluating: {q['question'][:50]}...")
            
            result = self.evaluate_question(q['question'], q.get('type', 'text'))
            
            results.append({
                'question': result.question,
                'type': result.question_type,
                'answer_length': len(result.answer),
                'citations_count': len(result.citations),
                'confidence': result.confidence,
                'has_answer': result.has_answer,
                'sources_used': result.sources_used,
                'avg_retrieval_score': sum(result.retrieval_scores) / max(len(result.retrieval_scores), 1),
                'text_sources': result.modality_distribution.get('text', 0),
                'table_sources': result.modality_distribution.get('table', 0),
                'image_sources': result.modality_distribution.get('image', 0)
            })
        
        df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = {
            'total_questions': len(df),
            'questions_with_answers': df['has_answer'].sum(),
            'answer_rate': df['has_answer'].mean(),
            'avg_confidence': df['confidence'].mean(),
            'avg_citations': df['citations_count'].mean(),
            'avg_sources': df['sources_used'].mean(),
            'text_based_accuracy': df[df['type'] == 'text']['has_answer'].mean() if 'text' in df['type'].unique() else None,
            'table_based_accuracy': df[df['type'] == 'table']['has_answer'].mean() if 'table' in df['type'].unique() else None,
            'image_based_accuracy': df[df['type'] == 'image']['has_answer'].mean() if 'image' in df['type'].unique() else None
        }
        
        logger.info("Evaluation complete")
        logger.info(f"Answer rate: {metrics['answer_rate']:.2%}")
        logger.info(f"Average confidence: {metrics['avg_confidence']:.2%}")
        
        return df, metrics
    
    def generate_report(self, df: pd.DataFrame, metrics: Dict) -> str:
        """Generate evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary metrics
        report.append("SUMMARY METRICS")
        report.append("-" * 40)
        report.append(f"Total Questions: {metrics['total_questions']}")
        report.append(f"Questions with Answers: {metrics['questions_with_answers']}")
        report.append(f"Answer Rate: {metrics['answer_rate']:.2%}")
        report.append(f"Average Confidence: {metrics['avg_confidence']:.2%}")
        report.append(f"Average Citations per Answer: {metrics['avg_citations']:.2f}")
        report.append(f"Average Sources per Question: {metrics['avg_sources']:.2f}")
        report.append("")
        
        # Modality-specific metrics
        report.append("MODALITY PERFORMANCE")
        report.append("-" * 40)
        
        if metrics['text_based_accuracy'] is not None:
            report.append(f"Text-based QA Accuracy: {metrics['text_based_accuracy']:.2%}")
        
        if metrics['table_based_accuracy'] is not None:
            report.append(f"Table-based QA Accuracy: {metrics['table_based_accuracy']:.2%}")
        
        if metrics['image_based_accuracy'] is not None:
            report.append(f"Image-based QA Accuracy: {metrics['image_based_accuracy']:.2%}")
        
        report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS")
        report.append("-" * 40)
        
        for _, row in df.iterrows():
            report.append(f"Question: {row['question'][:100]}...")
            report.append(f"  Type: {row['type']}, Has Answer: {'Yes' if row['has_answer'] else 'No'}")
            report.append(f"  Confidence: {row['confidence']:.2%}, Citations: {row['citations_count']}")
            report.append("")
        
        return "\n".join(report)

# Sample evaluation questions
SAMPLE_QUESTIONS = [
    {
        "question": "What is the main topic or subject of this document?",
        "type": "text",
        "description": "General document understanding"
    },
    {
        "question": "Summarize the key findings or conclusions from the document.",
        "type": "text",
        "description": "Conclusion extraction"
    },
    {
        "question": "Extract and present data from any table in the document.",
        "type": "table",
        "description": "Table data extraction"
    },
    {
        "question": "What numerical values or statistics are mentioned in the document?",
        "type": "mixed",
        "description": "Numerical information retrieval"
    },
    {
        "question": "Describe any charts, graphs, or images present in the document.",
        "type": "image",
        "description": "Image content description"
    },
    {
        "question": "What are the section headers or major divisions in the document?",
        "type": "text",
        "description": "Document structure understanding"
    },
    {
        "question": "Are there any dates, timelines, or chronological information mentioned?",
        "type": "mixed",
        "description": "Temporal information extraction"
    },
    {
        "question": "What references or citations are included in the document?",
        "type": "text",
        "description": "Reference extraction"
    },
    {
        "question": "Compare data from different tables if multiple tables exist.",
        "type": "table",
        "description": "Cross-table analysis"
    },
    {
        "question": "What is the source or authorship information for this document?",
        "type": "text",
        "description": "Metadata extraction"
    }
]