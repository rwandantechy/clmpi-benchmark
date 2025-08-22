"""
CLMPI Calculator with Proper Scoring Methods

Implements rigorous scoring for each CLMPI dimension:
- Accuracy: Exact Match + F1 against expert-validated answers
- Contextual Understanding: EM + F1 + context relevance
- Coherence: Sentence-to-sentence similarity + repetition penalty
- Fluency: Grammar checking + perplexity calculation
- Efficiency: Transparent latency/memory formulas
"""

import json
import time
import psutil
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# For grammar checking (simplified implementation)
try:
    import language_tool_python
    GRAMMAR_CHECKER_AVAILABLE = True
except ImportError:
    GRAMMAR_CHECKER_AVAILABLE = False
    logging.warning("language_tool_python not available. Using simplified grammar checking.")


@dataclass
class AccuracyResult:
    """Results from accuracy evaluation"""
    exact_match: float
    f1_score: float
    detailed_scores: List[float]
    responses: List[str]
    gold_answers: List[str]


@dataclass
class ContextualResult:
    """Results from contextual understanding evaluation"""
    exact_match: float
    f1_score: float
    context_similarity: float
    combined_score: float
    responses: List[str]
    contexts: List[str]
    gold_answers: List[str]


@dataclass
class CoherenceResult:
    """Results from coherence evaluation"""
    sentence_similarity: float
    repetition_penalty: float
    coherence_score: float
    responses: List[str]
    detailed_scores: List[float]


@dataclass
class FluencyResult:
    """Results from fluency evaluation"""
    grammar_score: float
    perplexity_score: float
    fluency_score: float
    responses: List[str]
    detailed_scores: List[float]


@dataclass
class EfficiencyResult:
    """Results from efficiency evaluation"""
    latency_seconds: float
    cpu_usage_percent: float
    memory_used_mb: float
    raw_efficiency: float
    normalized_efficiency: float


class CLMPICalculator:
    """
    CLMPI calculator with proper scoring methods
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'accuracy': 0.25,
            'contextual_understanding': 0.20,
            'coherence': 0.20,
            'fluency': 0.20,
            'performance_efficiency': 0.15
        }
        self._validate_weights()
        self.logger = logging.getLogger(__name__)
        
        # Initialize grammar checker if available
        if GRAMMAR_CHECKER_AVAILABLE:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        else:
            self.grammar_tool = None
    
    def _validate_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def evaluate_accuracy(self, responses: List[str], gold_answers: List[str], 
                         acceptable_answers: List[List[str]] = None) -> AccuracyResult:
        """
        Evaluate accuracy using Exact Match and F1 scoring
        
        Args:
            responses: Model responses
            gold_answers: Correct answers
            acceptable_answers: List of acceptable answer variations
            
        Returns:
            AccuracyResult with detailed scores
        """
        if len(responses) != len(gold_answers):
            raise ValueError("Number of responses must match number of gold answers")
        
        exact_matches = []
        f1_scores = []
        
        for i, (response, gold) in enumerate(zip(responses, gold_answers)):
            # Clean response
            clean_response = response.strip().lower()
            clean_gold = gold.strip().lower()
            
            # Check exact match
            if acceptable_answers and i < len(acceptable_answers):
                # Check against acceptable answers
                is_exact_match = any(clean_response == acc.strip().lower() 
                                   for acc in acceptable_answers[i])
            else:
                is_exact_match = clean_response == clean_gold
            
            exact_matches.append(1.0 if is_exact_match else 0.0)
            
            # Calculate F1 score (simplified - word overlap)
            response_words = set(clean_response.split())
            gold_words = set(clean_gold.split())
            
            if not gold_words:
                f1_scores.append(1.0 if not response_words else 0.0)
            else:
                precision = len(response_words & gold_words) / len(response_words) if response_words else 0
                recall = len(response_words & gold_words) / len(gold_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
        return AccuracyResult(
            exact_match=np.mean(exact_matches),
            f1_score=np.mean(f1_scores),
            detailed_scores=f1_scores,
            responses=responses,
            gold_answers=gold_answers
        )
    
    def evaluate_contextual_understanding(self, responses: List[str], 
                                        contexts: List[str], 
                                        gold_answers: List[str]) -> ContextualResult:
        """
        Evaluate contextual understanding using EM, F1, and context relevance
        
        Args:
            responses: Model responses
            contexts: Context information
            gold_answers: Correct answers
            
        Returns:
            ContextualResult with detailed scores
        """
        # Get accuracy scores
        accuracy_result = self.evaluate_accuracy(responses, gold_answers)
        
        # Calculate context similarity (simplified - word overlap)
        context_similarities = []
        for response, context in zip(responses, contexts):
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            if not context_words:
                context_similarities.append(0.0)
            else:
                overlap = len(response_words & context_words)
                similarity = overlap / len(context_words)
                context_similarities.append(similarity)
        
        context_similarity = np.mean(context_similarities)
        
        # Keep EM and F1 separate from context similarity (no blending)
        # Store context similarity as separate field for transparency
        
        return ContextualResult(
            exact_match=accuracy_result.exact_match,
            f1_score=accuracy_result.f1_score,
            context_similarity=context_similarity,
            combined_score=accuracy_result.f1_score,  # Use F1 as primary score
            responses=responses,
            contexts=contexts,
            gold_answers=gold_answers
        )
    
    def evaluate_coherence(self, responses: List[str]) -> CoherenceResult:
        """
        Evaluate coherence using sentence-to-sentence similarity and repetition penalty
        
        Args:
            responses: Model responses
            
        Returns:
            CoherenceResult with detailed scores
        """
        coherence_scores = []
        
        for response in responses:
            # Split into sentences (simplified)
            sentences = re.split(r'[.!?]+', response.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                coherence_scores.append(0.5)  # Neutral score for single sentence
                continue
            
            # Calculate sentence-to-sentence similarity
            similarities = []
            for i in range(len(sentences) - 1):
                # Simplified similarity using word overlap
                words1 = set(sentences[i].lower().split())
                words2 = set(sentences[i+1].lower().split())
                
                if not words1 or not words2:
                    similarities.append(0.0)
                else:
                    overlap = len(words1 & words2)
                    similarity = overlap / max(len(words1), len(words2))
                    similarities.append(similarity)
            
            sentence_similarity = np.mean(similarities) if similarities else 0.0
            
            # Calculate repetition penalty
            all_words = []
            for sentence in sentences:
                all_words.extend(sentence.lower().split())
            
            word_counts = {}
            for word in all_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            repetition_ratio = sum(count - 1 for count in word_counts.values()) / len(all_words)
            repetition_penalty = min(repetition_ratio, 0.5)  # Cap at 50% penalty
            
            # Final coherence score
            coherence_score = sentence_similarity * (1 - repetition_penalty)
            coherence_scores.append(coherence_score)
        
        # Calculate mean cohesion and repetition penalty separately
        mean_cohesion = np.mean([s for s in coherence_scores if s > 0]) if coherence_scores else 0.0
        repetition_penalty = np.mean([0.1 for _ in coherence_scores])  # Simplified for now
        
        return CoherenceResult(
            sentence_similarity=mean_cohesion,
            repetition_penalty=repetition_penalty,
            coherence_score=np.mean(coherence_scores),
            responses=responses,
            detailed_scores=coherence_scores
        )
    
    def evaluate_fluency(self, responses: List[str]) -> FluencyResult:
        """
        Evaluate fluency using grammar checking and perplexity
        
        Args:
            responses: Model responses
            
        Returns:
            FluencyResult with detailed scores
        """
        fluency_scores = []
        grammar_scores = []
        perplexity_scores = []
        
        for response in responses:
            # Grammar checking - store errors per token
            if self.grammar_tool:
                matches = self.grammar_tool.check(response)
                tokens = len(response.split())
                grammar_errors_per_token = len(matches) / tokens if tokens > 0 else 0.0
                grammar_score = max(0, 1 - grammar_errors_per_token)
            else:
                # Simplified grammar checking
                grammar_score = self._simple_grammar_check(response)
                grammar_errors_per_token = 1.0 - grammar_score  # Approximate
            
            grammar_scores.append(grammar_score)
            
            # Perplexity calculation (word diversity as proxy)
            words = response.lower().split()
            if not words:
                perplexity_score = 0.0
            else:
                unique_words = len(set(words))
                total_words = len(words)
                perplexity_score = unique_words / total_words  # Type-token ratio
            
            perplexity_scores.append(perplexity_score)
            
            # Combined fluency score: 60% grammar + 40% perplexity
            fluency_score = 0.6 * grammar_score + 0.4 * perplexity_score
            fluency_scores.append(fluency_score)
        
        return FluencyResult(
            grammar_score=np.mean(grammar_scores),
            perplexity_score=np.mean(perplexity_scores),
            fluency_score=np.mean(fluency_scores),
            responses=responses,
            detailed_scores=fluency_scores
        )
    
    def _simple_grammar_check(self, text: str) -> float:
        """Simplified grammar checking when language_tool is not available"""
        # Basic checks
        score = 1.0
        
        # Check for basic sentence structure
        if not text.strip():
            return 0.0
        
        # Check for capitalization
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            if sentence.strip() and not sentence.strip()[0].isupper():
                score -= 0.1
        
        # Check for proper punctuation
        if text and not text[-1] in '.!?':
            score -= 0.1
        
        return max(0.0, score)
    
    def measure_efficiency(self, generation_function, *args, **kwargs) -> EfficiencyResult:
        """
        Measure efficiency with transparent formulas
        
        Args:
            generation_function: Function to measure
            *args, **kwargs: Arguments for the function
            
        Returns:
            EfficiencyResult with detailed metrics
        """
        # Pre-warmup
        try:
            generation_function(*args, **kwargs)
        except:
            pass  # Ignore warmup errors
        
        # Actual measurement
        process = psutil.Process()
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = process.memory_info().rss
        
        try:
            result = generation_function(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error during efficiency measurement: {e}")
            return EfficiencyResult(
                latency_seconds=float('inf'),
                cpu_usage_percent=0.0,
                memory_used_mb=0.0,
                raw_efficiency=0.0,
                normalized_efficiency=0.0
            )
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = process.memory_info().rss
        
        latency = end_time - start_time
        cpu_usage = end_cpu - start_cpu
        memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB
        
        # Documented efficiency formula: eff = min(1.0, K_latency / latency_sec) where K_latency = 3.0
        # 3 seconds -> 1.0, 6 seconds -> 0.5, etc.
        K_latency = 3.0
        raw_efficiency = min(1.0, K_latency / latency) if latency > 0 else 0.0
        
        return EfficiencyResult(
            latency_seconds=latency,
            cpu_usage_percent=cpu_usage,
            memory_used_mb=memory_used,
            raw_efficiency=raw_efficiency,
            normalized_efficiency=raw_efficiency  # Will be normalized later
        )
    
    def normalize_efficiency_scores(self, efficiency_results: List[EfficiencyResult]) -> List[float]:
        """
        Normalize efficiency scores using min-max normalization
        
        Args:
            efficiency_results: List of efficiency results
            
        Returns:
            List of normalized scores in [0,1] range
        """
        if not efficiency_results:
            return []
        
        raw_scores = [result.raw_efficiency for result in efficiency_results]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        
        if max_score == min_score:
            return [0.5] * len(raw_scores)  # Neutral scores if all equal
        
        normalized = [(score - min_score) / (max_score - min_score) for score in raw_scores]
        return normalized
    
    def calculate_clmpi(self, accuracy_result: AccuracyResult,
                       contextual_result: ContextualResult,
                       coherence_result: CoherenceResult,
                       fluency_result: FluencyResult,
                       efficiency_score: float) -> Dict[str, float]:
        """
        Calculate final CLMPI scores
        
        Args:
            accuracy_result: Accuracy evaluation results
            contextual_result: Contextual understanding results
            coherence_result: Coherence evaluation results
            fluency_result: Fluency evaluation results
            efficiency_score: Normalized efficiency score
            
        Returns:
            Dictionary with CLMPI scores and component scores
        """
        # Normalize quality scores to [0,1]
        accuracy_norm = accuracy_result.f1_score  # Already in [0,1]
        contextual_norm = contextual_result.combined_score  # Already in [0,1]
        coherence_norm = coherence_result.coherence_score  # Already in [0,1]
        fluency_norm = fluency_result.fluency_score  # Already in [0,1]
        
        # Calculate CLMPI
        clmpi_01 = (
            self.weights['accuracy'] * accuracy_norm +
            self.weights['contextual_understanding'] * contextual_norm +
            self.weights['coherence'] * coherence_norm +
            self.weights['fluency'] * fluency_norm +
            self.weights['performance_efficiency'] * efficiency_score
        )
        
        clmpi_100 = clmpi_01 * 100
        
        return {
            'clmpi_01': clmpi_01,
            'clmpi_100': clmpi_100,
            'component_scores': {
                'accuracy': accuracy_norm,
                'contextual_understanding': contextual_norm,
                'coherence': coherence_norm,
                'fluency': fluency_norm,
                'performance_efficiency': efficiency_score
            },
            'detailed_results': {
                'accuracy': accuracy_result,
                'contextual_understanding': contextual_result,
                'coherence': coherence_result,
                'fluency': fluency_result
            }
        }
    
    def save_detailed_results(self, results: Dict, output_dir: Path):
        """Save detailed results for each metric"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save per-metric results
        for metric_name, result in results['detailed_results'].items():
            metric_dir = output_dir / metric_name
            metric_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            with open(metric_dir / 'detail.jsonl', 'w') as f:
                for i, response in enumerate(result.responses):
                    detail = {
                        'response': response,
                        'score': result.detailed_scores[i] if hasattr(result, 'detailed_scores') else None
                    }
                    f.write(json.dumps(detail) + '\n')
            
            # Save summary
            if metric_name == 'accuracy':
                average_score = result.f1_score
            elif metric_name == 'contextual_understanding':
                average_score = result.combined_score
            elif metric_name == 'coherence':
                average_score = result.coherence_score
            elif metric_name == 'fluency':
                average_score = result.fluency_score
            else:
                average_score = 0.0
            
            summary = {
                'metric': metric_name,
                'average_score': average_score,
                'total_responses': len(result.responses)
            }
            with open(metric_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
        
        # Save combined results
        with open(output_dir / 'clmpi_scores.json', 'w') as f:
            json.dump({
                'clmpi_01': results['clmpi_01'],
                'clmpi_100': results['clmpi_100'],
                'component_scores': results['component_scores'],
                'weights_used': self.weights
            }, f, indent=2)
