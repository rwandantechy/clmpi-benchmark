#!/usr/bin/env python3
"""
CLMPI Model Evaluation Script

This script implements the Comprehensive Language Model Performance Index (CLMPI)
framework to evaluate Large Language Models across multiple dimensions.

Usage:
    python evaluate_models.py --config config/model_config.yaml --output results/
"""

import argparse
import json
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from clmpi_calculator import CLMPICalculator, CLMPIScores, CLMPIWeights


class ModelEvaluator:
    """
    Main evaluator class for implementing CLMPI framework
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the model evaluator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.calculator = CLMPICalculator()
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('evaluation.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_evaluation_data(self) -> Dict[str, Any]:
        """
        Load evaluation datasets and prompts
        
        Returns:
            Dictionary containing evaluation data
        """
        data = {}
        
        # Load classification tasks
        with open('prompts/classification_tasks.json', 'r') as f:
            data['classification'] = json.load(f)
        
        # Load reasoning tasks
        with open('prompts/reasoning_tasks.json', 'r') as f:
            data['reasoning'] = json.load(f)
        
        return data
    
    def evaluate_accuracy(self, model_name: str, responses: List[str], 
                         expected_answers: List[str]) -> float:
        """
        Evaluate model accuracy
        
        Args:
            model_name: Name of the model being evaluated
            responses: Model-generated responses
            expected_answers: Expected correct answers
            
        Returns:
            Accuracy score (0-1)
        """
        self.logger.info(f"Evaluating accuracy for {model_name}")
        accuracy = self.calculator.evaluate_accuracy(responses, expected_answers)
        self.logger.info(f"Accuracy score: {accuracy:.3f}")
        return accuracy
    
    def evaluate_contextual_understanding(self, model_name: str, 
                                        responses: List[str]) -> float:
        """
        Evaluate contextual understanding (simplified automated version)
        
        Args:
            model_name: Name of the model being evaluated
            responses: Model-generated responses
            
        Returns:
            Contextual understanding score (0-5)
        """
        self.logger.info(f"Evaluating contextual understanding for {model_name}")
        
        # Simplified automated scoring based on response length and content
        # In a real implementation, this would use more sophisticated NLP techniques
        scores = []
        for response in responses:
            # Simple heuristics for context understanding
            score = 3.0  # Base score
            
            # Bonus for longer, more detailed responses
            if len(response.split()) > 20:
                score += 0.5
            
            # Bonus for responses that contain relevant keywords
            relevant_keywords = ['because', 'therefore', 'however', 'furthermore', 'consequently']
            if any(keyword in response.lower() for keyword in relevant_keywords):
                score += 0.5
            
            # Cap at 5.0
            score = min(score, 5.0)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Contextual understanding score: {avg_score:.3f}")
        return avg_score
    
    def evaluate_coherence(self, model_name: str, responses: List[str]) -> float:
        """
        Evaluate coherence (simplified automated version)
        
        Args:
            model_name: Name of the model being evaluated
            responses: Model-generated responses
            
        Returns:
            Coherence score (0-5)
        """
        self.logger.info(f"Evaluating coherence for {model_name}")
        
        # Simplified automated scoring
        scores = []
        for response in responses:
            score = 3.0  # Base score
            
            # Bonus for well-structured responses
            sentences = response.split('.')
            if len(sentences) > 2:
                score += 0.5
            
            # Bonus for logical connectors
            connectors = ['and', 'but', 'or', 'because', 'therefore', 'however']
            if any(connector in response.lower() for connector in connectors):
                score += 0.5
            
            # Cap at 5.0
            score = min(score, 5.0)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Coherence score: {avg_score:.3f}")
        return avg_score
    
    def evaluate_fluency(self, model_name: str, responses: List[str]) -> float:
        """
        Evaluate fluency (simplified automated version)
        
        Args:
            model_name: Name of the model being evaluated
            responses: Model-generated responses
            
        Returns:
            Fluency score (0-5)
        """
        self.logger.info(f"Evaluating fluency for {model_name}")
        
        # Simplified automated scoring
        scores = []
        for response in responses:
            score = 3.0  # Base score
            
            # Bonus for proper sentence structure
            if response.strip().endswith(('.', '!', '?')):
                score += 0.5
            
            # Bonus for varied vocabulary
            words = response.lower().split()
            unique_words = len(set(words))
            if unique_words > len(words) * 0.7:  # Good vocabulary diversity
                score += 0.5
            
            # Cap at 5.0
            score = min(score, 5.0)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"Fluency score: {avg_score:.3f}")
        return avg_score
    
    def measure_resource_efficiency(self, model_name: str, 
                                  evaluation_function, *args) -> float:
        """
        Measure resource efficiency
        
        Args:
            model_name: Name of the model being evaluated
            evaluation_function: Function to measure
            *args: Arguments for the function
            
        Returns:
            Efficiency score
        """
        self.logger.info(f"Measuring resource efficiency for {model_name}")
        
        time_taken, memory_used, efficiency = self.calculator.measure_resource_usage(
            evaluation_function, *args
        )
        
        self.logger.info(f"Time taken: {time_taken:.3f}s, Memory used: {memory_used:.2f}MB")
        self.logger.info(f"Efficiency score: {efficiency:.3f}")
        
        return efficiency
    
    def evaluate_model(self, model_name: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single model using CLMPI framework
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Starting evaluation for {model_name}")
        
        # Load evaluation data
        eval_data = self.load_evaluation_data()
        
        # Simulate model responses (in real implementation, this would call actual model APIs)
        # For demonstration, we'll use placeholder responses
        sample_responses = [
            "The answer is 42 because it represents the meaning of life according to Douglas Adams.",
            "Based on the context provided, the solution involves multiple steps.",
            "This is a coherent response that demonstrates understanding of the question.",
            "The model generates fluent and grammatically correct text.",
            "Resource efficiency is measured through time and memory usage."
        ]
        
        expected_answers = [
            "42",
            "multiple steps",
            "coherent response",
            "fluent text",
            "efficiency measured"
        ]
        
        # Evaluate each component
        accuracy = self.evaluate_accuracy(model_name, sample_responses, expected_answers)
        contextual_understanding = self.evaluate_contextual_understanding(model_name, sample_responses)
        coherence = self.evaluate_coherence(model_name, sample_responses)
        fluency = self.evaluate_fluency(model_name, sample_responses)
        
        # Measure resource efficiency
        def dummy_evaluation():
            time.sleep(0.1)  # Simulate processing time
            return "dummy result"
        
        resource_efficiency = self.measure_resource_efficiency(model_name, dummy_evaluation)
        
        # Create CLMPI scores
        scores = CLMPIScores(
            accuracy=accuracy,
            contextual_understanding=contextual_understanding,
            coherence=coherence,
            fluency=fluency,
            resource_efficiency=resource_efficiency
        )
        
        # Calculate CLMPI score
        clmpi_score = self.calculator.calculate_clmpi_normalized(scores)
        
        # Generate report
        report = self.calculator.generate_report(model_name, scores, clmpi_score)
        
        self.logger.info(f"Evaluation completed for {model_name}. CLMPI Score: {clmpi_score:.2f}")
        
        return report
    
    def evaluate_all_models(self) -> List[Dict[str, Any]]:
        """
        Evaluate all models specified in the configuration
        
        Returns:
            List of evaluation reports
        """
        results = []
        
        for model_name, model_config in self.config['models'].items():
            try:
                result = self.evaluate_model(model_name, model_config)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return results
    
    def generate_comparison_report(self, results: List[Dict[str, Any]], 
                                 output_dir: str):
        """
        Generate comparison report and visualizations
        
        Args:
            results: List of evaluation results
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create comparison DataFrame
        comparison_data = []
        for result in results:
            comparison_data.append({
                'Model': result['model_name'],
                'CLMPI_Score': result['clmpi_score'],
                'Accuracy': result['component_scores']['accuracy'],
                'Contextual_Understanding': result['component_scores']['contextual_understanding'],
                'Coherence': result['component_scores']['coherence'],
                'Fluency': result['component_scores']['fluency'],
                'Resource_Efficiency': result['component_scores']['resource_efficiency']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Save comparison table
        df.to_csv(output_path / 'model_comparison.csv', index=False)
        
        # Create visualizations
        self._create_visualizations(df, output_path)
        
        # Save detailed results
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Comparison report saved to {output_path}")
    
    def _create_visualizations(self, df: pd.DataFrame, output_path: Path):
        """Create visualization charts"""
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall CLMPI scores comparison
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Model'], df['CLMPI_Score'])
        plt.title('CLMPI Scores Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('CLMPI Score', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'clmpi_scores_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Component scores radar chart
        components = ['Accuracy', 'Contextual_Understanding', 'Coherence', 'Fluency', 'Resource_Efficiency']
        
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, model in enumerate(df['Model']):
            values = df.loc[df['Model'] == model, components].values[0].tolist()
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(components)
        ax.set_ylim(0, 5)
        ax.set_title('Component Scores Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_path / 'component_scores_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap of component scores
        plt.figure(figsize=(12, 8))
        heatmap_data = df.set_index('Model')[components]
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f', cbar_kws={'label': 'Score'})
        plt.title('Component Scores Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Components', fontsize=12)
        plt.ylabel('Models', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path / 'component_scores_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CLMPI Model Evaluation')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--output', default='results/', help='Output directory for results')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.config)
    
    # Run evaluations
    results = evaluator.evaluate_all_models()
    
    if results:
        # Generate comparison report
        evaluator.generate_comparison_report(results, args.output)
        
        # Print summary
        print("\n" + "="*50)
        print("CLMPI EVALUATION SUMMARY")
        print("="*50)
        
        for result in results:
            print(f"\n{result['model_name']}:")
            print(f"  CLMPI Score: {result['clmpi_score']:.2f}/25")
            print(f"  Interpretation: {result['interpretation']}")
            print(f"  Components:")
            for component, score in result['component_scores'].items():
                print(f"    {component.replace('_', ' ').title()}: {score:.3f}")
        
        print(f"\nDetailed results saved to: {args.output}")
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main() 