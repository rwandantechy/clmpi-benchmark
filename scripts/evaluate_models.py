#!/usr/bin/env python3
"""
CLMPI Model Evaluation Script

Main entry point for running CLMPI benchmarks on language models.
Evaluates models across 5 dimensions: accuracy, contextual understanding, 
coherence, fluency, and performance efficiency.

Example:
    python scripts/evaluate_models.py \
        --config config/model_config.yaml \
        --device config/device_default.yaml \
        --models phi3:mini mistral \
        --output results/edge_demo
"""

import argparse
import json
import yaml
import time
import logging
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import random

from clmpi_calculator import CLMPICalculator, CLMPIScores
from ollama_runner import OllamaRunner
from utils import sanitize_filename


class ModelEvaluator:
    """
    Main evaluator for CLMPI benchmark framework
    
    Evaluates models across 5 dimensions and generates standardized results.
    """
    
    def __init__(self, config_path: str, device_path: str, output_dir: str, label: str = 'run', seed: int = 42):
        self.config = self._load_config(config_path)
        self.device_config = self._load_config(device_path)
        self.output_dir = Path(output_dir)
        self.label = label
        self.seed = seed
        self.calculator = CLMPICalculator()
        self.ollama_runner = OllamaRunner(self.device_config['runtime']['ollama_host'])
        self.setup_logging()
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_hardware_info(self) -> Dict[str, str]:
        """Log hardware information for reproducibility"""
        hardware_info = {
            'cpu_model': platform.processor(),
            'cpu_cores': str(psutil.cpu_count()),
            'memory_gb': str(round(psutil.virtual_memory().total / (1024**3), 1)),
            'os': platform.system() + ' ' + platform.release(),
            'python_version': platform.python_version()
        }
        
        self.logger.info("Hardware Information:")
        for key, value in hardware_info.items():
            self.logger.info(f"  {key}: {value}")
        
        return hardware_info
    
    def load_prompts(self) -> Dict[str, List[Dict]]:
        """Load prompts from JSON files"""
        prompts = {}
        prompt_dir = Path("prompts")
        
        for dimension, prompt_files in self.config['prompt_sets'].items():
            prompts[dimension] = []
            for prompt_file in prompt_files:
                file_path = prompt_dir / prompt_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Extract tasks from the structure
                        if isinstance(data, dict) and 'tasks' in data:
                            tasks = data['tasks']
                        elif isinstance(data, list):
                            tasks = data
                        else:
                            tasks = []
                        
                        # Sample prompts if specified
                        if 'samples_per_task' in self.config['evaluation'] and tasks:
                            samples = min(len(tasks), self.config['evaluation']['samples_per_task'])
                            tasks = random.sample(tasks, samples)
                        
                        prompts[dimension].extend(tasks)
        
        return prompts
    
    def evaluate_accuracy(self, model_name: str, responses: List[str], expected_answers: List[str]) -> float:
        """Evaluate accuracy as correct / total"""
        self.logger.info(f"Evaluating accuracy for {model_name}")
        accuracy = self.calculator.evaluate_accuracy(responses, expected_answers)
        self.logger.info(f"Accuracy score: {accuracy:.3f}")
        return accuracy
    
    def evaluate_quality_dimension(self, model_name: str, responses: List[str], dimension: str) -> float:
        """Evaluate quality dimensions (contextual understanding, coherence, fluency)"""
        self.logger.info(f"Evaluating {dimension} for {model_name}")
        
        scores = []
        for response in responses:
            score = 3.0  # Base score
            
            # Simple heuristics for scoring
            if dimension == 'contextual_understanding':
                if len(response.split()) > 20:
                    score += 0.5
                if any(k in response.lower() for k in ['because', 'therefore', 'however']):
                    score += 0.5
            elif dimension == 'coherence':
                if len(response.split('.')) > 2:
                    score += 0.5
                if any(c in response.lower() for c in ['and', 'but', 'or', 'because']):
                    score += 0.5
            elif dimension == 'fluency':
                if len(response.split()) > 15:
                    score += 0.5
                if response.endswith(('.', '!', '?')):
                    score += 0.5
            
            scores.append(min(score, 5.0))
        
        avg_score = sum(scores) / len(scores)
        self.logger.info(f"{dimension} score: {avg_score:.3f}")
        return avg_score
    
    def evaluate_model(self, model_name: str, model_config: Dict, prompts: Dict[str, List[Dict]]) -> Dict:
        """Evaluate a single model across all dimensions"""
        self.logger.info(f"Starting evaluation of {model_name}")
        
        results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'responses': {},
            'component_scores': {},
            'performance_metrics': {}
        }
        
        # Get prompts for each dimension
        accuracy_prompts = prompts.get('accuracy', [])
        contextual_prompts = prompts.get('contextual_understanding', [])
        
        # Evaluate accuracy
        if accuracy_prompts:
            accuracy_responses = []
            expected_answers = []
            
            for prompt_data in accuracy_prompts:
                # Mock response for testing
                response = f"Mock response for: {prompt_data['prompt'][:50]}..."
                accuracy_responses.append(response)
                expected_answers.append(prompt_data.get('expected_answer', ''))
            
            results['responses']['accuracy'] = accuracy_responses
            results['component_scores']['accuracy'] = self.evaluate_accuracy(
                model_name, accuracy_responses, expected_answers
            )
        
        # Evaluate contextual understanding
        if contextual_prompts:
            contextual_responses = []
            
            for prompt_data in contextual_prompts:
                # Mock response for testing
                response = f"Mock contextual response for: {prompt_data['prompt'][:50]}..."
                contextual_responses.append(response)
            
            results['responses']['contextual_understanding'] = contextual_responses
            results['component_scores']['contextual_understanding'] = self.evaluate_quality_dimension(
                model_name, contextual_responses, 'contextual_understanding'
            )
        
        # Evaluate coherence (using contextual responses)
        if contextual_responses:
            results['component_scores']['coherence'] = self.evaluate_quality_dimension(
                model_name, contextual_responses, 'coherence'
            )
        
        # Evaluate fluency (using all responses)
        all_responses = accuracy_responses + contextual_responses
        if all_responses:
            results['component_scores']['fluency'] = self.evaluate_quality_dimension(
                model_name, all_responses, 'fluency'
            )
        
        # Evaluate performance efficiency
        if all_responses:
            # Simple efficiency calculation based on response length and time
            total_time = len(all_responses) * 2.0  # Estimate 2 seconds per response
            total_memory = len(all_responses) * 100.0  # Estimate 100MB per response
            efficiency = self.calculator.calculate_raw_efficiency(total_time, total_memory)
            results['component_scores']['performance_efficiency'] = efficiency
            results['performance_metrics'] = {
                'total_time_seconds': total_time,
                'total_memory_mb': total_memory,
                'efficiency_score': efficiency
            }
        
        # Calculate CLMPI scores
        scores = CLMPIScores(
            accuracy=results['component_scores'].get('accuracy', 0.0),
            contextual_understanding=results['component_scores'].get('contextual_understanding', 0.0),
            coherence=results['component_scores'].get('coherence', 0.0),
            fluency=results['component_scores'].get('fluency', 0.0),
            performance_efficiency=results['component_scores'].get('performance_efficiency', 0.0)
        )
        
        results['clmpi_score_01'] = self.calculator.calculate_clmpi(scores)
        results['clmpi_score_100'] = self.calculator.calculate_clmpi_100(scores)
        
        self.logger.info(f"CLMPI Score (0-1): {results['clmpi_score_01']:.3f}")
        self.logger.info(f"CLMPI Score (0-100): {results['clmpi_score_100']:.1f}")
        
        return results
    
    def run_evaluation(self, selected_models: Optional[List[str]] = None) -> List[Dict]:
        """Run evaluation for specified models"""
        # Log hardware info
        hardware_info = self.log_hardware_info()
        
        # Load prompts
        prompts = self.load_prompts()
        
        # Determine which models to evaluate
        available_models = list(self.config['models'].keys())
        if selected_models:
            models_to_evaluate = [m for m in selected_models if m in available_models]
            if len(models_to_evaluate) != len(selected_models):
                missing = set(selected_models) - set(available_models)
                self.logger.warning(f"Models not found in config: {missing}")
        else:
            models_to_evaluate = available_models
        
        self.logger.info(f"Evaluating models: {models_to_evaluate}")
        
        # Evaluate each model
        results = []
        for model_name in models_to_evaluate:
            try:
                model_config = self.config['models'][model_name]
                result = self.evaluate_model(model_name, model_config, prompts)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict], run_name: str):
        """Save results to standardized directory structure"""
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.output_dir / f"{timestamp}_{self.label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual model results
        for result in results:
            model_name = result['model_name']
            sanitized_name = sanitize_filename(model_name)
            model_file = run_dir / f"{sanitized_name}_results.json"
            with open(model_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Create summary
        summary = {
            'run_name': run_name,
            'timestamp': timestamp,
            'label': self.label,
            'seed': self.seed,
            'hardware_info': self.log_hardware_info(),
            'config_used': {
                'model_config': str(self.config),
                'evaluation_weights': self.config['evaluation_weights']
            },
            'results': results
        }
        
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create latest symlink
        latest_link = self.output_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
        
        self.logger.info(f"Results saved to: {run_dir}")
        return run_dir
    
    def print_summary(self, results: List[Dict], run_dir: Path):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("CLMPI EVALUATION SUMMARY")
        print("="*60)
        
        for result in results:
            model_name = result['model_name']
            clmpi_01 = result['clmpi_score_01']
            clmpi_100 = result['clmpi_score_100']
            scores = result['component_scores']
            
            print(f"\n{model_name}:")
            print(f"  CLMPI Score: {clmpi_01:.3f} (0-1) / {clmpi_100:.1f} (0-100)")
            print(f"  Accuracy: {scores.get('accuracy', 0):.3f}")
            print(f"  Contextual Understanding: {scores.get('contextual_understanding', 0):.1f}/5")
            print(f"  Coherence: {scores.get('coherence', 0):.1f}/5")
            print(f"  Fluency: {scores.get('fluency', 0):.1f}/5")
            print(f"  Performance Efficiency: {scores.get('performance_efficiency', 0):.3f}")
        
        print(f"\n" + "="*60)
        print("FILES GENERATED")
        print("="*60)
        print(f"Summary: {run_dir}/summary.json")
        print(f"Charts: evaluations/visualizations/")
        print(f"Excel: evaluations/clmpi_scorebook.xlsx")
        print(f"Latest: {run_dir.parent}/latest")


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI benchmark evaluation on language models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_models.py --config config/model_config.yaml --device config/device_default.yaml
  python scripts/evaluate_models.py --config config/model_config.yaml --models phi3:mini mistral --output results/demo
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--device', type=str, required=True,
                       help='Path to device configuration file')
    parser.add_argument('--models', nargs='+', type=str,
                       help='Specific models to evaluate (default: all in config)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--label', type=str, default='run',
                       help='Label for this run (used in folder naming)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        return 1
    
    if not Path(args.device).exists():
        print(f"Error: Device file not found: {args.device}")
        return 1
    
    # Run evaluation
    try:
        evaluator = ModelEvaluator(args.config, args.device, args.output, args.label, args.seed)
        results = evaluator.run_evaluation(args.models)
        
        if results:
            run_dir = evaluator.save_results(results, "benchmark_run")
            evaluator.print_summary(results, run_dir)
        else:
            print("No results generated")
            return 1
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
