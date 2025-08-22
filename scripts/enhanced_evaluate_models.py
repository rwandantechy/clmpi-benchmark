#!/usr/bin/env python3
"""
Enhanced CLMPI Model Evaluation Script

Implements rigorous benchmarking with:
- Standardized generation settings
- Curated expert-validated datasets
- Proper scoring methods for each dimension
- Granular logging and validation
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
import numpy as np

from enhanced_clmpi_calculator import EnhancedCLMPICalculator
from ollama_runner import OllamaRunner
from utils import sanitize_filename
from generation import load_generation_profile, get_generation_settings_for_metric


class EnhancedModelEvaluator:
    """
    Enhanced evaluator with rigorous benchmarking methodology
    """
    
    def __init__(self, model_config_path: str, generation_config_path: str, 
                 device_config_path: str, output_dir: str, label: str = 'run', seed: int = 42):
        self.model_config = self._load_config(model_config_path)
        self.generation_config = self._load_config(generation_config_path)
        self.device_config = self._load_config(device_config_path)
        self.output_dir = Path(output_dir)
        self.label = label
        self.seed = seed
        
        # Initialize calculator with weights from config
        weights = self.model_config.get('evaluation_weights', {})
        self.calculator = EnhancedCLMPICalculator(weights)
        self.ollama_runner = OllamaRunner(self.device_config['runtime']['ollama_host'])
        
        self.setup_logging()
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
    
    def log_configuration_info(self):
        """Log configuration information for transparency"""
        self.logger.info("Configuration Information:")
        self.logger.info(f"  Model Config: {len(self.model_config['models'])} models configured")
        self.logger.info(f"  Generation Profiles: {list(self.generation_config['generation_profiles'].keys())}")
        self.logger.info(f"  Device: {self.device_config['device']['name']}")
        self.logger.info(f"  Evaluation Weights: {self.model_config['evaluation_weights']}")
        self.logger.info(f"  Random Seed: {self.seed}")
    
    def load_curated_datasets(self) -> Dict[str, List[Dict]]:
        """Load curated datasets for each evaluation dimension"""
        datasets = {}
        prompt_dir = Path("prompts")
        
        for dimension, prompt_files in self.model_config['prompt_sets'].items():
            datasets[dimension] = []
            for prompt_file in prompt_files:
                file_path = prompt_dir / prompt_file
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Extract questions based on dataset structure
                        if 'questions' in data:
                            questions = data['questions']
                        elif 'conversations' in data:
                            questions = data['conversations']
                        elif 'prompts' in data:
                            questions = data['prompts']
                        else:
                            questions = []
                        
                        # Sample questions if specified
                        if 'samples_per_task' in self.model_config['evaluation'] and questions:
                            samples = min(len(questions), self.model_config['evaluation']['samples_per_task'])
                            questions = random.sample(questions, samples)
                        
                        datasets[dimension].extend(questions)
        
        return datasets
    
    def get_generation_settings(self, metric_name: str, model_name: str) -> Dict[str, Any]:
        """Get standardized generation settings for a metric"""
        # Use the centralized generation helper
        settings = get_generation_settings_for_metric(metric_name)
        
        # Apply model-specific overrides
        if model_name in self.generation_config['model_overrides']:
            overrides = self.generation_config['model_overrides'][model_name]
            settings.update(overrides)
        
        return settings
    
    def evaluate_accuracy(self, model_name: str, model_config: Dict, 
                         datasets: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Evaluate accuracy using curated factual dataset"""
        if 'accuracy' not in datasets or not datasets['accuracy']:
            self.logger.warning("No accuracy dataset found")
            return None
        
        self.logger.info(f"Evaluating accuracy for {model_name}")
        
        questions = datasets['accuracy']
        responses = []
        gold_answers = []
        acceptable_answers = []
        
        generation_settings = self.get_generation_settings('accuracy', model_name)
        
        for question in questions:
            # Generate response with deterministic settings
            response = self.ollama_runner.generate_response(
                model_name, question['question'], generation_settings
            )
            responses.append(response)
            gold_answers.append(question['correct_answer'])
            acceptable_answers.append(question.get('acceptable_answers', []))
        
        # Evaluate using enhanced calculator
        accuracy_result = self.calculator.evaluate_accuracy(
            responses, gold_answers, acceptable_answers
        )
        
        self.logger.info(f"Accuracy - EM: {accuracy_result.exact_match:.3f}, F1: {accuracy_result.f1_score:.3f}")
        
        return {
            'metric': 'accuracy',
            'result': accuracy_result,
            'responses': responses,
            'questions': questions
        }
    
    def evaluate_contextual_understanding(self, model_name: str, model_config: Dict,
                                        datasets: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Evaluate contextual understanding using curated dataset"""
        if 'contextual_understanding' not in datasets or not datasets['contextual_understanding']:
            self.logger.warning("No contextual understanding dataset found")
            return None
        
        self.logger.info(f"Evaluating contextual understanding for {model_name}")
        
        conversations = datasets['contextual_understanding']
        responses = []
        contexts = []
        gold_answers = []
        
        generation_settings = self.get_generation_settings('contextual_understanding', model_name)
        
        for conv in conversations:
            # Create prompt with context
            prompt = f"Context: {conv['context']}\n\nQuestion: {conv['question']}"
            
            response = self.ollama_runner.generate_response(
                model_name, prompt, generation_settings
            )
            responses.append(response)
            contexts.append(conv['context'])
            gold_answers.append(conv['correct_answer'])
        
        # Evaluate using enhanced calculator
        contextual_result = self.calculator.evaluate_contextual_understanding(
            responses, contexts, gold_answers
        )
        
        self.logger.info(f"Contextual Understanding - Combined: {contextual_result.combined_score:.3f}")
        
        return {
            'metric': 'contextual_understanding',
            'result': contextual_result,
            'responses': responses,
            'conversations': conversations
        }
    
    def evaluate_coherence(self, model_name: str, model_config: Dict,
                          datasets: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Evaluate coherence using dedicated coherence tasks"""
        if 'coherence' not in datasets or not datasets['coherence']:
            self.logger.warning("No coherence dataset found")
            return None
        
        self.logger.info(f"Evaluating coherence for {model_name}")
        
        prompts = datasets['coherence']
        responses = []
        
        generation_settings = self.get_generation_settings('coherence', model_name)
        
        for prompt_data in prompts:
            response = self.ollama_runner.generate_response(
                model_name, prompt_data['prompt'], generation_settings
            )
            responses.append(response)
        
        # Evaluate using enhanced calculator
        coherence_result = self.calculator.evaluate_coherence(responses)
        
        self.logger.info(f"Coherence Score: {coherence_result.coherence_score:.3f}")
        
        return {
            'metric': 'coherence',
            'result': coherence_result,
            'responses': responses,
            'prompts': prompts
        }
    
    def evaluate_fluency(self, model_name: str, model_config: Dict,
                        datasets: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Evaluate fluency using dedicated fluency tasks"""
        if 'fluency' not in datasets or not datasets['fluency']:
            self.logger.warning("No fluency dataset found")
            return None
        
        self.logger.info(f"Evaluating fluency for {model_name}")
        
        prompts = datasets['fluency']
        responses = []
        
        generation_settings = self.get_generation_settings('fluency', model_name)
        
        for prompt_data in prompts:
            response = self.ollama_runner.generate_response(
                model_name, prompt_data['prompt'], generation_settings
            )
            responses.append(response)
        
        # Evaluate using enhanced calculator
        fluency_result = self.calculator.evaluate_fluency(responses)
        
        self.logger.info(f"Fluency Score: {fluency_result.fluency_score:.3f}")
        
        return {
            'metric': 'fluency',
            'result': fluency_result,
            'responses': responses,
            'prompts': prompts
        }
    
    def evaluate_efficiency(self, model_name: str, model_config: Dict,
                           datasets: Dict[str, List[Dict]]) -> Optional[Dict]:
        """Evaluate efficiency using accuracy tasks"""
        if 'accuracy' not in datasets or not datasets['accuracy']:
            self.logger.warning("No accuracy dataset found for efficiency measurement")
            return None
        
        self.logger.info(f"Evaluating efficiency for {model_name}")
        
        questions = datasets['accuracy'][:3]  # Use first 3 questions for efficiency
        efficiency_results = []
        
        generation_settings = self.get_generation_settings('accuracy', model_name)
        
        for question in questions:
            # Measure efficiency for each generation
            def generate_with_measurement():
                return self.ollama_runner.generate_response(
                    model_name, question['question'], generation_settings
                )
            
            efficiency_result = self.calculator.measure_efficiency(generate_with_measurement)
            efficiency_results.append(efficiency_result)
        
        # Normalize efficiency scores
        normalized_scores = self.calculator.normalize_efficiency_scores(efficiency_results)
        
        # Average the normalized scores
        avg_efficiency = np.mean(normalized_scores)
        
        self.logger.info(f"Efficiency Score: {avg_efficiency:.3f}")
        
        return {
            'metric': 'efficiency',
            'results': efficiency_results,
            'normalized_score': avg_efficiency,
            'questions': questions
        }
    
    def evaluate_model(self, model_name: str, model_config: Dict, 
                      datasets: Dict[str, List[Dict]]) -> Dict:
        """Evaluate a single model across all dimensions"""
        self.logger.info(f"Starting enhanced evaluation of {model_name}")
        
        results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # Evaluate each metric separately
        accuracy_eval = self.evaluate_accuracy(model_name, model_config, datasets)
        if accuracy_eval:
            results['metrics']['accuracy'] = accuracy_eval
        
        contextual_eval = self.evaluate_contextual_understanding(model_name, model_config, datasets)
        if contextual_eval:
            results['metrics']['contextual_understanding'] = contextual_eval
        
        coherence_eval = self.evaluate_coherence(model_name, model_config, datasets)
        if coherence_eval:
            results['metrics']['coherence'] = coherence_eval
        
        fluency_eval = self.evaluate_fluency(model_name, model_config, datasets)
        if fluency_eval:
            results['metrics']['fluency'] = fluency_eval
        
        efficiency_eval = self.evaluate_efficiency(model_name, model_config, datasets)
        if efficiency_eval:
            results['metrics']['efficiency'] = efficiency_eval
        
        # Calculate CLMPI scores
        if all(metric in results['metrics'] for metric in ['accuracy', 'contextual_understanding', 'coherence', 'fluency', 'efficiency']):
            clmpi_results = self.calculator.calculate_clmpi(
                accuracy_eval['result'],
                contextual_eval['result'],
                coherence_eval['result'],
                fluency_eval['result'],
                efficiency_eval['normalized_score']
            )
            
            results['clmpi_scores'] = clmpi_results
            
            self.logger.info(f"CLMPI Score: {clmpi_results['clmpi_01']:.3f} (0-1) / {clmpi_results['clmpi_100']:.1f} (0-100)")
        
        return results
    
    def run_evaluation(self, selected_models: Optional[List[str]] = None) -> List[Dict]:
        """Run enhanced evaluation for specified models"""
        # Log hardware and configuration info
        hardware_info = self.log_hardware_info()
        self.log_configuration_info()
        
        # Resolve generation profiles (do not hardcode temperature/top_p/top_k anywhere else)
        gen_det = load_generation_profile("deterministic")  # for Accuracy, Context
        gen_cre = load_generation_profile("creative")       # for Coherence, Fluency
        print("=== RUN GENERATION PROFILES ===")
        print("deterministic:", gen_det)
        print("creative:", gen_cre)
        
        # Load curated datasets
        datasets = self.load_curated_datasets()
        
        # Determine which models to evaluate
        available_models = list(self.model_config['models'].keys())
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
                model_config = self.model_config['models'][model_name]
                result = self.evaluate_model(model_name, model_config, datasets)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        return results
    
    def save_enhanced_results(self, results: List[Dict], run_name: str):
        """Save enhanced results with granular logging"""
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = self.output_dir / f"{timestamp}_{self.label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        def make_serializable(obj):
            """Recursively convert dataclass objects to dictionaries"""
            if hasattr(obj, '__dict__'):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        # Save individual model results
        for result in results:
            model_name = result['model_name']
            sanitized_name = sanitize_filename(model_name)
            model_file = run_dir / f"{sanitized_name}_results.json"
            
            # Convert all dataclass objects to dictionaries for JSON serialization
            serializable_result = make_serializable(result)
            
            with open(model_file, 'w') as f:
                json.dump(serializable_result, f, indent=2)
        
        # Save detailed results for each metric
        for result in results:
            if 'clmpi_scores' in result:
                model_name = sanitize_filename(result['model_name'])
                detailed_dir = run_dir / f"{model_name}_detailed"
                self.calculator.save_detailed_results(result['clmpi_scores'], detailed_dir)
        
        # Create run banner with weights, device, and formulas
        gen_det = load_generation_profile("deterministic")
        gen_cre = load_generation_profile("creative")
        
        run_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device_config": self.device_config,
            "weights": self.model_config.get('evaluation_weights', {}),
            "generation_profiles": {"deterministic": gen_det, "creative": gen_cre},
            "notes": "Deterministic for Accuracy/Context; Creative for Coherence/Fluency."
        }
        with open(run_dir / "run_info.json", "w") as f:
            json.dump(run_info, f, indent=2)
        
        # Create summary
        summary = {
            'run_name': run_name,
            'timestamp': timestamp,
            'label': self.label,
            'seed': self.seed,
            'hardware_info': self.log_hardware_info(),
            'config_used': {
                'model_config': str(self.model_config),
                'generation_config': str(self.generation_config),
                'evaluation_weights': self.model_config.get('evaluation_weights', {})
            },
            'results': make_serializable(results)
        }
        
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create latest symlink
        latest_link = self.output_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
        
        self.logger.info(f"Enhanced results saved to: {run_dir}")
        return run_dir
    
    def print_enhanced_summary(self, results: List[Dict], run_dir: Path):
        """Print enhanced evaluation summary"""
        print("\n" + "="*60)
        print("ENHANCED CLMPI EVALUATION SUMMARY")
        print("="*60)
        
        for result in results:
            model_name = result['model_name']
            print(f"\n{model_name}:")
            
            if 'clmpi_scores' in result:
                clmpi_01 = result['clmpi_scores']['clmpi_01']
                clmpi_100 = result['clmpi_scores']['clmpi_100']
                component_scores = result['clmpi_scores']['component_scores']
                
                print(f"  CLMPI Score: {clmpi_01:.3f} (0-1) / {clmpi_100:.1f} (0-100)")
                print(f"  Accuracy: {component_scores['accuracy']:.3f}")
                print(f"  Contextual Understanding: {component_scores['contextual_understanding']:.3f}")
                print(f"  Coherence: {component_scores['coherence']:.3f}")
                print(f"  Fluency: {component_scores['fluency']:.3f}")
                print(f"  Performance Efficiency: {component_scores['performance_efficiency']:.3f}")
            else:
                print("  [WARNING] Incomplete evaluation - missing some metrics")
        
        print(f"\n" + "="*60)
        print("FILES GENERATED")
        print("="*60)
        print(f"Summary: {run_dir}/summary.json")
        print(f"Detailed Results: {run_dir}/*_detailed/")
        print(f"Charts: evaluations/visualizations/")
        print(f"Excel: evaluations/clmpi_scorebook.xlsx")
        print(f"Latest: {run_dir.parent}/latest")


def main():
    """Main function with enhanced CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run enhanced CLMPI benchmark evaluation with rigorous methodology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/enhanced_evaluate_models.py --model-config config/model_config.yaml --generation-config config/generation_config.yaml --device-config config/device_default.yaml
  python scripts/enhanced_evaluate_models.py --model-config config/model_config.yaml --generation-config config/generation_config.yaml --models phi3:mini mistral --label enhanced_demo
        """
    )
    
    parser.add_argument('--model-config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--generation-config', type=str, required=True,
                       help='Path to generation configuration file')
    parser.add_argument('--device-config', type=str, required=True,
                       help='Path to device configuration file')
    parser.add_argument('--models', nargs='+', type=str,
                       help='Specific models to evaluate (default: all in config)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--label', type=str, default='enhanced_run',
                       help='Label for this run (used in folder naming)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate files exist
    for config_file in [args.model_config, args.generation_config, args.device_config]:
        if not Path(config_file).exists():
            print(f"Error: Config file not found: {config_file}")
            return 1
    
    # Run enhanced evaluation
    try:
        evaluator = EnhancedModelEvaluator(
            args.model_config, args.generation_config, args.device_config,
            args.output, args.label, args.seed
        )
        results = evaluator.run_evaluation(args.models)
        
        if results:
            run_dir = evaluator.save_enhanced_results(results, "enhanced_benchmark_run")
            evaluator.print_enhanced_summary(results, run_dir)
        else:
            print("No results generated")
            return 1
            
    except Exception as e:
        print(f"Error during enhanced evaluation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
