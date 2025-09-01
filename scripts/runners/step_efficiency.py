#!/usr/bin/env python3
"""
CLMPI Efficiency Step Evaluation Script
Measure actual Ollama model performance: model size, inference time, memory usage
"""

import argparse
import json
import yaml
import time
import logging
import psutil
import requests
import subprocess
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from clmpi_calculator import CLMPICalculator
from ollama_runner import OllamaRunner
from generation import load_generation_profile
from logger import save_responses_markdown

def load_dataset(dataset_path: str) -> dict:
    """Load dataset from path"""
    import json
    from pathlib import Path
    
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def get_ollama_model_info(model_name: str) -> dict:
    """Get model information from Ollama API"""
    try:
        response = requests.post(
            "http://localhost:11434/api/show",
            json={"name": model_name},
            timeout=10
        )
        if response.status_code == 200:
            model_info = response.json()
            
            # Get actual model file size
            if "modelfile" in model_info and "FROM" in model_info["modelfile"]:
                # Extract the model path from the modelfile
                modelfile_lines = model_info["modelfile"].split('\n')
                for line in modelfile_lines:
                    if line.startswith("FROM "):
                        model_path = line.replace("FROM ", "").strip()
                        if model_path.startswith("/"):
                            try:
                                import os
                                if os.path.exists(model_path):
                                    model_size_bytes = os.path.getsize(model_path)
                                    model_size_mb = model_size_bytes / (1024 * 1024)
                                    model_info["actual_size_mb"] = model_size_mb
                                    model_info["actual_size_gb"] = model_size_mb / 1024
                            except Exception as e:
                                logging.warning(f"Error getting model file size: {e}")
                        break
            
            return model_info
        else:
            logging.warning(f"Failed to get model info: {response.status_code}")
            return {}
    except Exception as e:
        logging.warning(f"Error getting model info: {e}")
        return {}


def get_ollama_process_info() -> dict:
    """Get Ollama process resource usage"""
    try:
        # Find Ollama process
        ollama_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    ollama_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        if not ollama_processes:
            return {"error": "No Ollama processes found"}
        
        # Get the main Ollama process (usually the first one)
        main_proc = ollama_processes[0]
        
        # Get memory and CPU info
        memory_info = main_proc.memory_info()
        cpu_percent = main_proc.cpu_percent()
        
        return {
            "pid": main_proc.pid,
            "memory_rss_mb": memory_info.rss / (1024 * 1024),
            "memory_vms_mb": memory_info.vms / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "process_count": len(ollama_processes)
        }
    except Exception as e:
        logging.warning(f"Error getting Ollama process info: {e}")
        return {"error": str(e)}


def measure_model_performance(model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float = 1.0, top_k: int = 40) -> dict:
    """Measure actual model performance metrics"""
    
    # Get initial Ollama process state
    initial_ollama = get_ollama_process_info()
    
    # Get model info
    model_info = get_ollama_model_info(model_name)
    model_size_mb = model_info.get("actual_size_mb", 0)  # Use actual file size
    if model_size_mb == 0:
        # Fallback to API size if actual size not available
        model_size_mb = model_info.get("size", 0) / (1024 * 1024) if model_info else 0
    
    # Initialize Ollama runner
    ollama_runner = OllamaRunner("http://localhost:11434")
    
    # Measure inference time
    start_time = time.time()
    try:
        response, _ = ollama_runner.generate_response(model_name, prompt, max_tokens, temperature, top_p, top_k)
        end_time = time.time()
        success = True
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        response = ""
        end_time = start_time
        success = False
    
    # Get final Ollama process state
    final_ollama = get_ollama_process_info()
    
    # Calculate metrics
    inference_time = end_time - start_time
    
    # Memory delta (if available)
    memory_delta_mb = 0
    if "error" not in initial_ollama and "error" not in final_ollama:
        memory_delta_mb = final_ollama["memory_rss_mb"] - initial_ollama["memory_rss_mb"]
    
    # Peak memory (use final state as approximation)
    peak_memory_mb = final_ollama.get("memory_rss_mb", 0) if "error" not in final_ollama else 0
    
    # CPU usage (average during inference)
    cpu_usage = final_ollama.get("cpu_percent", 0) if "error" not in final_ollama else 0
    
    return {
        "success": success,
        "response": response,
        "inference_time_seconds": inference_time,
        "model_size_mb": model_size_mb,
        "peak_memory_mb": peak_memory_mb,
        "memory_delta_mb": memory_delta_mb,
        "cpu_usage_percent": cpu_usage,
        "ollama_processes": final_ollama.get("process_count", 0),
        "model_info": model_info,
        "initial_ollama_state": initial_ollama,
        "final_ollama_state": final_ollama
    }


def calculate_efficiency_score(performance_data: dict, accuracy_score: float = 1.0) -> float:
    """Calculate efficiency score based on performance metrics and accuracy"""
    
    if not performance_data["success"]:
        return 0.0
    
    inference_time = performance_data["inference_time_seconds"]
    model_size_mb = performance_data["model_size_mb"]
    
    # Time-based scoring (faster = better)
    if inference_time <= 1.0:
        time_score = 1.0
    elif inference_time <= 3.0:
        time_score = 0.8
    elif inference_time <= 5.0:
        time_score = 0.6
    elif inference_time <= 10.0:
        time_score = 0.4
    else:
        time_score = 0.2
    
    # Size-based scoring (smaller = better for edge)
    if model_size_mb <= 1000:  # 1GB
        size_score = 1.0
    elif model_size_mb <= 3000:  # 3GB
        size_score = 0.8
    elif model_size_mb <= 7000:  # 7GB
        size_score = 0.6
    else:
        size_score = 0.4
    
    # Base efficiency score (70% time, 30% size)
    base_efficiency = 0.7 * time_score + 0.3 * size_score
    
    # Apply accuracy penalty: wrong answers get 0 efficiency
    final_efficiency = base_efficiency * accuracy_score
    
    return final_efficiency


def check_answer_accuracy(response: str, expected_answers: list) -> float:
    """Check if the model's answer is correct"""
    try:
        # Try to parse JSON response
        import json
        parsed = json.loads(response.strip())
        model_answer = parsed.get("answer", "").lower().strip()
        
        # Check against expected answers
        for expected in expected_answers:
            if expected.lower().strip() == model_answer:
                return 1.0  # Correct answer
        
        return 0.0  # Wrong answer
        
    except (json.JSONDecodeError, KeyError, AttributeError):
        # If response is not valid JSON or missing answer field
        return 0.0


def find_latest_run_directory() -> Path:
    """Find the latest run directory, or create one if none exists"""
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()
    
    # Look for existing stepwise runs
    stepwise_runs = list(results_dir.glob("*_stepwise"))
    if stepwise_runs:
        # Use the most recent one
        latest = max(stepwise_runs, key=lambda p: p.stat().st_mtime)
        return latest
    else:
        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = results_dir / f"{timestamp}_stepwise"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


def save_efficiency_responses_markdown(model_name: str, task_data: dict, performance_data: dict, 
                                     efficiency: float, accuracy_score: float = 1.0) -> str:
    """Save efficiency responses with detailed Ollama model metrics in Markdown format"""
    
    # Create model-specific directory
    model_dir = Path("results/model_responses") / model_name.replace(":", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create markdown content
    markdown_content = f"""# {model_name} - Efficiency Responses

**Evaluation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Metric**: Efficiency
**Total Questions**: 1

---

## Question 1: {task_data.get("id", "eff_001")}

**Question/Prompt**: 
```
{task_data.get("prompt", "")}
```

**Model Response**: 
```
{performance_data.get("response", "")}
```

**Ollama Model Performance Metrics**:
- **Inference Time**: {performance_data.get("inference_time_seconds", 0):.3f} seconds
- **Model Size**: {performance_data.get("model_size_mb", 0):.1f} MB ({performance_data.get("model_size_mb", 0)/1024:.1f} GB)
- **Peak Memory Usage**: {performance_data.get("peak_memory_mb", 0):.1f} MB
- **Memory Delta**: {performance_data.get("memory_delta_mb", 0):.1f} MB
- **CPU Usage**: {performance_data.get("cpu_usage_percent", 0):.1f}%
- **Ollama Processes**: {performance_data.get("ollama_processes", 0)}

**Answer Accuracy**: {accuracy_score:.3f} ({'Correct' if accuracy_score == 1.0 else 'Incorrect'})

**Efficiency Score**: {efficiency:.3f}

**Expected Answer**: {task_data.get("reference", ["au"])}

**Score**: {efficiency:.3f}

---
"""
    
    # Save to file
    filename = "efficiency_responses.md"
    filepath = model_dir / filename
    
    with open(filepath, 'w') as f:
        f.write(markdown_content)
    
    return str(filepath)


def run_efficiency_evaluation(model_name: str, verbose: bool = False) -> dict:
    """Run efficiency evaluation measuring actual Ollama model performance"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load efficiency task
    dataset = load_dataset("prompts/efficiency_tasks.json")
    task = dataset[0]  # Just one task
    prompt = task.get("prompt", "What is 15 + 27?")
    
    # Load generation profile
    profile = load_generation_profile("deterministic")
    
    if verbose:
        logger.info(f"Testing Ollama model efficiency: {model_name}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Profile: deterministic")
    
    # Extract generation parameters
    max_tokens = profile.get("max_tokens", 1000)
    temperature = profile.get("temperature", 0.0)
    top_p = profile.get("top_p", 1.0)
    top_k = profile.get("top_k", 40)
    
    # Measure actual model performance
    performance_data = measure_model_performance(model_name, prompt, max_tokens, temperature, top_p, top_k)
    
    # Check answer accuracy
    expected_answers = task.get("reference", ["au"])  # Default to gold symbol
    accuracy_score = check_answer_accuracy(performance_data.get("response", ""), expected_answers)
    
    # Calculate efficiency score (including accuracy)
    efficiency = calculate_efficiency_score(performance_data, accuracy_score)
    
    if verbose:
        logger.info(f"Model Size: {performance_data['model_size_mb']:.1f} MB")
        logger.info(f"Inference Time: {performance_data['inference_time_seconds']:.3f}s")
        logger.info(f"Peak Memory: {performance_data['peak_memory_mb']:.1f} MB")
        logger.info(f"CPU Usage: {performance_data['cpu_usage_percent']:.1f}%")
        logger.info(f"Answer Accuracy: {accuracy_score:.3f} ({'Correct' if accuracy_score == 1.0 else 'Incorrect'})")
        logger.info(f"Efficiency Score: {efficiency:.3f}")
    
    # Find or create run directory
    run_dir = find_latest_run_directory()
    metric_dir = run_dir / "efficiency"
    metric_dir.mkdir(exist_ok=True)
    
    # Save responses with detailed metrics
    response_file = save_efficiency_responses_markdown(
        model_name, task, performance_data, efficiency, accuracy_score
    )
    
    # Save detailed results
    with open(metric_dir / "detail.jsonl", "w") as f:
        detail = {
            "task_id": task.get("id", "eff_001"),
            "prompt": prompt,
            "response": performance_data.get("response", ""),
            "expected_answers": task.get("reference", ["au"]),
            "accuracy_score": accuracy_score,
            "inference_time_seconds": performance_data.get("inference_time_seconds", 0),
            "model_size_mb": performance_data.get("model_size_mb", 0),
            "peak_memory_mb": performance_data.get("peak_memory_mb", 0),
            "memory_delta_mb": performance_data.get("memory_delta_mb", 0),
            "cpu_usage_percent": performance_data.get("cpu_usage_percent", 0),
            "ollama_processes": performance_data.get("ollama_processes", 0),
            "efficiency": efficiency,
            "success": performance_data.get("success", False),
            "model_info": performance_data.get("model_info", {}),
            "initial_ollama_state": performance_data.get("initial_ollama_state", {}),
            "final_ollama_state": performance_data.get("final_ollama_state", {})
        }
        f.write(json.dumps(detail) + "\n")
    
    # Save summary
    summary = {
        "metric": "efficiency",
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "efficiency": efficiency,
        "accuracy_score": accuracy_score,
        "inference_time_seconds": performance_data.get("inference_time_seconds", 0),
        "model_size_mb": performance_data.get("model_size_mb", 0),
        "peak_memory_mb": performance_data.get("peak_memory_mb", 0),
        "cpu_usage_percent": performance_data.get("cpu_usage_percent", 0),
        "success": performance_data.get("success", False),
        "generation_profile": "deterministic",
        "dataset_path": "prompts/efficiency_tasks.json"
    }
    
    with open(metric_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print single line summary
    print(f"[EFF] {model_name} efficiency={efficiency:.3f}")
    
    if verbose:
        logger.info(f"Results saved to: {metric_dir}")
        logger.info(f"Response file: {response_file}")
    
    return {
        "metric": "efficiency",
        "score": efficiency,
        "run_dir": str(run_dir),
        "metric_dir": str(metric_dir)
    }


def main():
    """Main function with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Run CLMPI efficiency evaluation step',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/step_efficiency.py --model phi3:3.8b
  python scripts/step_efficiency.py --model phi3:3.8b --verbose
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model name to evaluate (must be available via Ollama)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        result = run_efficiency_evaluation(args.model, args.verbose)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
