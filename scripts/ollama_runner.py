#!/usr/bin/env python3
"""
Ollama Runner - Integration script for running local LLMs via Ollama

This module provides integration with Ollama for running open-source models
locally and collecting performance metrics for CLMPI evaluation.
"""

import requests
import json
import time
import psutil
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    latency_ms: float
    memory_used_mb: float
    cpu_usage_percent: float
    token_count: int
    tokens_per_second: float
    timestamp: float


class OllamaRunner:
    """
    Main class for running models via Ollama and collecting performance metrics
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize Ollama runner
        
        Args:
            host: Ollama server host URL
        """
        self.host = host
        self.logger = logging.getLogger(__name__)
        self.performance_thread = None
        self.monitoring_active = False
        self.performance_data = []
        
    def check_ollama_status(self) -> bool:
        """
        Check if Ollama server is running
        
        Returns:
            True if Ollama is available, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama server not available: {e}")
            return False
    
    def list_available_models(self) -> List[str]:
        """
        Get list of available models in Ollama
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                self.logger.error(f"Failed to get models: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def start_performance_monitoring(self):
        """Start background performance monitoring"""
        self.monitoring_active = True
        self.performance_data = []
        self.performance_thread = threading.Thread(target=self._monitor_performance)
        self.performance_thread.daemon = True
        self.performance_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_performance_monitoring(self):
        """Stop background performance monitoring"""
        self.monitoring_active = False
        if self.performance_thread:
            self.performance_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_performance(self):
        """Background thread for monitoring system performance"""
        process = psutil.Process()
        start_time = time.time()
        
        while self.monitoring_active:
            try:
                # Get current memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Get current CPU usage
                cpu_percent = process.cpu_percent()
                
                # Record data point
                self.performance_data.append({
                    'timestamp': time.time() - start_time,
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                break
    
    def generate_response(self, model_name: str, prompt: str, 
                         max_tokens: int = 1000, temperature: float = 0.1) -> Tuple[str, PerformanceMetrics]:
        """
        Generate response from Ollama model with performance monitoring
        
        Args:
            model_name: Name of the model to use
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (response_text, performance_metrics)
        """
        # Start performance monitoring
        self.start_performance_monitoring()
        
        try:
            # Prepare request payload
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
            
            # Record start time
            start_time = time.time()
            
            # Make request to Ollama
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=60
            )
            
            # Record end time
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                token_count = result.get("eval_count", 0)
                
                # Stop monitoring and calculate metrics
                self.stop_performance_monitoring()
                
                # Calculate performance metrics
                latency_ms = (end_time - start_time) * 1000
                tokens_per_second = token_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
                
                # Get average memory and CPU usage
                if self.performance_data:
                    avg_memory = sum(p['memory_mb'] for p in self.performance_data) / len(self.performance_data)
                    avg_cpu = sum(p['cpu_percent'] for p in self.performance_data) / len(self.performance_data)
                else:
                    avg_memory = 0
                    avg_cpu = 0
                
                metrics = PerformanceMetrics(
                    latency_ms=latency_ms,
                    memory_used_mb=avg_memory,
                    cpu_usage_percent=avg_cpu,
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    timestamp=end_time
                )
                
                self.logger.info(f"Generated response from {model_name}: {latency_ms:.2f}ms, {avg_memory:.2f}MB, {avg_cpu:.1f}% CPU")
                
                return response_text, metrics
                
            else:
                self.stop_performance_monitoring()
                self.logger.error(f"Ollama request failed: {response.status_code} - {response.text}")
                raise Exception(f"Ollama request failed: {response.status_code}")
                
        except Exception as e:
            self.stop_performance_monitoring()
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def run_batch_evaluation(self, model_name: str, prompts: List[str], 
                           max_tokens: int = 1000, temperature: float = 0.1) -> List[Tuple[str, PerformanceMetrics]]:
        """
        Run batch evaluation on multiple prompts
        
        Args:
            model_name: Name of the model to use
            prompts: List of prompts to evaluate
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature
            
        Returns:
            List of (response, metrics) tuples
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)} for {model_name}")
            
            try:
                response, metrics = self.generate_response(
                    model_name, prompt, max_tokens, temperature
                )
                results.append((response, metrics))
                
                # Small delay between requests to prevent overwhelming the system
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing prompt {i+1}: {e}")
                # Add empty result to maintain indexing
                results.append(("", PerformanceMetrics(0, 0, 0, 0, 0, time.time())))
        
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        try:
            response = requests.get(f"{self.host}/api/show", params={"name": model_name})
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"Model {model_name} not found")
                return None
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Pulling model: {model_name}")
            
            payload = {"name": model_name}
            response = requests.post(f"{self.host}/api/pull", json=payload, stream=True)
            
            if response.status_code == 200:
                # Stream the pull progress
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        if 'status' in data:
                            self.logger.info(f"Pull status: {data['status']}")
                
                self.logger.info(f"Successfully pulled model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def cleanup_model(self, model_name: str) -> bool:
        """
        Remove a model from local storage (optional cleanup)
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"name": model_name}
            response = requests.delete(f"{self.host}/api/delete", json=payload)
            
            if response.status_code == 200:
                self.logger.info(f"Successfully removed model: {model_name}")
                return True
            else:
                self.logger.error(f"Failed to remove model {model_name}: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing model {model_name}: {e}")
            return False


def example_usage():
    """Example usage of the Ollama runner"""
    
    # Initialize runner
    runner = OllamaRunner()
    
    # Check if Ollama is running
    if not runner.check_ollama_status():
        print("Ollama server is not running. Please start Ollama first.")
        return
    
    # List available models
    models = runner.list_available_models()
    print(f"Available models: {models}")
    
    # Example prompt
    prompt = "Explain the concept of machine learning in simple terms."
    
    # Generate response (if you have a model available)
    if models:
        model_name = models[0]  # Use first available model
        try:
            response, metrics = runner.generate_response(model_name, prompt)
            print(f"\nResponse: {response}")
            print(f"\nPerformance Metrics:")
            print(f"  Latency: {metrics.latency_ms:.2f}ms")
            print(f"  Memory: {metrics.memory_used_mb:.2f}MB")
            print(f"  CPU: {metrics.cpu_usage_percent:.1f}%")
            print(f"  Tokens: {metrics.token_count}")
            print(f"  Tokens/sec: {metrics.tokens_per_second:.2f}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No models available. Please pull a model first using 'ollama pull <model_name>'.")


if __name__ == "__main__":
    example_usage() 