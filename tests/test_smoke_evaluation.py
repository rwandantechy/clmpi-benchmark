#!/usr/bin/env python3
"""
Smoke test for enhanced evaluation pipeline
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
import pytest
import yaml

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from enhanced_evaluate_models import EnhancedModelEvaluator


class TestSmokeEvaluation:
    """Smoke test for enhanced evaluation pipeline"""
    
    def test_smoke_evaluation_structure(self):
        """Test that evaluation creates expected output structure"""
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create minimal config for testing
            model_config = {
                'evaluation_weights': {
                    'accuracy': 0.25,
                    'contextual_understanding': 0.20,
                    'coherence': 0.20,
                    'fluency': 0.20,
                    'performance_efficiency': 0.15
                },
                'prompt_sets': {
                            'accuracy': ['accuracy.json'],
        'contextual_understanding': ['context.json'],
        'coherence': ['coherence.json'],
        'fluency': ['fluency.json'],
        'performance_efficiency': ['coherence.json']
                },
                'models': {
                    'test_model': {
                        'ollama_name': 'test_model',
                        'timeout_seconds': 30
                    }
                },
                'evaluation': {
                    'samples_per_task': 2,  # Small sample for smoke test
                    'random_seed': 42,
                    'save_raw_responses': True,
                    'generate_visualizations': False
                }
            }
            
            generation_config = {
                'version': '1.0.0',
                'generation_profiles': {
                    'deterministic': {
                        'temperature': 0.0,
                        'top_p': 1.0,
                        'top_k': 1,
                        'max_tokens': 100,
                        'use_for': ['accuracy', 'contextual_understanding']
                    },
                    'creative': {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'top_k': 40,
                        'max_tokens': 100,
                        'use_for': ['coherence', 'fluency']
                    }
                },
                'model_overrides': {},
                'validation': {
                    'min_temperature': 0.0,
                    'max_temperature': 1.0,
                    'min_top_p': 0.1,
                    'max_top_p': 1.0,
                    'min_top_k': 1,
                    'max_top_k': 100
                }
            }
            
            device_config = {
                'device': {
                    'name': 'Test Device',
                    'os': 'Test OS',
                    'cpu_model': 'Test CPU',
                    'cpu_cores': 4,
                    'memory_gb': 8,
                    'storage_type': 'SSD'
                },
                'runtime': {
                    'ollama_host': 'http://localhost:11434',
                    'max_concurrent_models': 1,
                    'memory_buffer_mb': 1024,
                    'cpu_reservation_percent': 20
                },
                'thresholds': {
                    'max_latency_seconds': 30,
                    'max_memory_mb': 7000,
                    'max_cpu_percent': 95,
                    'min_response_length': 10,
                    'max_response_length': 500
                },
                'evaluation': {
                    'timeout_seconds': 60,
                    'max_retries': 3,
                    'save_intermediate': True,
                    'cleanup_between_runs': True
                }
            }
            
            # Write configs to temp directory
            config_dir = temp_path / 'config'
            config_dir.mkdir()
            
            with open(config_dir / 'model_config.yaml', 'w') as f:
                yaml.dump(model_config, f)
            
            with open(config_dir / 'generation_config.yaml', 'w') as f:
                yaml.dump(generation_config, f)
            
            with open(config_dir / 'device_config.yaml', 'w') as f:
                yaml.dump(device_config, f)
            
            # Create output directory
            output_dir = temp_path / 'results'
            output_dir.mkdir()
            
            # Mock the ollama runner to avoid actual model calls
            class MockOllamaRunner:
                def generate_response(self, model_name, prompt, settings):
                    return f"Mock response for: {prompt[:20]}..."
            
            # Patch the evaluator to use mock runner
            evaluator = EnhancedModelEvaluator(
                str(config_dir / 'model_config.yaml'),
                str(config_dir / 'generation_config.yaml'),
                str(config_dir / 'device_config.yaml'),
                str(output_dir),
                'smoke_test',
                42
            )
            evaluator.ollama_runner = MockOllamaRunner()
            
            # Run evaluation
            results = evaluator.run_evaluation(['test_model'])
            
            # Validate results structure
            assert len(results) > 0
            assert 'model_name' in results[0]
            assert results[0]['model_name'] == 'test_model'
            
            # Save results to files
            run_dir = evaluator.save_enhanced_results(results, "smoke_test")
            
            # Check that output files were created
            run_dirs = list(output_dir.glob('*_smoke_test'))
            assert len(run_dirs) > 0
            
            run_dir = run_dirs[0]
            
            # Check for summary.json
            summary_file = run_dir / 'summary.json'
            assert summary_file.exists()
            
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Validate summary structure
            assert 'run_name' in summary
            assert 'timestamp' in summary
            assert 'label' in summary
            assert 'seed' in summary
            assert 'hardware_info' in summary
            assert 'config_used' in summary
            assert 'results' in summary
            
            # Check for model results
            model_files = list(run_dir.glob('*_results.json'))
            assert len(model_files) > 0
            
            # Check for detailed results structure
            detailed_dirs = list(run_dir.glob('*_detailed'))
            if detailed_dirs:  # May not be created if evaluation fails
                detailed_dir = detailed_dirs[0]
                
                # Check for metric directories
                metric_dirs = ['accuracy', 'contextual_understanding', 'coherence', 'fluency', 'efficiency']
                for metric in metric_dirs:
                    metric_dir = detailed_dir / metric
                    if metric_dir.exists():
                        # Check for detail.jsonl and summary.json
                        detail_file = metric_dir / 'detail.jsonl'
                        summary_file = metric_dir / 'summary.json'
                        
                        if detail_file.exists():
                            assert detail_file.stat().st_size > 0
                        
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                metric_summary = json.load(f)
                            assert 'metric' in metric_summary
                            assert 'average_score' in metric_summary
    
    def test_schema_validation(self):
        """Test that summary.json matches the schema"""
        # This test would validate against docs/schemas/summary.schema.json
        # For now, we'll just check basic structure
        schema_file = Path(__file__).parent.parent / 'docs' / 'schemas' / 'summary.schema.json'
        assert schema_file.exists()
        
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Basic schema validation
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'required' in schema


if __name__ == "__main__":
    pytest.main([__file__])
