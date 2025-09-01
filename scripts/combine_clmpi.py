#!/usr/bin/env python3
"""
Automated CLMPI score combination script
Reads individual dimension results and calculates the combined CLMPI score
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLMPICombiner:
    """Automated CLMPI score combination"""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'accuracy': 0.25,
            'contextual_understanding': 0.20,
            'coherence': 0.20,
            'fluency': 0.20,
            'performance_efficiency': 0.15
        }
        self._validate_weights()
    
    def _validate_weights(self):
        """Ensure weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    def read_dimension_results(self, results_dir: Path) -> Dict[str, Any]:
        """Read all dimension results from the results directory"""
        dimension_scores = {}
        dimension_details = {}
        
        # Expected dimension directories
        dimensions = ['accuracy', 'context', 'coherence', 'fluency', 'efficiency']
        
        for dim in dimensions:
            summary_file = results_dir / dim / 'summary.json'
            if not summary_file.exists():
                logger.warning(f"Summary file not found for {dim}: {summary_file}")
                continue
            
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                # Extract the main score for each dimension
                if dim == 'accuracy':
                    score = summary.get('f1_score', 0.0)
                elif dim == 'context':
                    score = summary.get('combined_score', 0.0)
                elif dim == 'coherence':
                    score = summary.get('coherence_score', 0.0)
                elif dim == 'fluency':
                    score = summary.get('fluency_score', 0.0)
                elif dim == 'efficiency':
                    score = summary.get('efficiency', 0.0)
                else:
                    score = 0.0
                
                dimension_scores[dim] = score
                dimension_details[dim] = summary
                logger.info(f"Loaded {dim}: {score}")
                
            except Exception as e:
                logger.error(f"Error reading {dim} results: {e}")
                dimension_scores[dim] = 0.0
                dimension_details[dim] = {}
        
        return dimension_scores, dimension_details
    
    def calculate_clmpi(self, dimension_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate the combined CLMPI score"""
        
        # Map dimension names to weight keys
        dimension_mapping = {
            'accuracy': 'accuracy',
            'context': 'contextual_understanding',
            'coherence': 'coherence',
            'fluency': 'fluency',
            'efficiency': 'performance_efficiency'
        }
        
        # Calculate weighted contributions
        contributions = {}
        total_score = 0.0
        
        for dim, score in dimension_scores.items():
            weight_key = dimension_mapping.get(dim, dim)
            weight = self.weights.get(weight_key, 0.0)
            contribution = weight * score
            contributions[dim] = {
                'score': score,
                'weight': weight,
                'contribution': contribution
            }
            total_score += contribution
        
        # Calculate CLMPI scores in different scales
        clmpi_01 = total_score  # 0-1 scale
        clmpi_25 = total_score * 25  # 0-25 scale
        clmpi_100 = total_score * 100  # 0-100 scale
        
        return {
            'clmpi_01': clmpi_01,
            'clmpi_25': clmpi_25,
            'clmpi_100': clmpi_100,
            'component_scores': contributions,
            'total_score': total_score
        }
    
    def create_summary(self, results_dir: Path, dimension_scores: Dict[str, float], 
                      dimension_details: Dict[str, Any], clmpi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive CLMPI summary"""
        
        # Get model name and timestamp from first available dimension
        model_name = "unknown"
        timestamp = "unknown"
        evaluation_run = results_dir.name
        
        for dim_details in dimension_details.values():
            if dim_details:
                model_name = dim_details.get('model', model_name)
                timestamp = dim_details.get('timestamp', timestamp)
                break
        
        # Create generation profiles mapping
        generation_profiles = {}
        datasets_used = {}
        
        for dim, details in dimension_details.items():
            if details:
                generation_profiles[dim] = details.get('generation_profile', 'unknown')
                datasets_used[dim] = details.get('dataset_path', 'unknown')
        
        return {
            'model': model_name,
            'timestamp': timestamp,
            'evaluation_run': evaluation_run,
            
            'clmpi_scores': {
                'clmpi_01': clmpi_results['clmpi_01'],
                'clmpi_25': clmpi_results['clmpi_25'],
                'clmpi_100': clmpi_results['clmpi_100']
            },
            
            'component_scores': clmpi_results['component_scores'],
            
            'weights_used': self.weights,
            
            'detailed_results': dimension_details,
            
            'evaluation_metadata': {
                'generation_profiles_used': generation_profiles,
                'datasets_used': datasets_used
            }
        }
    
    def combine_results(self, results_dir: Path) -> Dict[str, Any]:
        """Main method to combine all CLMPI results"""
        logger.info(f"Combining CLMPI results from: {results_dir}")
        
        # Read dimension results
        dimension_scores, dimension_details = self.read_dimension_results(results_dir)
        
        # Calculate CLMPI
        clmpi_results = self.calculate_clmpi(dimension_scores)
        
        # Create summary
        summary = self.create_summary(results_dir, dimension_scores, dimension_details, clmpi_results)
        
        # Save summary
        output_file = results_dir / 'clmpi_summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"CLMPI summary saved to: {output_file}")
        logger.info(f"Final CLMPI score: {clmpi_results['clmpi_01']:.3f} (0-1 scale)")
        logger.info(f"Final CLMPI score: {clmpi_results['clmpi_25']:.3f} (0-25 scale)")
        logger.info(f"Final CLMPI score: {clmpi_results['clmpi_100']:.1f} (0-100 scale)")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description='Combine CLMPI dimension results into final score')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory containing dimension evaluations')
    parser.add_argument('--weights', type=str, default=None,
                       help='JSON string of custom weights (optional)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1
    
    # Parse custom weights if provided
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid weights JSON: {e}")
            return 1
    
    # Combine results
    try:
        combiner = CLMPICombiner(weights)
        summary = combiner.combine_results(results_dir)
        logger.info("CLMPI combination completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Error combining CLMPI results: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
