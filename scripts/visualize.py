#!/usr/bin/env python3
"""
CLMPI Results Visualization
Creates charts and graphs from CLMPI evaluation results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
import numpy as np

def load_clmpi_results(results_dir):
    """Load CLMPI results from a results directory"""
    clmpi_file = results_dir / "clmpi_summary.json"
    if not clmpi_file.exists():
        print(f"Error: CLMPI summary not found in {results_dir}")
        return None
    
    with open(clmpi_file, 'r') as f:
        return json.load(f)

def create_component_chart(results, output_dir, model_name):
    """Create component scores bar chart"""
    component_scores = results.get('component_scores', {})
    
    if not component_scores:
        print("No component scores found")
        return
    
    # Extract data
    components = list(component_scores.keys())
    scores = [comp['score'] for comp in component_scores.values()]
    weights = [comp['weight'] for comp in component_scores.values()]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Component scores
    bars1 = ax1.bar(components, scores, color='skyblue', alpha=0.7)
    ax1.set_title(f'CLMPI Component Scores - {model_name}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score (0-1)', fontsize=12)
    ax1.set_ylim(0, 1.1)
    
    # Add score labels on bars
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Component weights
    bars2 = ax2.bar(components, weights, color='lightcoral', alpha=0.7)
    ax2.set_title(f'Component Weights in CLMPI - {model_name}', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.set_ylim(0, max(weights) * 1.2)
    
    # Add weight labels on bars
    for bar, weight in zip(bars2, weights):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_file = output_dir / f"component_scores_{model_name.replace(':', '_')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Component chart saved: {chart_file}")
    plt.close()

def create_clmpi_scale_chart(results, output_dir, model_name):
    """Create CLMPI scores across different scales"""
    clmpi_scores = results.get('clmpi_scores', {})
    
    if not clmpi_scores:
        print("No CLMPI scores found")
        return
    
    scales = list(clmpi_scores.keys())
    scores = list(clmpi_scores.values())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(scales, scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    ax.set_title(f'CLMPI Scores Across Different Scales - {model_name}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save chart
    chart_file = output_dir / f"clmpi_scales_{model_name.replace(':', '_')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"CLMPI scales chart saved: {chart_file}")
    plt.close()

def create_radar_chart(results, output_dir, model_name):
    """Create radar chart for component scores"""
    component_scores = results.get('component_scores', {})
    
    if not component_scores:
        print("No component scores found")
        return
    
    # Extract data
    components = list(component_scores.keys())
    scores = [comp['score'] for comp in component_scores.values()]
    
    # Number of variables
    num_vars = len(components)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * 3.14159 for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Add the first score again to complete the circle
    scores += scores[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, scores, 'o-', linewidth=2, color='#FF6B6B')
    ax.fill(angles, scores, alpha=0.25, color='#FF6B6B')
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(components, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    
    ax.set_title(f'CLMPI Component Performance - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True)
    
    # Save chart
    chart_file = output_dir / f"radar_chart_{model_name.replace(':', '_')}.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved: {chart_file}")
    plt.close()

def create_comparison_charts(results_dir, output_dir):
    """Create comparison charts when multiple models are available"""
    # Find all CLMPI summary files
    summary_files = list(results_dir.glob("*/clmpi_summary.json"))
    
    if len(summary_files) < 2:
        print("Need at least 2 models for comparison charts")
        return
    
    print(f"Found {len(summary_files)} models for comparison")
    
    # Load all results
    all_results = []
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r') as f:
                result = json.load(f)
                # Extract model name
                model_name = result.get('model', 'Unknown')
                if model_name == 'unknown':
                    # Try to get from evaluation_run
                    evaluation_run = result.get('evaluation_run', 'Unknown')
                    if '_' in evaluation_run:
                        model_name = evaluation_run.split('_')[0].replace('_', ' ')
                    else:
                        model_name = evaluation_run
                
                result['display_name'] = model_name
                all_results.append(result)
        except Exception as e:
            print(f"Error loading {summary_file}: {e}")
            continue
    
    if len(all_results) < 2:
        print("Could not load enough models for comparison")
        return
    
    # Create comparison charts
    create_model_comparison_chart(all_results, output_dir)
    create_clmpi_comparison_chart(all_results, output_dir)

def create_model_comparison_chart(all_results, output_dir):
    """Create comparison chart of component scores across models"""
    # Extract data
    models = [r['display_name'] for r in all_results]
    components = ['accuracy', 'context', 'coherence', 'fluency', 'efficiency']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar positions
    x = np.arange(len(components))
    width = 0.8 / len(models)
    
    # Create bars for each model
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for i, (result, color) in enumerate(zip(all_results, colors)):
        scores = []
        for comp in components:
            comp_data = result.get('component_scores', {}).get(comp, {})
            scores.append(comp_data.get('score', 0.0))
        
        bar_pos = x + (i - len(models)/2 + 0.5) * width
        bars = ax.bar(bar_pos, scores, width, label=result['display_name'], 
                     color=color, alpha=0.8)
        
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Components', fontsize=12)
    ax.set_ylabel('Score (0-1)', fontsize=12)
    ax.set_title('CLMPI Component Scores Comparison Across Models', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([comp.title() for comp in components])
    ax.set_ylim(0, 1.1)
    ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_file = output_dir / "model_comparison.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Model comparison chart saved: {chart_file}")
    plt.close()

def create_clmpi_comparison_chart(all_results, output_dir):
    """Create comparison chart of overall CLMPI scores across models"""
    # Extract data
    models = [r['display_name'] for r in all_results]
    clmpi_100_scores = [r.get('clmpi_scores', {}).get('clmpi_100', 0.0) for r in all_results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, clmpi_100_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'], alpha=0.8)
    ax.set_title('Overall CLMPI Scores Comparison (Scale 0-100)', fontsize=16, fontweight='bold')
    ax.set_ylabel('CLMPI Score', fontsize=12)
    
    # Add score labels on bars
    for bar, score in zip(bars, clmpi_100_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(clmpi_100_scores) * 0.01,
               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save chart
    chart_file = output_dir / "clmpi_comparison.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"CLMPI comparison chart saved: {chart_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize CLMPI benchmark results')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Path to results directory (default: most recent)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for charts (default: visualizations)')
    
    args = parser.parse_args()
    
    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Find most recent stepwise directory
        results_base = Path("results")
        if not results_base.exists():
            print("Error: No results directory found")
            return 1
        
        stepwise_dirs = list(results_base.glob("*_stepwise"))
        if not stepwise_dirs:
            print("Error: No stepwise evaluation results found")
            return 1
        
        results_dir = max(stepwise_dirs, key=lambda p: p.stat().st_mtime)
        print(f"Using results from: {results_dir}")
    
    # Load results
    results = load_clmpi_results(results_dir)
    if not results:
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Extract model name
    model_name = results.get('model', 'Unknown Model')
    if model_name == 'unknown':
        # Try to get from evaluation_run
        evaluation_run = results.get('evaluation_run', 'Unknown')
        if '_' in evaluation_run:
            model_name = evaluation_run.split('_')[0].replace('_', ' ')
        else:
            model_name = evaluation_run
    
    print(f"Model: {model_name}")
    
    try:
        # Create individual model charts
        create_component_chart(results, output_dir, model_name)
        create_clmpi_scale_chart(results, output_dir, model_name)
        create_radar_chart(results, output_dir, model_name)
        
        print(f"\nIndividual model charts saved to: {output_dir}")
        print("Files created:")
        print(f"  - component_scores_{model_name.replace(':', '_')}.png")
        print(f"  - clmpi_scales_{model_name.replace(':', '_')}.png")
        print(f"  - radar_chart_{model_name.replace(':', '_')}.png")
        
        # Check if we can create comparison charts
        print("\nChecking for multiple models...")
        create_comparison_charts(results_dir, output_dir)
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
