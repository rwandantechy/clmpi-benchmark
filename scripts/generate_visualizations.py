"""
Generate visualizations for CLMPI benchmark results

Creates radar charts and bar charts from evaluation results.
Uses matplotlib for consistent, publication-ready charts.

Example:
    python scripts/generate_visualizations.py --input results/2024-12-19_143022_demo --output evaluations/
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


def load_results(results_dir: Path) -> List[Dict]:
    """
    Load results from a run directory
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of model result dictionaries
    """
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
        return data.get('results', [])


def create_radar_chart(results: List[Dict], output_path: Path):
    """
    Create radar chart comparing component scores across models
    
    Args:
        results: List of model result dictionaries
        output_path: Path to save the chart
    """
    # Extract component names and scores
    components = ['accuracy', 'contextual_understanding', 'coherence', 'fluency', 'performance_efficiency']
    component_labels = ['Accuracy', 'Contextual\nUnderstanding', 'Coherence', 'Fluency', 'Performance\nEfficiency']
    
    # Number of variables
    num_vars = len(components)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        model_name = result['model_name']
        scores = []
        
        for component in components:
            # Normalize scores to [0,1] if needed
            score = result['component_scores'][component]
            if component in ['contextual_understanding', 'coherence', 'fluency']:
                score = score / 5.0  # Normalize from [0,5] to [0,1]
            scores.append(score)
        
        scores += scores[:1]  # Complete the circle
        
        ax.plot(angles, scores, 'o-', linewidth=2, label=model_name, color=colors[i])
        ax.fill(angles, scores, alpha=0.25, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('CLMPI Component Scores Comparison', pad=20, size=16, weight='bold')
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_bar_chart(results: List[Dict], output_path: Path):
    """
    Create bar chart comparing overall CLMPI scores
    
    Args:
        results: List of model result dictionaries
        output_path: Path to save the chart
    """
    # Extract model names and CLMPI scores
    model_names = [result['model_name'] for result in results]
    clmpi_scores = [result['clmpi_score_100'] for result in results]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars
    bars = ax.bar(model_names, clmpi_scores, color=plt.cm.Set3(np.linspace(0, 1, len(results))))
    
    # Customize the plot
    ax.set_xlabel('Models', fontsize=12, weight='bold')
    ax.set_ylabel('CLMPI Score (0-100)', fontsize=12, weight='bold')
    ax.set_title('CLMPI Scores Comparison', fontsize=16, weight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, clmpi_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels if needed
    if len(model_names) > 4:
        plt.xticks(rotation=45, ha='right')
    
    # Set y-axis limits
    ax.set_ylim(0, max(clmpi_scores) * 1.1)
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def update_excel_scorebook(results: List[Dict], output_path: Path):
    """
    Update Excel scorebook with new results
    
    Args:
        results: List of model result dictionaries
        output_path: Path to Excel file
    """
    # Prepare data for Excel
    data = []
    for result in results:
        row = {
            'Model': result['model_name'],
            'CLMPI_Score_100': result['clmpi_score_100'],
            'CLMPI_Score_01': result['clmpi_score_01'],
            'Accuracy': result['component_scores']['accuracy'],
            'Contextual_Understanding': result['component_scores']['contextual_understanding'],
            'Coherence': result['component_scores']['coherence'],
            'Fluency': result['component_scores']['fluency'],
            'Performance_Efficiency': result['component_scores']['performance_efficiency']
        }
        data.append(row)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False, sheet_name='CLMPI_Results')


def main():
    """Main function for generating visualizations"""
    parser = argparse.ArgumentParser(description='Generate CLMPI benchmark visualizations')
    parser.add_argument('--input', type=str, required=True, 
                       help='Path to results directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output directory for visualizations')
    
    args = parser.parse_args()
    
    # Load results
    results_dir = Path(args.input)
    results = load_results(results_dir)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print(f"Generating visualizations for {len(results)} models...")
    
    # Radar chart
    radar_path = output_dir / "visualizations" / "component_scores_radar.png"
    create_radar_chart(results, radar_path)
    print(f"Radar chart saved to: {radar_path}")
    
    # Bar chart
    bar_path = output_dir / "visualizations" / "clmpi_scores_bar.png"
    create_bar_chart(results, bar_path)
    print(f"Bar chart saved to: {bar_path}")
    
    # Update Excel scorebook
    excel_path = output_dir / "clmpi_scorebook.xlsx"
    update_excel_scorebook(results, excel_path)
    print(f"Excel scorebook updated: {excel_path}")
    
    print("Visualization generation complete!")


if __name__ == "__main__":
    main()
