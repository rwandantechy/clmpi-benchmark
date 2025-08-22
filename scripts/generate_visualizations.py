"""
Generate intuitive visualizations for CLMPI benchmark results

Creates professional, publication-ready charts optimized for benchmarking data:
- Horizontal bar charts for easy model comparison
- Heatmaps for component score analysis  
- Performance matrices for edge deployment insights

Example:
    python scripts/generate_visualizations.py --input results/2024-12-19_143022_demo --output evaluations/
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_results(results_dir: Path) -> List[Dict]:
    """
    Load results from a run directory
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of model result dictionaries
    """
    # Try stepwise CLMPI summary first
    clmpi_summary = results_dir / "clmpi_summary.json"
    if clmpi_summary.exists():
        with open(clmpi_summary, 'r') as f:
            data = json.load(f)
            return [data]  # Single model for stepwise
    
    # Try legacy summary format
    summary_file = results_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data = json.load(f)
            return data.get('results', [])
    
    raise FileNotFoundError(f"No results found in {results_dir}")


def create_horizontal_bar_chart(results: List[Dict], output_path: Path):
    """
    Create horizontal bar chart for CLMPI scores - most intuitive for model comparison
    
    Horizontal bars are easier to read and compare, especially with long model names
    """
    # Extract model names and CLMPI scores
    model_names = [result['model'] for result in results]
    clmpi_scores = [result['clmpi_scores']['clmpi_25'] for result in results]
    
    # Sort by score for better visualization
    sorted_data = sorted(zip(model_names, clmpi_scores), key=lambda x: x[1], reverse=True)
    model_names, clmpi_scores = zip(*sorted_data)
    
    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(12, max(6, len(model_names) * 0.8)))
    
    # Create horizontal bars with color gradient
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))
    bars = ax.barh(model_names, clmpi_scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add value labels on bars
    for bar, score in zip(bars, clmpi_scores):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', ha='left', va='center', fontweight='bold', fontsize=11)
    
    # Customize the plot
    ax.set_xlabel('CLMPI Score (0-25)', fontsize=14, weight='bold')
    ax.set_title('CLMPI Benchmark Results - Model Performance Ranking', 
                fontsize=16, weight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 25)
    
    # Add reference lines for performance tiers
    ax.axvline(x=20, color='green', linestyle='--', alpha=0.7, label='Excellent (≥20)')
    ax.axvline(x=15, color='orange', linestyle='--', alpha=0.7, label='Good (≥15)')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Fair (≥10)')
    ax.legend(loc='lower right')
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_component_heatmap(results: List[Dict], output_path: Path):
    """
    Create heatmap for component scores - intuitive for identifying strengths/weaknesses
    """
    # Extract component scores
    components = ['accuracy', 'contextual_understanding', 'coherence', 'fluency', 'performance_efficiency']
    component_labels = ['Accuracy', 'Context', 'Coherence', 'Fluency', 'Efficiency']
    
    # Prepare data for heatmap
    data = []
    model_names = []
    
    for result in results:
        model_names.append(result['model'])
        row = []
        for component in components:
            score = result['clmpi_scores']['component_scores'][component]
            row.append(score)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, index=model_names, columns=component_labels)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(model_names) * 0.6)))
    
    # Use diverging colormap for better interpretation
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                cbar_kws={'label': 'Score (0-1)'}, ax=ax, linewidths=0.5)
    
    ax.set_title('CLMPI Component Score Analysis', fontsize=16, weight='bold', pad=20)
    ax.set_xlabel('Evaluation Dimensions', fontsize=12, weight='bold')
    ax.set_ylabel('Models', fontsize=12, weight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_performance_matrix(results: List[Dict], output_path: Path):
    """
    Create performance matrix showing quality vs efficiency trade-offs
    """
    # Calculate quality score (average of accuracy, context, coherence, fluency)
    quality_scores = []
    efficiency_scores = []
    model_names = []
    
    for result in results:
        scores = result['clmpi_scores']['component_scores']
        quality = (scores['accuracy'] + scores['contextual_understanding'] + 
                  scores['coherence'] + scores['fluency']) / 4
        efficiency = scores['performance_efficiency']
        
        quality_scores.append(quality)
        efficiency_scores.append(efficiency)
        model_names.append(result['model'])
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with model labels
    scatter = ax.scatter(efficiency_scores, quality_scores, s=200, alpha=0.7, 
                        c=range(len(results)), cmap='viridis', edgecolors='white', linewidth=2)
    
    # Add model labels
    for i, model in enumerate(model_names):
        ax.annotate(model, (efficiency_scores[i], quality_scores[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10, weight='bold')
    
    # Add quadrant lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add quadrant labels
    ax.text(0.25, 0.75, 'High Quality\nLow Efficiency', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    ax.text(0.75, 0.75, 'High Quality\nHigh Efficiency', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    ax.text(0.25, 0.25, 'Low Quality\nLow Efficiency', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    ax.text(0.75, 0.25, 'Low Quality\nHigh Efficiency', ha='center', va='center', 
            fontsize=12, weight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel('Performance Efficiency Score', fontsize=14, weight='bold')
    ax.set_ylabel('Quality Score (Avg of ACC, CON, COH, FLU)', fontsize=14, weight='bold')
    ax.set_title('CLMPI Performance Matrix: Quality vs Efficiency', fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def create_edge_deployment_chart(results: List[Dict], output_path: Path):
    """
    Create specialized chart for edge deployment decision making
    """
    # Extract efficiency metrics
    model_names = []
    latency_scores = []
    memory_scores = []
    clmpi_scores = []
    
    for result in results:
        model_names.append(result['model'])
        clmpi_scores.append(result['clmpi_scores']['clmpi_25'])
        
        # Get efficiency details from individual metric summaries
        efficiency_summary = result.get('individual_metrics', {}).get('performance_efficiency', {})
        latency_scores.append(efficiency_summary.get('avg_latency_seconds', 0))
        memory_scores.append(efficiency_summary.get('avg_memory_mb', 0))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Latency comparison
    bars1 = ax1.bar(model_names, latency_scores, color='skyblue', alpha=0.8, edgecolor='navy')
    ax1.set_title('Response Latency Comparison', fontsize=14, weight='bold')
    ax1.set_ylabel('Average Latency (seconds)', fontsize=12, weight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, latency_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}s', ha='center', va='bottom', fontweight='bold')
    
    # Memory usage comparison
    bars2 = ax2.bar(model_names, memory_scores, color='lightcoral', alpha=0.8, edgecolor='darkred')
    ax2.set_title('Memory Usage Comparison', fontsize=14, weight='bold')
    ax2.set_ylabel('Average Memory (MB)', fontsize=12, weight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars2, memory_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score:.0f}MB', ha='center', va='bottom', fontweight='bold')
    
    # Rotate x-axis labels
    for ax in [ax1, ax2]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('Edge Deployment Performance Metrics', fontsize=16, weight='bold', y=1.02)
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def update_excel_scorebook(results: List[Dict], output_path: Path):
    """
    Update Excel scorebook with comprehensive results
    """
    # Prepare data for Excel
    data = []
    for result in results:
        clmpi_scores = result['clmpi_scores']
        component_scores = clmpi_scores['component_scores']
        
        row = {
            'Model': result['model'],
            'CLMPI_Score_25': clmpi_scores['clmpi_25'],
            'CLMPI_Score_01': clmpi_scores['clmpi_01'],
            'Accuracy': component_scores['accuracy'],
            'Contextual_Understanding': component_scores['contextual_understanding'],
            'Coherence': component_scores['coherence'],
            'Fluency': component_scores['fluency'],
            'Performance_Efficiency': component_scores['performance_efficiency']
        }
        data.append(row)
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='CLMPI_Results', index=False)
        
        # Add summary statistics
        summary_data = {
            'Metric': ['Best Model', 'Worst Model', 'Average Score', 'Score Range'],
            'CLMPI_25': [
                df['CLMPI_Score_25'].max(),
                df['CLMPI_Score_25'].min(),
                df['CLMPI_Score_25'].mean(),
                df['CLMPI_Score_25'].max() - df['CLMPI_Score_25'].min()
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary_Stats', index=False)


def main():
    """Main function for generating visualizations"""
    parser = argparse.ArgumentParser(description='Generate intuitive CLMPI benchmark visualizations')
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
    print(f"Generating intuitive visualizations for {len(results)} models...")
    
    # 1. Horizontal bar chart - most intuitive for model comparison
    bar_path = output_dir / "visualizations" / "clmpi_scores_horizontal.png"
    create_horizontal_bar_chart(results, bar_path)
    print(f"✓ Horizontal bar chart saved to: {bar_path}")
    
    # 2. Component heatmap - intuitive for identifying strengths/weaknesses
    heatmap_path = output_dir / "visualizations" / "component_scores_heatmap.png"
    create_component_heatmap(results, heatmap_path)
    print(f"✓ Component heatmap saved to: {heatmap_path}")
    
    # 3. Performance matrix - intuitive for quality vs efficiency trade-offs
    matrix_path = output_dir / "visualizations" / "performance_matrix.png"
    create_performance_matrix(results, matrix_path)
    print(f"✓ Performance matrix saved to: {matrix_path}")
    
    # 4. Edge deployment chart - specialized for resource-constrained devices
    edge_path = output_dir / "visualizations" / "edge_deployment_metrics.png"
    create_edge_deployment_chart(results, edge_path)
    print(f"✓ Edge deployment chart saved to: {edge_path}")
    
    # 5. Update Excel scorebook
    excel_path = output_dir / "clmpi_scorebook.xlsx"
    update_excel_scorebook(results, excel_path)
    print(f"✓ Excel scorebook updated: {excel_path}")
    
    print("\n🎯 Visualization generation complete!")
    print("📊 Charts optimized for intuitive interpretation and professional presentation")


if __name__ == "__main__":
    main()
