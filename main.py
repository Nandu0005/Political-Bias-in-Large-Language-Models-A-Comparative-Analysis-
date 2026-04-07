"""
main.py
Main entry point for the Political Bias Analysis Pipeline
Runs complete pipeline for all datasets
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

from config_loader import config
from collect_responses import collect_all_datasets
from preprocessing import ResponsePreprocessor
from bias_scoring import BiasScorer, score_all_datasets
from compare_models import ModelComparer, compare_all_datasets
from visualize_results import PaperStyleVisualizer

ROOT = Path(__file__).parent

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.get_log_level()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BiasAnalysisPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self.models = config.models
        self.weights = config.weights
        
        logger.info(f"Initializing pipeline (simulate={simulate})")
        logger.info(f"Models: {self.models}")
        logger.info(f"Weights: {self.weights}")
    
    def print_header(self, text: str):
        """Print formatted header"""
        print("\n" + "=" * 70)
        print(f" {text}")
        print("=" * 70)
    
    def run_full_pipeline(self):
        """Run complete pipeline for all datasets"""
        
        start_time = datetime.now()
        
        self.print_header("STEP 1: COLLECTING RESPONSES")
        if self.simulate:
            print("Mode: SIMULATED responses (no API calls)")
        else:
            print("Mode: REAL API calls")
        
        collect_all_datasets(simulate=self.simulate)
        
        self.print_header("STEP 2: PREPROCESSING RESPONSES")
        preprocessor = ResponsePreprocessor()
        preprocessor.process_all_datasets()
        
        self.print_header("STEP 3: CALCULATING BIAS SCORES")
        scorer = BiasScorer(self.weights)
        score_all_datasets()
        
        self.print_header("STEP 4: STATISTICAL COMPARISON")
        comparer = ModelComparer()
        compare_all_datasets()
        
        self.print_header("STEP 5: GENERATING PAPER FIGURES")
        visualizer = PaperStyleVisualizer()
        visualizer.generate_all_figures()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Total time: {duration:.2f} seconds")
        print(f"\nOutputs saved in:")
        print("  📁 results/pew/")
        print("  📁 results/compass/")
        print("  📁 results/isidewith/")
        print("  📁 results/basepaper/")
        print("\nGenerated Figures:")
        print("  📊 Figure 7  - pew/fig7_pew_typology.png")
        print("  📊 Figure 8  - compass/fig8_9_political_compass.png")
        print("  📊 Figure 9  - compass/fig8_9_political_compass.png")
        print("  📊 Figure 10 - compass/fig10_combined_compass.png")
        print("  📊 Figure 11 - isidewith/fig11_isidewith_heatmap.png")
        print("  📊 Figure 12 - isidewith/fig12_isidewith_linechart.png")
        print("  📊 Figure 13 - basepaper/fig13_ideological_percentages.png")

# Add this to your main.py

def generate_bias_with_images():
    """
    Run complete bias analysis and generate images
    """
    print("\n" + "="*70)
    print("📊 GENERATING BIAS SCORES AND IMAGES")
    print("="*70)
    
    # Import and run the complete pipeline
    from generate_all_bias import main as bias_main
    bias_main()
    
    print("\n✅ Bias scores and images generated successfully!")
    print("   Check results folder for CSV files and images")
def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Political Bias Analysis Pipeline')
    parser.add_argument('--simulate', action='store_true', 
                       help='Use simulated responses (no API calls)')
    parser.add_argument('--real', action='store_true',
                       help='Use real API calls (requires API keys)')
    parser.add_argument('--dataset', type=str, 
                       choices=['pew', 'compass', 'isidewith', 'basepaper', 'all'],
                       default='all', help='Dataset to process')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip collection (use existing raw files)')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Determine simulate mode
    if args.real:
        simulate = False
    elif args.simulate:
        simulate = True
    else:
        simulate = config.get_simulate_mode()
    
    # Print banner
    print("\n" + "=" * 70)
    print(" POLITICAL BIAS ANALYSIS IN LARGE LANGUAGE MODELS")
    print(" Based on IEEE Access Paper (January 2025)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Models: ChatGPT-4, Claude, Google Gemini, Perplexity")
    print(f"  • Mode: {'SIMULATED' if simulate else 'REAL API'}")
    print(f"  • Dataset: {args.dataset}")
    print("=" * 70 + "\n")
    
    # Run pipeline
    pipeline = BiasAnalysisPipeline(simulate=simulate)
    
    if args.dataset == 'all':
        pipeline.run_full_pipeline()
    else:
        logger.error("Single dataset processing not implemented in this version")
        logger.info("Use --dataset all for complete analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()