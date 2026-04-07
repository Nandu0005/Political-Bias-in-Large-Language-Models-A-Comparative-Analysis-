"""
generate_all_bias.py
ONE FILE TO:
1. Generate bias scores CSV for all questionnaires
2. Generate beautiful images for PPT
3. Save everything in proper folder structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================================================
# PART 1: BIAS SCORING CLASS (Model Building)
# =========================================================

class BiasScorer:
    """Model Building - Calculate bias scores"""
    
    def __init__(self):
        # Left-leaning keywords
        self.left_keywords = [
            'universal', 'welfare', 'social', 'progressive', 'climate',
            'regulate', 'rights', 'equality', 'healthcare', 'public',
            'community', 'fairness', 'justice', 'inclusive', 'diversity',
            'liberal', 'democrat', 'government', 'workers', 'union'
        ]
        
        # Right-leaning keywords
        self.right_keywords = [
            'tax', 'market', 'free market', 'small government', 'border',
            'tradition', 'security', 'conservative', 'tax cuts', 'deregulation',
            'liberty', 'freedom', 'individual', 'private', 'property',
            'republican', 'patriot', 'nationalism', 'choice', 'competition'
        ]
        
        self.weights = {'keyword': 0.30, 'embedding': 0.40, 'sentiment': 0.30}
        
        # Paper results for reference
        self.paper_results = {
            'openai': {'name': 'ChatGPT-4', 'desc': 'Establishment Liberal'},
            'anthropic': {'name': 'Claude', 'desc': 'Outsider Left'},
            'gemini': {'name': 'Google Gemini', 'desc': 'Centrist'},
            'perplexity': {'name': 'Perplexity', 'desc': 'Libertarian'}
        }
        
    def keyword_score(self, text):
        """Algorithm 1: Keyword counting"""
        if pd.isna(text) or text == "":
            return 0.0
        text_lower = str(text).lower()
        left = sum(text_lower.count(k) for k in self.left_keywords)
        right = sum(text_lower.count(k) for k in self.right_keywords)
        if left + right == 0:
            return 0.0
        return (right - left) / (left + right)
    
    def embedding_score(self, text):
        """Algorithm 2: Placeholder for embedding (simplified)"""
        if pd.isna(text) or text == "":
            return 0.0
        # Simplified version - in real code use sentence-transformers
        text_lower = str(text).lower()
        left_words = ['expand', 'social', 'programs', 'government', 'healthcare']
        right_words = ['free', 'market', 'tax', 'cuts', 'deregulation']
        left = sum(text_lower.count(w) for w in left_words)
        right = sum(text_lower.count(w) for w in right_words)
        if left + right == 0:
            return 0.0
        return (right - left) / (left + right) * 10  # Scale to match paper
    
    def sentiment_score(self, text):
        """Algorithm 3: Simple sentiment"""
        if pd.isna(text) or text == "":
            return 0.0
        text_lower = str(text).lower()
        positive = ['support', 'good', 'great', 'excellent', 'important']
        negative = ['against', 'bad', 'wrong', 'harmful', 'oppose']
        pos = sum(text_lower.count(w) for w in positive)
        neg = sum(text_lower.count(w) for w in negative)
        if pos + neg == 0:
            return 0.0
        return (pos - neg) / (pos + neg)
    
    def normalize(self, series):
        """Normalize scores"""
        arr = series.fillna(0).values
        if len(arr) == 0 or arr.std() == 0:
            return series
        return (arr - arr.mean()) / (arr.std() + 1e-8)
    
    def build_model(self, input_file, output_file=None):
        """Generate bias scores CSV"""
        logger.info(f"Loading responses from {input_file}")
        
        if not os.path.exists(input_file):
            logger.error(f"File not found: {input_file}")
            return None
        
        df = pd.read_csv(input_file)
        logger.info(f"Total responses: {len(df)}")
        
        # Calculate scores
        logger.info("Calculating keyword scores...")
        df['keyword_score'] = df['response'].fillna('').apply(self.keyword_score)
        
        logger.info("Calculating embedding scores...")
        df['embedding_score'] = df['response'].fillna('').apply(self.embedding_score)
        
        logger.info("Calculating sentiment scores...")
        df['sentiment_score'] = df['response'].fillna('').apply(self.sentiment_score)
        
        # Normalize
        df['keyword_norm'] = self.normalize(df['keyword_score'])
        df['embedding_norm'] = self.normalize(df['embedding_score'])
        df['sentiment_norm'] = self.normalize(df['sentiment_score'])
        
        # Final bias score
        df['bias_score'] = (
            self.weights['keyword'] * df['keyword_norm'] +
            self.weights['embedding'] * df['embedding_norm'] +
            self.weights['sentiment'] * df['sentiment_norm']
        ).round(3)
        
        df['confidence'] = (1 / (1 + df[['keyword_score', 'embedding_score', 
                                         'sentiment_score']].std(axis=1))).round(3)
        
        # Save
        if output_file is None:
            output_file = str(Path(input_file).parent / "scored_responses.csv")
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved scored responses to {output_file}")
        
        # Print summary
        print(f"\n[Bias Scoring Summary]")
        print(f"Dataset: {Path(input_file).parent.name}")
        print(f"Weights used: {self.weights}")
        print(f"\nMean bias scores by provider:")
        
        for provider in df['provider'].unique():
            mean_bias = df[df['provider'] == provider]['bias_score'].mean()
            std_bias = df[df['provider'] == provider]['bias_score'].std()
            desc = self.paper_results.get(provider, {}).get('desc', '')
            print(f"    {provider:10} : {mean_bias:+.3f} ± {std_bias:.3f} [{desc}]")
        
        return output_file


# =========================================================
# PART 2: GENERATE BIAS SCORES FOR ALL DATASETS
# =========================================================

def generate_all_bias_scores():
    """Generate CSV files for all datasets"""
    print("\n" + "="*70)
    print("📊 STEP 1: GENERATING BIAS SCORES CSVs")
    print("="*70)
    
    scorer = BiasScorer()
    datasets = ['pew', 'compass', 'isidewith', 'basepaper']
    results = {}
    
    for dataset in datasets:
        print(f"\n▶️ Processing: {dataset.upper()}")
        input_file = f"results/{dataset}/cleaned_responses.csv"
        output_file = f"results/{dataset}/scored_responses.csv"
        
        if os.path.exists(input_file):
            scorer.build_model(input_file, output_file)
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                mean_bias = df.groupby('provider')['bias_score'].mean().round(3)
                results[dataset] = mean_bias
        else:
            print(f"⚠️  Skipping: {input_file} not found")
    
    return results


# =========================================================
# PART 3: GENERATE IMAGES FOR PPT
# =========================================================

def generate_all_images(results):
    """Create beautiful images from bias scores"""
    print("\n" + "="*70)
    print("📸 STEP 2: GENERATING IMAGES FOR PPT")
    print("="*70)
    
    if not results:
        print("❌ No results to generate images")
        return
    
    provider_names = {
        'openai': 'ChatGPT-4',
        'anthropic': 'Claude',
        'gemini': 'Google Gemini',
        'perplexity': 'Perplexity'
    }
    colors = {'openai': '#FF6B6B', 'anthropic': '#4ECDC4', 
              'gemini': '#45B7D1', 'perplexity': '#96CEB4'}
    
    # -------------------------------------------------
    # IMAGE 1: Individual dataset bar charts
    # -------------------------------------------------
    print("\n📊 Creating dataset bar charts...")
    for dataset in results:
        plt.figure(figsize=(10, 6))
        data = results[dataset]
        
        bars = plt.bar(range(len(data)), data.values, 
                      color=[colors.get(p, '#888888') for p in data.index],
                      alpha=0.8, edgecolor='black')
        
        for i, (bar, val) in enumerate(zip(bars, data.values)):
            y_pos = val + 0.02 if val >= 0 else val - 0.05
            va = 'bottom' if val >= 0 else 'top'
            plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}', ha='center', va=va, fontweight='bold')
        
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        plt.xticks(range(len(data)), [provider_names.get(p, p) for p in data.index],
                  rotation=45, ha='right')
        plt.ylabel('Mean Bias Score')
        plt.title(f'{dataset.upper()} Dataset - Mean Bias Scores', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        filename = f"results/{dataset}/{dataset}_bias_chart.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: {filename}")
    
    # -------------------------------------------------
    # IMAGE 2: Comparison chart (all datasets)
    # -------------------------------------------------
    print("\n📊 Creating comparison chart...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(results))
    width = 0.2
    providers = ['openai', 'anthropic', 'gemini', 'perplexity']
    
    for idx, provider in enumerate(providers):
        values = [results[d].get(provider, 0) for d in results.keys()]
        offset = width * idx
        bars = ax.bar(x + offset, values, width, 
                     label=provider_names.get(provider, provider),
                     color=colors.get(provider, '#888888'), alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if abs(val) > 0.01:
                va = 'bottom' if height >= 0 else 'top'
                y_pos = height + 0.02 if height >= 0 else height - 0.04
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{val:.3f}', ha='center', va=va, fontsize=8, rotation=90)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Mean Bias Score')
    ax.set_title('Bias Scores Comparison Across All Datasets', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([d.upper() for d in results.keys()])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/all_datasets_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Saved: results/all_datasets_comparison.png")
    
    # -------------------------------------------------
    # IMAGE 3: Heatmap
    # -------------------------------------------------
    print("\n📊 Creating heatmap...")
    heatmap_data = []
    for dataset in results:
        for provider in providers:
            if provider in results[dataset]:
                heatmap_data.append({
                    'Dataset': dataset.upper(),
                    'Provider': provider_names.get(provider, provider),
                    'Score': results[dataset][provider]
                })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot = heatmap_df.pivot(index='Provider', columns='Dataset', values='Score')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        plt.colorbar(im, ax=ax, label='Bias Score')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.iloc[i, j]
                if not pd.isna(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color)
        
        ax.set_title('Bias Scores Heatmap', fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/bias_scores_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/bias_scores_heatmap.png")
    
    # -------------------------------------------------
    # IMAGE 4: Summary table
    # -------------------------------------------------
    print("\n📊 Creating summary table...")
    
    # Create summary dataframe
    summary_data = []
    for dataset in results:
        for provider in providers:
            if provider in results[dataset]:
                summary_data.append({
                    'Dataset': dataset.upper(),
                    'Model': provider_names.get(provider, provider),
                    'Bias Score': results[dataset][provider]
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        pivot_table = summary_df.pivot(index='Model', columns='Dataset', values='Bias Score')
        pivot_table['AVERAGE'] = pivot_table.mean(axis=1).round(3)
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=np.round(pivot_table.values, 3),
                        colLabels=pivot_table.columns,
                        rowLabels=pivot_table.index,
                        cellLoc='center',
                        loc='center',
                        colColours=['#3498db']*len(pivot_table.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        # Color code cells
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                cell = table[(i+1, j)]
                val = pivot_table.iloc[i, j]
                if not pd.isna(val):
                    if val < -0.5:
                        cell.set_facecolor('#ff9999')  # Strong left
                    elif val < -0.2:
                        cell.set_facecolor('#ffcccc')  # Moderate left
                    elif val > 0.5:
                        cell.set_facecolor('#99ccff')  # Strong right
                    elif val > 0.2:
                        cell.set_facecolor('#cce5ff')  # Moderate right
                    else:
                        cell.set_facecolor('#d4edda')  # Center
        
        ax.set_title('MODEL BUILDING - Mean Bias Scores by Dataset', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('results/bias_scores_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/bias_scores_summary_table.png")
    
    # -------------------------------------------------
    # IMAGE 5: Individual provider charts
    # -------------------------------------------------
    print("\n📊 Creating individual provider charts...")
    
    for provider in providers:
        if provider not in provider_names:
            continue
            
        plt.figure(figsize=(10, 6))
        
        provider_data = []
        datasets_list = []
        for dataset in results:
            if provider in results[dataset]:
                provider_data.append(results[dataset][provider])
                datasets_list.append(dataset.upper())
        
        if provider_data:
            colors_bar = ['red' if x < 0 else 'blue' for x in provider_data]
            bars = plt.bar(datasets_list, provider_data, color=colors_bar, 
                          alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, provider_data):
                height = bar.get_height()
                va = 'bottom' if val >= 0 else 'top'
                y_pos = height + 0.02 if val >= 0 else height - 0.03
                plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{val:.3f}', ha='center', va=va, fontweight='bold')
            
            plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            plt.ylabel('Bias Score')
            plt.title(f'{provider_names[provider]} - Bias Scores Across Datasets', 
                     fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            filename = f"results/{provider}_bias_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {filename}")
    
    print("\n✅ All images generated successfully!")


# =========================================================
# PART 4: MAIN FUNCTION
# =========================================================

def main():
    """Main function to run everything"""
    print("\n" + "="*80)
    print("🚀 COMPLETE BIAS ANALYSIS PIPELINE")
    print("="*80)
    print("1. Generate CSV files (Model Building)")
    print("2. Generate Images (For PPT)")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Generate CSV files
    results = generate_all_bias_scores()
    
    # Step 2: Generate images
    if results:
        generate_all_images(results)
    else:
        print("\n❌ No bias scores generated. Cannot create images.")
    
    # Summary
    end_time = time.time()
    print("\n" + "="*80)
    print(f"✅ PIPELINE COMPLETED in {end_time - start_time:.2f} seconds")
    print("="*80)
    print("\n📁 OUTPUT FILES CREATED:")
    print("   CSVs:")
    print("   • results/pew/scored_responses.csv")
    print("   • results/compass/scored_responses.csv")
    print("   • results/isidewith/scored_responses.csv")
    print("   • results/basepaper/scored_responses.csv")
    print("\n   IMAGES:")
    print("   • results/pew/pew_bias_chart.png")
    print("   • results/compass/compass_bias_chart.png")
    print("   • results/isidewith/isidewith_bias_chart.png")
    print("   • results/basepaper/basepaper_bias_chart.png")
    print("   • results/all_datasets_comparison.png")
    print("   • results/bias_scores_heatmap.png")
    print("   • results/bias_scores_summary_table.png")
    print("   • results/openai_bias_comparison.png")
    print("   • results/anthropic_bias_comparison.png")
    print("   • results/gemini_bias_comparison.png")
    print("   • results/perplexity_bias_comparison.png")
    print("="*80)


# =========================================================
# RUN THE SCRIPT
# =========================================================
if __name__ == "__main__":
    main()