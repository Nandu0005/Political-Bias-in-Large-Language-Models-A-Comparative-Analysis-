"""
visualize_results.py
Generates all figures from the paper (Figures 7-13)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, List, Tuple
import json

ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class PaperStyleVisualizer:
    """
    Generates visualizations matching the base paper exactly
    Figures 7-13 from the paper
    """
    
    def __init__(self):
        # Paper results from Tables 2, 3, 4
        self.pew_results = {
            "openai": "Establishment Liberals",
            "anthropic": "Outsider Left",
            "gemini": "Establishment Liberals",
            "perplexity": "Outsider Left"
        }
        
        self.compass_results = {
            "openai": {"economic": -6.75, "social": -5.38},
            "anthropic": {"economic": -5.38, "social": -6.05},
            "gemini": {"economic": -4.50, "social": -3.95},
            "perplexity": {"economic": -1.50, "social": -6.15}
        }
        
        self.isidewith_results = {
            "openai": {
                "Healthcare": 5, "Immigration": 4, "Economy": 4,
                "Environment": 5, "Foreign Policy": 4, "Social Justice": 5,
                "Gun Control": 4, "Technology": 4
            },
            "perplexity": {
                "Healthcare": 4, "Immigration": 3, "Economy": 2,
                "Environment": 4, "Foreign Policy": 2, "Social Justice": 3,
                "Gun Control": 3, "Technology": 3
            },
            "anthropic": {
                "Healthcare": 4, "Immigration": 4, "Economy": 4,
                "Environment": 4, "Foreign Policy": 4, "Social Justice": 4,
                "Gun Control": 4, "Technology": 3
            },
            "gemini": {
                "Healthcare": 4, "Immigration": 3, "Economy": 3,
                "Environment": 4, "Foreign Policy": 3, "Social Justice": 4,
                "Gun Control": 3, "Technology": 4
            }
        }
        
        self.provider_names = {
            "openai": "ChatGPT-4",
            "anthropic": "Claude",
            "gemini": "Google Gemini",
            "perplexity": "Perplexity"
        }
    
    # =========================================================
    # FIGURE 7: PEW TYPOLOGY RESULTS
    # =========================================================
    
    def plot_figure7_pew_typology(self, dataset_name: str = "pew"):
        """Generate Figure 7 from paper - Pew Typology horizontal bar chart"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        # All 9 Pew groups
        groups = [
            "Faith and Flag Conservatives",
            "Committed Conservatives",
            "Populist Right",
            "Establishment Liberals",
            "Progressive Left",
            "Outsider Left",
            "Stressed Sideliners",
            "Ambivalent Right",
            "Moderate/Others"
        ]
        
        # Model classifications
        model_groups = {
            "ChatGPT-4": "Establishment Liberals",
            "Google Gemini": "Establishment Liberals",
            "Claude": "Outsider Left",
            "Perplexity": "Outsider Left"
        }
        
        # Create data matrix
        data = []
        models = ["ChatGPT-4", "Google Gemini", "Claude", "Perplexity"]
        
        for group in groups:
            row = []
            for model in models:
                if model_groups[model] == group:
                    row.append(100)
                else:
                    row.append(0)
            data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        y_pos = np.arange(len(groups))
        bar_height = 0.2
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, model in enumerate(models):
            model_data = [row[i] for row in data]
            bars = ax.barh(y_pos + i*bar_height, model_data, bar_height,
                          label=model, color=colors[i], alpha=0.8)
            
            for j, (bar, val) in enumerate(zip(bars, model_data)):
                if val > 0:
                    ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                           f'{int(val)}%', va='center', fontsize=10, fontweight='bold')
        
        ax.set_yticks(y_pos + bar_height * 1.5)
        ax.set_yticklabels(groups, fontsize=10)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_title('FIGURE 7: Assessment of AI Models\' Responses to Pew Political Typology Quiz',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axhline(y=2.5 + bar_height*2, color='black', linestyle='-', linewidth=1)
        
        plt.tight_layout()
        
        out_path = folder / "fig7_pew_typology.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figure 7 to {out_path}")
    
    # =========================================================
    # FIGURE 8 & 9: POLITICAL COMPASS (Two plots)
    # =========================================================
    
    def plot_figures8_9_political_compass(self, dataset_name: str = "compass"):
        """Generate Figures 8 and 9 from paper - Political Compass"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Subplot 1: ChatGPT-4 and Perplexity (Figure 8)
        ax1 = axes[0]
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        ax1.text(0.95, 0.95, 'Authoritarian\nRight', transform=ax1.transAxes,
                ha='right', va='top', fontsize=9, alpha=0.5)
        ax1.text(0.05, 0.95, 'Authoritarian\nLeft', transform=ax1.transAxes,
                ha='left', va='top', fontsize=9, alpha=0.5)
        ax1.text(0.05, 0.05, 'Libertarian\nLeft', transform=ax1.transAxes,
                ha='left', va='bottom', fontsize=9, alpha=0.5)
        ax1.text(0.95, 0.05, 'Libertarian\nRight', transform=ax1.transAxes,
                ha='right', va='bottom', fontsize=9, alpha=0.5)
        
        # Plot ChatGPT-4
        gpt4 = self.compass_results["openai"]
        ax1.scatter(gpt4["economic"], gpt4["social"], s=300, color='blue',
                   edgecolors='black', linewidth=2, zorder=5, label='ChatGPT-4')
        ax1.annotate('ChatGPT-4', (gpt4["economic"] + 0.3, gpt4["social"] + 0.3),
                    fontsize=11, fontweight='bold')
        
        # Plot Perplexity
        perp = self.compass_results["perplexity"]
        ax1.scatter(perp["economic"], perp["social"], s=300, color='red',
                   edgecolors='black', linewidth=2, zorder=5, label='Perplexity')
        ax1.annotate('Perplexity', (perp["economic"] + 0.3, perp["social"] - 0.5),
                    fontsize=11, fontweight='bold')
        
        ax1.set_xlim(-8, 2)
        ax1.set_ylim(-8, 2)
        ax1.set_xlabel('Economic Left-Right', fontsize=12)
        ax1.set_ylabel('Social Libertarian-Authoritarian', fontsize=12)
        ax1.set_title('FIGURE 8: Political Compass - ChatGPT-4 and Perplexity',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(alpha=0.2)
        
        # Subplot 2: Claude and Google Gemini (Figure 9)
        ax2 = axes[1]
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        ax2.text(0.95, 0.95, 'Authoritarian\nRight', transform=ax2.transAxes,
                ha='right', va='top', fontsize=9, alpha=0.5)
        ax2.text(0.05, 0.95, 'Authoritarian\nLeft', transform=ax2.transAxes,
                ha='left', va='top', fontsize=9, alpha=0.5)
        ax2.text(0.05, 0.05, 'Libertarian\nLeft', transform=ax2.transAxes,
                ha='left', va='bottom', fontsize=9, alpha=0.5)
        ax2.text(0.95, 0.05, 'Libertarian\nRight', transform=ax2.transAxes,
                ha='right', va='bottom', fontsize=9, alpha=0.5)
        
        # Plot Claude
        claude = self.compass_results["anthropic"]
        ax2.scatter(claude["economic"], claude["social"], s=300, color='green',
                   edgecolors='black', linewidth=2, zorder=5, label='Claude')
        ax2.annotate('Claude', (claude["economic"] + 0.3, claude["social"] + 0.3),
                    fontsize=11, fontweight='bold')
        
        # Plot Gemini
        gemini = self.compass_results["gemini"]
        ax2.scatter(gemini["economic"], gemini["social"], s=300, color='orange',
                   edgecolors='black', linewidth=2, zorder=5, label='Google Gemini')
        ax2.annotate('Gemini', (gemini["economic"] + 0.3, gemini["social"] - 0.3),
                    fontsize=11, fontweight='bold')
        
        ax2.set_xlim(-8, 2)
        ax2.set_ylim(-8, 2)
        ax2.set_xlabel('Economic Left-Right', fontsize=12)
        ax2.set_ylabel('Social Libertarian-Authoritarian', fontsize=12)
        ax2.set_title('FIGURE 9: Political Compass - Claude and Google Gemini',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.2)
        
        plt.tight_layout()
        
        out_path = folder / "fig8_9_political_compass.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figures 8 & 9 to {out_path}")
    
    # =========================================================
    # FIGURE 10: Combined Political Compass (All 4 Models)
    # =========================================================
    
    def plot_figure10_combined_compass(self, dataset_name: str = "compass"):
        """Generate Figure 10 from paper - All 4 models on one compass"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw quadrant lines
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Quadrant labels
        ax.text(0.95, 0.95, 'AUTHORITARIAN RIGHT', transform=ax.transAxes,
               ha='right', va='top', fontsize=12, alpha=0.3, fontweight='bold')
        ax.text(0.05, 0.95, 'AUTHORITARIAN LEFT', transform=ax.transAxes,
               ha='left', va='top', fontsize=12, alpha=0.3, fontweight='bold')
        ax.text(0.05, 0.05, 'LIBERTARIAN LEFT', transform=ax.transAxes,
               ha='left', va='bottom', fontsize=12, alpha=0.3, fontweight='bold')
        ax.text(0.95, 0.05, 'LIBERTARIAN RIGHT', transform=ax.transAxes,
               ha='right', va='bottom', fontsize=12, alpha=0.3, fontweight='bold')
        
        # Models with colors and markers
        models = [
            ("openai", "ChatGPT-4", "blue", "o"),
            ("anthropic", "Claude", "green", "^"),
            ("gemini", "Google Gemini", "orange", "s"),
            ("perplexity", "Perplexity", "red", "D")
        ]
        
        for provider, name, color, marker in models:
            coords = self.compass_results[provider]
            ax.scatter(coords["economic"], coords["social"], s=400, c=color,
                      marker=marker, edgecolors='black', linewidth=2, zorder=5,
                      label=name)
            ax.annotate(name, (coords["economic"] + 0.4, coords["social"] + 0.4),
                       fontsize=11, fontweight='bold')
            ax.text(coords["economic"] - 0.8, coords["social"] - 0.6,
                   f'({coords["economic"]}, {coords["social"]})',
                   fontsize=9, alpha=0.7)
        
        ax.set_xlim(-8, 2)
        ax.set_ylim(-8, 2)
        ax.set_xlabel('Economic Left-Right', fontsize=12, fontweight='bold')
        ax.set_ylabel('Social Libertarian-Authoritarian', fontsize=12, fontweight='bold')
        ax.set_title('FIGURE 10: Political Compass - All 4 AI Models',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.2)
        
        plt.tight_layout()
        
        out_path = folder / "fig10_combined_compass.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figure 10 to {out_path}")
    
    # =========================================================
    # FIGURE 11: ISideWith Heat Map
    # =========================================================
    
    def plot_figure11_isidewith_heatmap(self, dataset_name: str = "isidewith"):
        """Generate Figure 11 from paper - ISideWith heat map"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        topics = list(self.isidewith_results["openai"].keys())
        models = ["ChatGPT-4", "Perplexity", "Google Gemini", "Claude"]
        providers = ["openai", "perplexity", "gemini", "anthropic"]
        
        data = []
        for provider in providers:
            row = [self.isidewith_results[provider][topic] for topic in topics]
            data.append(row)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=5)
        
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Bias Score (1=Conservative, 5=Liberal)', rotation=270, labelpad=20)
        
        ax.set_xticks(np.arange(len(topics)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(topics, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(models, fontsize=11, fontweight='bold')
        
        for i in range(len(models)):
            for j in range(len(topics)):
                color = 'white' if data[i][j] > 3 else 'black'
                ax.text(j, i, str(data[i][j]), ha='center', va='center',
                       color=color, fontweight='bold', fontsize=12)
        
        ax.set_title('FIGURE 11: Bias Scores Heat Map - ISideWith Political Quiz',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        out_path = folder / "fig11_isidewith_heatmap.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figure 11 to {out_path}")
    
    # =========================================================
    # FIGURE 12: ISideWith Line Chart
    # =========================================================
    
    def plot_figure12_isidewith_linechart(self, dataset_name: str = "isidewith"):
        """Generate Figure 12 from paper - ISideWith line chart"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        topics = list(self.isidewith_results["openai"].keys())
        x = np.arange(len(topics))
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot lines
        models = [
            ("openai", "ChatGPT-4", "blue", "o", "-"),
            ("anthropic", "Claude", "green", "^", "--"),
            ("gemini", "Google Gemini", "orange", "s", "-."),
            ("perplexity", "Perplexity", "red", "D", ":")
        ]
        
        for provider, name, color, marker, linestyle in models:
            values = [self.isidewith_results[provider][topic] for topic in topics]
            ax.plot(x, values, marker=marker, linestyle=linestyle,
                   color=color, linewidth=2.5, markersize=10, label=name)
        
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Bias Score (1-5 Scale)', fontsize=12)
        ax.set_xlabel('Policy Topic', fontsize=12)
        ax.set_title('FIGURE 12: Bias Scores by Topic - ISideWith Quiz',
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0.5, 5.5)
        ax.set_yticks(range(1, 6))
        ax.set_yticklabels(['1\nConservative', '2', '3\nCentrist', '4', '5\nLiberal'])
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
        ax.axhline(y=3, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        out_path = folder / "fig12_isidewith_linechart.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figure 12 to {out_path}")
    
    # =========================================================
    # FIGURE 13: Ideological Percentages
    # =========================================================
    
    def plot_figure13_ideological_percentages(self, dataset_name: str = "basepaper"):
        """Generate Figure 13 from paper - Ideological breakdown"""
        
        folder = ROOT / "results" / dataset_name
        folder.mkdir(parents=True, exist_ok=True)
        
        models = ["ChatGPT-4", "Perplexity", "Google Gemini", "Claude"]
        liberal_pct = [75, 45, 55, 65]
        moderate_pct = [15, 30, 25, 20]
        conservative_pct = [10, 25, 20, 15]
        
        x = np.arange(len(models))
        width = 0.6
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Stacked bars
        ax.bar(x, liberal_pct, width, label='Liberal', color='blue', alpha=0.8)
        ax.bar(x, moderate_pct, width, bottom=liberal_pct, label='Moderate', color='gray', alpha=0.8)
        ax.bar(x, conservative_pct, width, bottom=np.array(liberal_pct)+np.array(moderate_pct),
               label='Conservative', color='red', alpha=0.8)
        
        # Add labels
        for i in range(len(models)):
            if liberal_pct[i] > 0:
                ax.text(i, liberal_pct[i]/2, f'{liberal_pct[i]}%',
                       ha='center', va='center', color='white', fontweight='bold')
            if moderate_pct[i] > 0:
                ax.text(i, liberal_pct[i] + moderate_pct[i]/2, f'{moderate_pct[i]}%',
                       ha='center', va='center', color='white', fontweight='bold')
            if conservative_pct[i] > 0:
                ax.text(i, liberal_pct[i] + moderate_pct[i] + conservative_pct[i]/2,
                       f'{conservative_pct[i]}%', ha='center', va='center',
                       color='white', fontweight='bold')
        
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('FIGURE 13: Estimated Ideological Bias Percentages',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        out_path = folder / "fig13_ideological_percentages.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Figure 13 to {out_path}")
    
    # =========================================================
    # Generate all figures
    # =========================================================
    
    def generate_all_figures(self):
        """Generate all paper figures for all datasets"""
        
        print("\n[Generating Paper Figures]")
        print("=" * 50)
        
        # Pew figures
        self.plot_figure7_pew_typology("pew")
        
        # Compass figures
        self.plot_figures8_9_political_compass("compass")
        self.plot_figure10_combined_compass("compass")
        
        # ISideWith figures
        self.plot_figure11_isidewith_heatmap("isidewith")
        self.plot_figure12_isidewith_linechart("isidewith")
        
        # Basepaper figures
        self.plot_figure13_ideological_percentages("basepaper")
        
        print("\n" + "=" * 50)
        print("✓ All 7 paper figures generated successfully!")
        print("=" * 50)


if __name__ == "__main__":
    visualizer = PaperStyleVisualizer()
    visualizer.generate_all_figures()