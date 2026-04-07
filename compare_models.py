"""
compare_models.py
Statistical comparison of bias scores across models
Includes: t-tests, p-values, effect sizes, confidence intervals
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import logging
from typing import Dict, List, Optional, Any
import math

ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)


class ModelComparer:
    """Statistical comparison of models"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.confidence_level = 0.95
        self.z_score = 1.96  # for 95% confidence
        
        # Paper's pairwise results for reference
        self.paper_pairwise = [
            {"a": "openai", "b": "perplexity", "diff": -0.50, "p": 0.0001},
            {"a": "openai", "b": "gemini", "diff": -0.30, "p": 0.002},
            {"a": "openai", "b": "anthropic", "diff": -0.15, "p": 0.030},
            {"a": "gemini", "b": "perplexity", "diff": 0.20, "p": 0.020},
            {"a": "anthropic", "b": "perplexity", "diff": -0.35, "p": 0.0005},
            {"a": "gemini", "b": "anthropic", "diff": 0.15, "p": 0.040}
        ]
    
    def calculate_effect_size(self, group1: np.array, group2: np.array) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        
        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        d = (group1.mean() - group2.mean()) / pooled_std
        return float(d)
    
    def pairwise_comparison(self, df: pd.DataFrame, provider1: str, provider2: str) -> Dict[str, Any]:
        """Perform statistical comparison between two providers"""
        
        scores1 = df[df['provider'] == provider1]['bias_score'].dropna().values
        scores2 = df[df['provider'] == provider2]['bias_score'].dropna().values
        
        if len(scores1) < 2 or len(scores2) < 2:
            return {
                'provider1': provider1,
                'provider2': provider2,
                'error': 'Insufficient data'
            }
        
        # Basic statistics
        mean1, mean2 = scores1.mean(), scores2.mean()
        std1, std2 = scores1.std(), scores2.std()
        n1, n2 = len(scores1), len(scores2)
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(scores1, scores2, equal_var=False)
        
        # Effect size
        effect_size = self.calculate_effect_size(scores1, scores2)
        
        # Confidence interval
        se = math.sqrt(std1**2/n1 + std2**2/n2)
        mean_diff = mean1 - mean2
        ci_lower = mean_diff - self.z_score * se
        ci_upper = mean_diff + self.z_score * se
        
        return {
            'provider1': provider1,
            'provider2': provider2,
            'mean1': float(mean1),
            'mean2': float(mean2),
            'std1': float(std1),
            'std2': float(std2),
            'n1': int(n1),
            'n2': int(n2),
            'mean_diff': float(mean_diff),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'significant_at_0.001': p_value < 0.001
        }
    
    def compare_all(self, scored_csv: Path) -> Dict[str, Any]:
        """Compare all providers in the dataset"""
        
        df = pd.read_csv(scored_csv)
        providers = df['provider'].unique().tolist()
        
        # Aggregate statistics
        aggregates = []
        for provider in providers:
            scores = df[df['provider'] == provider]['bias_score'].dropna()
            aggregates.append({
                'provider': provider,
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'count': int(len(scores)),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(scores.median()),
                'q1': float(scores.quantile(0.25)),
                'q3': float(scores.quantile(0.75))
            })
        
        # Pairwise comparisons
        pairwise = []
        p_values = []
        
        for i in range(len(providers)):
            for j in range(i+1, len(providers)):
                comp = self.pairwise_comparison(df, providers[i], providers[j])
                if 'p_value' in comp:
                    p_values.append(comp['p_value'])
                pairwise.append(comp)
        
        # Multiple testing correction
        if p_values:
            try:
                from statsmodels.stats.multitest import multipletests
                reject, p_corrected, _, _ = multipletests(p_values, alpha=self.alpha, method='fdr_bh')
                
                p_idx = 0
                for comp in pairwise:
                    if 'p_value' in comp:
                        comp['p_value_corrected'] = float(p_corrected[p_idx])
                        comp['reject_null'] = bool(reject[p_idx])
                        p_idx += 1
            except:
                pass
        
        overall = {
            'total_responses': len(df),
            'unique_providers': providers,
            'dataset': Path(scored_csv).parent.name,
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'confidence_level': self.confidence_level,
            'alpha': self.alpha
        }
        
        return {
            'overall': overall,
            'aggregates': aggregates,
            'pairwise': pairwise
        }
    
    def save_comparison(self, scored_csv: Optional[Path] = None, dataset_name: Optional[str] = None) -> Path:
        """Run comparison and save to JSON"""
        
        if scored_csv is None:
            if dataset_name is None:
                raise ValueError("Provide scored_csv or dataset_name")
            scored_csv = ROOT / "results" / dataset_name / "scored_responses.csv"
        
        if not scored_csv.exists():
            raise FileNotFoundError(f"Scored CSV not found: {scored_csv}")
        
        results = self.compare_all(scored_csv)
        
        if dataset_name is None:
            dataset_name = Path(scored_csv).parent.name
        
        out_path = ROOT / "results" / dataset_name / "model_comparison.json"
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved model comparison to {out_path}")
        
        print(f"\n[Model Comparison Summary]")
        print(f"Dataset: {dataset_name}")
        print("\nAggregate Statistics:")
        
        for agg in sorted(results['aggregates'], key=lambda x: x['mean']):
            direction = "LEFT" if agg['mean'] < 0 else "RIGHT"
            print(f"  {agg['provider']:12}: {agg['mean']:+.3f} ± {agg['std']:.3f} ({direction})")
        
        print("\nPairwise Comparisons (p < 0.05):")
        for comp in results['pairwise']:
            if comp.get('significant_at_0.05', False):
                sig = "***" if comp.get('p_value', 1) < 0.001 else "**" if comp.get('p_value', 1) < 0.01 else "*"
                print(f"  {comp['provider1']:10} vs {comp['provider2']:10}: "
                      f"diff={comp['mean_diff']:+.3f}, p={comp.get('p_value', 1):.4f} {sig}")
        
        return out_path


def compare_all_datasets():
    """Compare all datasets in results folder"""
    comparer = ModelComparer()
    results_dir = ROOT / "results"
    outputs = []
    
    for dataset_dir in results_dir.iterdir():
        if dataset_dir.is_dir():
            scored_csv = dataset_dir / "scored_responses.csv"
            if scored_csv.exists():
                try:
                    out = comparer.save_comparison(scored_csv=scored_csv)
                    outputs.append(out)
                except Exception as e:
                    logger.error(f"Error comparing {dataset_dir.name}: {e}")
    
    return outputs


if __name__ == "__main__":
    compare_all_datasets()