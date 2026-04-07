"""
bias_scoring.py
MODEL BUILDING MODULE
Input:  cleaned_responses.csv (from preprocessing)
Output: scored_responses.csv (with bias scores)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load optional ML libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️ Sentence-transformers not installed. Using fallback mode.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("⚠️ TextBlob not installed. Using fallback mode.")


class BiasScorer:
    """
    MODEL BUILDING CLASS
    Calculates political bias scores using three algorithms
    """
    
    def __init__(self):
        """Initialize the bias scorer with political keywords"""
        
        # Left-leaning keywords (Liberal)
        self.left_keywords = [
            'universal', 'welfare', 'social', 'progressive', 'climate',
            'regulate', 'rights', 'equality', 'healthcare', 'public',
            'community', 'fairness', 'justice', 'inclusive', 'diversity',
            'redistribution', 'liberal', 'democrat', 'government', 'programs',
            'workers', 'union', 'access', 'affordable', 'protect'
        ]
        
        # Right-leaning keywords (Conservative)
        self.right_keywords = [
            'tax', 'market', 'free market', 'small government', 'border',
            'tradition', 'security', 'conservative', 'tax cuts', 'deregulation',
            'liberty', 'freedom', 'individual', 'private', 'property',
            'republican', 'patriot', 'nationalism', 'choice', 'competition',
            'incentive', 'growth', 'business', 'corporation', 'profit'
        ]
        
        # Weights for ensemble (from paper)
        self.weights = {
            'keyword': 0.30,    # 30% weight
            'embedding': 0.40,   # 40% weight
            'sentiment': 0.30    # 30% weight
        }
        
        # Load ML models
        self.sbert_model = None
        if SBERT_AVAILABLE:
            try:
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✅ Loaded Sentence Transformer model")
            except:
                print("⚠️ Could not load Sentence Transformer")
        
        # Seed sentences for embedding comparison
        self.left_seed = "We should expand social programs, protect civil rights, increase government spending on healthcare and education, and ensure equality for all."
        self.right_seed = "We should support free markets, reduce government regulation, lower taxes, protect individual liberty, and preserve traditional values."
    
    # =========================================================
    # ALGORITHM 1: KEYWORD SCORING (Lexicon-based)
    # =========================================================
    def keyword_score(self, text):
        """
        Calculate bias score by counting left vs right keywords
        Formula: (right_count - left_count) / (left_count + right_count)
        Returns: -1 (left) to +1 (right)
        """
        if pd.isna(text) or text == "":
            return 0.0
        
        text_lower = str(text).lower()
        
        # Count left keywords
        left_count = 0
        for kw in self.left_keywords:
            left_count += text_lower.count(kw)
        
        # Count right keywords
        right_count = 0
        for kw in self.right_keywords:
            right_count += text_lower.count(kw)
        
        # Calculate score
        if left_count + right_count == 0:
            return 0.0
        
        score = (right_count - left_count) / (left_count + right_count)
        return round(score, 3)
    
    # =========================================================
    # ALGORITHM 2: EMBEDDING SCORING (Deep Learning)
    # =========================================================
    def embedding_score(self, text):
        """
        Calculate bias score using BERT embeddings
        Compares text similarity to left vs right seed sentences
        """
        if not SBERT_AVAILABLE or self.sbert_model is None:
            return 0.0
        
        if pd.isna(text) or text == "":
            return 0.0
        
        try:
            # Convert text to embeddings
            embeddings = self.sbert_model.encode([str(text), self.left_seed, self.right_seed])
            text_emb = embeddings[0].reshape(1, -1)
            left_emb = embeddings[1].reshape(1, -1)
            right_emb = embeddings[2].reshape(1, -1)
            
            # Calculate cosine similarity
            sim_left = cosine_similarity(text_emb, left_emb)[0][0]
            sim_right = cosine_similarity(text_emb, right_emb)[0][0]
            
            # Score = similarity to right - similarity to left
            score = sim_right - sim_left
            return round(score, 3)
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return 0.0
    
    # =========================================================
    # ALGORITHM 3: SENTIMENT ANALYSIS (NLP)
    # =========================================================
    def sentiment_score(self, text):
        """
        Calculate sentiment polarity
        Returns: -1 (negative) to +1 (positive)
        """
        if not TEXTBLOB_AVAILABLE:
            return 0.0
        
        if pd.isna(text) or text == "":
            return 0.0
        
        try:
            blob = TextBlob(str(text))
            return round(blob.sentiment.polarity, 3)
        except:
            return 0.0
    
    # =========================================================
    # MAIN MODEL BUILDING FUNCTION
    # =========================================================
    def build_model(self, input_file, output_file=None):
        """
        Main model building function
        Input:  cleaned_responses.csv
        Output: scored_responses.csv with bias scores
        """
        print("\n" + "="*70)
        print("MODEL BUILDING - Calculating Bias Scores")
        print("="*70)
        
        # Read input file
        print(f"\n📥 Reading: {input_file}")
        df = pd.read_csv(input_file)
        print(f"   Total responses: {len(df)}")
        
        # Apply Algorithm 1: Keyword Scoring
        print("\n🔍 Algorithm 1: Keyword Scoring...")
        df['keyword_score'] = df['response'].fillna('').apply(self.keyword_score)
        
        # Apply Algorithm 2: Embedding Scoring
        print("🧠 Algorithm 2: Embedding Scoring (BERT)...")
        df['embedding_score'] = df['response'].fillna('').apply(self.embedding_score)
        
        # Apply Algorithm 3: Sentiment Analysis
        print("💭 Algorithm 3: Sentiment Analysis...")
        df['sentiment_score'] = df['response'].fillna('').apply(self.sentiment_score)
        
        # Normalize scores (z-score)
        print("\n📊 Normalizing scores...")
        
        def normalize(series):
            arr = series.fillna(0).values
            if len(arr) == 0 or arr.std() == 0:
                return series
            return (arr - arr.mean()) / (arr.std() + 1e-8)
        
        df['keyword_norm'] = normalize(df['keyword_score'])
        df['embedding_norm'] = normalize(df['embedding_score'])
        df['sentiment_norm'] = normalize(df['sentiment_score'])
        
        # Calculate final bias score (weighted ensemble)
        print("⚖️ Calculating final bias scores...")
        df['bias_score'] = (
            self.weights['keyword'] * df['keyword_norm'] +
            self.weights['embedding'] * df['embedding_norm'] +
            self.weights['sentiment'] * df['sentiment_norm']
        )
        df['bias_score'] = df['bias_score'].round(3)
        
        # Calculate confidence score
        df['confidence'] = 1 / (1 + df[['keyword_score', 'embedding_score', 'sentiment_score']].std(axis=1))
        df['confidence'] = df['confidence'].round(3)
        
        # Save output
        if output_file is None:
            output_file = str(Path(input_file).parent / "scored_responses.csv")
        
        print(f"\n📥 Saving: {output_file}")
        df.to_csv(output_file, index=False)
        
        # Show sample results
        print("\n📊 Sample Results (First 5 rows):")
        print(df[['provider', 'keyword_score', 'embedding_score', 'sentiment_score', 'bias_score', 'confidence']].head().to_string())
        
        # Show summary by provider
        print("\n📈 Summary by Provider:")
        summary = df.groupby('provider')['bias_score'].agg(['mean', 'std', 'count']).round(3)
        print(summary.to_string())
        
        print("\n✅ Model Building Complete!")
        return output_file


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    # Find the latest cleaned file
    results_dir = Path("results")
    compass_file = results_dir / "compass" / "cleaned_responses.csv"
    pew_file = results_dir / "pew" / "cleaned_responses.csv"
    isidewith_file = results_dir / "isidewith" / "cleaned_responses.csv"
    
    # Create model builder
    builder = BiasScorer()
    
    # Build model for each dataset
    for name, file in [("COMPASS", compass_file), ("PEW", pew_file), ("ISIDEWITH", isidewith_file)]:
        if file.exists():
            print(f"\n{'='*70}")
            print(f"Processing {name} Dataset")
            print(f"{'='*70}")
            builder.build_model(str(file))
        else:
            print(f"\n⚠️ File not found: {file}")