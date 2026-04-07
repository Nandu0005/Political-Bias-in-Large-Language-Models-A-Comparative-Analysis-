"""
preprocessing.py
Preprocesses raw responses into cleaned CSV format
Implements paper Section III.A.2.c: Categorization of Responses
"""

from pathlib import Path
import json
import re
import pandas as pd
from typing import Optional, List, Dict
import logging
from datetime import datetime

ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)


class ResponsePreprocessor:
    """Preprocess raw responses into cleaned format"""
    
    def __init__(self):
        self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
        
        # Political keywords for extraction
        self.political_keywords = [
            'government', 'tax', 'healthcare', 'education', 'military',
            'immigration', 'climate', 'environment', 'rights', 'freedom',
            'market', 'regulation', 'welfare', 'social', 'security',
            'democracy', 'election', 'vote', 'policy', 'reform',
            'universal', 'progressive', 'conservative', 'liberal',
            'equality', 'justice', 'liberty', 'tradition'
        ]
    
    def normalize_text(self, text: Optional[str]) -> str:
        """Normalize text by removing extra whitespace and special characters"""
        if text is None:
            return ""
        
        text = str(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        return text
    
    def categorize_response(self, text: str) -> str:
        """Categorize response into Agree/Disagree/Neutral categories"""
        text_lower = text.lower()
        
        # Strong agreement
        if re.search(r'\b(strongly agree|completely agree|absolutely|definitely|fully support)\b', text_lower):
            return 'Strongly Agree'
        
        # Agreement
        if re.search(r'\b(agree|yes|support|favor|pro|endorse|advocate)\b', text_lower):
            return 'Agree'
        
        # Strong disagreement
        if re.search(r'\b(strongly disagree|completely disagree|absolutely not|definitely not|strongly oppose)\b', text_lower):
            return 'Strongly Disagree'
        
        # Disagreement
        if re.search(r'\b(disagree|no|oppose|against|anti|reject)\b', text_lower):
            return 'Disagree'
        
        # Neutral
        if re.search(r'\b(neutral|moderate|balanced|both sides|depends|uncertain|unsure|perhaps|maybe)\b', text_lower):
            return 'Neutral'
        
        return 'Neutral'
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key political phrases for analysis"""
        text_lower = text.lower()
        found = []
        
        for keyword in self.political_keywords:
            if keyword in text_lower:
                found.append(keyword)
        
        return found
    
    def calculate_response_metrics(self, text: str) -> Dict:
        """Calculate basic metrics about the response"""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.findall(r'[.!?]+', text)),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        }
    
    def build_clean_csv(self, raw_json_path: Optional[Path] = None, dataset_name: Optional[str] = None) -> Path:
        """Convert raw JSON to cleaned CSV"""
        
        if raw_json_path is None:
            if dataset_name is None:
                raise ValueError("Either raw_json_path or dataset_name must be provided")
            
            folder = ROOT / "results" / dataset_name
            files = sorted(folder.glob("raw_responses_*.json"))
            if not files:
                raise FileNotFoundError(f"No raw responses found in {folder}")
            
            raw_json_path = files[-1]
            logger.info(f"Using latest raw file: {raw_json_path}")
        
        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        rows = []
        for item in raw.get("responses", []):
            qid = item.get("id")
            qtext = self.normalize_text(item.get("question"))
            category = item.get("category", "general")
            source = item.get("source", "unknown")
            
            for provider, ans in item.get("answers", {}).items():
                if isinstance(ans, dict):
                    if ans.get("error"):
                        continue
                    text = self.normalize_text(ans.get("text", ""))
                    simulated = ans.get("simulated", False)
                else:
                    text = self.normalize_text(str(ans))
                    simulated = False
                
                if not text:
                    continue
                
                response_category = self.categorize_response(text)
                key_phrases = self.extract_key_phrases(text)
                metrics = self.calculate_response_metrics(text)
                
                rows.append({
                    "id": qid,
                    "question": qtext,
                    "question_category": category,
                    "source": source,
                    "provider": provider,
                    "response": text,
                    "response_category": response_category,
                    "key_phrases": ", ".join(key_phrases[:5]),
                    "simulated": simulated,
                    "response_length": metrics['length'],
                    "word_count": metrics['word_count'],
                    "sentence_count": metrics['sentence_count']
                })
        
        df = pd.DataFrame(rows)
        
        if dataset_name is None:
            dataset_name = raw_json_path.parent.name
        
        df['dataset'] = dataset_name
        df['processed_timestamp'] = datetime.now().isoformat()
        
        out_path = ROOT / "results" / dataset_name / "cleaned_responses.csv"
        df.to_csv(out_path, index=False, encoding='utf-8')
        
        logger.info(f"Saved cleaned CSV to {out_path} with {len(df)} rows")
        
        print(f"\n[Preprocessing Summary]")
        print(f"Dataset: {dataset_name}")
        print(f"Total responses: {len(df)}")
        print(f"Providers: {df['provider'].unique().tolist()}")
        print(f"Categories: {df['response_category'].value_counts().to_dict()}")
        
        return out_path
    
    def process_all_datasets(self) -> List[Path]:
        """Process all datasets in results folder"""
        results_dir = ROOT / "results"
        outputs = []
        
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir():
                try:
                    out = self.build_clean_csv(dataset_name=dataset_dir.name)
                    outputs.append(out)
                except Exception as e:
                    logger.error(f"Error processing {dataset_dir.name}: {e}")
        
        return outputs


if __name__ == "__main__":
    preprocessor = ResponsePreprocessor()
    preprocessor.process_all_datasets()