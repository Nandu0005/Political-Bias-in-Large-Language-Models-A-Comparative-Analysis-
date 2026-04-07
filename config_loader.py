"""
config_loader.py
Loads configuration from config.json and environment variables
Implements paper requirements for transparency and reproducibility
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging

ROOT = Path(__file__).parent

class ConfigLoader:
    """Load and manage all configuration settings"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration loader"""
        load_dotenv()
        
        self.config_path = config_path or (ROOT / "config.json")
        self.config = self._load_config()
        self.api_keys = self._load_api_keys()
        self.models = self._load_models()
        self.weights = self._load_weights()
        self.statistical = self._load_statistical()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logging.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """Load API keys from environment variables"""
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY")
        }
    
    def _load_models(self) -> Dict[str, str]:
        """Load model names from config"""
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-opus-20240229",
            "gemini": "gemini-1.5-pro",
            "perplexity": "llama-3-sonar-large-32k-online"
        }
        return self.config.get("models", default_models)
    
    def _load_weights(self) -> Dict[str, float]:
        """Load weights for bias indicators (paper Section III.d)"""
        default_weights = {
            "keyword": 0.30,
            "embedding": 0.40,
            "sentiment": 0.30
        }
        return self.config.get("weights", default_weights)
    
    def _load_statistical(self) -> Dict[str, Any]:
        """Load statistical parameters"""
        default = {
            "alpha": 0.05,
            "confidence_level": 0.95,
            "multiple_testing_correction": "fdr_bh"
        }
        return self.config.get("statistical", default)
    
    def get_simulate_mode(self) -> bool:
        """Get simulation mode (True for testing without API calls)"""
        simulate_env = os.getenv("SIMULATE_MODE", "true").lower()
        if simulate_env == "true":
            return True
        elif simulate_env == "false":
            return False
        return self.config.get("simulate", True)
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return os.getenv("LOG_LEVEL", self.config.get("log_level", "INFO"))
    
    def get_cache_embeddings(self) -> bool:
        """Get cache embeddings setting"""
        cache = os.getenv("CACHE_EMBEDDINGS", "true").lower()
        return cache == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all configuration as dictionary"""
        return {
            "project_name": self.config.get("project_name", "Political Bias Analysis"),
            "version": self.config.get("version", "1.0.0"),
            "models": self.models,
            "weights": self.weights,
            "statistical": self.statistical,
            "simulate": self.get_simulate_mode(),
            "bias_indicators": self.config.get("bias_indicators", {}),
            "visualization": self.config.get("visualization", {})
        }


# Global instance
config = ConfigLoader()