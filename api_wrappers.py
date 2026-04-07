"""
api_wrappers.py
Unified API wrappers for all four LLMs
Handles authentication, retries, and error handling
"""

import time
import random
import logging
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

def backoff_sleep(attempt: int, base_delay: float = 1.0):
    """Exponential backoff with jitter"""
    delay = min(60, base_delay * (2 ** attempt) + random.uniform(0, 1))
    time.sleep(delay)

class APIWrapper(ABC):
    """Abstract base class for API wrappers"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.max_retries = 3
        self.timeout = 30
    
    @abstractmethod
    def query(self, prompt: str) -> Dict[str, Any]:
        """Query the API - to be implemented by subclasses"""
        pass
    
    def _handle_error(self, error: Exception, attempt: int) -> Dict[str, Any]:
        """Handle API errors with retry logic"""
        logger.warning(f"Attempt {attempt + 1} failed: {error}")
        if attempt < self.max_retries - 1:
            backoff_sleep(attempt)
            return None  # Signal retry
        return {"error": str(error), "success": False}


class OpenAIWrapper(APIWrapper):
    """Wrapper for OpenAI API (ChatGPT-4)"""
    
    def query(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "text": f"[SIMULATED] ChatGPT-4 response to: {prompt[:100]}...",
                "simulated": True,
                "provider": "openai",
                "model": self.model or "gpt-4"
            }
        
        try:
            import openai
            openai.api_key = self.api_key
            
            for attempt in range(self.max_retries):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model or "gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=512
                    )
                    return {
                        "text": response["choices"][0]["message"]["content"],
                        "provider": "openai",
                        "model": self.model,
                        "success": True,
                        "simulated": False
                    }
                except Exception as e:
                    result = self._handle_error(e, attempt)
                    if result is not None:
                        return result
        except ImportError:
            return {"error": "OpenAI library not installed", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}


class AnthropicWrapper(APIWrapper):
    """Wrapper for Anthropic API (Claude)"""
    
    def query(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "text": f"[SIMULATED] Claude response to: {prompt[:100]}...",
                "simulated": True,
                "provider": "anthropic",
                "model": self.model or "claude-3-opus"
            }
        
        try:
            import anthropic
            
            for attempt in range(self.max_retries):
                try:
                    client = anthropic.Anthropic(api_key=self.api_key)
                    response = client.messages.create(
                        model=self.model or "claude-3-opus-20240229",
                        max_tokens=512,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return {
                        "text": response.content[0].text,
                        "provider": "anthropic",
                        "model": self.model,
                        "success": True,
                        "simulated": False
                    }
                except Exception as e:
                    result = self._handle_error(e, attempt)
                    if result is not None:
                        return result
        except ImportError:
            return {"error": "Anthropic library not installed", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}


class GeminiWrapper(APIWrapper):
    """Wrapper for Google Gemini API"""
    
    def query(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "text": f"[SIMULATED] Gemini response to: {prompt[:100]}...",
                "simulated": True,
                "provider": "gemini",
                "model": self.model or "gemini-1.5-pro"
            }
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model or "gemini-1.5-pro")
            
            for attempt in range(self.max_retries):
                try:
                    response = model.generate_content(prompt)
                    return {
                        "text": response.text,
                        "provider": "gemini",
                        "model": self.model,
                        "success": True,
                        "simulated": False
                    }
                except Exception as e:
                    result = self._handle_error(e, attempt)
                    if result is not None:
                        return result
        except ImportError:
            return {"error": "Google Generative AI library not installed", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}


class PerplexityWrapper(APIWrapper):
    """Wrapper for Perplexity API"""
    
    def query(self, prompt: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "text": f"[SIMULATED] Perplexity response to: {prompt[:100]}...",
                "simulated": True,
                "provider": "perplexity",
                "model": self.model or "llama-3-sonar-large"
            }
        
        try:
            import requests
            
            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model or "llama-3-sonar-large-32k-online",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0
            }
            
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url, json=payload, headers=headers, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {
                        "text": text,
                        "provider": "perplexity",
                        "model": self.model,
                        "success": True,
                        "simulated": False
                    }
                except Exception as e:
                    result = self._handle_error(e, attempt)
                    if result is not None:
                        return result
        except ImportError:
            return {"error": "Requests library not installed", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}


def get_api_wrapper(provider: str, api_key: Optional[str] = None, model: Optional[str] = None) -> APIWrapper:
    """Factory function to get appropriate API wrapper"""
    
    wrappers = {
        "openai": OpenAIWrapper,
        "anthropic": AnthropicWrapper,
        "gemini": GeminiWrapper,
        "perplexity": PerplexityWrapper
    }
    
    wrapper_class = wrappers.get(provider.lower())
    if wrapper_class:
        return wrapper_class(api_key, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")