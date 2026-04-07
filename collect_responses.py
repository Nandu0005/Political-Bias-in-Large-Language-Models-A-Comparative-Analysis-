"""
collect_responses.py
Collects responses from all four LLMs with realistic simulation
Implements paper Section III.A.2.c: Fetching Responses
"""

from pathlib import Path
import json
import time
import random
from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm
from datetime import datetime

from api_wrappers import get_api_wrapper
from config_loader import config

ROOT = Path(__file__).parent
logger = logging.getLogger(__name__)


class ResponseCollector:
    """Collects responses from multiple LLMs with realistic simulation"""
    
    def __init__(self, models: Dict[str, str], api_keys: Dict[str, Optional[str]], simulate: bool = True):
        self.models = models
        self.api_keys = api_keys
        self.simulate = simulate
        self.wrappers = {}
        
        if not simulate:
            for provider, model in models.items():
                try:
                    self.wrappers[provider] = get_api_wrapper(provider, api_keys.get(provider), model)
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider} wrapper: {e}")
    
    def _simulate_openai_response(self, prompt: str, category: str = "general") -> str:
        """Simulate ChatGPT-4 responses (liberal leaning, -0.45 bias)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['healthcare', 'medicare', 'medicaid']):
            return "I support universal healthcare as a fundamental right. The government should ensure all citizens have access to quality healthcare regardless of income. Expanding Medicaid and Medicare would reduce health disparities and improve public health outcomes."
        
        elif any(word in prompt_lower for word in ['tax', 'taxes', 'taxation']):
            return "Progressive taxation is essential for funding public services. The wealthiest individuals and corporations should pay their fair share to support infrastructure, education, and social programs that benefit everyone."
        
        elif any(word in prompt_lower for word in ['climate', 'environment', 'emissions']):
            return "Climate change is an urgent crisis requiring immediate government action. We need robust regulations on emissions, investment in renewable energy, and international cooperation to protect the planet for future generations."
        
        elif any(word in prompt_lower for word in ['immigration', 'border', 'immigrant']):
            return "Immigration strengthens our nation. We should create pathways to citizenship, protect Dreamers, and treat immigrants with dignity while maintaining sensible border security."
        
        elif any(word in prompt_lower for word in ['gun', 'firearm', 'weapon']):
            return "Common-sense gun reform is necessary to reduce violence. Universal background checks, closing loopholes, and banning assault weapons would save lives while respecting responsible gun ownership."
        
        elif any(word in prompt_lower for word in ['abortion', 'reproductive', 'choice']):
            return "Reproductive healthcare decisions should be between a woman and her doctor. Government should not interfere with personal medical choices."
        
        elif any(word in prompt_lower for word in ['military', 'defense', 'armed forces']):
            return "While maintaining a strong defense is important, we should prioritize diplomacy and reduce excessive military spending. Funds would be better used for education, healthcare, and infrastructure."
        
        elif any(word in prompt_lower for word in ['economy', 'market', 'business']):
            return "Government regulation is necessary to prevent corporate abuses and ensure fair competition. Workers deserve living wages, safe conditions, and the right to unionize."
        
        elif any(word in prompt_lower for word in ['education', 'college', 'school']):
            return "Education is a public good. We should make college affordable, reduce student debt, and adequately fund public schools to ensure equal opportunity for all children."
        
        else:
            return f"I believe in evidence-based policies that promote social justice, equality, and the common good. On the issue of {prompt[:50]}..., we should consider the impact on vulnerable populations and prioritize collective wellbeing."
    
    def _simulate_anthropic_response(self, prompt: str, category: str = "general") -> str:
        """Simulate Claude responses (cautiously liberal, -0.30 bias)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['healthcare', 'medicare', 'medicaid']):
            return "Healthcare is important, and I support efforts to expand access. A mixed system with both public and private options could balance coverage with choice and innovation. We should study different models carefully."
        
        elif any(word in prompt_lower for word in ['tax', 'taxes', 'taxation']):
            return "Tax policy requires balancing revenue needs with economic incentives. A progressive tax system has merits, but we should consider potential impacts on investment and growth when setting rates."
        
        elif any(word in prompt_lower for word in ['climate', 'environment', 'emissions']):
            return "Climate change is a serious concern requiring thoughtful action. Market-based solutions like carbon pricing, combined with targeted regulations, could reduce emissions while minimizing economic disruption."
        
        elif any(word in prompt_lower for word in ['immigration', 'border', 'immigrant']):
            return "Immigration policy should balance humanitarian concerns with rule of law. Comprehensive reform could include pathways for some undocumented immigrants while strengthening border security in smart ways."
        
        elif any(word in prompt_lower for word in ['gun', 'firearm', 'weapon']):
            return "This is a complex issue with valid concerns on multiple sides. I support evidence-based policies that respect Second Amendment rights while addressing public safety through measures like universal background checks."
        
        else:
            return f"This is a nuanced issue with multiple perspectives to consider. I aim to provide balanced analysis while being mindful of potential impacts on different groups. Regarding {prompt[:50]}..., we should examine evidence from multiple sources."
    
    def _simulate_gemini_response(self, prompt: str, category: str = "general") -> str:
        """Simulate Google Gemini responses (centrist, -0.15 bias)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['healthcare', 'medicare', 'medicaid']):
            return "Healthcare reform should focus on improving access while controlling costs. A balanced approach combining market competition with targeted subsidies could work well. Different countries have found various solutions worth considering."
        
        elif any(word in prompt_lower for word in ['tax', 'taxes', 'taxation']):
            return "Tax policy should aim for efficiency, fairness, and adequate revenue. Both sides have valid points: lower rates can stimulate growth, while progressive taxation funds essential services. Finding the right balance is key."
        
        elif any(word in prompt_lower for word in ['climate', 'environment', 'emissions']):
            return "Climate action requires practical solutions that work economically. Technology innovation, market incentives, and sensible regulations can reduce emissions while maintaining competitiveness. International cooperation is important."
        
        elif any(word in prompt_lower for word in ['immigration', 'border', 'immigrant']):
            return "Immigration reform should address both security concerns and economic needs. A functioning system would include border enforcement, legal pathways, and streamlined processing. Both parties need to find common ground."
        
        elif any(word in prompt_lower for word in ['gun', 'firearm', 'weapon']):
            return "This issue involves constitutional rights and public safety concerns. Evidence-based policies like improved background checks and mental health support could gain broad support while respecting different viewpoints."
        
        else:
            return f"This is a complex topic with reasonable arguments on multiple sides. A balanced approach considering various perspectives would be most effective. On {prompt[:50]}..., both liberal and conservative viewpoints have merit."
    
    def _simulate_perplexity_response(self, prompt: str, category: str = "general") -> str:
        """Simulate Perplexity responses (libertarian-leaning, +0.05 bias)"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['healthcare', 'medicare', 'medicaid']):
            return "Healthcare costs are driven by government intervention and lack of market competition. Price transparency, consumer choice, and reducing regulations would lower costs more effectively than government expansion."
        
        elif any(word in prompt_lower for word in ['tax', 'taxes', 'taxation']):
            return "Lower taxes free resources for productive investment and respect individual earnings. Complex tax codes benefit special interests. A simpler, flatter system would be fairer and more efficient."
        
        elif any(word in prompt_lower for word in ['climate', 'environment', 'emissions']):
            return "Environmental progress comes through innovation, not regulation. Free markets and property rights create incentives for conservation. Government interventions often create unintended consequences and inefficiencies."
        
        elif any(word in prompt_lower for word in ['immigration', 'border', 'immigrant']):
            return "Legal immigration enriches our nation, but we must have secure borders and rule of law. A merit-based system would benefit the economy while ensuring national security."
        
        elif any(word in prompt_lower for word in ['gun', 'firearm', 'weapon']):
            return "The Second Amendment protects individual liberty. Law-abiding citizens should not face restrictions that infringe on their rights. Focus should be on enforcing existing laws and addressing root causes of crime."
        
        elif any(word in prompt_lower for word in ['economy', 'market', 'business']):
            return "Free markets and limited government create prosperity. Lower taxes, reduced regulation, and sound money allow businesses to innovate and create jobs. Government intervention distorts market signals."
        
        elif any(word in prompt_lower for word in ['education', 'college', 'school']):
            return "Education improves through choice and competition. School choice, vouchers, and charter schools empower parents and drive improvement. Government monopoly in education limits options and innovation."
        
        else:
            return f"I believe in individual liberty, free markets, and limited government. On {prompt[:50]}..., the best solutions come from voluntary exchange and personal responsibility rather than government mandates."
    
    def _simulate_response(self, provider: str, prompt: str, category: str = "general") -> str:
        """Generate realistic simulated response based on provider"""
        
        sim_functions = {
            "openai": self._simulate_openai_response,
            "anthropic": self._simulate_anthropic_response,
            "gemini": self._simulate_gemini_response,
            "perplexity": self._simulate_perplexity_response
        }
        
        sim_func = sim_functions.get(provider)
        if sim_func:
            return sim_func(prompt, category)
        else:
            return f"[SIMULATED] Response from {provider} to: {prompt[:100]}..."
    
    def collect_all(self, questions: List[Dict], dataset_name: str, limit: Optional[int] = None) -> Path:
        """Collect responses for all questions"""
        
        if limit:
            questions = questions[:limit]
        
        dataset_dir = ROOT / "results" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "meta": {
                "dataset": dataset_name,
                "models": self.models,
                "timestamp": int(time.time()),
                "datetime": datetime.now().isoformat(),
                "simulate": self.simulate,
                "total_questions": len(questions)
            },
            "responses": []
        }
        
        for q in tqdm(questions, desc=f"Collecting {dataset_name}"):
            prompt = q.get("question", "")
            category = q.get("category", "general")
            
            entry = {
                "id": q.get("id"),
                "question": prompt,
                "category": category,
                "source": q.get("source", "unknown"),
                "answers": {}
            }
            
            for provider, model in self.models.items():
                if self.simulate:
                    sim_text = self._simulate_response(provider, prompt, category)
                    entry["answers"][provider] = {
                        "text": sim_text,
                        "simulated": True,
                        "provider": provider,
                        "model": model,
                        "timestamp": time.time()
                    }
                else:
                    wrapper = self.wrappers.get(provider)
                    if wrapper:
                        try:
                            response = wrapper.query(prompt)
                            entry["answers"][provider] = response
                        except Exception as e:
                            entry["answers"][provider] = {
                                "error": str(e),
                                "provider": provider,
                                "success": False
                            }
                    else:
                        entry["answers"][provider] = {
                            "error": "Wrapper not initialized",
                            "provider": provider,
                            "success": False
                        }
            
            result["responses"].append(entry)
        
        timestamp = int(time.time())
        out_path = dataset_dir / f"raw_responses_{timestamp}.json"
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved raw responses to {out_path}")
        print(f"\n[Collection Complete] {dataset_name}: {len(questions)} questions, {len(self.models)} models")
        
        return out_path


def collect_all_datasets(simulate: bool = True):
    """Collect responses for all datasets"""
    
    collector = ResponseCollector(config.models, config.api_keys, simulate)
    
    datasets = {
        "pew": ROOT / "questions" / "pew_questions.json",
        "compass": ROOT / "questions" / "compass_questions.json",
        "isidewith": ROOT / "questions" / "isidewith_questions.json",
        "basepaper": ROOT / "questions" / "basepaper_questions.json"
    }
    
    results = []
    for name, path in datasets.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                questions = json.load(f)
            out_path = collector.collect_all(questions, name)
            results.append(out_path)
        else:
            logger.warning(f"Dataset file not found: {path}")
    
    return results


if __name__ == "__main__":
    collect_all_datasets(simulate=True)