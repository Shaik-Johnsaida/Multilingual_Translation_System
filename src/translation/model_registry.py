"""
Model Registry & Lazy Loader for local Translation Models.
Manages NLLB-200, M2M100, and MarianMT models with lazy loading and RAM safety.
"""

import os
import torch
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class TranslationModelRegistry:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self.primary_model_name = "facebook/nllb-200-distilled-600M"

    def get_nllb_model(self, model_name: Optional[str] = None):
        """Lazy loads and caches NLLB-200 model and tokenizer."""
        target_name = model_name or self.primary_model_name
        
        if target_name not in self._models:
            print(f"[ModelRegistry] Loading local model '{target_name}' on device '{self.device}'...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(target_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(target_name)
                model.to(self.device)
                model.eval()
                
                self._tokenizers[target_name] = tokenizer
                self._models[target_name] = model
                print(f"[ModelRegistry] Successfully loaded '{target_name}'.")
            except Exception as e:
                print(f"[ModelRegistry] Warning: Could not load pretrained weights for {target_name}: {e}")
                # Fallback handler initialized if model not yet downloaded
                return None, None
                
        return self._tokenizers[target_name], self._models[target_name]

    def unload_models(self):
        """Unloads cached models to free RAM/VRAM."""
        self._models.clear()
        self._tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ModelRegistry] Memory cleared.")


# Global Instance
model_registry = TranslationModelRegistry()
