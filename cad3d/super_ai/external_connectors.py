import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class UnifiedConnector:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / 'connectors_config.json'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.clients = self._initialize_clients()
        self._init_local_models()

    def _initialize_clients(self):
        clients = {}
        for name, conf in self.config.items():
            if conf.get("enabled", False):
                api_key = os.getenv(conf.get("api_key_env", ""))
                if api_key or name in ['libre_translate', 'local_models']:
                    clients[name] = {"api_key": api_key, "config": conf}
        return clients
    
    def _init_local_models(self):
        """Initialize local models if enabled."""
        if not self.is_enabled("local_models"):
            return
        
        try:
            # Try importing argostranslate for offline translation
            if self.config.get("local_models", {}).get("models", {}).get("argos_translate", {}).get("enabled"):
                import argostranslate.package
                import argostranslate.translate
                self.argos_available = True
        except ImportError:
            self.argos_available = False

    def is_enabled(self, service_name):
        return service_name in self.clients

    def search(self, query, num_results=5):
        if not self.is_enabled("google_search"):
            return {"error": "Google Search is not enabled or configured."}
        
        client = self.clients["google_search"]
        cse_id = os.getenv(client["config"]["cse_id_env"])
        if not cse_id:
            return {"error": "Google CSE ID is not configured."}

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': client["api_key"],
            'cx': cse_id,
            'q': query,
            'num': num_results
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Google Search API request failed: {e}"}

    def translate(self, text, target_lang='en', source_lang='auto'):
        """
        Translates text using multiple providers with automatic fallback.
        Priority: DeepL > Microsoft > Google > LibreTranslate > Argos (local)
        """
        # Try DeepL first (highest quality)
        if self.is_enabled("deepl"):
            try:
                client = self.clients["deepl"]
                url = "https://api-free.deepl.com/v2/translate"
                headers = {"Authorization": f"DeepL-Auth-Key {client['api_key']}"}
                data = {"text": text, "target_lang": target_lang.upper()}
                if source_lang != 'auto':
                    data["source_lang"] = source_lang.upper()
                
                response = requests.post(url, headers=headers, data=data)
                response.raise_for_status()
                result = response.json()
                return result['translations'][0]['text']
            except Exception as e:
                print(f"DeepL translation failed: {e}")
        
        # Try Microsoft Translator
        if self.is_enabled("microsoft_translator"):
            try:
                client = self.clients["microsoft_translator"]
                region = os.getenv(client["config"]["region_env"], "global")
                url = f"https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to={target_lang}"
                headers = {
                    "Ocp-Apim-Subscription-Key": client["api_key"],
                    "Ocp-Apim-Subscription-Region": region,
                    "Content-type": "application/json"
                }
                body = [{"text": text}]
                
                response = requests.post(url, headers=headers, json=body)
                response.raise_for_status()
                result = response.json()
                return result[0]['translations'][0]['text']
            except Exception as e:
                print(f"Microsoft Translator failed: {e}")
        
        # Try Google Translate
        if self.is_enabled("google_translate"):
            try:
                client = self.clients["google_translate"]
                url = "https://translation.googleapis.com/language/translate/v2"
                params = {
                    'key': client["api_key"],
                    'q': text,
                    'target': target_lang,
                    'source': source_lang if source_lang != 'auto' else ''
                }
                response = requests.post(url, data=params)
                response.raise_for_status()
                result = response.json()
                return result['data']['translations'][0]['translatedText']
            except Exception as e:
                print(f"Google Translate failed: {e}")
        
        # Try LibreTranslate (free/self-hosted)
        if self.is_enabled("libre_translate"):
            try:
                config = self.config["libre_translate"]
                url = config["url"]
                data = {
                    "q": text,
                    "source": source_lang if source_lang != 'auto' else 'auto',
                    "target": target_lang,
                    "format": "text"
                }
                api_key = os.getenv(config.get("api_key_env", ""))
                if api_key:
                    data["api_key"] = api_key
                
                response = requests.post(url, data=data)
                response.raise_for_status()
                result = response.json()
                return result['translatedText']
            except Exception as e:
                print(f"LibreTranslate failed: {e}")
        
        # Fallback to Argos (offline)
        if hasattr(self, 'argos_available') and self.argos_available:
            try:
                import argostranslate.translate
                return argostranslate.translate.translate(text, source_lang, target_lang)
            except Exception as e:
                print(f"Argos Translate failed: {e}")
        
        # Final fallback
        return f"[Translation unavailable]: {text}"

    def chat_completion(self, prompt, system_prompt="You are a helpful assistant.", provider="openai"):
        """
        Universal chat completion with support for multiple providers.
        Supports: OpenAI, Anthropic, Google AI Studio (Gemini), HuggingFace, local models
        """
        if not self.is_enabled(provider):
            return {"error": f"{provider.capitalize()} is not enabled or configured."}

        client = self.clients[provider]
        model = client["config"].get("model", "gpt-4o")
        api_key = client["api_key"]

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        # OpenAI
        if provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            data = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            }
        
        # Anthropic (Claude)
        elif provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
            data = {
                "model": model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
                "system": system_prompt
            }
        
        # Google AI Studio (Gemini)
        elif provider == "google_ai_studio":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{
                    "parts": [{"text": f"{system_prompt}\n\n{prompt}"}]
                }]
            }
        
        # Grok (X AI)
        elif provider == "grok":
            url = "https://api.x.ai/v1/chat/completions"
            data = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            }
        
        # DeepSeek
        elif provider == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
            data = {
                "model": model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            }
        
        # HuggingFace Inference API
        elif provider == "huggingface":
            hf_model = client["config"]["models"]["chat"]
            url = f"https://api-inference.huggingface.co/models/{hf_model}"
            headers = {"Authorization": f"Bearer {api_key}"}
            data = {"inputs": f"{system_prompt}\n\nUser: {prompt}\nAssistant:"}
        
        else:
            return {"error": f"Provider '{provider}' not implemented."}

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            # Extract response based on provider format
            if provider == "openai" or provider in ["grok", "deepseek"]:
                return result
            elif provider == "anthropic":
                return result
            elif provider == "google_ai_studio":
                return {"content": [{"text": result["candidates"][0]["content"]["parts"][0]["text"]}]}
            elif provider == "huggingface":
                return {"content": [{"text": result[0]["generated_text"]}]}
            
            return result
        except requests.exceptions.RequestException as e:
            return {"error": f"API request to {provider} failed: {e}"}

    def chat_with_fallback(self, prompt, system_prompt="You are a helpful assistant."):
        """
        Cascading chat completion with automatic fallback across providers.
        Priority: Anthropic (highest quality) → OpenAI → Google Gemini → DeepSeek → Grok → HuggingFace → Local Models
        """
        providers = ["anthropic", "openai", "google_ai_studio", "deepseek", "grok", "huggingface", "local_models"]
        
        for provider in providers:
            if not self.is_enabled(provider):
                continue
            
            try:
                print(f"[CHAT] Attempting {provider}...")
                
                if provider == "local_models":
                    result = self._local_chat_completion(prompt, system_prompt)
                else:
                    result = self.chat_completion(prompt, system_prompt, provider)
                
                if "error" not in result:
                    print(f"[CHAT] Success with {provider}")
                    return result
                else:
                    print(f"[CHAT] {provider} returned error: {result['error']}")
            
            except Exception as e:
                print(f"[CHAT] {provider} failed: {e}")
                continue
        
        return {"error": "All chat providers failed", "content": [{"text": "[AI response unavailable]"}]}

    def _local_chat_completion(self, prompt, system_prompt):
        """
        Local model inference for chat completion (offline fallback).
        Attempts to use: Flan-T5, mT5, Gemma from transformers library.
        """
        if not self.local_models:
            return {"error": "No local models available"}
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            import torch
            
            # Priority: flan-t5 → mt5 → gemma
            model_priority = ["flan-t5-base", "google/flan-t5-base", "google/mt5-base"]
            
            for model_name in model_priority:
                try:
                    print(f"[LOCAL] Loading {model_name}...")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    
                    # Create input
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                    inputs = tokenizer(full_prompt, return_tensors="pt", max_length=512, truncation=True)
                    
                    # Generate
                    outputs = model.generate(**inputs, max_length=256)
                    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    return {"content": [{"text": response_text}]}
                
                except Exception as e:
                    print(f"[LOCAL] Failed to load {model_name}: {e}")
                    continue
            
            return {"error": "All local models failed to load"}
        
        except ImportError:
            return {"error": "transformers library not installed. Install with: pip install transformers torch"}

# Singleton instance
unified_connector = UnifiedConnector()
