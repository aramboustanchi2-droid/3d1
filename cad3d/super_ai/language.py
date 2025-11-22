import logging
import json
import os
import random

logger = logging.getLogger(__name__)

# Import external connector for real translation
try:
    from .external_connectors import unified_connector
    EXTERNAL_CONNECTORS_AVAILABLE = True
except ImportError:
    EXTERNAL_CONNECTORS_AVAILABLE = False
    logger.warning("External connectors not available. Using fallback translation.")

class LanguageModule:
    def __init__(self, knowledge_base_path="super_ai_knowledge_base.json"):
        self.knowledge_base_path = knowledge_base_path
        self.supported_languages = ["en", "zh", "fa"]
        self.use_external_translation = EXTERNAL_CONNECTORS_AVAILABLE
        self.current_fluency = {
            "en": 1.0,
            "zh": 0.1,  # Initial low fluency
            "fa": 0.1   # Initial low fluency
        }
        self.vocabulary = {
            "en": {},
            "zh": {
                "architecture": "建筑", "design": "设计", "plan": "平面图", "structure": "结构",
                "analysis": "分析", "decision": "决策", "leadership": "领导", "execution": "执行",
                "proposal": "提案", "approved": "批准", "training": "训练", "system": "系统",
                "intelligence": "智能", "council": "委员会", "verdict": "裁决"
            },
            "fa": {
                "Architecture": "معماری", "architecture": "معماری",
                "Design": "طراحی", "design": "طراحی",
                "Plan": "نقشه", "plan": "نقشه",
                "Structure": "سازه", "structure": "سازه",
                "Analysis": "تحلیل", "analysis": "تحلیل",
                "Decision": "تصمیم‌گیری", "decision": "تصمیم‌گیری",
                "Leadership": "رهبری", "leadership": "رهبری",
                "Execution": "اجرا", "execution": "اجرا",
                "Execute": "اجرای", "execute": "اجرای",
                "Proposal": "پیشنهاد", "proposal": "پیشنهاد",
                "Approved": "تایید شد", "approved": "تایید شد",
                "Training": "آموزش", "training": "آموزش",
                "System": "سیستم", "system": "سیستم",
                "Intelligence": "هوش", "intelligence": "هوش",
                "Agent": "عامل", "agent": "عامل",
                "Council": "شورا", "council": "شورا",
                "Central": "مرکزی", "central": "مرکزی",
                "Economic": "اقتصادی", "economic": "اقتصادی",
                "Computational": "محاسباتی", "computational": "محاسباتی",
                "Ideation": "ایده‌پردازی", "ideation": "ایده‌پردازی",
                "Verdict": "حکم", "verdict": "حکم",
                "The": "", "the": "",
                "Has": "است", "has": "است",
                "Decided": "تصمیم گرفت", "decided": "تصمیم گرفت",
                "Project": "پروژه", "project": "پروژه",
                "Blueprint": "نقشه", "blueprint": "نقشه",
                "Generated": "تولید شد", "generated": "تولید شد",
                "Of": "از", "of": "از",
                "A": "یک", "a": "یک"
            }
        }
        self._load_knowledge()

    def _load_knowledge(self):
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "language_module" in data:
                        lang_data = data["language_module"]
                        loaded_fluency = lang_data.get("fluency", {})
                        loaded_vocab = lang_data.get("vocabulary", {})
                        
                        # Update existing dicts instead of overwriting to preserve new defaults
                        self.current_fluency.update(loaded_fluency)
                        self.vocabulary.update(loaded_vocab)
                        
                        # Ensure 'fa' exists if not loaded
                        if "fa" not in self.vocabulary:
                            self.vocabulary["fa"] = {}
                        if "fa" not in self.current_fluency:
                            self.current_fluency["fa"] = 0.1
            except Exception as e:
                logger.error(f"Failed to load language knowledge: {e}")

    def save_knowledge(self):
        # Load existing data first to avoid overwriting other modules
        data = {}
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                pass
        
        data["language_module"] = {
            "fluency": self.current_fluency,
            "vocabulary": self.vocabulary
        }
        
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def train_language(self, language_code, dataset_name="general"):
        """
        Simulates training on a language dataset.
        """
        logger.info(f"Training Language Module on {language_code} ({dataset_name})...")
        
        if language_code not in self.supported_languages:
            logger.warning(f"Language {language_code} not fully supported yet.")
            return
            
        # Simulate learning curve
        current = self.current_fluency.get(language_code, 0.0)
        improvement = random.uniform(0.1, 0.3)
        new_level = min(1.0, current + improvement)
        self.current_fluency[language_code] = new_level
        
        # Add simulated vocabulary
        if language_code == "zh":
            self.vocabulary["zh"].update({
                "architecture": "建筑",
                "design": "设计",
                "plan": "平面图",
                "structure": "结构",
                "analysis": "分析",
                "decision": "决策",
                "leadership": "领导",
                "execution": "执行",
                "proposal": "提案",
                "approved": "批准",
                "training": "训练",
                "system": "系统",
                "intelligence": "智能"
            })
        elif language_code == "fa":
            self.vocabulary["fa"].update({
                "architecture": "معماری",
                "design": "طراحی",
                "plan": "نقشه",
                "structure": "سازه",
                "analysis": "تحلیل",
                "decision": "تصمیم‌گیری",
                "leadership": "رهبری",
                "execution": "اجرا",
                "proposal": "پیشنهاد",
                "approved": "تایید شد",
                "training": "آموزش",
                "system": "سیستم",
                "intelligence": "هوش",
                "agent": "ایجنت",
                "council": "شورا",
                "central": "مرکزی",
                "economic": "اقتصادی",
                "computational": "محاسباتی",
                "ideation": "ایده‌پردازی"
            })
            
        self.save_knowledge()
        return f"Fluency in {language_code} increased to {new_level:.2f}"

    def detect_language(self, text):
        """
        Simple heuristic for language detection.
        """
        # Check for Chinese characters
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return "zh"
        
        # Check for Persian/Arabic characters
        # Basic range for Arabic script which covers Persian
        for char in text:
            if '\u0600' <= char <= '\u06FF':
                return "fa"

        return "en"

    def translate(self, text, target_lang):
        """
        Translates text using external API (Google Translate) if available,
        otherwise falls back to dictionary-based translation.
        """
        # Try external translation first if enabled
        if self.use_external_translation and EXTERNAL_CONNECTORS_AVAILABLE:
            try:
                result = unified_connector.translate(text, target_lang=target_lang, source_lang='auto')
                if not isinstance(result, dict):  # Success case - result is string
                    logger.info(f"Successfully translated to {target_lang} using external API")
                    return result
                else:  # Error dict returned
                    logger.warning(f"External translation failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.warning(f"External translation error: {e}, falling back to dictionary")
        
        # Fallback to dictionary-based translation
        if target_lang == "zh":
            # Mock translation to Chinese
            # Replace known keywords
            translated = text
            for en_word, zh_word in self.vocabulary.get("zh", {}).items():
                translated = translated.replace(en_word, zh_word)
                translated = translated.replace(en_word.capitalize(), zh_word)
            return translated

        elif target_lang == "fa":
            # Mock translation to Persian
            translated = text
            for en_word, fa_word in self.vocabulary.get("fa", {}).items():
                translated = translated.replace(en_word, fa_word)
                translated = translated.replace(en_word.capitalize(), fa_word)
                translated = translated.replace(en_word.upper(), fa_word)
            return translated
                
        elif target_lang == "en":
            # Try external API for reverse translation
            if self.use_external_translation and EXTERNAL_CONNECTORS_AVAILABLE:
                try:
                    result = unified_connector.translate(text, target_lang='en', source_lang='auto')
                    if not isinstance(result, dict):
                        return result
                except Exception:
                    pass
            return text
            
        return text
