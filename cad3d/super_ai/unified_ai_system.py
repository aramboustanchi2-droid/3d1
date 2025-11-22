"""
Unified AI System - یکپارچه‌سازی 5 مدل هوش مصنوعی
ترکیب RAG + Fine-Tuning + LoRA + Prompt Engineering + PEFT + Security

این سیستم جامع همه روش‌های AI را با سیستم امنیتی پیشرفته یکپارچه می‌کند
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum, auto
import time

logger = logging.getLogger(__name__)

# ===========================
# AI Method Types
# ===========================

class AIMethodType(Enum):
    """انواع روش‌های AI"""
    RAG = "retrieval_augmented_generation"
    FINE_TUNING = "fine_tuning"
    LORA = "low_rank_adaptation"
    PROMPT_ENGINEERING = "prompt_engineering"
    PEFT = "peft"

class AITaskType(Enum):
    """انواع وظایف AI"""
    CAD_ANALYSIS = auto()
    ARCHITECTURAL_DESIGN = auto()
    STRUCTURAL_CALCULATION = auto()
    MEP_OPTIMIZATION = auto()
    CODE_COMPLIANCE = auto()
    MATERIAL_ESTIMATION = auto()
    GENERAL_QUERY = auto()

# ===========================
# Unified AI System
# ===========================

class UnifiedAISystem:
    """
    سیستم یکپارچه AI با 5 روش:
    1. RAG - بازیابی و تولید
    2. Fine-Tuning - آموزش عمیق
    3. LoRA - تطبیق کم‌رتبه
    4. Prompt Engineering - مهندسی پرامپت
    5. PEFT - تنظیم کارآمد پارامترها
    
    + یکپارچگی کامل با سیستم امنیتی
    """
    
    def __init__(self, storage_dir: str = "models/unified_ai"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # سیستم‌های فرعی
        self.rag_system = None
        self.fine_tuning_system = None
        self.lora_system = None
        self.prompt_system = None
        self.peft_system = None
        self.security_dashboard = None
        self.meta_controller = None
        
        # آمار استفاده
        self.usage_stats = {
            "rag_calls": 0,
            "fine_tuning_calls": 0,
            "lora_calls": 0,
            "prompt_calls": 0,
            "peft_calls": 0,
            "hybrid_calls": 0,
            "total_queries": 0
        }
        
        # تنظیمات روتینگ خودکار
        self.auto_routing = True
        
        self._initialize_systems()
    
    def _initialize_systems(self):
        """راه‌اندازی تمام سیستم‌های فرعی"""
        logger.info("="*80)
        logger.info("🚀 INITIALIZING UNIFIED AI SYSTEM")
        logger.info("="*80)
        
        # 1. RAG System
        try:
            from .rag_system import RAGSystem
            self.rag_system = RAGSystem(storage_dir=os.path.join(self.storage_dir, "rag"))
            logger.info("✅ RAG System initialized")
        except Exception as e:
            logger.error(f"❌ RAG System failed: {e}")
        
        # 2. Fine-Tuning System (شبیه‌سازی)
        self.fine_tuning_system = {
            "status": "ready",
            "models": ["cad_analysis_v1", "architectural_design_v2"],
            "last_training": "2025-11-20"
        }
        logger.info("✅ Fine-Tuning System initialized")
        
        # 3. LoRA System (شبیه‌سازی)
        self.lora_system = {
            "status": "ready",
            "adapters": ["structural_calc", "mep_optimization"],
            "rank": 8
        }
        logger.info("✅ LoRA System initialized")
        
        # 4. Prompt Engineering System
        from .prompt_engineering import PromptEngineeringManager
        self.prompt_system = PromptEngineeringManager()
        logger.info("✅ Prompt Engineering System initialized")
        
        # 5. PEFT System
        try:
            from .peft_system import PEFTManager
            self.peft_system = PEFTManager()
            logger.info("✅ PEFT System initialized")
        except Exception as e:
            logger.warning(f"⚠️ PEFT System not available: {e}")
        
        # 6. Meta-Controller (AI Method Selector)
        try:
            from .meta_controller import MetaController
            self.meta_controller = MetaController()
            logger.info("✅ Meta-Controller initialized")
        except Exception as e:
            logger.warning(f"⚠️ Meta-Controller not available: {e}")
        
        # 7. Security Dashboard
        try:
            from .advanced_security import SecurityDashboard
            self.security_dashboard = SecurityDashboard()
            logger.info("✅ Security Dashboard integrated")
        except Exception as e:
            logger.warning(f"⚠️ Security Dashboard not available: {e}")
        
        logger.info("="*80)
        logger.info("🎉 UNIFIED AI SYSTEM READY")
        logger.info("="*80 + "\n")
    
    def query(
        self,
        query: str,
        method: Optional[AIMethodType] = None,
        task_type: Optional[AITaskType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        پرسش یکپارچه با انتخاب خودکار یا دستی روش
        
        Args:
            query: پرسش کاربر
            method: روش AI (اختیاری - خودکار انتخاب می‌شود)
            task_type: نوع وظیفه (اختیاری)
            **kwargs: پارامترهای اضافی
        
        Returns:
            پاسخ کامل با متادیتا
        """
        # بررسی امنیتی
        if self.security_dashboard:
            if not self._security_check(query):
                return {
                    "error": "Security check failed",
                    "status": "blocked"
                }
        
        # انتخاب خودکار روش
        decision_explanation = None
        if method is None and self.auto_routing:
            method, decision_explanation = self._select_best_method(query, task_type)
        
        # اجرای درخواست
        start_time = time.time()
        response = self._execute_query(query, method, task_type, **kwargs)
        execution_time = time.time() - start_time
        
        # اضافه کردن توضیحات تصمیم
        if decision_explanation:
            response["selection_reasoning"] = decision_explanation
        
        # به‌روزرسانی آمار
        self.usage_stats["total_queries"] += 1
        if method == AIMethodType.RAG:
            self.usage_stats["rag_calls"] += 1
        elif method == AIMethodType.FINE_TUNING:
            self.usage_stats["fine_tuning_calls"] += 1
        elif method == AIMethodType.LORA:
            self.usage_stats["lora_calls"] += 1
        elif method == AIMethodType.PROMPT_ENGINEERING:
            self.usage_stats["prompt_calls"] += 1
        elif method == AIMethodType.PEFT:
            self.usage_stats["peft_calls"] += 1
        
        # به‌روزرسانی عملکرد در Meta-Controller
        if self.meta_controller and decision_explanation:
            success = response.get("status") == "success"
            method_name = self._method_enum_to_name(method)
            self.meta_controller.update_performance(method_name, success, execution_time)
        
        return response
    
    def _security_check(self, query: str) -> bool:
        """بررسی امنیتی پرسش"""
        # بررسی الگوهای مشکوک
        suspicious_patterns = [
            "delete", "drop", "truncate", "exec",
            "system", "os.", "subprocess", "__import__"
        ]
        
        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if pattern in query_lower:
                logger.warning(f"🚨 Suspicious pattern detected: {pattern}")
                return False
        
        return True
    
    def _select_best_method(
        self,
        query: str,
        task_type: Optional[AITaskType]
    ) -> tuple[AIMethodType, Optional[Dict]]:
        """
        انتخاب خودکار بهترین روش با Meta-Controller هوشمند
        
        اگر Meta-Controller موجود باشد، از تحلیل پیشرفته استفاده می‌کند
        در غیر این صورت، از روش ساده کلیدواژه‌ای استفاده می‌شود
        """
        
        query_lower = query.lower()
        
        # بررسی کلیدواژه‌های صریح (اولویت بالا)
        explicit_methods = {
            "rag": AIMethodType.RAG,
            "retrieval": AIMethodType.RAG,
            "fine-tun": AIMethodType.FINE_TUNING,
            "fine tun": AIMethodType.FINE_TUNING,
            "lora": AIMethodType.LORA,
            "peft": AIMethodType.PEFT,
            "prompt": AIMethodType.PROMPT_ENGINEERING
        }
        
        for keyword, method in explicit_methods.items():
            if keyword in query_lower:
                return method, {
                    "controller": "Explicit Keyword",
                    "keyword": keyword,
                    "note": f"User explicitly requested {method.value}"
                }
        
        # اگر Meta-Controller موجود است، از آن استفاده کن
        if self.meta_controller:
            features = self.meta_controller.analyze_query(
                query,
                task_type.name if task_type else None
            )
            
            method_name, score = self.meta_controller.select_best_method(features)
            
            # تبدیل نام به enum
            method_enum = self._name_to_method_enum(method_name)
            
            # توضیحات تصمیم
            explanation = {
                "controller": "Meta-Controller (Intelligent)",
                "selected_method": method_name,
                "score": f"{score.score:.1f}",
                "reasoning": score.reasoning,
                "features": {
                    "complexity": features.complexity.value,
                    "domain": features.domain,
                    "confidence_needed": f"{features.confidence_needed:.0%}"
                },
                "scores": {
                    "speed": f"{score.speed_score:.1f}",
                    "accuracy": f"{score.accuracy_score:.1f}",
                    "cost": f"{score.cost_score:.1f}"
                }
            }
            
            return method_enum, explanation
        
        # روش ساده (Fallback)
        return self._select_method_simple(query, task_type), {
            "controller": "Simple (Keyword-based)",
            "note": "Meta-Controller not available"
        }
    
    def _select_method_simple(
        self,
        query: str,
        task_type: Optional[AITaskType]
    ) -> AIMethodType:
        """
        روش ساده انتخاب بر اساس کلیدواژه (Fallback)
        """
        query_lower = query.lower()
        
        # کلمات کلیدی برای RAG (نیاز به دانش واقعی)
        rag_keywords = [
            "محاسبه", "چقدر", "چند", "مساحت", "حجم",
            "استاندارد", "ضوابط", "مبحث", "قانون",
            "calculate", "how much", "how many", "area", "volume"
        ]
        
        # کلمات کلیدی برای Fine-Tuning (تحلیل پیچیده)
        fine_tuning_keywords = [
            "تحلیل", "طراحی", "بهینه", "پیشنهاد",
            "analyze", "design", "optimize", "suggest"
        ]
        
        # کلمات کلیدی برای LoRA (تطبیق سریع)
        lora_keywords = [
            "سازه", "تاسیسات", "برق", "لوله‌کشی",
            "structural", "mep", "electrical", "plumbing"
        ]
        
        # کلمات کلیدی برای PEFT (فناوری‌های پارامتر-کارا)
        peft_keywords = [
            "peft", "prefix", "p-tuning", "ptuning", "ia3", "adalora", "qlora", "adapter"
        ]
        
        # بررسی کلمات کلیدی
        rag_score = sum(1 for kw in rag_keywords if kw in query_lower)
        ft_score = sum(1 for kw in fine_tuning_keywords if kw in query_lower)
        lora_score = sum(1 for kw in lora_keywords if kw in query_lower)
        peft_score = sum(1 for kw in peft_keywords if kw in query_lower)
        
        # تصمیم‌گیری
        if rag_score >= 2:
            return AIMethodType.RAG
        elif ft_score >= 1 and task_type in [AITaskType.CAD_ANALYSIS, AITaskType.ARCHITECTURAL_DESIGN]:
            return AIMethodType.FINE_TUNING
        elif peft_score >= 1:
            return AIMethodType.PEFT
        elif lora_score >= 1:
            return AIMethodType.LORA
        else:
            return AIMethodType.PROMPT_ENGINEERING
    
    def _name_to_method_enum(self, name: str) -> AIMethodType:
        """تبدیل نام روش به enum"""
        mapping = {
            "RAG": AIMethodType.RAG,
            "Fine-Tuning": AIMethodType.FINE_TUNING,
            "LoRA": AIMethodType.LORA,
            "Prompt Engineering": AIMethodType.PROMPT_ENGINEERING,
            "PEFT": AIMethodType.PEFT
        }
        return mapping.get(name, AIMethodType.PROMPT_ENGINEERING)
    
    def _method_enum_to_name(self, method: AIMethodType) -> str:
        """تبدیل enum به نام روش"""
        mapping = {
            AIMethodType.RAG: "RAG",
            AIMethodType.FINE_TUNING: "Fine-Tuning",
            AIMethodType.LORA: "LoRA",
            AIMethodType.PROMPT_ENGINEERING: "Prompt Engineering",
            AIMethodType.PEFT: "PEFT"
        }
        return mapping.get(method, "Prompt Engineering")
    
    def _execute_query(
        self,
        query: str,
        method: AIMethodType,
        task_type: Optional[AITaskType],
        **kwargs
    ) -> Dict[str, Any]:
        """اجرای پرسش با روش انتخاب شده"""
        
        response = {
            "query": query,
            "method": method.value,
            "task_type": task_type.name if task_type else None,
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
        
        try:
            if method == AIMethodType.RAG:
                result = self._execute_rag(query, **kwargs)
            elif method == AIMethodType.FINE_TUNING:
                result = self._execute_fine_tuning(query, **kwargs)
            elif method == AIMethodType.LORA:
                result = self._execute_lora(query, **kwargs)
            elif method == AIMethodType.PROMPT_ENGINEERING:
                result = self._execute_prompt(query, **kwargs)
            elif method == AIMethodType.PEFT:
                result = self._execute_peft(query, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            response.update(result)
            
        except Exception as e:
            logger.error(f"❌ Query execution failed: {e}")
            response["status"] = "error"
            response["error"] = str(e)
        
        return response
    
    def _execute_rag(self, query: str, **kwargs) -> Dict[str, Any]:
        """اجرای RAG"""
        if not self.rag_system:
            return {"error": "RAG system not available"}
        
        top_k = kwargs.get("top_k", 3)
        
        # بازیابی اسناد
        results = self.rag_system.retrieve(query, top_k=top_k)
        
        # تولید پاسخ
        rag_response = self.rag_system.generate_rag_response(query, top_k=top_k)
        
        return {
            "method_details": "RAG - Retrieval-Augmented Generation",
            "retrieved_documents": rag_response["retrieved_documents"],
            "num_docs": len(results),
            "prompt": rag_response["prompt"]
        }
    
    def _execute_fine_tuning(self, query: str, **kwargs) -> Dict[str, Any]:
        """اجرای Fine-Tuning"""
        # شبیه‌سازی Fine-Tuning
        model_name = kwargs.get("model", "cad_analysis_v1")
        
        return {
            "method_details": "Fine-Tuning - Specialized trained model",
            "model_used": model_name,
            "training_date": self.fine_tuning_system["last_training"],
            "note": "Using fine-tuned model for specialized CAD analysis"
        }
    
    def _execute_lora(self, query: str, **kwargs) -> Dict[str, Any]:
        """اجرای LoRA"""
        # شبیه‌سازی LoRA
        adapter = kwargs.get("adapter", "structural_calc")
        
        return {
            "method_details": "LoRA - Low-Rank Adaptation",
            "adapter_used": adapter,
            "rank": self.lora_system["rank"],
            "note": "Using LoRA adapter for efficient domain adaptation"
        }
    
    def _execute_prompt(self, query: str, **kwargs) -> Dict[str, Any]:
        """اجرای Prompt Engineering"""
        if not self.prompt_system:
            return {"error": "Prompt system not available"}
        
        template_type = kwargs.get("template", "architectural_analysis")
        
        # تولید پرامپت
        prompt_data = {
            "user_query": query,
            "domain": "architecture",
            "language": "fa"
        }
        # Generate prompt using manager
        try:
            gen = self.prompt_system.generate_prompt(
                query=query,
                task_type="architectural",
                template_name=template_type,
                **prompt_data
            )
            prompt = gen["prompt"] if isinstance(gen, dict) else gen
        except Exception as e:
            return {"status": "error", "error": str(e)}
        
        return {
            "method_details": "Prompt Engineering - Carefully crafted prompts",
            "template_used": template_type,
            "prompt": prompt
        }
    
    def _execute_peft(self, query: str, **kwargs) -> Dict[str, Any]:
        """اجرای PEFT (Parameter-Efficient Fine-Tuning)"""
        if not self.peft_system:
            return {"error": "PEFT system not available"}
        adapter = kwargs.get("adapter", None)
        technique = kwargs.get("technique", None)
        task = kwargs.get("task", None)
        result = self.peft_system.apply(query=query, task_type=task, adapter=adapter, technique=technique)
        return {
            "method_details": "PEFT - Parameter-Efficient Fine-Tuning",
            "technique": result.get("technique"),
            "adapter_used": result.get("adapter"),
            "peft_available": result.get("peft_available", False),
            "note": result.get("note")
        }
    
    def hybrid_query(
        self,
        query: str,
        methods: List[AIMethodType],
        **kwargs
    ) -> Dict[str, Any]:
        """
        پرسش ترکیبی با چند روش همزمان
        
        Example: RAG + Prompt Engineering
        """
        self.usage_stats["hybrid_calls"] += 1
        
        responses = {}
        for method in methods:
            result = self._execute_query(query, method, None, **kwargs)
            responses[method.value] = result
        
        return {
            "query": query,
            "methods_used": [m.value for m in methods],
            "timestamp": datetime.now().isoformat(),
            "individual_responses": responses,
            "hybrid": True
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """دریافت وضعیت کامل سیستم"""
        status = {
            "unified_ai_system": {
                "status": "operational",
                "methods_available": []
            },
            "rag": None,
            "fine_tuning": None,
            "lora": None,
            "prompt_engineering": None,
            "peft": None,
            "security": None,
            "usage_statistics": self.usage_stats
        }
        
        # RAG
        if self.rag_system:
            status["unified_ai_system"]["methods_available"].append("RAG")
            status["rag"] = self.rag_system.get_statistics()
        
        # Fine-Tuning
        if self.fine_tuning_system:
            status["unified_ai_system"]["methods_available"].append("Fine-Tuning")
            status["fine_tuning"] = self.fine_tuning_system
        
        # LoRA
        if self.lora_system:
            status["unified_ai_system"]["methods_available"].append("LoRA")
            status["lora"] = self.lora_system
        
        # Prompt Engineering
        if self.prompt_system:
            status["unified_ai_system"]["methods_available"].append("Prompt Engineering")
            status["prompt_engineering"] = {"status": "ready"}
        
        # PEFT
        if self.peft_system:
            status["unified_ai_system"]["methods_available"].append("PEFT")
            status["peft"] = self.peft_system.get_status()
        
        # Meta-Controller
        if self.meta_controller:
            status["meta_controller"] = self.meta_controller.get_performance_stats()
        
        # Security
        if self.security_dashboard:
            status["security"] = {
                "status": self.security_dashboard.current_status.value,
                "mother_key_locked": self.security_dashboard.mother_key.is_locked,
                "agents_created": self.security_dashboard.agent_manager.total_created
            }
        
        return status
    
    def compare_methods(self) -> Dict[str, Any]:
        """مقایسه 5 روش AI"""
        return {
            "comparison": {
                "RAG": {
                    "setup_time": "Minutes",
                    "cost": "Low ($0-$10)",
                    "quality": "Excellent for facts",
                    "gpu_required": False,
                    "best_for": [
                        "Knowledge-based queries",
                        "Frequently updated info",
                        "Multi-document reasoning",
                        "Transparent sources"
                    ],
                    "when_to_use": "Need accurate, source-backed answers"
                },
                "Fine-Tuning": {
                    "setup_time": "Hours to Days",
                    "cost": "Medium ($100-$1000)",
                    "quality": "Excellent for specialized tasks",
                    "gpu_required": True,
                    "best_for": [
                        "Domain-specific tasks",
                        "Complex reasoning",
                        "Consistent style",
                        "Production deployment"
                    ],
                    "when_to_use": "Have labeled data, need specialized model"
                },
                "LoRA": {
                    "setup_time": "Hours",
                    "cost": "Low ($10-$100)",
                    "quality": "Very good, efficient",
                    "gpu_required": True,
                    "best_for": [
                        "Quick adaptation",
                        "Multiple domains",
                        "Resource-constrained",
                        "Frequent updates"
                    ],
                    "when_to_use": "Need fast adaptation with less data"
                },
                "Prompt Engineering": {
                    "setup_time": "Minutes",
                    "cost": "Very Low ($0-$5)",
                    "quality": "Good to Excellent",
                    "gpu_required": False,
                    "best_for": [
                        "Quick prototyping",
                        "General tasks",
                        "No training data",
                        "Flexible requirements"
                    ],
                    "when_to_use": "No training data, quick iteration needed"
                },
                "PEFT": {
                    "setup_time": "Minutes to Hours",
                    "cost": "Low ($0-$50)",
                    "quality": "Very Good",
                    "gpu_required": False,
                    "best_for": [
                        "Adapter-based domain updates",
                        "Limited compute",
                        "Multiple adapters",
                        "Efficient updates"
                    ],
                    "when_to_use": "Need fast efficient fine-tuning without full retrain"
                }
            },
            "recommendation": {
                "start_with": "RAG + Prompt Engineering (lowest cost, fastest)",
                "scale_to": "PEFT or LoRA or Fine-Tuning (better quality, specialized)",
                "best_hybrid": "RAG + Prompt Engineering (knowledge + structure) + PEFT for adapters",
                "production": "Fine-Tuning + RAG (specialized + updated info)"
            }
        }
    
    def save_configuration(self, name: str = "default"):
        """ذخیره تنظیمات سیستم"""
        config = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "auto_routing": self.auto_routing,
            "usage_stats": self.usage_stats,
            "systems": {
                "rag": self.rag_system is not None,
                "fine_tuning": self.fine_tuning_system is not None,
                "lora": self.lora_system is not None,
                "prompt": self.prompt_system is not None,
                "peft": self.peft_system is not None,
                "security": self.security_dashboard is not None
            }
        }
        
        filepath = os.path.join(self.storage_dir, f"{name}_config.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Configuration saved: {filepath}")
    
    def explain_selection(self, query: str, task_type: Optional[AITaskType] = None) -> Dict:
        """
        توضیح دلیل انتخاب روش برای یک پرسش (بدون اجرا)
        
        Args:
            query: پرسش کاربر
            task_type: نوع وظیفه (اختیاری)
        
        Returns:
            توضیحات کامل تصمیم‌گیری
        """
        if not self.meta_controller:
            return {
                "error": "Meta-Controller not available",
                "fallback": "Using simple keyword-based selection"
            }
        
        features = self.meta_controller.analyze_query(
            query,
            task_type.name if task_type else None
        )
        
        # محاسبه امتیاز همه روش‌ها
        all_methods = ["RAG", "Fine-Tuning", "LoRA", "Prompt Engineering", "PEFT"]
        scores = []
        
        for method in all_methods:
            score = self.meta_controller._score_method(method, features)
            scores.append(score)
        
        scores.sort(key=lambda x: x.score, reverse=True)
        
        best_method = scores[0].method
        best_score = scores[0]
        
        return self.meta_controller.explain_decision(
            query,
            features,
            best_method,
            best_score,
            scores
        )
    
    def get_performance_stats(self) -> Dict:
        """دریافت آمار عملکرد Meta-Controller"""
        if not self.meta_controller:
            return {"error": "Meta-Controller not available"}
        
        return self.meta_controller.get_performance_stats()

# ===========================
# Global Instance
# ===========================

unified_ai = UnifiedAISystem()
