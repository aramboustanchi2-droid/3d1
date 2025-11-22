"""
Meta-Controller for Intelligent AI Method Selection

تحلیل‌گر هوشمند پرسش و انتخاب بهترین روش AI
بر اساس ویژگی‌های ورودی، تخصصی بودن، نیاز به سرعت/دقت
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class QueryComplexity(Enum):
    """پیچیدگی پرسش"""
    SIMPLE = "simple"           # ساده - محاسبه مستقیم
    MODERATE = "moderate"       # متوسط - نیاز به استدلال
    COMPLEX = "complex"         # پیچیده - تحلیل چندمرحله‌ای
    VERY_COMPLEX = "very_complex"  # بسیار پیچیده - نیاز به تخصص


class QueryUrgency(Enum):
    """فوریت پاسخ"""
    REALTIME = "realtime"       # لحظه‌ای < 100ms
    FAST = "fast"               # سریع < 1s
    NORMAL = "normal"           # عادی < 5s
    BATCH = "batch"             # دسته‌ای - بدون محدودیت


@dataclass
class QueryFeatures:
    """ویژگی‌های استخراج‌شده از پرسش"""
    length: int                          # طول متن
    has_numbers: bool                    # شامل اعداد
    has_technical_terms: bool            # اصطلاحات تخصصی
    requires_calculation: bool           # نیاز به محاسبه
    requires_knowledge: bool             # نیاز به دانش پایه
    requires_reasoning: bool             # نیاز به استدلال
    is_specialized: bool                 # تخصصی (معماری، سازه، تاسیسات)
    complexity: QueryComplexity          # پیچیدگی
    urgency: QueryUrgency               # فوریت
    domain: Optional[str]               # حوزه (architecture, structural, MEP, etc.)
    confidence_needed: float            # دقت موردنیاز (0-1)


@dataclass
class MethodScore:
    """امتیاز هر روش برای پرسش"""
    method: str
    score: float                        # امتیاز کلی (0-100)
    speed_score: float                  # امتیاز سرعت
    accuracy_score: float               # امتیاز دقت
    cost_score: float                   # امتیاز هزینه
    reasoning: str                      # دلیل انتخاب


class MetaController:
    """
    کنترلر هوشمند انتخاب روش AI
    
    تحلیل ویژگی‌های پرسش:
    - طول و پیچیدگی
    - تخصصی بودن
    - نیاز به داده به‌روز
    - سرعت پاسخ
    - حساسیت به خطا
    
    تصمیم‌گیری بر اساس:
    - امتیازدهی چندمعیاره
    - قوانین خبره
    - تاریخچه عملکرد
    """
    
    def __init__(self):
        # الگوهای تشخیص
        self.number_pattern = re.compile(r'\d+\.?\d*')
        self.calculation_keywords = [
            "محاسبه", "حساب", "چقدر", "چند", "جمع", "تفریق", "ضرب", "تقسیم",
            "calculate", "compute", "how much", "how many", "sum", "total"
        ]
        self.technical_terms = [
            "معماری", "سازه", "تاسیسات", "برق", "لوله‌کشی", "تهویه", "بتن", "فولاد",
            "architecture", "structural", "mep", "hvac", "concrete", "steel", "beam", "column"
        ]
        self.knowledge_keywords = [
            "استاندارد", "ضابطه", "مبحث", "قانون", "آیین‌نامه", "تعریف", "چیست",
            "بهینه", "طراحی", "ساختار", "ساختمان", "سازه‌ای",
            "standard", "code", "regulation", "definition", "what is", "explain",
            "design", "structure", "building", "structural", "optimize", "optimization"
        ]
        self.reasoning_keywords = [
            "تحلیل", "بررسی", "مقایسه", "ارزیابی", "بهینه", "پیشنهاد", "چرا",
            "analyze", "compare", "evaluate", "optimize", "suggest", "why", "because"
        ]
        
        # وزن‌ها برای تصمیم‌گیری (قابل تنظیم)
        self.weights = {
            "speed": 0.3,      # اهمیت سرعت
            "accuracy": 0.4,   # اهمیت دقت
            "cost": 0.3        # اهمیت هزینه
        }
        
        # آمار عملکرد روش‌ها (برای یادگیری)
        self.method_performance = {
            "RAG": {"success_rate": 0.95, "avg_time": 0.5},
            "Fine-Tuning": {"success_rate": 0.98, "avg_time": 1.2},
            "LoRA": {"success_rate": 0.92, "avg_time": 0.8},
            "Prompt Engineering": {"success_rate": 0.85, "avg_time": 0.3},
            "PEFT": {"success_rate": 0.90, "avg_time": 0.6}
        }
    
    def analyze_query(self, query: str, task_type: Optional[str] = None) -> QueryFeatures:
        """
        تحلیل کامل پرسش و استخراج ویژگی‌ها
        
        Args:
            query: متن پرسش
            task_type: نوع وظیفه (اختیاری)
        
        Returns:
            ویژگی‌های استخراج‌شده
        """
        query_lower = query.lower()
        
        # طول و اعداد
        length = len(query)
        has_numbers = bool(self.number_pattern.search(query))
        
        # تشخیص نیازها
        requires_calculation = any(kw in query_lower for kw in self.calculation_keywords)
        requires_knowledge = any(kw in query_lower for kw in self.knowledge_keywords)
        requires_reasoning = any(kw in query_lower for kw in self.reasoning_keywords)
        
        # تخصصی بودن
        has_technical_terms = any(term in query_lower for term in self.technical_terms)
        is_specialized = has_technical_terms or (task_type and task_type in [
            "CAD_ANALYSIS", "STRUCTURAL_CALCULATION", "MEP_OPTIMIZATION"
        ])
        
        # تشخیص حوزه
        domain = self._detect_domain(query_lower)
        
        # پیچیدگی
        complexity = self._assess_complexity(
            length, has_numbers, requires_calculation,
            requires_knowledge, requires_reasoning, is_specialized
        )
        
        # فوریت (پیش‌فرض بر اساس پیچیدگی)
        urgency = self._assess_urgency(complexity, requires_calculation)
        
        # دقت موردنیاز
        confidence_needed = self._assess_confidence_requirement(
            requires_calculation, is_specialized, has_numbers
        )
        
        return QueryFeatures(
            length=length,
            has_numbers=has_numbers,
            has_technical_terms=has_technical_terms,
            requires_calculation=requires_calculation,
            requires_knowledge=requires_knowledge,
            requires_reasoning=requires_reasoning,
            is_specialized=is_specialized,
            complexity=complexity,
            urgency=urgency,
            domain=domain,
            confidence_needed=confidence_needed
        )
    
    def _detect_domain(self, query_lower: str) -> Optional[str]:
        """تشخیص حوزه تخصصی"""
        domains = {
            "structural": ["سازه", "تیر", "ستون", "بتن", "فولاد", "ساختار سازه", "structural", "beam", "column", "concrete", "steel"],
            "mep": ["تاسیسات", "برق", "لوله‌کشی", "تهویه", "فاضلاب", "ظرفیت لوله", "mep", "hvac", "electrical", "plumbing", "pipe", "drainage"],
            "calculation": ["محاسبه", "حساب", "عدد", "calculate", "compute", "number"],
            "architecture": ["معماری", "طراحی", "اتاق", "خانه", "ساختمان", "فضا", "چیدمان", "architecture", "design", "room", "building", "layout"],
            "general": []
        }
        
        for domain, keywords in domains.items():
            if any(kw in query_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _assess_complexity(
        self,
        length: int,
        has_numbers: bool,
        requires_calculation: bool,
        requires_knowledge: bool,
        requires_reasoning: bool,
        is_specialized: bool
    ) -> QueryComplexity:
        """ارزیابی پیچیدگی پرسش"""
        
        score = 0
        
        # طول
        if length > 200: score += 2
        elif length > 100: score += 1
        
        # نیازها
        if requires_calculation: score += 1
        if requires_knowledge: score += 1
        if requires_reasoning: score += 2
        if is_specialized: score += 2
        if has_numbers and requires_calculation: score += 1
        
        if score >= 6:
            return QueryComplexity.VERY_COMPLEX
        elif score >= 4:
            return QueryComplexity.COMPLEX
        elif score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _assess_urgency(self, complexity: QueryComplexity, requires_calculation: bool) -> QueryUrgency:
        """ارزیابی فوریت پاسخ"""
        
        # محاسبات ساده نیاز به سرعت بالا دارند
        if requires_calculation and complexity == QueryComplexity.SIMPLE:
            return QueryUrgency.FAST
        
        # پرسش‌های پیچیده می‌توانند کندتر باشند
        if complexity == QueryComplexity.VERY_COMPLEX:
            return QueryUrgency.BATCH
        elif complexity == QueryComplexity.COMPLEX:
            return QueryUrgency.NORMAL
        else:
            return QueryUrgency.FAST
    
    def _assess_confidence_requirement(
        self,
        requires_calculation: bool,
        is_specialized: bool,
        has_numbers: bool
    ) -> float:
        """ارزیابی سطح دقت موردنیاز (0-1)"""
        
        confidence = 0.7  # پایه
        
        if requires_calculation: confidence += 0.15
        if is_specialized: confidence += 0.10
        if has_numbers: confidence += 0.05
        
        return min(confidence, 1.0)
    
    def select_best_method(
        self,
        features: QueryFeatures,
        available_methods: Optional[List[str]] = None
    ) -> Tuple[str, MethodScore]:
        """
        انتخاب بهترین روش بر اساس ویژگی‌های پرسش
        
        Args:
            features: ویژگی‌های پرسش
            available_methods: روش‌های در دسترس (همه اگر None)
        
        Returns:
            (روش انتخاب‌شده, امتیازات)
        """
        
        if available_methods is None:
            available_methods = ["RAG", "Fine-Tuning", "LoRA", "Prompt Engineering", "PEFT"]
        
        # محاسبه امتیاز هر روش
        scores = []
        
        for method in available_methods:
            score = self._score_method(method, features)
            scores.append(score)
        
        # مرتب‌سازی بر اساس امتیاز کلی
        scores.sort(key=lambda x: x.score, reverse=True)
        
        best_method = scores[0].method
        best_score = scores[0]
        
        return best_method, best_score
    
    def _score_method(self, method: str, features: QueryFeatures) -> MethodScore:
        """محاسبه امتیاز یک روش برای پرسش"""
        
        # امتیازات پایه هر روش
        base_scores = {
            "RAG": {
                "speed": 85,      # سریع
                "accuracy": 90,   # دقیق (با منابع)
                "cost": 90        # ارزان
            },
            "Fine-Tuning": {
                "speed": 70,      # کندتر
                "accuracy": 95,   # بسیار دقیق
                "cost": 40        # گران
            },
            "LoRA": {
                "speed": 75,      # متوسط
                "accuracy": 88,   # خوب
                "cost": 70        # متوسط
            },
            "Prompt Engineering": {
                "speed": 95,      # خیلی سریع
                "accuracy": 75,   # متوسط
                "cost": 95        # خیلی ارزان
            },
            "PEFT": {
                "speed": 80,      # سریع
                "accuracy": 85,   # خوب
                "cost": 80        # ارزان
            }
        }
        
        scores = base_scores[method].copy()
        reasoning_parts = []
        
        # تنظیمات بر اساس ویژگی‌های پرسش
        
        # RAG برای دانش و محاسبات عالی است
        if method == "RAG":
            if features.requires_knowledge:
                scores["accuracy"] += 5
                reasoning_parts.append("نیاز به دانش پایه")
            if features.requires_calculation:
                scores["accuracy"] += 3
                reasoning_parts.append("محاسبه با منابع")
            if features.urgency == QueryUrgency.FAST:
                scores["speed"] += 5
                reasoning_parts.append("پاسخ سریع")
        
        # Fine-Tuning برای کارهای تخصصی پیچیده
        elif method == "Fine-Tuning":
            if features.is_specialized:
                scores["accuracy"] += 10
                reasoning_parts.append("تخصصی")
            if features.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
                scores["accuracy"] += 15
                scores["speed"] -= 10  # کندتر اما دقیق‌تر
                reasoning_parts.append("پیچیدگی بالا")
            if features.confidence_needed > 0.9:
                scores["accuracy"] += 8
                reasoning_parts.append("نیاز به دقت بالا")
            if features.urgency == QueryUrgency.BATCH:
                scores["speed"] += 10  # زمان مهم نیست
            if features.requires_reasoning:
                scores["accuracy"] += 10
                reasoning_parts.append("استدلال پیچیده")
        
        # LoRA برای تطبیق سریع
        elif method == "LoRA":
            if features.is_specialized and features.complexity == QueryComplexity.MODERATE:
                scores["accuracy"] += 8
                reasoning_parts.append("تخصصی متوسط")
            if features.domain in ["structural", "mep"]:
                scores["accuracy"] += 10
                scores["speed"] += 5
                reasoning_parts.append(f"حوزه {features.domain}")
            if features.requires_calculation and features.is_specialized:
                scores["accuracy"] += 5
                reasoning_parts.append("محاسبه تخصصی")
        
        # Prompt Engineering برای سرعت
        elif method == "Prompt Engineering":
            if features.urgency == QueryUrgency.FAST:
                scores["speed"] += 5
                reasoning_parts.append("نیاز به سرعت")
            if features.complexity == QueryComplexity.SIMPLE:
                scores["accuracy"] += 15
                scores["speed"] += 5
                reasoning_parts.append("پرسش ساده")
            if not features.is_specialized:
                scores["accuracy"] += 8
                reasoning_parts.append("عمومی")
            if not features.requires_reasoning:
                scores["accuracy"] += 5
                reasoning_parts.append("بدون استدلال پیچیده")
        
        # PEFT برای کارآیی
        elif method == "PEFT":
            if features.is_specialized and features.confidence_needed > 0.8:
                scores["accuracy"] += 10
                scores["speed"] += 5
                reasoning_parts.append("تخصصی با دقت")
            if features.complexity == QueryComplexity.MODERATE:
                scores["accuracy"] += 8
                reasoning_parts.append("پیچیدگی متوسط")
            if features.domain == "mep":
                scores["accuracy"] += 5
                reasoning_parts.append("مناسب MEP")
        
        # محاسبه امتیاز نهایی
        speed_score = min(scores["speed"], 100)
        accuracy_score = min(scores["accuracy"], 100)
        cost_score = min(scores["cost"], 100)
        
        # امتیاز وزن‌دار
        final_score = (
            speed_score * self.weights["speed"] +
            accuracy_score * self.weights["accuracy"] +
            cost_score * self.weights["cost"]
        )
        
        # اضافه کردن تاریخچه عملکرد (با وزن کمتر)
        perf = self.method_performance[method]
        performance_factor = 0.7 + (perf["success_rate"] * 0.3)  # 70% base + 30% history
        final_score *= performance_factor
        
        reasoning = f"{method}: " + ", ".join(reasoning_parts) if reasoning_parts else f"{method}: امتیاز پایه"
        
        return MethodScore(
            method=method,
            score=final_score,
            speed_score=speed_score,
            accuracy_score=accuracy_score,
            cost_score=cost_score,
            reasoning=reasoning
        )
    
    def explain_decision(
        self,
        query: str,
        features: QueryFeatures,
        selected_method: str,
        score: MethodScore,
        all_scores: Optional[List[MethodScore]] = None
    ) -> Dict:
        """
        توضیح دلیل انتخاب روش
        
        Returns:
            توضیحات کامل تصمیم
        """
        
        explanation = {
            "query": query,
            "selected_method": selected_method,
            "reasoning": score.reasoning,
            "features": {
                "complexity": features.complexity.value,
                "urgency": features.urgency.value,
                "domain": features.domain,
                "requires_knowledge": features.requires_knowledge,
                "requires_calculation": features.requires_calculation,
                "is_specialized": features.is_specialized,
                "confidence_needed": f"{features.confidence_needed:.0%}"
            },
            "scores": {
                "final": f"{score.score:.1f}",
                "speed": f"{score.speed_score:.1f}",
                "accuracy": f"{score.accuracy_score:.1f}",
                "cost": f"{score.cost_score:.1f}"
            }
        }
        
        if all_scores:
            explanation["alternatives"] = [
                {
                    "method": s.method,
                    "score": f"{s.score:.1f}",
                    "reasoning": s.reasoning
                }
                for s in all_scores[:3]  # تاپ 3
            ]
        
        return explanation
    
    def update_performance(self, method: str, success: bool, execution_time: float):
        """به‌روزرسانی آمار عملکرد برای یادگیری"""
        
        if method not in self.method_performance:
            return
        
        perf = self.method_performance[method]
        
        # به‌روزرسانی نرخ موفقیت (با میانگین متحرک)
        alpha = 0.1  # ضریب یادگیری
        perf["success_rate"] = (
            (1 - alpha) * perf["success_rate"] +
            alpha * (1.0 if success else 0.0)
        )
        
        # به‌روزرسانی زمان متوسط
        perf["avg_time"] = (
            (1 - alpha) * perf["avg_time"] +
            alpha * execution_time
        )
    
    def get_performance_stats(self) -> Dict:
        """دریافت آمار عملکرد همه روش‌ها"""
        return {
            method: {
                "success_rate": f"{stats['success_rate']:.1%}",
                "avg_time": f"{stats['avg_time']:.2f}s"
            }
            for method, stats in self.method_performance.items()
        }
