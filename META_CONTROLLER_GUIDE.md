# Meta-Controller System - Complete Implementation

## خلاصه سیستم (System Overview)

یک سیستم Meta-Controller هوشمند برای انتخاب خودکار بهترین روش AI از بین 5 روش تکمیلی:

### روش‌های AI (5 Methods)

1. **RAG** (Retrieval Augmented Generation) - دانش پایه + پاسخ سریع
2. **Fine-Tuning** - تخصصی + دقت بالا
3. **LoRA** (Low-Rank Adaptation) - تطبیق سریع حوزه‌ای
4. **Prompt Engineering** - سریع + ارزان + ساده
5. **PEFT** (Parameter-Efficient Fine-Tuning) - کارآمد + تخصصی

## معماری (Architecture)

```
User Query → Meta-Controller → Method Selection → Execution → Performance Update
                    ↓
            Feature Analysis
                    ↓
         Multi-Criteria Scoring
                    ↓
         Adaptive Learning
```

### تحلیل ویژگی‌ها (Feature Analysis)

Meta-Controller 11 ویژگی را تحلیل می‌کند:

1. **length** - طول متن (کاراکتر)
2. **has_numbers** - وجود اعداد
3. **has_technical_terms** - اصطلاحات فنی
4. **requires_calculation** - نیاز به محاسبه
5. **requires_knowledge** - نیاز به دانش
6. **requires_reasoning** - نیاز به استدلال
7. **is_specialized** - تخصصی بودن
8. **complexity** - پیچیدگی (4 سطح)
   - SIMPLE
   - MODERATE
   - COMPLEX
   - VERY_COMPLEX
9. **urgency** - فوریت (4 سطح)
   - REALTIME
   - FAST
   - NORMAL
   - BATCH
10. **domain** - حوزه (5 دامنه)
    - structural (سازه)
    - mep (تاسیسات)
    - calculation (محاسبه)
    - architecture (معماری)
    - general (عمومی)
11. **confidence_needed** - دقت موردنیاز (0.7-1.0)

## الگوریتم امتیازدهی (Scoring Algorithm)

### امتیازات پایه (Base Scores)

| Method | Speed | Accuracy | Cost |
|--------|-------|----------|------|
| RAG | 85 | 90 | 90 |
| Fine-Tuning | 70 | 95 | 40 |
| LoRA | 75 | 88 | 70 |
| Prompt Engineering | 95 | 75 | 95 |
| PEFT | 80 | 85 | 80 |

### تنظیمات بر اساس ویژگی (Feature-Based Adjustments)

**RAG:**

- +5 accuracy if requires_knowledge
- +3 accuracy if requires_calculation
- +5 speed if urgency=FAST

**Fine-Tuning:**

- +10 accuracy if is_specialized
- +15 accuracy, -10 speed if complexity=COMPLEX/VERY_COMPLEX
- +8 accuracy if confidence_needed > 0.9
- +10 accuracy if requires_reasoning

**LoRA:**

- +8 accuracy if specialized + moderate complexity
- +10 accuracy, +5 speed if domain=structural/mep
- +5 accuracy if calculation + specialized

**Prompt Engineering:**

- +5 speed if urgency=FAST
- +15 accuracy, +5 speed if complexity=SIMPLE
- +8 accuracy if not specialized
- +5 accuracy if not requires_reasoning

**PEFT:**

- +10 accuracy, +5 speed if specialized + confidence > 0.8
- +8 accuracy if complexity=MODERATE
- +5 accuracy if domain=mep

### امتیاز نهایی (Final Score)

```python
weighted_score = (
    speed * 0.30 +    # وزن سرعت: 30%
    accuracy * 0.40 + # وزن دقت: 40%
    cost * 0.30       # وزن هزینه: 30%
)

performance_factor = 0.7 + (success_rate * 0.3)  # 70% base + 30% history
final_score = weighted_score * performance_factor
```

## یادگیری تطبیقی (Adaptive Learning)

سیستم پس از هر اجرا عملکرد را به‌روزرسانی می‌کند:

```python
# Exponential Moving Average (alpha = 0.1)
new_success_rate = (1 - alpha) * old_rate + alpha * (1 if success else 0)
new_avg_time = (1 - alpha) * old_time + alpha * execution_time
```

## نمونه تصمیم‌گیری (Decision Examples)

### Test 1: محاسبه ساده

```
Query: "محاسبه مساحت اتاق 5 در 4 متر"
Features:
  - Complexity: moderate
  - Domain: calculation
  - Requires Calculation: true
  - Confidence Needed: 90%
Selected: RAG
Reasoning: "RAG: محاسبه با منابع, پاسخ سریع"
```

### Test 2: تحلیل پیچیده

```
Query: "تحلیل جامع سیستم سازه‌ای ساختمان 20 طبقه..."
Features:
  - Complexity: very_complex
  - Domain: structural
  - Requires Knowledge: true
  - Requires Reasoning: true
  - Is Specialized: true
Selected: RAG (but close to Fine-Tuning)
Reasoning: "RAG: نیاز به دانش پایه"
Note: با تنظیم وزن‌ها می‌توان Fine-Tuning را ترجیح داد
```

### Test 3: پرسش بسیار ساده

```
Query: "hi"
Features:
  - Complexity: simple
  - Domain: general
  - No specialized requirements
Selected: Prompt Engineering
Reasoning: "Prompt Engineering: نیاز به سرعت, پرسش ساده"
```

### Test 4: کلیدواژه صریح

```
Query: "use lora adapter for structural analysis"
Selected: LoRA (Explicit Keyword Detection)
Note: کلیدواژه‌های صریح اولویت بالا دارند
```

## نتایج آزمایش (Test Results)

### تنوع روش‌ها (Method Diversity)

```
Total Queries: 7
  RAG: 2
  Fine-Tuning: 1
  LoRA: 1
  Prompt Engineering: 2
  PEFT: 1

✓ تمام 5 روش استفاده شدند
```

### آمار عملکرد (Performance Stats)

```
RAG: Success=96.7%, AvgTime=0.37s
Fine-Tuning: Success=98.0%, AvgTime=1.20s
LoRA: Success=92.0%, AvgTime=0.80s
Prompt Engineering: Success=86.5%, AvgTime=0.27s
PEFT: Success=90.0%, AvgTime=0.60s
```

## فایل‌های کلیدی (Key Files)

### 1. `cad3d/super_ai/meta_controller.py` (520 سطر)

```python
class MetaController:
    def analyze_query(query, task_type) -> QueryFeatures
    def select_best_method(features) -> (method, score)
    def update_performance(method, success, time)
    def explain_decision(...) -> dict
```

### 2. `cad3d/super_ai/unified_ai_system.py` (720 سطر)

```python
class UnifiedAISystem:
    def __init__():
        self.meta_controller = MetaController()
    
    def query(query, task_type):
        # 1. Security check
        # 2. Select method (Meta-Controller)
        # 3. Execute query
        # 4. Update performance
        # 5. Return response + selection_reasoning
```

### 3. تست‌ها (Tests)

- `test_meta_controller.py` - آزمایش کامل Meta-Controller
- `test_meta_diverse.py` - آزمایش تنوع انتخاب روش‌ها
- `test_unified_simple.py` - آزمایش اولیه 5 روش

## استفاده (Usage)

### استفاده ساده

```python
from cad3d.super_ai.unified_ai_system import UnifiedAISystem

# Initialize
system = UnifiedAISystem()

# Query with auto-selection
response = system.query("محاسبه مساحت 5 در 4 متر")

print(response['method'])  # متد انتخاب‌شده
print(response['result'])  # نتیجه
print(response['selection_reasoning'])  # توضیحات تصمیم
```

### توضیح بدون اجرا (Dry-Run)

```python
# Get explanation without execution
explanation = system.explain_selection(
    "تحلیل سازه 10 طبقه",
    AITaskType.STRUCTURAL_CALCULATION
)

print(explanation['selected_method'])
print(explanation['reasoning'])
print(explanation['features'])
print(explanation['scores'])
```

### مشاهده آمار

```python
status = system.get_system_status()
print(status['meta_controller'])  # آمار عملکرد
print(status['usage_statistics'])  # آمار استفاده
```

## ویژگی‌های کلیدی (Key Features)

✅ **تحلیل هوشمند**: 11 ویژگی از پرسش استخراج می‌شود
✅ **امتیازدهی چندمعیاره**: سرعت (30%) + دقت (40%) + هزینه (30%)
✅ **یادگیری تطبیقی**: عملکرد بعد از هر اجرا به‌روز می‌شود
✅ **شفافیت**: توضیحات کامل برای هر تصمیم
✅ **کلیدواژه صریح**: اولویت به درخواست صریح کاربر
✅ **Fallback**: روش ساده اگر Meta-Controller خطا داد
✅ **تنوع**: تمام 5 روش قابل انتخاب

## تنظیمات پیشرفته (Advanced Configuration)

### تغییر وزن‌ها

```python
controller = system.meta_controller
controller.weights = {
    "speed": 0.2,      # کاهش اهمیت سرعت
    "accuracy": 0.6,   # افزایش اهمیت دقت
    "cost": 0.2        # کاهش اهمیت هزینه
}
```

### آماده‌سازی اولیه عملکرد

```python
controller.method_performance["Fine-Tuning"]["success_rate"] = 0.99
controller.method_performance["Fine-Tuning"]["avg_time"] = 0.9
```

## نتیجه‌گیری

Meta-Controller یک سیستم هوشمند برای انتخاب خودکار روش AI است که:

- **تحلیل دقیق**: پرسش را به‌طور کامل تحلیل می‌کند
- **تصمیم بهینه**: بر اساس معیارهای چندگانه انتخاب می‌کند
- **یادگیری مداوم**: از تاریخچه اجرا یاد می‌گیرد
- **شفاف**: دلیل تصمیم را توضیح می‌دهد
- **انعطاف‌پذیر**: قابل تنظیم و سفارشی‌سازی است

---

**تاریخ**: 2025
**وضعیت**: ✅ تکمیل شده و تست شده
**تعداد روش‌ها**: 5
**تعداد ویژگی‌ها**: 11
**الگوریتم**: Multi-Criteria Decision Making + Adaptive Learning
