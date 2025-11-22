# 🎯 راهنمای کامل Prompt Engineering در KURDO-AI

## مقدمه

**Prompt Engineering** سومین روش آموزش (بدون آموزش!) در KURDO-AI است که مکمل Fine-Tuning و LoRA می‌باشد.

### چرا Prompt Engineering؟

- ⚡ **فوری**: بدون نیاز به آموزش!
- 💰 **رایگان**: فقط هزینه inference
- 🚀 **انعطاف**: برای هر تسکی قابل استفاده
- 🎯 **ساده**: نیازی به GPU ندارید

---

## 📚 تکنیک‌های موجود

### 1️⃣ Zero-Shot Prompting

استفاده از مدل بدون هیچ مثالی:

```python
from cad3d.super_ai.brain import SuperAIBrain

brain = SuperAIBrain()

# استفاده مستقیم از template
prompt = brain.use_prompt_template(
    "arch_calculation",
    task="محاسبه مساحت اتاق",
    given_values="طول: 6 متر، عرض: 4 متر",
    required_output="مساحت به متر مربع"
)

print(prompt)
```

**خروجی:**

```
You are KURDO-AI, an expert architectural calculator.

Task: محاسبه مساحت اتاق
Given: طول: 6 متر، عرض: 4 متر
Required: مساحت به متر مربع

Show your calculation steps clearly.
Use appropriate units (metric: meters, square meters, cubic meters).
Provide practical recommendations when relevant.

Answer:
```

### 2️⃣ Few-Shot Learning

یادگیری از چند مثال:

```python
# تعریف مثال‌ها
examples = [
    {
        "input": "محاسبه مساحت اتاق 5×4 متر",
        "output": "مساحت = طول × عرض = 5 × 4 = 20 متر مربع"
    },
    {
        "input": "مساحت اتاق 6×3.5 متر؟",
        "output": "مساحت = 6 × 3.5 = 21 متر مربع"
    },
    {
        "input": "Calculate area of 8m × 5m room",
        "output": "Area = length × width = 8 × 5 = 40 square meters"
    }
]

# ایجاد few-shot prompt
prompt = brain.create_few_shot_prompt(
    task_description="Calculate room area in square meters. Show formula and result.",
    examples=examples,
    current_input="محاسبه مساحت اتاق 7.5×6 متر",
    max_examples=3
)

print(prompt)
```

**خروجی:**

```
# Task
Calculate room area in square meters. Show formula and result.

# Examples

Example 1:
Input: محاسبه مساحت اتاق 5×4 متر
Output: مساحت = طول × عرض = 5 × 4 = 20 متر مربع

Example 2:
Input: مساحت اتاق 6×3.5 متر؟
Output: مساحت = 6 × 3.5 = 21 متر مربع

Example 3:
Input: Calculate area of 8m × 5m room
Output: Area = length × width = 8 × 5 = 40 square meters

# Your Turn

Input: محاسبه مساحت اتاق 7.5×6 متر
Output: 
```

### 3️⃣ Chain-of-Thought Reasoning

استدلال گام‌به‌گام برای مسائل پیچیده:

```python
problem = """
یک ساختمان 5 طبقه با ابعاد هر طبقه 12×15 متر می‌خواهیم بسازیم.
ارتفاع هر طبقه 3 متر است.
چند آجر و چند تن سیمان برای ساخت دیوارهای خارجی نیاز داریم؟
(ضخامت دیوار خارجی 30 سانتی‌متر)
"""

prompt = brain.create_chain_of_thought_prompt(
    problem=problem.strip(),
    domain="architectural engineering"
)

print(prompt)
```

**خروجی:**

```
You are KURDO-AI, an expert in architectural engineering.

Problem: یک ساختمان 5 طبقه با ابعاد هر طبقه 12×15 متر می‌خواهیم بسازیم...

Solve this step-by-step:
1. Understand: What is being asked?
2. Identify: What information do we have?
3. Plan: What approach should we use?
4. Calculate: Work through the solution
5. Verify: Does the answer make sense?
6. Conclude: State the final answer clearly

Let's work through this:
```

### 4️⃣ Cached System Prompt

ذخیره prompt برای استفاده مکرر (سبک Anthropic):

```python
# داده‌های آموزشی
training_examples = [
    {
        "input": "محاسبه مساحت اتاق 5×4 متر",
        "output": "مساحت = 5 × 4 = 20 متر مربع"
    },
    {
        "input": "چند آجر برای دیوار 10 متری نیاز است؟",
        "output": "مساحت = 10 × 3 = 30 m²\nآجر = 30 × 60 = 1,800 عدد"
    },
    # ... مثال‌های بیشتر
]

# ایجاد cached prompt
cached = brain.create_cached_system_prompt(
    system_role="KURDO-AI - Expert Architectural Assistant",
    training_examples=training_examples,
    max_examples=20
)

print(f"Cache ID: {cached['cache_id']}")
print(f"Examples cached: {cached['num_examples']}")
print(f"Estimated tokens: {cached['estimated_tokens']}")
```

**نتیجه:**

```json
{
  "cache_id": "cached_prompt_20241121_143022",
  "num_examples": 20,
  "estimated_tokens": 450,
  "usage": "Use this cached content as system message in API calls"
}
```

---

## 🎨 Template های آماده

KURDO-AI شامل template های آماده برای کارهای مختلف است:

### لیست Template ها

```python
templates = brain.list_prompt_templates()
print(templates)
```

**خروجی:**

```python
[
    'arch_calculation',         # محاسبات معماری
    'code_generation',          # تولید کد
    'technical_analysis',       # تحلیل فنی
    'design_review',            # بررسی طراحی
    'technical_translation'     # ترجمه فنی
]
```

### مثال: استفاده از Template محاسبات

```python
prompt = brain.use_prompt_template(
    "arch_calculation",
    task="محاسبه حجم اتاق",
    given_values="طول: 6م، عرض: 4م، ارتفاع: 2.8م",
    required_output="حجم به متر مکعب"
)
```

### مثال: استفاده از Template تولید کد

```python
prompt = brain.use_prompt_template(
    "code_generation",
    language="Python",
    task="Calculate room area and volume",
    requirements="""
- Take length, width, height as input
- Calculate area and volume
- Return both values
- Add input validation
    """
)
```

### مثال: استفاده از Template بررسی طراحی

```python
prompt = brain.use_prompt_template(
    "design_review",
    project_name="برج مسکونی تهران",
    design_element="طراحی پی ساختمان 10 طبقه",
    applicable_standards="مبحث 19، استاندارد 2800"
)
```

---

## 🔄 مقایسه با روش‌های دیگر

```python
comparison = brain.compare_prompt_vs_training()
print(comparison)
```

### خلاصه مقایسه

| ویژگی | Prompt Engineering | LoRA | Fine-Tuning |
|-------|-------------------|------|-------------|
| **زمان آماده‌سازی** | 0 (فوری) | 1-3 ساعت | 4-10 ساعت |
| **هزینه** | $0 | $0 (محلی) | $10-50 |
| **GPU مورد نیاز** | خیر | بله (6GB+) | بله (40GB+) |
| **کیفیت** | خوب | خیلی خوب | عالی |
| **انعطاف** | خیلی بالا | متوسط | کم |
| **مناسب برای** | نمونه‌سازی، تست | چند تسک | تولید |

### چه موقع از کدام استفاده کنیم؟

**استفاده از Prompt Engineering زمانی که:**

- ✅ داده آموزشی ندارید (کمتر از 10 مثال)
- ✅ نیاز به نتیجه فوری دارید
- ✅ GPU ندارید
- ✅ تسک مرتباً تغییر می‌کند
- ✅ در حال نمونه‌سازی هستید
- ✅ بودجه محدود دارید

**استفاده از LoRA زمانی که:**

- ✅ داده آموزشی دارید (50-500 مثال)
- ✅ چند تسک مختلف دارید
- ✅ GPU محدود دارید (6-12GB)
- ✅ نیاز به آموزش سریع دارید

**استفاده از Fine-Tuning زمانی که:**

- ✅ داده زیاد دارید (500+ مثال)
- ✅ نیاز به بهترین کیفیت دارید
- ✅ GPU قدرتمند دارید (40GB+)
- ✅ برای تولید نهایی است

---

## 💡 استراتژی ترکیبی (Hybrid)

بهترین روش: **ترکیب هر سه روش**!

### مرحله 1: شروع با Prompt Engineering

```python
# شروع سریع
examples = [
    {"input": "مساحت 5×4", "output": "20 متر مربع"},
    {"input": "حجم 6×4×3", "output": "72 متر مکعب"}
]

prompt = brain.create_few_shot_prompt(
    task_description="محاسبات معماری",
    examples=examples,
    current_input="مساحت 7×6؟"
)
# استفاده فوری در پروژه
```

### مرحله 2: جمع‌آوری داده واقعی

```python
# ثبت query های واقعی کاربران
real_queries = []
while True:
    user_input = get_user_query()
    ai_response = generate_response(user_input)
    
    # ذخیره برای آموزش
    real_queries.append({
        "input": user_input,
        "output": ai_response
    })
    
    # وقتی 50-100 مثال جمع شد...
    if len(real_queries) >= 50:
        break
```

### مرحله 3: آموزش LoRA

```python
# حالا که داده کافی داریم، LoRA آموزش می‌دهیم
result = brain.auto_train(
    training_data=real_queries,
    adapter_name="kurdo-real-usage",
    provider="local"
)
```

### مرحله 4: ترکیب هر دو

```python
# برای query های رایج: از LoRA
if query_type == "common":
    response = use_lora_adapter("kurdo-real-usage", query)

# برای query های نادر: از Prompt Engineering
else:
    prompt = brain.create_few_shot_prompt(
        task_description=task,
        examples=similar_examples,
        current_input=query
    )
    response = call_api(prompt)
```

---

## 🧪 تست و آزمایش

### اجرای تست‌های تعاملی

```bash
python cad3d/super_ai/test_prompt_engineering.py
```

**منوی تعاملی:**

```
1. 📋 Prompt Templates (Built-in)
2. 📚 Few-Shot Learning (No Training)
3. 🧠 Chain-of-Thought Reasoning
4. 💾 Cached System Prompt (Anthropic)
5. 📊 Usage Statistics
6. ⚖️  Comparison: Prompt vs Training
7. 🎯 All Three Methods Demo
8. 🚀 Run All Tests
9. ❌ Exit
```

### اجرای تست خاص

```bash
# فقط templates
python cad3d/super_ai/test_prompt_engineering.py --templates

# فقط few-shot
python cad3d/super_ai/test_prompt_engineering.py --few-shot

# فقط chain-of-thought
python cad3d/super_ai/test_prompt_engineering.py --cot

# فقط cached prompts
python cad3d/super_ai/test_prompt_engineering.py --cached

# مقایسه روش‌ها
python cad3d/super_ai/test_prompt_engineering.py --compare

# دموی هر سه روش
python cad3d/super_ai/test_prompt_engineering.py --three-methods

# همه تست‌ها
python cad3d/super_ai/test_prompt_engineering.py --all
```

---

## 📊 مثال کامل: پروژه واقعی

```python
from cad3d.super_ai.brain import SuperAIBrain

brain = SuperAIBrain()

# مرحله 1: داده‌های اولیه (حتی 5 مثال کافیست!)
initial_examples = [
    {
        "input": "محاسبه مساحت اتاق 5×4 متر",
        "output": "مساحت = طول × عرض = 5 × 4 = 20 متر مربع"
    },
    {
        "input": "چند آجر برای دیوار 10 متری؟",
        "output": "مساحت = 10 × 3 = 30 m²\nآجر = 30 × 60 = 1,800 عدد"
    },
    {
        "input": "حداقل ارتفاع سقف؟",
        "output": "طبق مبحث 19: حداقل 2.4 متر"
    },
    {
        "input": "عمق پی 3 طبقه؟",
        "output": "حداقل 1.5 متر زیر تراز یخبندان"
    },
    {
        "input": "Calculate volume 6×4×3",
        "output": "Volume = 6 × 4 × 3 = 72 cubic meters"
    }
]

# مرحله 2: ایجاد cached prompt
cached = brain.create_cached_system_prompt(
    system_role="KURDO-AI - Expert Architectural Calculator",
    training_examples=initial_examples,
    max_examples=5
)

print(f"✅ Cached prompt created: {cached['cache_id']}")
print(f"📊 {cached['num_examples']} examples cached")

# مرحله 3: استفاده با API
def answer_query(user_query):
    # استفاده از cached system prompt + user query
    system_message = cached['cached_content']
    
    # فراخوانی API (مثلاً OpenAI یا Anthropic)
    response = call_api(
        system=system_message,
        user=user_query
    )
    
    return response

# تست
print(answer_query("محاسبه مساحت اتاق 8×6 متر"))
# خروجی: "مساحت = 8 × 6 = 48 متر مربع"

# مرحله 4: توسعه با داده بیشتر
# بعد از جمع‌آوری 50+ query واقعی:
more_data = collect_real_queries()

if len(more_data) >= 50:
    print("🚀 آموزش LoRA با داده واقعی...")
    result = brain.auto_train(
        training_data=more_data,
        adapter_name="kurdo-production"
    )
    
    if result['status'] == 'success':
        print(f"✅ LoRA trained: {result['adapter_name']}")
        print("💡 از این به بعد از LoRA استفاده کنید!")
```

---

## 🎓 نکات مهم

### 1. کیفیت مثال‌ها

```python
# ❌ مثال بد
bad_examples = [
    {"input": "room?", "output": "20"}
]

# ✅ مثال خوب
good_examples = [
    {
        "input": "محاسبه مساحت اتاق 5×4 متر",
        "output": "مساحت = طول × عرض = 5 × 4 = 20 متر مربع\nاین مساحت برای اتاق خواب مناسب است."
    }
]
```

### 2. تنوع مثال‌ها

```python
diverse_examples = [
    # فارسی
    {"input": "مساحت 5×4", "output": "20 متر مربع"},
    # انگلیسی
    {"input": "area 6×3", "output": "18 square meters"},
    # با جزئیات
    {"input": "محاسبه دقیق مساحت اتاق خواب", "output": "..."},
    # ساده
    {"input": "5×4؟", "output": "20 m²"}
]
```

### 3. Context Window

```python
# محدودیت: معمولاً 4k-128k توکن

# برای GPT-4: max_examples=10
# برای Claude: max_examples=20
# برای Gemini Pro: max_examples=50

prompt = brain.create_few_shot_prompt(
    examples=examples,
    max_examples=10  # تنظیم بر اساس مدل
)
```

---

## 🎉 خلاصه

حالا KURDO-AI **سه روش مکمل** دارد:

1. **Prompt Engineering** 🎯
   - فوری، رایگان، بدون GPU
   - برای شروع و نمونه‌سازی

2. **LoRA** ⚡
   - سریع، کارآمد، GPU متوسط
   - برای چند تسک

3. **Fine-Tuning** 💪
   - بهترین کیفیت، GPU قدرتمند
   - برای تولید نهایی

**شروع کنید با Prompt Engineering، توسعه دهید با LoRA، نهایی کنید با Fine-Tuning!**

🚀 **موفق باشید!**
