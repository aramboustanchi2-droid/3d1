# 🎓 راهنمای جامع Fine-Tuning در KURDO-AI
## آموزش شخصی‌سازی مدل‌های هوش مصنوعی

---

## 📋 معرفی Fine-Tuning

**Fine-Tuning** یعنی آموزش یک مدل هوش مصنوعی از پیش آموزش‌دیده روی داده‌های خاص خودت. با این کار می‌تونی:

✅ مدل رو روی حوزه معماری و مهندسی تخصصی کنی  
✅ دقت پاسخ‌ها رو تا ۱۰ برابر افزایش بدی  
✅ یه AI اختصاصی برای پروژه‌های خودت داشته باشی  
✅ از دانش فارسی و تخصصی معماری ایران استفاده کنی  

---

## 🎯 چه زمانی باید Fine-Tune کنم؟

### موارد مناسب:
- ✅ وقتی پاسخ‌های عمومی کافی نیستن
- ✅ وقتی می‌خوای مدل زبان فارسی رو بهتر بفهمه
- ✅ وقتی داده‌های تخصصی معماری ایران داری
- ✅ وقتی می‌خوای سبک خاصی از طراحی رو یاد بده

### موارد نامناسب:
- ❌ وقتی داده کمتر از ۵۰ نمونه داری
- ❌ وقتی فقط یک سوال ساده داری
- ❌ وقتی بودجه کافی برای API نداری

---

## 🌐 پلتفرم‌های پشتیبانی‌شده

### 1️⃣ OpenAI (توصیه می‌شه)
- ✅ **مدل‌های پشتیبانی**: GPT-4o-mini, GPT-3.5-turbo
- ✅ **کیفیت**: عالی
- ✅ **سرعت**: سریع (۲۰-۴۰ دقیقه)
- ⚠️ **هزینه**: از $۰.۰۰۸/۱K token شروع میشه

### 2️⃣ HuggingFace (رایگان/محلی)
- ✅ **مدل‌های پشتیبانی**: Flan-T5, mT5, BART و ۱۰۰,۰۰۰+ مدل دیگه
- ✅ **کیفیت**: خوب تا عالی (بستگی به مدل داره)
- ✅ **سرعت**: متوسط (۱-۴ ساعت روی CPU)
- ✅ **هزینه**: رایگان! (روی سیستم خودت اجرا میشه)

### 3️⃣ Anthropic (شبیه‌سازی با Prompt Caching)
- ✅ **روش**: Few-shot learning با cached prompts
- ✅ **کیفیت**: خیلی خوب
- ✅ **سرعت**: فوری
- ⚠️ **محدودیت**: واقعاً fine-tune نیست، ولی خیلی موثره

---

## 🔧 راه‌اندازی

### مرحله ۱: نصب کتابخانه‌های اضافی

```bash
# برای OpenAI (فقط requests کافیه - قبلاً نصب شده)
# نیازی به نصب اضافی نیست!

# برای HuggingFace (اختیاری)
pip install transformers datasets accelerate
```

### مرحله ۲: بررسی کلیدهای API

فایل `.env` رو چک کن:

```bash
# برای OpenAI fine-tuning
OPENAI_API_KEY=sk-proj-XXXXXXXXXXXXXXXX

# برای HuggingFace (اختیاری)
HUGGINGFACE_API_KEY=hf_XXXXXXXXXXXXXXXX
```

### مرحله ۳: آماده‌سازی داده‌های آموزشی

دو راه داری:

#### راه ۱: استفاده از Corpus معماری (خودکار) ✅ توصیه می‌شه
سیستم خودکار از داده‌های موجود در `datasets/persian_corpus/architecture/` استفاده می‌کنه.

#### راه ۲: داده‌های سفارشی خودت

فرمت OpenAI:
```python
training_data = [
    {
        "messages": [
            {"role": "system", "content": "تو یک مشاور معماری حرفه‌ای هستی"},
            {"role": "user", "content": "امکان‌سنجی یک برج ۲۰ طبقه چطوری انجام میشه؟"},
            {"role": "assistant", "content": "امکان‌سنجی برج ۲۰ طبقه شامل این مراحل است: ۱) بررسی زمین ۲) محاسبه فضای قابل ساخت ۳) ارزیابی اقتصادی..."}
        ]
    },
    # حداقل ۵۰-۱۰۰ نمونه دیگه...
]
```

فرمت HuggingFace:
```python
training_data = [
    {
        "input": "امکان‌سنجی یک برج ۲۰ طبقه",
        "output": "امکان‌سنجی شامل: بررسی زمین، محاسبه فضا، ارزیابی اقتصادی..."
    },
    # ...
]
```

---

## 🚀 استفاده

### روش ۱: استفاده مستقیم از Brain

```python
from cad3d.super_ai.brain import SuperAIBrain

brain = SuperAIBrain()

# Fine-tune با OpenAI (توصیه می‌شه)
result = brain.fine_tune_model(
    provider="openai",
    use_architectural_corpus=True  # از داده‌های معماری استفاده می‌کنه
)

print(result)
# Output: {
#   "status": "running",
#   "provider": "openai",
#   "job_id": "ftjob-abc123...",
#   "training_file_id": "file-xyz789..."
# }
```

### روش ۲: Fine-tune با HuggingFace (رایگان/محلی)

```python
result = brain.fine_tune_model(
    provider="huggingface",
    base_model="google/flan-t5-base",
    use_architectural_corpus=True
)

# این روی سیستم خودت اجرا میشه
# ممکنه ۱-۴ ساعت طول بکشه (بستگی به CPU/GPU داره)
```

### روش ۳: Anthropic با Few-Shot Learning

```python
result = brain.fine_tune_model(
    provider="anthropic",
    use_architectural_corpus=True
)

# این یه prompt کش‌شده می‌سازه با نمونه‌های آموزشی
# فوری اجرا میشه!
```

### روش ۴: استفاده مستقیم از ماژول Fine-Tuning

```python
from cad3d.super_ai.fine_tuning import fine_tuning_manager

# ۱. آماده‌سازی داده
training_data = [
    {
        "messages": [
            {"role": "system", "content": "تو KURDO-AI هستی"},
            {"role": "user", "content": "سلام"},
            {"role": "assistant", "content": "سلام! چطور می‌تونم کمکت کنم؟"}
        ]
    },
    # ...
]

# ۲. شروع Fine-tuning
result = fine_tuning_manager.full_fine_tune_workflow(
    provider="openai",
    training_data=training_data,
    base_model="gpt-4o-mini-2024-07-18",
    custom_suffix="my-custom-model"
)
```

---

## 📊 بررسی وضعیت

### چک کردن وضعیت Job

```python
# برای OpenAI
job_id = "ftjob-abc123..."
status = brain.check_fine_tune_status(job_id, provider="openai")

print(status)
# Output: {
#   "id": "ftjob-abc123...",
#   "status": "succeeded",  # یا "running", "failed"
#   "fine_tuned_model": "ft:gpt-4o-mini-2024-07-18:kurdo-ai-arch::abc123",
#   "trained_tokens": 125000,
#   ...
# }
```

### لیست تمام Fine-tune های قبلی

```python
history = brain.list_fine_tuned_models()

for job in history:
    print(f"Provider: {job['provider']}")
    print(f"Status: {job['status']}")
    print(f"Date: {job['timestamp']}")
    print("---")
```

---

## 💡 بهترین روش‌ها (Best Practices)

### 1️⃣ تعداد داده‌های آموزشی

| حالت | تعداد نمونه | نتیجه |
|------|-------------|-------|
| ❌ خیلی کم | < ۵۰ | احتمال Overfitting بالا |
| ✅ خوب | ۵۰-۵۰۰ | نتایج معمولی |
| 🏆 عالی | ۵۰۰-۱۰,۰۰۰ | نتایج عالی |
| ⚠️ خیلی زیاد | > ۱۰,۰۰۰ | گران میشه |

### 2️⃣ کیفیت داده

✅ **خوب:**
- داده‌های واقعی از پروژه‌های قبلی
- پاسخ‌های دقیق و کامل
- تنوع در سوالات و موضوعات

❌ **بد:**
- داده‌های تکراری
- پاسخ‌های کوتاه و ناقص
- همه یک نوع سوال

### 3️⃣ انتخاب مدل پایه

| مدل | موارد استفاده | هزینه |
|-----|---------------|-------|
| GPT-4o-mini | پروژه‌های واقعی (توصیه) | $$ متوسط |
| GPT-3.5-turbo | تست و توسعه | $ کم |
| Flan-T5-base | آزمایشی/آفلاین | رایگان |
| Flan-T5-large | پروژه محلی جدی | رایگان |

### 4️⃣ Hyperparameters

برای OpenAI:
```python
hyperparameters = {
    "n_epochs": 3,  # ۳ تا ۵ معمولاً خوبه
    "batch_size": 1,  # کوچیک‌تر = دقیق‌تر (ولی کندتر)
    "learning_rate_multiplier": 1.0  # بین ۰.۵ تا ۲.۰
}
```

---

## 💰 مدیریت هزینه

### هزینه‌های OpenAI Fine-Tuning (تقریبی)

**مرحله Training:**
- GPT-4o-mini: $۰.۰۰۸/۱K tokens
- GPT-3.5-turbo: $۰.۰۰۸/۱K tokens

**مرحله Inference (استفاده از مدل):**
- GPT-4o-mini fine-tuned: $۰.۰۳/۱K tokens (۳ برابر مدل معمولی)
- GPT-3.5-turbo fine-tuned: $۰.۰۱۲/۱K tokens

**مثال محاسبه:**
```
۱۰۰ نمونه آموزشی × ۵۰۰ token = ۵۰,۰۰۰ tokens
۳ epochs = ۱۵۰,۰۰۰ tokens آموزش
هزینه training = ۱۵۰ × $۰.۰۰۸ = $۱.۲۰ 💰
```

### راه‌های کاهش هزینه

1. ✅ **شروع با HuggingFace** (رایگان)
2. ✅ **استفاده از GPT-4o-mini** به جای GPT-4
3. ✅ **تعداد epochs رو کم کن** (۲-۳ کافیه)
4. ✅ **داده‌ها رو فیلتر کن** (فقط بهترین‌ها)

---

## 🎯 نمونه‌های کاربردی

### نمونه ۱: Fine-tune برای معماری ایرانی

```python
from cad3d.super_ai.brain import SuperAIBrain

brain = SuperAIBrain()

# داده‌های سفارشی درباره معماری ایرانی
persian_architecture_data = [
    {
        "messages": [
            {"role": "system", "content": "تو متخصص معماری ایرانی هستی"},
            {"role": "user", "content": "ایوان در معماری ایران چه کاربردی داره؟"},
            {"role": "assistant", "content": "ایوان یکی از عناصر اصلی معماری سنتی ایران است که..."}
        ]
    },
    # ۵۰+ نمونه دیگه...
]

# Fine-tune
result = brain.fine_tune_model(
    provider="openai",
    training_data=persian_architecture_data,
    base_model="gpt-4o-mini-2024-07-18"
)

print(f"Job started: {result['job_id']}")
```

### نمونه ۲: Fine-tune محلی با Flan-T5

```python
# این روی سیستم خودت اجرا میشه (رایگان!)
result = brain.fine_tune_model(
    provider="huggingface",
    base_model="google/flan-t5-base",
    training_data=persian_architecture_data
)

# بعد از آموزش، مدل در این آدرس ذخیره میشه:
model_path = result['output_dir']  # models/fine_tuned/
```

### نمونه ۳: استفاده از مدل Fine-tuned شده

```python
from cad3d.super_ai.external_connectors import unified_connector

# استفاده از مدل fine-tuned OpenAI
response = unified_connector.chat_completion(
    prompt="یک ساختمان ۱۵ طبقه در تهران بسازیم. امکان‌سنجی کن.",
    system_prompt="تو KURDO-AI هستی که روی معماری ایران fine-tune شدی.",
    provider="openai"
    # مدل fine-tuned رو باید در connectors_config.json تنظیم کنی
)
```

---

## 🔧 استفاده از Corpus معماری موجود

سیستم یه corpus آماده معماری داره که می‌تونی ازش استفاده کنی:

### ساختار Corpus:

```
datasets/persian_corpus/
├── architecture/
│   ├── architectural_terms.txt          # اصطلاحات معماری
│   ├── feasibility_guidelines.txt       # راهنمای امکان‌سنجی
│   ├── design_principles.txt            # اصول طراحی
│   └── iranian_architecture_history.txt # تاریخچه
├── structure/
│   └── structural_engineering.txt       # مهندسی سازه
└── urban_planning/
    └── urban_design_principles.txt      # شهرسازی
```

### نحوه استفاده:

```python
# خودکار از corpus استفاده می‌کنه
result = brain.fine_tune_model(
    provider="openai",
    use_architectural_corpus=True  # ✅ این رو true کن
)

# یا مستقیم corpus رو بخون
from cad3d.super_ai.fine_tuning import fine_tuning_manager

training_data = fine_tuning_manager.prepare_architectural_training_data(
    source_dir="datasets/persian_corpus/architecture"
)

print(f"Loaded {len(training_data)} training examples")
```

---

## 🐛 عیب‌یابی

### مشکل: "Fine-tuning module not available"

**راه‌حل:**
```bash
# چک کن که فایل وجود داره
ls cad3d/super_ai/fine_tuning.py

# اگه نیست، دوباره ایجاد کن
```

### مشکل: "Training file upload failed"

**علت‌های محتمل:**
- کلید OpenAI API اشتباهه
- فرمت داده درست نیست
- اتصال اینترنت قطعه

**راه‌حل:**
```python
# تست کن کلیدت کار می‌کنه
import os
print(os.getenv("OPENAI_API_KEY"))

# فرمت داده رو چک کن
# باید دقیقاً مثل نمونه بالا باشه
```

### مشکل: "Insufficient quota"

**علت:** اعتبار OpenAI کافی نیست

**راه‌حل:**
1. به [OpenAI Billing](https://platform.openai.com/account/billing) برو
2. اعتبار اضافه کن (حداقل $۵)
3. یا از HuggingFace استفاده کن (رایگان)

### مشکل: HuggingFace خیلی کنده

**راه‌حل:**
```bash
# اگه GPU داری
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# یا مدل کوچیک‌تر استفاده کن
base_model="google/flan-t5-small"  # به جای base
```

---

## 📈 بهبود مستمر

### چرخه Fine-Tuning:

```
۱. جمع‌آوری داده (۱۰۰+ نمونه) 
    ↓
۲. Fine-tune مدل (اولین بار)
    ↓
۳. تست و ارزیابی
    ↓
۴. جمع‌آوری feedback
    ↓
۵. اضافه کردن داده‌های جدید
    ↓
۶. Fine-tune دوباره (بهبود)
    ↓
برگشت به مرحله ۳
```

### متریک‌های ارزیابی:

```python
# بعد از fine-tune، تست کن
test_prompts = [
    "امکان‌سنجی یک برج ۱۰ طبقه",
    "محاسبه فضای قابل ساخت",
    "بررسی نور طبیعی ساختمان"
]

for prompt in test_prompts:
    response = unified_connector.chat_completion(
        prompt=prompt,
        provider="openai"  # از مدل fine-tuned استفاده می‌کنه
    )
    print(f"Q: {prompt}")
    print(f"A: {response}")
    print("---")
```

---

## 🎉 نتیجه‌گیری

با Fine-Tuning می‌تونی:

✅ **دقت رو ۵-۱۰ برابر افزایش بدی** در حوزه تخصصی  
✅ **یه AI اختصاصی داشته باشی** برای پروژه‌های خودت  
✅ **از دانش فارسی و ایرانی استفاده کنی** در مدل‌ها  
✅ **هزینه استفاده طولانی‌مدت رو کاهش بدی** (نیاز به prompt کمتر)  
✅ **سرعت پاسخ رو افزایش بدی** (مدل قبلاً یاد گرفته)  

### مراحل شروع سریع:

```bash
# ۱. چک کن همه چیز آمادهس
python -c "from cad3d.super_ai.fine_tuning import fine_tuning_manager; print('✅ Ready!')"

# ۲. یه fine-tune تستی بزن
python -c "
from cad3d.super_ai.brain import SuperAIBrain
brain = SuperAIBrain()
result = brain.fine_tune_model(provider='anthropic', use_architectural_corpus=True)
print(result)
"

# ۳. برای OpenAI واقعی، اعتبار اضافه کن و دوباره تست کن!
```

**KURDO-AI حالا قابلیت Fine-Tuning داره و می‌تونه روی هر حوزه‌ای آموزش ببینه! 🎓🤖**

---

*آخرین بروزرسانی: نوامبر ۲۰۲۵*
*نسخه: ۲.۱ (Fine-Tuning Enabled)*
