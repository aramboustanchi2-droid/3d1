# University Knowledge System - سیستم یادگیری از دانشگاه‌های برتر دنیا

## خلاصه سیستم

یک سیستم کامل برای یادگیری خودکار و مداوم از **10 دانشگاه برتر دنیا** بدون نیاز به API:

### 🎓 دانشگاه‌های پوشش‌داده‌شده

1. **MIT** (USA) - AI, Robotics, Computer Science, Architecture
2. **Stanford** (USA) - AI, Machine Learning, Computer Vision, NLP
3. **Cambridge** (UK) - Engineering, Mathematics, Computer Science
4. **Oxford** (UK) - Engineering, Materials Science, Computer Science
5. **Berkeley** (USA) - AI, Machine Learning, Architecture
6. **ETH Zurich** (Switzerland) - Architecture, Civil Engineering, Structural
7. **Caltech** (USA) - Physics, Engineering, Computer Science
8. **Imperial** (UK) - Engineering, AI, Civil Engineering, Architecture
9. **Carnegie Mellon** (USA) - AI, Robotics, Architecture
10. **TU Delft** (Netherlands) - Architecture, Civil Engineering, Urban Planning

### 🌐 منابع قابل دسترسی

برای هر دانشگاه:

- **OpenCourseWare**: کورس‌های رایگان، ویدیو، نوت‌های درسی
- **Research Repositories**: مقالات، پژوهش‌ها، thesis ها
- **Department Pages**: اطلاعات تخصصی هر بخش
- **Publications**: انتشارات علمی و تحقیقاتی

همه منابع **بدون نیاز به API** قابل دسترسی هستند!

## معماری سیستم

```
┌─────────────────────────────────────────────────────────────┐
│                    UniversityKnowledgeIntegration           │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Scraper    │───▶│    Agent     │───▶│  RAG System  │ │
│  │              │    │   Manager    │    │              │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │         │
│         ▼                    ▼                    ▼         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   10 Unis    │    │  10 Agents   │    │  Knowledge   │ │
│  │   Websites   │    │  Learning    │    │    Base      │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                              │                             │
│                              ▼                             │
│                    ┌──────────────┐                        │
│                    │  Scheduler   │                        │
│                    │  Auto-Update │                        │
│                    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## فایل‌های سیستم

### 1. `university_config.py`

تنظیمات و لیست دانشگاه‌ها:

```python
UNIVERSITIES = {
    "MIT": {
        "name": "Massachusetts Institute of Technology",
        "resources": {
            "opencourseware": "https://ocw.mit.edu",
            "research": "https://dspace.mit.edu",
            "ai_lab": "https://www.csail.mit.edu/research"
        },
        "focus_areas": ["AI", "Robotics", "Architecture"]
    },
    # ... 9 other universities
}
```

### 2. `university_scraper.py`

استخراج محتوا از وب‌سایت‌ها:

- `UniversityScraper`: Web scraping با مدیریت خطا و cache
- `UniversityResourceCollector`: جمع‌آوری از چندین دانشگاه
- ویژگی‌ها:
  - Rate limiting برای جلوگیری از ban
  - Retry mechanism
  - Cache برای کاهش درخواست‌ها
  - استخراج لینک‌های PDF
  - پردازش HTML به متن

### 3. `university_agents.py`

ایجنت‌های هوشمند:

- `UniversityAgent`: یک ایجنت برای هر دانشگاه
  - جمع‌آوری دوره‌ای محتوا
  - ردیابی تغییرات
  - به‌روزرسانی RAG
  - ذخیره state
- `UniversityAgentManager`: مدیریت 10 ایجنت
  - یادگیری موازی
  - آمارگیری
  - گزارش‌دهی

### 4. `university_scheduler.py`

زمان‌بندی خودکار:

- اجرای دوره‌ای (روزانه/هفتگی/ماهانه)
- Background thread
- Logging
- مدیریت خطا

### 5. `test_university_integration.py`

تست و دمو کامل سیستم

## نصب Dependencies

```bash
pip install requests beautifulsoup4 schedule
```

## استفاده

### استفاده ساده

```python
from cad3d.super_ai.university_config import UNIVERSITIES, AGENT_CONFIG
from cad3d.super_ai.university_agents import UniversityAgentManager
from cad3d.super_ai.rag_system import RAGSystem

# Initialize
rag_system = RAGSystem()
agent_manager = UniversityAgentManager(UNIVERSITIES, AGENT_CONFIG, rag_system)

# یادگیری از MIT
result = agent_manager.learn_from_specific(['MIT'])

# یادگیری از 5 دانشگاه برتر
top_5 = ["MIT", "Stanford", "Cambridge", "Oxford", "Berkeley"]
result = agent_manager.learn_from_specific(top_5)

# یادگیری از همه
result = agent_manager.learn_from_all()

# آمار
stats = agent_manager.get_statistics()
print(f"Total documents: {stats['total_documents_collected']}")
```

### استفاده با Scheduler

```python
from cad3d.super_ai.university_scheduler import UniversityLearningScheduler

# Initialize scheduler
scheduler = UniversityLearningScheduler(agent_manager)

# تنظیم زمان‌بندی پیش‌فرض
# Top 5: روزانه ساعت 2 صبح
# Next 5: هفتگی یکشنبه‌ها ساعت 3 صبح
scheduler.setup_default_schedules()

# شروع (background thread)
scheduler.start()

# اجرای فوری برای تست
scheduler.run_now(['MIT'])

# توقف
scheduler.stop()
```

### ادغام با UnifiedAISystem

```python
from cad3d.super_ai.unified_ai_system import UnifiedAISystem

# سیستم AI با دانش دانشگاهی
system = UnifiedAISystem()

# پرسش (RAG از دانش دانشگاهی استفاده می‌کند)
response = system.query("What is the latest AI research at MIT?")
print(response['result'])
```

## اجرای دمو

```bash
cd e:\3d
.\.venv\Scripts\python.exe test_university_integration.py
```

خروجی:

- لیست 10 دانشگاه
- یادگیری از MIT (نمونه)
- آمار جمع‌آوری
- تست RAG query

## ویژگی‌های کلیدی

### ✅ بدون API

- همه منابع از طریق web scraping
- هیچ نیاز به API key یا ثبت‌نام
- دسترسی آزاد به محتوای عمومی

### ✅ Cache هوشمند

- ذخیره محتوای دریافت‌شده
- کاهش درخواست‌های تکراری
- مدیریت خودکار حافظه

### ✅ Rate Limiting

- تاخیر بین درخواست‌ها
- جلوگیری از ban شدن
- احترام به robots.txt

### ✅ یادگیری مداوم

- به‌روزرسانی خودکار دوره‌ای
- ردیابی تغییرات
- اضافه شدن محتوای جدید به RAG

### ✅ مقیاس‌پذیر

- 10 ایجنت موازی
- قابل افزایش تا دانشگاه‌های بیشتر
- مدیریت حافظه و منابع

### ✅ قابل نظارت

- Logging کامل
- آمارگیری دقیق
- گزارش‌های دوره‌ای

## Configuration

### تنظیم Scraping

```python
AGENT_CONFIG = {
    "scraping": {
        "user_agent": "Mozilla/5.0...",
        "timeout": 30,           # ثانیه
        "retry_attempts": 3,     # تعداد تلاش مجدد
        "rate_limit": 2          # تاخیر بین درخواست‌ها
    }
}
```

### تنظیم یادگیری

```python
AGENT_CONFIG = {
    "learning": {
        "update_frequency": "daily",  # یا "weekly", "monthly"
        "max_documents_per_session": 50,
        "content_types": ["PDF", "HTML"],
        "languages": ["en", "fa"]
    }
}
```

### تنظیم Cache

```python
AGENT_CONFIG = {
    "storage": {
        "cache_dir": "university_cache",
        "embeddings_dir": "university_embeddings",
        "max_cache_size_gb": 10
    }
}
```

## مثال‌های کاربردی

### مثال 1: یادگیری از MIT OpenCourseWare

```python
# یادگیری کورس‌های AI از MIT
result = agent_manager.learn_from_specific(['MIT'])

# جستجوی اطلاعات
from cad3d.super_ai.rag_system import RAGSystem
rag = RAGSystem()
results = rag.search("machine learning algorithms", top_k=5)

for r in results:
    if r['metadata']['university'] == 'MIT':
        print(f"Course: {r['metadata']['resource']}")
        print(f"Content: {r['content'][:200]}...")
```

### مثال 2: جمع‌آوری تحقیقات معماری

```python
# دانشگاه‌های قوی در معماری
architecture_unis = ["ETH_Zurich", "TU_Delft", "MIT", "Carnegie_Mellon"]
result = agent_manager.learn_from_specific(architecture_unis)

# جستجو در تحقیقات معماری
results = rag.search("sustainable architecture design", top_k=10)
```

### مثال 3: به‌روزرسانی خودکار شبانه

```python
scheduler = UniversityLearningScheduler(agent_manager)

# Top 5 دانشگاه هر شب ساعت 2
for uni in ["MIT", "Stanford", "Cambridge", "Oxford", "Berkeley"]:
    scheduler.add_university_schedule(uni, 'daily', '02:00')

# شروع
scheduler.start()

# سیستم هر شب به‌صورت خودکار یاد می‌گیرد
```

## Troubleshooting

### خطای Connection

```python
# افزایش timeout
AGENT_CONFIG['scraping']['timeout'] = 60

# افزایش retry
AGENT_CONFIG['scraping']['retry_attempts'] = 5
```

### مشکل حافظه

```python
# کاهش تعداد صفحات
AGENT_CONFIG['learning']['max_documents_per_session'] = 20

# پاکسازی cache قدیمی
import shutil
shutil.rmtree('university_cache')
```

### Rate Limiting

```python
# افزایش تاخیر
AGENT_CONFIG['scraping']['rate_limit'] = 5  # 5 ثانیه بین درخواست‌ها
```

## آمار سیستم

بعد از یادگیری از همه دانشگاه‌ها:

```
Total Universities: 10
Total Resources: 30+ sources
Potential Documents: 1000s of pages
Knowledge Domains: 
  - AI & Machine Learning
  - Architecture & Urban Design
  - Civil & Structural Engineering
  - Computer Science
  - Robotics
  - Materials Science
  - MEP Systems
```

## نتیجه‌گیری

این سیستم یک **پایگاه دانش زنده** از برترین دانشگاه‌های دنیا ایجاد می‌کند که:

- ✅ به‌صورت خودکار به‌روز می‌شود
- ✅ بدون هزینه و API عمل می‌کند
- ✅ با سیستم RAG ادغام شده
- ✅ قابل استفاده در UnifiedAISystem
- ✅ دانش تخصصی معماری، مهندسی و AI

---

**تاریخ ایجاد**: نوامبر 2025  
**وضعیت**: ✅ آماده برای استفاده  
**تعداد دانشگاه‌ها**: 10  
**تعداد منابع**: 30+  
**نیاز به API**: ❌ خیر
