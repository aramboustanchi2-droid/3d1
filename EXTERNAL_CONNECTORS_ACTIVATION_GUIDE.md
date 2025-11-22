# KURDO-AI External Connectors - Complete Activation Guide

# راهنمای کامل فعال‌سازی اتصالات خارجی KURDO-AI

---

## 🌐 Overview / نمای کلی

KURDO-AI now supports permanent connection to **15+ major AI platforms and services** for continuous learning and real-world operation. This guide covers complete activation.

KURDO-AI اکنون از اتصال دائمی به **بیش از ۱۵ پلتفرم و سرویس اصلی هوش مصنوعی** برای یادگیری مستمر و عملیات واقعی پشتیبانی می‌کند.

---

## 📋 Supported Services / سرویس‌های پشتیبانی‌شده

### Translation Services / سرویس‌های ترجمه

1. **Google Translate** - Google Cloud Translation API
2. **Microsoft Translator** - Azure Cognitive Services
3. **DeepL** - High-quality translation (highest priority)
4. **Amazon Translate** - AWS translation service
5. **LibreTranslate** - Free/self-hosted option
6. **Argos Translate** - Offline translation (no API key needed)

### Chat & LLM Services / سرویس‌های گفتگو و مدل‌های زبانی

1. **OpenAI** - GPT-4o, GPT-4, GPT-3.5
2. **Anthropic** - Claude-3 (Opus, Sonnet, Haiku)
3. **Google AI Studio** - Gemini Pro, Gemini Ultra
4. **Grok** - X AI's language model
5. **DeepSeek** - DeepSeek-V2
6. **HuggingFace** - Access to 100,000+ models via Inference API

### Search Services / سرویس‌های جستجو

1. **Google Custom Search** - For online research and context enrichment

### Local Models (Offline) / مدل‌های محلی (آفلاین)

1. **Flan-T5** - Google's instruction-tuned T5
2. **mT5** - Multilingual T5
3. **BERT** - Text understanding and classification
4. **Gemma** - Google's open model
5. **Argos Translate** - Offline translation

---

## 🔧 Installation / نصب

### Step 1: Install Required Packages / مرحله ۱: نصب بسته‌های مورد نیاز

```bash
# Basic requirements (already installed)
pip install requests beautifulsoup4 python-dotenv

# Optional: For local models
pip install transformers torch sentencepiece argostranslate
```

### Step 2: Configure API Keys / مرحله ۲: تنظیم کلیدهای API

1. Copy `.env.example` to `.env`:

   ```bash
   cp .env.example .env
   ```

2. Open `.env` and add your API keys (see below for how to obtain them)

---

## 🔑 Obtaining API Keys / دریافت کلیدهای API

### Google Services

#### Google Translate API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Cloud Translation API"
4. Create credentials → API Key
5. Copy to `.env` as `GOOGLE_TRANSLATE_API_KEY`

#### Google AI Studio (Gemini)

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy to `.env` as `GOOGLE_AI_STUDIO_KEY`

#### Google Custom Search (for research)

1. Get API key from [Google Cloud Console](https://console.cloud.google.com/)
2. Enable "Custom Search API"
3. Create Search Engine at [Programmable Search](https://programmablesearchengine.google.com/)
4. Copy Search Engine ID as `GOOGLE_SEARCH_CX`
5. Copy API key as `GOOGLE_SEARCH_API_KEY`

### Microsoft Azure

#### Azure Translator

1. Go to [Azure Portal](https://portal.azure.com/)
2. Create "Translator" resource
3. Copy Key 1 to `.env` as `AZURE_TRANSLATOR_KEY`
4. Copy Region (e.g., "westus2") as `AZURE_TRANSLATOR_REGION`

### OpenAI

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create new API key
3. Copy to `.env` as `OPENAI_API_KEY`

### Anthropic (Claude)

1. Visit [Anthropic Console](https://console.anthropic.com/)
2. Create API key
3. Copy to `.env` as `ANTHROPIC_API_KEY`

### Grok (X AI)

1. Visit [X AI Console](https://console.x.ai/)
2. Generate API key
3. Copy to `.env` as `GROK_API_KEY`

### DeepSeek

1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Create API key
3. Copy to `.env` as `DEEPSEEK_API_KEY`

### DeepL

1. Visit [DeepL API](https://www.deepl.com/pro-api)
2. Sign up for Free or Pro plan
3. Copy Authentication Key to `.env` as `DEEPL_API_KEY`

### Amazon Translate

1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Create IAM user with "TranslateFullAccess" policy
3. Generate access keys
4. Copy to `.env` as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### LibreTranslate (Free)

1. Use public instance: `https://libretranslate.com/translate` (no key needed)
2. Or self-host: [LibreTranslate GitHub](https://github.com/LibreTranslate/LibreTranslate)
3. Optional: Get API key from [LibreTranslate](https://libretranslate.com/)

### HuggingFace

1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create new token (read access)
3. Copy to `.env` as `HUGGINGFACE_API_KEY`

---

## ⚙️ Configuration / پیکربندی

### Enabling Services / فعال‌سازی سرویس‌ها

Edit `cad3d/super_ai/connectors_config.json`:

```json
{
  "google_translate": {
    "enabled": true,  // Set to true to enable
    "api_key_env": "GOOGLE_TRANSLATE_API_KEY"
  },
  "openai": {
    "enabled": true,
    "api_key_env": "OPENAI_API_KEY",
    "model": "gpt-4o"
  }
  // ... etc
}
```

**Important:** Only enable services you have valid API keys for!

---

## 🚀 Usage / استفاده

### Automatic Cascading Fallback / سقوط خودکار پلکانی

The system automatically tries providers in priority order:

#### Translation Priority

1. **DeepL** (highest quality)
2. Microsoft Translator
3. Google Translate
4. LibreTranslate (free)
5. Argos Translate (offline)

#### Chat/LLM Priority

1. **Anthropic Claude** (highest quality)
2. OpenAI GPT-4o
3. Google Gemini
4. DeepSeek
5. Grok
6. HuggingFace API
7. Local models (offline)

### Example Code / کد نمونه

```python
from cad3d.super_ai.external_connectors import unified_connector

# Translation with automatic fallback
result = unified_connector.translate("Hello world", "fa")
print(result)  # "سلام دنیا"

# Chat with automatic fallback
response = unified_connector.chat_with_fallback(
    prompt="Explain feasibility analysis",
    system_prompt="You are an architectural AI expert."
)
print(response['content'][0]['text'])

# Online research
search_results = unified_connector.search("modern architecture trends", num_results=5)
```

### Using in KURDO-AI Brain / استفاده در مغز KURDO-AI

The brain automatically uses external connectors when processing requests:

```python
from cad3d.super_ai.brain import SuperAIBrain

brain = SuperAIBrain()

# Process in Persian - automatically translates using external APIs
result = brain.process_request("امکان‌سنجی یک برج ۲۰ طبقه در زمین ۱۰۰۰ متری")

# Brain will:
# 1. Detect Persian language
# 2. Translate to English using cascading fallback
# 3. Perform online research if enabled
# 4. Generate AI-enhanced response
# 5. Translate back to Persian
```

---

## 📊 Testing Services / تست سرویس‌ها

Create a test script `test_connectors.py`:

```python
from cad3d.super_ai.external_connectors import unified_connector

# Test translation
print("Testing Translation...")
result = unified_connector.translate("Hello", "fa")
print(f"Result: {result}")

# Test chat
print("\nTesting Chat...")
response = unified_connector.chat_with_fallback(
    prompt="Hello, who are you?",
    system_prompt="You are KURDO-AI."
)
print(f"Response: {response.get('content', [{}])[0].get('text', 'N/A')}")

# Test search
if unified_connector.is_enabled("google_search"):
    print("\nTesting Search...")
    results = unified_connector.search("AI architecture", num_results=3)
    print(f"Found {len(results.get('items', []))} results")
```

Run:

```bash
python test_connectors.py
```

---

## 🔍 Monitoring & Debugging / نظارت و اشکال‌زدایی

### Check Service Status / بررسی وضعیت سرویس‌ها

```python
from cad3d.super_ai.external_connectors import unified_connector

# Check which services are enabled
for service in ["google_translate", "openai", "anthropic", "deepl"]:
    status = "✅ Enabled" if unified_connector.is_enabled(service) else "❌ Disabled"
    print(f"{service}: {status}")
```

### Enable Console Logging / فعال‌سازی لاگ کنسول

The system automatically prints status messages:

```
[TRANSLATION] Attempting deepl...
[TRANSLATION] Success with deepl
```

```
[CHAT] Attempting anthropic...
[CHAT] Success with anthropic
```

---

## 💡 Best Practices / بهترین روش‌ها

### Cost Management / مدیریت هزینه

1. **Start with Free Tiers**: LibreTranslate (free), HuggingFace (generous limits)
2. **Use Local Models**: Install transformers for offline fallback
3. **Enable Strategic Services**: Only enable what you need
4. **Monitor Usage**: Check provider dashboards regularly

### Quality Optimization / بهینه‌سازی کیفیت

1. **DeepL for Translation**: Best quality, enable if available
2. **Claude for Critical Tasks**: Most accurate LLM
3. **Gemini for Speed**: Fast and cost-effective
4. **Local Models for Testing**: Free and private

### Security / امنیت

1. **Never commit `.env`**: Already in `.gitignore`
2. **Rotate keys regularly**: Monthly recommended
3. **Use IAM roles in production**: For AWS services
4. **Restrict API key permissions**: Minimum required only

---

## 🐛 Troubleshooting / عیب‌یابی

### Issue: "Translation unavailable"

- Check `.env` file exists and has correct keys
- Verify service is enabled in `connectors_config.json`
- Test API key directly using provider's console

### Issue: "All chat providers failed"

- At least one chat provider must be enabled
- Check API key format (some require prefixes)
- Verify internet connection

### Issue: Local models fail to load

- Install transformers: `pip install transformers torch`
- First run downloads models (~500MB-2GB)
- Check disk space and internet connection

### Issue: Rate limiting errors

- Reduce request frequency
- Upgrade to paid tier
- Enable more fallback providers

---

## 📦 Local Models Setup / راه‌اندازی مدل‌های محلی

### For Offline Translation (Argos)

```bash
pip install argostranslate
python -c "import argostranslate.package; argostranslate.package.update_package_index(); argostranslate.package.install_from_path('en_fa')"
```

### For Chat/LLM (Flan-T5)

```bash
pip install transformers torch

# First use will download ~800MB model
python -c "from transformers import AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')"
```

---

## 🎯 Next Steps / مراحل بعدی

1. ✅ Install packages: `pip install requests beautifulsoup4 python-dotenv`
2. ✅ Create `.env` from `.env.example`
3. ✅ Add API keys for desired services
4. ✅ Enable services in `connectors_config.json`
5. ✅ Test with `test_connectors.py`
6. ✅ Run KURDO-AI with external intelligence!

---

## 📞 Support / پشتیبانی

For issues or questions:

- Review this guide carefully
- Check provider documentation
- Test services individually
- Enable verbose logging for debugging

**KURDO-AI is now connected to the world's leading AI platforms! 🌍🤖**

---

*Last updated: 2024*
*Version: 2.0 (Multi-Provider Cascade)*
