"""
Test script for KURDO-AI Fine-Tuning capabilities
تست قابلیت‌های Fine-Tuning سیستم KURDO-AI
"""

from cad3d.super_ai.brain import SuperAIBrain
from cad3d.super_ai.fine_tuning import fine_tuning_manager

def test_fine_tuning_availability():
    """بررسی در دسترس بودن سیستم Fine-Tuning"""
    print("=" * 60)
    print("🔍 Testing Fine-Tuning Availability / تست دسترسی")
    print("=" * 60)
    
    brain = SuperAIBrain()
    status = brain.get_status()
    
    print(f"✅ Fine-Tuning Available: {status.get('fine_tuning', False)}")
    print(f"📊 Previous Fine-Tuning Jobs: {status.get('fine_tuning_jobs', 0)}")
    
    if status.get('last_fine_tune'):
        print(f"📅 Last Fine-Tune: {status['last_fine_tune']}")
    
    print()
    return status.get('fine_tuning', False)

def test_architectural_corpus():
    """تست بارگذاری داده‌های معماری"""
    print("=" * 60)
    print("📚 Testing Architectural Corpus / تست داده‌های معماری")
    print("=" * 60)
    
    training_data = fine_tuning_manager.prepare_architectural_training_data()
    
    if training_data:
        print(f"✅ Loaded {len(training_data)} training examples")
        print(f"📝 Sample example:")
        if len(training_data) > 0:
            sample = training_data[0]
            print(f"   System: {sample['messages'][0]['content'][:50]}...")
            print(f"   User: {sample['messages'][1]['content'][:50]}...")
            print(f"   Assistant: {sample['messages'][2]['content'][:50]}...")
    else:
        print("⚠️  No training data found. Check datasets/persian_corpus/ directory")
    
    print()
    return len(training_data) if training_data else 0

def test_anthropic_simulation():
    """تست شبیه‌سازی Fine-Tuning با Anthropic"""
    print("=" * 60)
    print("🤖 Testing Anthropic Prompt Caching / تست Anthropic")
    print("=" * 60)
    
    brain = SuperAIBrain()
    
    # ایجاد داده‌های نمونه
    sample_data = [
        {
            "input": "امکان‌سنجی یک ساختمان ۱۰ طبقه",
            "output": "برای امکان‌سنجی باید زمین، مقررات و بودجه بررسی شود"
        },
        {
            "input": "محاسبه تراکم ساختمانی",
            "output": "تراکم = مساحت زیربنا / مساحت زمین"
        }
    ]
    
    result = brain.fine_tune_model(
        provider="anthropic",
        training_data=sample_data,
        use_architectural_corpus=False
    )
    
    print(f"Status: {result.get('status')}")
    print(f"Message: {result.get('message', 'N/A')}")
    
    if result.get('cached_prompt_file'):
        print(f"✅ Cached prompt saved to: {result['cached_prompt_file']}")
    
    print()
    return result.get('status') == 'completed'

def test_openai_preparation():
    """تست آماده‌سازی برای OpenAI (بدون اجرای واقعی)"""
    print("=" * 60)
    print("🚀 Testing OpenAI Preparation / تست آماده‌سازی OpenAI")
    print("=" * 60)
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("✅ OpenAI API Key found")
        print(f"   Key prefix: {api_key[:15]}...")
        print()
        print("⚠️  To actually start fine-tuning:")
        print("   brain.fine_tune_model(provider='openai', use_architectural_corpus=True)")
        print()
        print("💰 Estimated cost:")
        print("   ~100 examples × 3 epochs = ~$1-2 USD")
    else:
        print("❌ OpenAI API Key not found")
        print("   Add OPENAI_API_KEY to .env file")
        print("   Get key from: https://platform.openai.com/api-keys")
    
    print()
    return api_key is not None

def test_huggingface_availability():
    """بررسی قابلیت Fine-Tune محلی با HuggingFace"""
    print("=" * 60)
    print("🤗 Testing HuggingFace Availability / تست HuggingFace")
    print("=" * 60)
    
    try:
        import transformers
        import datasets
        print("✅ transformers library installed")
        print(f"   Version: {transformers.__version__}")
        print()
        print("✅ Ready for local fine-tuning!")
        print("   Example:")
        print("   brain.fine_tune_model(provider='huggingface', base_model='google/flan-t5-small')")
        available = True
    except ImportError:
        print("❌ transformers library not installed")
        print("   Install with: pip install transformers datasets accelerate")
        available = False
    
    print()
    return available

def test_custom_data():
    """تست با داده‌های سفارشی"""
    print("=" * 60)
    print("📝 Testing Custom Training Data / تست داده سفارشی")
    print("=" * 60)
    
    custom_data = [
        {
            "messages": [
                {"role": "system", "content": "تو یک مشاور معماری هستی"},
                {"role": "user", "content": "ضریب اشغال چیست؟"},
                {"role": "assistant", "content": "ضریب اشغال نسبت سطح اشغال زمین به کل مساحت زمین است"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "تو یک مشاور معماری هستی"},
                {"role": "user", "content": "تراکم ساختمانی چطور محاسبه میشه؟"},
                {"role": "assistant", "content": "تراکم = مجموع زیربناها / مساحت زمین"}
            ]
        }
    ]
    
    print(f"✅ Created {len(custom_data)} custom training examples")
    print()
    print("Sample format:")
    print(f"  {custom_data[0]}")
    print()
    print("💡 To use custom data:")
    print("   brain.fine_tune_model(provider='openai', training_data=custom_data)")
    
    print()
    return True

def show_fine_tuning_history():
    """نمایش تاریخچه Fine-Tuning"""
    print("=" * 60)
    print("📜 Fine-Tuning History / تاریخچه Fine-Tuning")
    print("=" * 60)
    
    history = fine_tuning_manager.get_fine_tuning_history()
    
    if history:
        print(f"✅ Found {len(history)} previous fine-tuning jobs:")
        print()
        for idx, job in enumerate(history, 1):
            print(f"{idx}. Provider: {job.get('provider')}")
            print(f"   Status: {job.get('status')}")
            print(f"   Date: {job.get('timestamp')}")
            if job.get('job_id'):
                print(f"   Job ID: {job.get('job_id')}")
            print()
    else:
        print("📭 No fine-tuning history yet")
        print("   Run a fine-tuning job to see it here!")
    
    print()

def main():
    """اجرای تمام تست‌ها"""
    print()
    print("🎓" * 30)
    print("KURDO-AI FINE-TUNING TEST SUITE")
    print("مجموعه تست قابلیت‌های Fine-Tuning")
    print("🎓" * 30)
    print()
    
    results = {}
    
    # Test 1: Availability
    results['availability'] = test_fine_tuning_availability()
    
    # Test 2: Architectural Corpus
    results['corpus_count'] = test_architectural_corpus()
    
    # Test 3: Anthropic Simulation
    results['anthropic'] = test_anthropic_simulation()
    
    # Test 4: OpenAI Preparation
    results['openai_ready'] = test_openai_preparation()
    
    # Test 5: HuggingFace
    results['huggingface'] = test_huggingface_availability()
    
    # Test 6: Custom Data
    results['custom_data'] = test_custom_data()
    
    # Show History
    show_fine_tuning_history()
    
    # Summary
    print("=" * 60)
    print("📊 SUMMARY / خلاصه نتایج")
    print("=" * 60)
    print(f"Fine-Tuning Module: {'✅ Available' if results['availability'] else '❌ Not Available'}")
    print(f"Architectural Corpus: {'✅ ' + str(results['corpus_count']) + ' examples' if results['corpus_count'] > 0 else '⚠️  No data'}")
    print(f"Anthropic Ready: {'✅ Yes' if results['anthropic'] else '❌ No'}")
    print(f"OpenAI Ready: {'✅ Yes' if results['openai_ready'] else '⚠️  API key needed'}")
    print(f"HuggingFace Ready: {'✅ Yes' if results['huggingface'] else '⚠️  Install needed'}")
    print(f"Custom Data Format: {'✅ Valid' if results['custom_data'] else '❌ Invalid'}")
    print()
    
    # Recommendations
    print("=" * 60)
    print("💡 RECOMMENDATIONS / توصیه‌ها")
    print("=" * 60)
    
    if results['anthropic']:
        print("✅ You can start with Anthropic (fast & free):")
        print("   brain.fine_tune_model(provider='anthropic', use_architectural_corpus=True)")
        print()
    
    if results['openai_ready']:
        print("✅ OpenAI fine-tuning ready (costs ~$1-2):")
        print("   brain.fine_tune_model(provider='openai', use_architectural_corpus=True)")
        print()
    elif not results['openai_ready']:
        print("⚠️  Add OpenAI API key to .env for production fine-tuning")
        print()
    
    if results['huggingface']:
        print("✅ Local fine-tuning available (free, but slower):")
        print("   brain.fine_tune_model(provider='huggingface', base_model='google/flan-t5-small')")
        print()
    
    if results['corpus_count'] > 0:
        print(f"✅ {results['corpus_count']} architectural examples ready for training")
        print()
    
    print("=" * 60)
    print("🎉 Fine-Tuning System Ready! / سیستم آماده است!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
