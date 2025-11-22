#!/usr/bin/env python3
"""
KURDO-AI Architectural Training - Real-World Example
Demonstrates automatic training with architectural data
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain
import json


# Sample architectural training data
ARCHITECTURAL_TRAINING_DATA = [
    # Room area calculations
    {
        "prompt": "محاسبه مساحت اتاق با ابعاد 5 متر در 4 متر",
        "completion": "مساحت = طول × عرض = 5 × 4 = 20 متر مربع"
    },
    {
        "prompt": "مساحت اتاق 6×3.5 متر چقدر است؟",
        "completion": "مساحت = 6 × 3.5 = 21 متر مربع"
    },
    {
        "prompt": "Calculate area of room 8m × 5m",
        "completion": "Area = length × width = 8 × 5 = 40 square meters"
    },
    
    # Volume calculations
    {
        "prompt": "حجم یک اتاق با ابعاد 4×3×2.8 متر را محاسبه کن",
        "completion": "حجم = طول × عرض × ارتفاع = 4 × 3 × 2.8 = 33.6 متر مکعب"
    },
    {
        "prompt": "Volume of room 5m × 4m with 3m height?",
        "completion": "Volume = 5 × 4 × 3 = 60 cubic meters"
    },
    
    # Material calculations
    {
        "prompt": "چند آجر برای دیوار 10 متری با ارتفاع 3 متر نیاز است؟",
        "completion": "برای هر متر مربع دیوار معمولاً 60 آجر نیاز است.\nمساحت دیوار = 10 × 3 = 30 متر مربع\nتعداد آجر = 30 × 60 = 1800 آجر"
    },
    {
        "prompt": "How many bricks for 15m wall, 3m height?",
        "completion": "Wall area = 15 × 3 = 45 m²\nBricks needed (60 per m²) = 45 × 60 = 2,700 bricks"
    },
    {
        "prompt": "چند کیلوگرم سیمان برای 100 متر مربع سفت‌کاری؟",
        "completion": "برای سفت‌کاری 1 متر مربع (ضخامت 2 سانتی‌متر): حدود 5 کیلوگرم سیمان\nبرای 100 متر مربع = 100 × 5 = 500 کیلوگرم سیمان"
    },
    
    # Structural calculations
    {
        "prompt": "حداکثر طول دهانه تیر فلزی IPE200 بدون ستون؟",
        "completion": "تیر IPE200 با فرض بار متوسط مسکونی:\nحداکثر دهانه ایمن: حدود 4-5 متر\nبرای دهانه بیشتر باید از تیرهای سنگین‌تر یا ستون میانی استفاده شود."
    },
    {
        "prompt": "What is the maximum span for IPE240 beam?",
        "completion": "IPE240 beam with residential load:\nMaximum safe span: approximately 5-6 meters\nFor longer spans, use heavier beams or intermediate columns."
    },
    
    # Foundation calculations
    {
        "prompt": "عمق پی ساختمان 3 طبقه در تهران چقدر باید باشد؟",
        "completion": "برای ساختمان 3 طبقه در تهران:\n- عمق حداقل پی: 1.5 متر (زیر تراز یخبندان)\n- پهنای پی: حداقل 80 سانتی‌متر\n- توصیه: انجام آزمایش خاک برای تعیین دقیق"
    },
    {
        "prompt": "Foundation depth for 5-story building?",
        "completion": "For 5-story building:\n- Minimum depth: 2-2.5 meters below ground\n- Width: minimum 1 meter\n- Recommendation: soil test required for precise calculation"
    },
    
    # Cost estimation
    {
        "prompt": "تخمین هزینه ساخت آپارتمان 100 متری در تهران",
        "completion": "هزینه تقریبی ساخت (1402):\n- اسکلت فلزی: 8-10 میلیون تومان/متر\n- اسکلت بتنی: 7-9 میلیون تومان/متر\n- نازک‌کاری: 3-4 میلیون تومان/متر\n- جمع برای 100 متر: حدود 1-1.5 میلیارد تومان\n(قیمت‌ها متغیر است)"
    },
    {
        "prompt": "Estimate construction cost for 150 sqm apartment",
        "completion": "Approximate construction cost:\n- Structure (concrete): $200-250/sqm\n- Finishing: $100-150/sqm\n- Total for 150 sqm: $45,000-60,000\n(Prices vary by location and quality)"
    },
    
    # Code compliance
    {
        "prompt": "حداقل ارتفاع سقف آپارتمان مسکونی در ایران؟",
        "completion": "طبق مبحث 19 مقررات ملی ساختمان:\n- حداقل ارتفاع سقف اتاق‌های اصلی: 2.4 متر\n- حداقل ارتفاع سقف راهرو: 2.1 متر\n- حداقل ارتفاع سقف سرویس‌های بهداشتی: 2.1 متر"
    },
    {
        "prompt": "Minimum ceiling height for residential building?",
        "completion": "According to building codes:\n- Living rooms, bedrooms: minimum 2.4 meters (7.9 feet)\n- Hallways: minimum 2.1 meters (6.9 feet)\n- Bathrooms: minimum 2.1 meters (6.9 feet)"
    },
    
    # Earthquake resistance
    {
        "prompt": "چگونه ساختمان را در برابر زلزله مقاوم کنیم؟",
        "completion": "روش‌های افزایش مقاومت در برابر زلزله:\n1. استفاده از اسکلت فلزی یا بتنی با آرماتوربندی صحیح\n2. اتصالات قوی بین عناصر سازه\n3. توزیع متقارن جرم و سختی\n4. استفاده از دیوارهای برشی\n5. پایه‌ریزی صحیح و عمیق\n6. اجرای استاندارد 2800 ایران"
    },
    {
        "prompt": "How to make building earthquake-resistant?",
        "completion": "Earthquake resistance methods:\n1. Strong structural frame (steel or reinforced concrete)\n2. Robust connections between elements\n3. Symmetric mass and stiffness distribution\n4. Shear walls\n5. Deep and proper foundation\n6. Follow seismic codes (like UBC, IBC)"
    },
    
    # Energy efficiency
    {
        "prompt": "راه‌های کاهش مصرف انرژی در ساختمان چیست؟",
        "completion": "روش‌های کاهش مصرف انرژی:\n1. عایق‌کاری دیوارها (حداقل 5 سانتی‌متر)\n2. پنجره‌های دوجداره (کاهش 30-40% اتلاف حرارت)\n3. عایق سقف و کف\n4. استفاده از پنل‌های خورشیدی\n5. جهت‌گیری مناسب ساختمان\n6. سیستم گرمایش مرکزی\n7. روشنایی LED"
    },
    {
        "prompt": "Ways to reduce building energy consumption?",
        "completion": "Energy efficiency methods:\n1. Wall insulation (min 5cm)\n2. Double-glazed windows (30-40% heat loss reduction)\n3. Roof and floor insulation\n4. Solar panels\n5. Proper building orientation\n6. Central heating system\n7. LED lighting"
    },
    
    # Parking calculations
    {
        "prompt": "ابعاد استاندارد یک پارکینگ چقدر است؟",
        "completion": "ابعاد استاندارد پارکینگ:\n- پارکینگ عمودی: 2.5 × 5 متر (حداقل)\n- پارکینگ موازی: 2 × 6 متر\n- پارکینگ جانبی (45 درجه): 2.5 × 5.5 متر\n- عرض راهرو دسترسی: حداقل 6 متر\n- ارتفاع سقف: حداقل 2.2 متر"
    },
    {
        "prompt": "Standard parking space dimensions?",
        "completion": "Standard parking dimensions:\n- Perpendicular: 2.5 × 5 meters (8.2 × 16.4 ft)\n- Parallel: 2 × 6 meters (6.6 × 19.7 ft)\n- Angled (45°): 2.5 × 5.5 meters\n- Aisle width: minimum 6 meters (19.7 ft)\n- Ceiling height: minimum 2.2 meters (7.2 ft)"
    },
    
    # Staircase design
    {
        "prompt": "ابعاد استاندارد پله در ساختمان مسکونی؟",
        "completion": "ابعاد استاندارد پله:\n- ارتفاع پله (ضلع قائم): 17-18 سانتی‌متر\n- عرض پله (ضلع افقی): 28-30 سانتی‌متر\n- رابطه بلون: 2h + d = 63 سانتی‌متر\n- عرض راه پله: حداقل 90 سانتی‌متر (ترجیحاً 120 سانتی‌متر)\n- ارتفاع نرده: 90-100 سانتی‌متر"
    },
    {
        "prompt": "Standard staircase dimensions for residential?",
        "completion": "Standard stair dimensions:\n- Riser height: 17-18 cm (6.7-7.1 inches)\n- Tread depth: 28-30 cm (11-11.8 inches)\n- Blondel's formula: 2h + d = 63 cm (24.8 in)\n- Staircase width: minimum 90 cm (preferably 120 cm)\n- Handrail height: 90-100 cm (35-39 inches)"
    },
    
    # Window sizing
    {
        "prompt": "نسبت مساحت پنجره به مساحت کف اتاق چقدر باید باشد؟",
        "completion": "نسبت استاندارد پنجره به کف اتاق:\n- حداقل: 1/8 (12.5% مساحت کف)\n- ترجیحی: 1/6 تا 1/5 (16-20%)\n- مثال: اتاق 20 متری → حداقل 2.5 متر مربع پنجره\n- برای نور کافی: 2-3 متر مربع پنجره در هر اتاق"
    },
    {
        "prompt": "Window to floor area ratio?",
        "completion": "Standard window to floor ratio:\n- Minimum: 1/8 (12.5% of floor area)\n- Preferred: 1/6 to 1/5 (16-20%)\n- Example: 20 sqm room → minimum 2.5 sqm window\n- For adequate light: 2-3 sqm window per room"
    },
    
    # Plumbing
    {
        "prompt": "حداقل شیب لوله فاضلاب چقدر باشد؟",
        "completion": "شیب استاندارد لوله فاضلاب:\n- لوله‌های 50-100 میلی‌متر: 2-3 درصد (2-3 سانتی‌متر در هر متر)\n- لوله‌های بزرگ‌تر (بیش از 100 میلی‌متر): 1-2 درصد\n- حداقل مطلق: 1 درصد\n- مثال: لوله 5 متری → حداقل 5 سانتی‌متر اختلاف ارتفاع"
    },
    {
        "prompt": "Minimum slope for drainage pipes?",
        "completion": "Standard drainage pipe slope:\n- Pipes 50-100mm: 2-3% (2-3 cm per meter)\n- Large pipes (>100mm): 1-2%\n- Absolute minimum: 1%\n- Example: 5-meter pipe → minimum 5 cm height difference"
    },
]


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    """Main training workflow."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          🏗️  KURDO-AI ARCHITECTURAL TRAINING - REAL EXAMPLE 🏗️             ║
║                                                                              ║
║  Training KURDO-AI on architectural knowledge and calculations               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    brain = SuperAIBrain()
    
    # Step 1: Show training data summary
    print_section("STEP 1: TRAINING DATA OVERVIEW")
    print(f"📊 Total training samples: {len(ARCHITECTURAL_TRAINING_DATA)}")
    print("\n🏗️  Categories covered:")
    print("  • Room area and volume calculations")
    print("  • Material quantity estimation")
    print("  • Structural engineering basics")
    print("  • Foundation design")
    print("  • Cost estimation")
    print("  • Building codes compliance")
    print("  • Earthquake resistance")
    print("  • Energy efficiency")
    print("  • Parking design")
    print("  • Staircase design")
    print("  • Window sizing")
    print("  • Plumbing standards")
    
    print("\n📝 Sample entries:")
    for i, sample in enumerate(ARCHITECTURAL_TRAINING_DATA[:3], 1):
        print(f"\n  [{i}] Prompt: {sample['prompt'][:60]}...")
        print(f"      Response: {sample['completion'][:80]}...")
    
    # Step 2: Get intelligent recommendation
    print_section("STEP 2: INTELLIGENT TRAINING RECOMMENDATION")
    print("🤖 Analyzing available resources and recommending best method...")
    
    recommendation = brain.recommend_training_method(
        dataset_size=len(ARCHITECTURAL_TRAINING_DATA),
        provider="local"
    )
    
    print(f"\n✅ Recommended Method: {recommendation.get('recommended_method', 'Unknown')}")
    print(f"🎯 Confidence: {recommendation.get('confidence', 0) * 100:.0f}%")
    print(f"⏱️  Estimated Time: {recommendation.get('estimated_time_hours', 0):.1f} hours")
    print(f"💰 Estimated Cost: ${recommendation.get('estimated_cost_usd', 0):.2f}")
    
    if recommendation.get('gpu_memory_available'):
        print(f"🖥️  GPU Memory Available: {recommendation['gpu_memory_available']:.1f} GB")
    
    print("\n📋 Reasoning:")
    for reason in recommendation.get('reasoning', []):
        print(f"  • {reason}")
    
    print("\n📦 Requirements:")
    for req in recommendation.get('requirements', []):
        print(f"  • {req}")
    
    if recommendation.get('alternatives'):
        print("\n🔄 Alternative Methods:")
        for alt in recommendation['alternatives']:
            print(f"  • {alt.get('method')}: {alt.get('estimated_time_hours', 0):.1f}h, ${alt.get('estimated_cost_usd', 0):.2f}")
    
    # Step 3: User confirmation
    print_section("STEP 3: TRAINING CONFIRMATION")
    print("⚠️  This will start actual model training (may take hours).")
    print("📁 Adapter will be saved as: 'kurdo-arch-knowledge-v1'")
    print("🎯 Base model: meta-llama/Llama-2-7b-hf")
    
    proceed = input("\n🤔 Proceed with training? (yes/no): ").strip().lower()
    
    if proceed not in ['yes', 'y']:
        print("\n⏸️  Training cancelled by user.")
        print("💡 To train later, run:")
        print("   from cad3d.super_ai.brain import SuperAIBrain")
        print("   brain = SuperAIBrain()")
        print("   brain.auto_train(training_data=your_data, adapter_name='your-name')")
        return
    
    # Step 4: Execute training
    print_section("STEP 4: AUTO-TRAINING")
    print("🚀 Starting automatic training with recommended method...")
    print("⏳ This may take a while. Please be patient...\n")
    
    result = brain.auto_train(
        training_data=ARCHITECTURAL_TRAINING_DATA,
        adapter_name="kurdo-arch-knowledge-v1",
        model_name="meta-llama/Llama-2-7b-hf",
        provider="local"
    )
    
    # Step 5: Show results
    print_section("STEP 5: TRAINING RESULTS")
    
    if result.get("status") == "success":
        print("✅ Training completed successfully!")
        print(f"\n📁 Adapter Name: {result.get('adapter_name', 'Unknown')}")
        print(f"📂 Adapter Path: {result.get('adapter_path', 'Unknown')}")
        
        if "training_time_seconds" in result:
            mins = result["training_time_seconds"] / 60
            print(f"⏱️  Training Time: {mins:.1f} minutes")
        
        if "metrics" in result:
            print("\n📊 Training Metrics:")
            for key, value in result["metrics"].items():
                print(f"  • {key}: {value}")
        
        print("\n🎉 KURDO-AI has learned architectural knowledge!")
        print("\n💡 How to use:")
        print("   from cad3d.super_ai.brain import SuperAIBrain")
        print("   brain = SuperAIBrain()")
        print("   # Load the adapter and generate responses")
        
    else:
        print(f"❌ Training failed: {result.get('message', 'Unknown error')}")
        print("\n💡 Troubleshooting:")
        print("  • Check GPU memory (may need more)")
        print("  • Try OpenAI fine-tuning instead (set provider='openai')")
        print("  • Ensure all dependencies installed: pip install peft bitsandbytes")
    
    # Step 6: Show all adapters
    print_section("STEP 6: ALL TRAINED ADAPTERS")
    adapters = brain.list_lora_adapters()
    
    if adapters.get("adapters"):
        print("🎯 Available LoRA Adapters:")
        for adapter in adapters["adapters"]:
            print(f"  • {adapter}")
    else:
        print("📝 No adapters trained yet.")
    
    if adapters.get("training_history"):
        print("\n📜 Training History:")
        for entry in adapters["training_history"]:
            print(f"  • {entry.get('adapter_name', 'Unknown')} - {entry.get('timestamp', 'Unknown')}")
    
    print("\n✅ Training workflow complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
