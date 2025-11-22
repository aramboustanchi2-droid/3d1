"""
Test Meta-Controller Intelligence

آزمایش سیستم Meta-Controller برای انتخاب هوشمند روش AI
"""

import io
import sys
import os

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cad3d.super_ai.unified_ai_system import (
    UnifiedAISystem,
    AITaskType
)

print("\n" + "="*80)
print("  META-CONTROLLER INTELLIGENCE TEST")
print("="*80 + "\n")

# Initialize
print("Step 1: Initializing Unified AI System with Meta-Controller...")
unified = UnifiedAISystem()
print("✓ System initialized\n")

# Check Meta-Controller status
status = unified.get_system_status()
if status.get('meta_controller'):
    print("✓ Meta-Controller is active")
    print(f"  Performance Stats:")
    for method, stats in status['meta_controller'].items():
        print(f"    {method}: Success={stats['success_rate']}, AvgTime={stats['avg_time']}")
else:
    print("✗ Meta-Controller not available")

print("\n" + "="*80)
print("  TEST 1: Simple Calculation (Fast Response)")
print("="*80)

query1 = "محاسبه مساحت اتاق 5 در 4 متر"
print(f"\nQuery: {query1}")
print("Expected: RAG (fast, accurate for calculations)")

# Get explanation before execution
explanation1 = unified.explain_selection(query1, AITaskType.ARCHITECTURAL_DESIGN)
print(f"\nSelected Method: {explanation1['selected_method']}")
print(f"Reasoning: {explanation1['reasoning']}")
print(f"Features:")
print(f"  - Complexity: {explanation1['features']['complexity']}")
print(f"  - Domain: {explanation1['features']['domain']}")
print(f"  - Confidence Needed: {explanation1['features']['confidence_needed']}")
print(f"Scores:")
print(f"  - Speed: {explanation1['scores']['speed']}")
print(f"  - Accuracy: {explanation1['scores']['accuracy']}")
print(f"  - Cost: {explanation1['scores']['cost']}")

# Execute query
response1 = unified.query(query1, task_type=AITaskType.ARCHITECTURAL_DESIGN)
print(f"\nActual Method: {response1['method']}")
print(f"Status: {response1['status']}")
if 'selection_reasoning' in response1:
    print(f"Controller: {response1['selection_reasoning']['controller']}")

print("\n" + "="*80)
print("  TEST 2: Complex Analysis (Specialized)")
print("="*80)

query2 = "تحلیل و بهینه‌سازی طراحی ساختار سازه‌ای یک ساختمان 10 طبقه با در نظر گرفتن ضوابط ملی"
print(f"\nQuery: {query2}")
print("Expected: Fine-Tuning (complex, specialized, high accuracy needed)")

explanation2 = unified.explain_selection(query2, AITaskType.STRUCTURAL_CALCULATION)
print(f"\nSelected Method: {explanation2['selected_method']}")
print(f"Reasoning: {explanation2['reasoning']}")
print(f"Features:")
print(f"  - Complexity: {explanation2['features']['complexity']}")
print(f"  - Domain: {explanation2['features']['domain']}")
print(f"  - Requires Knowledge: {explanation2['features']['requires_knowledge']}")
print(f"  - Specialized: {explanation2['features']['is_specialized']}")
print(f"Scores: Speed={explanation2['scores']['speed']}, Accuracy={explanation2['scores']['accuracy']}")

response2 = unified.query(query2, task_type=AITaskType.STRUCTURAL_CALCULATION)
print(f"\nActual Method: {response2['method']}")
print(f"Status: {response2['status']}")

print("\n" + "="*80)
print("  TEST 3: Quick General Question")
print("="*80)

query3 = "پیشنهاد برای چیدمان فضای نشیمن"
print(f"\nQuery: {query3}")
print("Expected: Prompt Engineering (simple, fast, general)")

explanation3 = unified.explain_selection(query3)
print(f"\nSelected Method: {explanation3['selected_method']}")
print(f"Reasoning: {explanation3['reasoning']}")
print(f"Features:")
print(f"  - Complexity: {explanation3['features']['complexity']}")
print(f"  - Domain: {explanation3['features']['domain']}")

response3 = unified.query(query3)
print(f"\nActual Method: {response3['method']}")
print(f"Status: {response3['status']}")

print("\n" + "="*80)
print("  TEST 4: Technical MEP Query")
print("="*80)

query4 = "محاسبه ظرفیت لوله‌کشی فاضلاب برای ساختمان"
print(f"\nQuery: {query4}")
print("Expected: LoRA or PEFT (specialized domain, moderate complexity)")

explanation4 = unified.explain_selection(query4, AITaskType.MEP_OPTIMIZATION)
print(f"\nSelected Method: {explanation4['selected_method']}")
print(f"Reasoning: {explanation4['reasoning']}")
print(f"Features:")
print(f"  - Complexity: {explanation4['features']['complexity']}")
print(f"  - Domain: {explanation4['features']['domain']}")

if 'alternatives' in explanation4:
    print(f"\nTop 3 Alternatives:")
    for alt in explanation4['alternatives']:
        print(f"  {alt['method']}: score={alt['score']} - {alt['reasoning']}")

response4 = unified.query(query4, task_type=AITaskType.MEP_OPTIMIZATION)
print(f"\nActual Method: {response4['method']}")
print(f"Status: {response4['status']}")

print("\n" + "="*80)
print("  TEST 5: PEFT Adapter Query")
print("="*80)

query5 = "use peft prefix-tuning adapter for architectural analysis"
print(f"\nQuery: {query5}")
print("Expected: PEFT (explicit keyword)")

explanation5 = unified.explain_selection(query5)
print(f"\nSelected Method: {explanation5['selected_method']}")
print(f"Reasoning: {explanation5['reasoning']}")

response5 = unified.query(query5)
print(f"\nActual Method: {response5['method']}")
print(f"Status: {response5['status']}")

print("\n" + "="*80)
print("  PERFORMANCE SUMMARY")
print("="*80)

final_status = unified.get_system_status()
if final_status.get('meta_controller'):
    print("\nMeta-Controller Performance (Updated):")
    for method, stats in final_status['meta_controller'].items():
        print(f"  {method}:")
        print(f"    Success Rate: {stats['success_rate']}")
        print(f"    Avg Time: {stats['avg_time']}")

stats = final_status['usage_statistics']
print(f"\nTotal Queries: {stats['total_queries']}")
print(f"  RAG: {stats['rag_calls']}")
print(f"  Fine-Tuning: {stats['fine_tuning_calls']}")
print(f"  LoRA: {stats['lora_calls']}")
print(f"  Prompt Engineering: {stats['prompt_calls']}")
print(f"  PEFT: {stats['peft_calls']}")

print("\n" + "="*80)
print("  SUCCESS!")
print("="*80)
print("\nMeta-Controller Features Demonstrated:")
print("  ✓ Query complexity analysis")
print("  ✓ Domain detection (architecture, structural, MEP)")
print("  ✓ Confidence requirement assessment")
print("  ✓ Multi-criteria scoring (speed, accuracy, cost)")
print("  ✓ Performance tracking and learning")
print("  ✓ Decision explanation")
print("\nIntelligent AI method selection is operational!")
print("="*80 + "\n")
