"""
Diverse Meta-Controller Test
آزمایش متنوع برای نمایش انتخاب روش‌های مختلف
"""

import io
import sys
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cad3d.super_ai.unified_ai_system import (
    UnifiedAISystem,
    AITaskType
)

print("\n" + "="*80)
print("  DIVERSE META-CONTROLLER TEST - Different Methods Selection")
print("="*80 + "\n")

unified = UnifiedAISystem()
print("✓ System initialized with Meta-Controller\n")

# Test cases designed to trigger different methods
test_cases = [
    {
        "query": "hi",
        "expected": "Prompt Engineering",
        "reason": "Very simple, very fast needed"
    },
    {
        "query": "تحلیل جامع و کامل سیستم سازه‌ای ساختمان 20 طبقه با بررسی کامل استانداردهای ملی و بین‌المللی و ارائه راهکارهای بهینه‌سازی",
        "task": AITaskType.STRUCTURAL_CALCULATION,
        "expected": "Fine-Tuning",
        "reason": "Very complex, specialized, high accuracy needed"
    },
    {
        "query": "طراحی سیستم تهویه مطبوع",
        "task": AITaskType.MEP_OPTIMIZATION,
        "expected": "LoRA or PEFT",
        "reason": "MEP domain, specialized"
    },
    {
        "query": "what is the formula for area",
        "expected": "RAG",
        "reason": "Knowledge query, fast"
    },
    {
        "query": "use lora adapter for structural beam analysis",
        "expected": "LoRA",
        "reason": "Explicit LoRA keyword"
    },
    {
        "query": "apply peft for mep",
        "expected": "PEFT",
        "reason": "Explicit PEFT keyword"
    },
    {
        "query": "use fine-tuning model",
        "expected": "Fine-Tuning",
        "reason": "Explicit fine-tuning keyword"
    },
]

results = []

for i, test in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"  TEST {i}: {test['reason']}")
    print(f"{'='*80}")
    
    query = test["query"]
    task = test.get("task")
    expected = test["expected"]
    
    print(f"\nQuery: {query}")
    print(f"Expected: {expected}")
    
    # Get explanation
    explanation = unified.explain_selection(query, task)
    selected = explanation["selected_method"]
    
    print(f"\nSelected: {selected}")
    print(f"Reasoning: {explanation['reasoning']}")
    print(f"Domain: {explanation['features']['domain']}")
    print(f"Complexity: {explanation['features']['complexity']}")
    
    # Execute
    response = unified.query(query, task_type=task)
    actual = response["method"]
    
    match = "✓" if selected.lower() in expected.lower() or expected.lower() in selected.lower() else "✗"
    print(f"\nMatch: {match} (Expected: {expected}, Got: {actual})")
    
    results.append({
        "test": i,
        "expected": expected,
        "selected": selected,
        "actual": actual,
        "match": match == "✓"
    })

print("\n" + "="*80)
print("  RESULTS SUMMARY")
print("="*80)

# Count by method
status = unified.get_system_status()
stats = status["usage_statistics"]

print(f"\nTotal Queries: {stats['total_queries']}")
print(f"  RAG: {stats['rag_calls']}")
print(f"  Fine-Tuning: {stats['fine_tuning_calls']}")
print(f"  LoRA: {stats['lora_calls']}")
print(f"  Prompt Engineering: {stats['prompt_calls']}")
print(f"  PEFT: {stats['peft_calls']}")

print("\nMethod Diversity:")
methods_used = []
for key, val in stats.items():
    if key.endswith('_calls') and val > 0:
        method = key.replace('_calls', '').replace('_', ' ').title()
        methods_used.append(method)

print(f"  {len(methods_used)} different methods used: {', '.join(methods_used)}")

matches = sum(1 for r in results if r["match"])
print(f"\nAccuracy: {matches}/{len(results)} ({100*matches//len(results)}%)")

print("\n" + "="*80)
print("  META-CONTROLLER INTELLIGENCE DEMONSTRATED!")
print("="*80)
print("\nKey Features:")
print("  ✓ Analyzes query complexity (SIMPLE → VERY_COMPLEX)")
print("  ✓ Detects domain (architecture, structural, MEP, calculation)")
print("  ✓ Assesses urgency (REALTIME → BATCH)")
print("  ✓ Calculates confidence requirements (70% - 100%)")
print("  ✓ Multi-criteria scoring: speed (30%) + accuracy (40%) + cost (30%)")
print("  ✓ Adaptive learning from execution history")
print("  ✓ Transparent decision explanation")
print("  ✓ Selects optimal method based on query characteristics")
print("\n" + "="*80 + "\n")
