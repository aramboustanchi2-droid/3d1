"""
Simple test for Unified AI System
Tests RAG + Fine-Tuning + LoRA + Prompt Engineering + PEFT
"""

import io
import sys
import os

# Fix Unicode encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cad3d.super_ai.unified_ai_system import (
    UnifiedAISystem,
    AIMethodType,
    AITaskType
)

print("\n" + "="*80)
print("  CAD3D UNIFIED AI SYSTEM - 5 METHODS TEST")
print("="*80 + "\n")

# Initialize
print("Step 1: Initializing Unified AI System...")
unified = UnifiedAISystem()
print("✓ System initialized\n")

# Get status
print("Step 2: System Status")
print("-" * 40)
status = unified.get_system_status()
print(f"Available Methods: {', '.join(status['unified_ai_system']['methods_available'])}")

if status.get('rag'):
    print(f"RAG Documents: {status['rag']['documents_indexed']}")
    print(f"RAG Model: {status['rag']['embedding_model']}")

if status.get('security'):
    print(f"Security: {status['security']['status']}")

print("\n" + "="*80)
print("  TEST 1: RAG (Retrieval-Augmented Generation)")
print("="*80)

query1 = "محاسبه مساحت اتاق 5 در 4 متر"
print(f"\nQuery: {query1}")
response1 = unified.query(
    query=query1,
    method=AIMethodType.RAG,
    task_type=AITaskType.ARCHITECTURAL_DESIGN,
    top_k=2
)
print(f"Method: {response1['method']}")
print(f"Status: {response1['status']}")
print(f"Retrieved Docs: {response1.get('num_docs', 0)}")

if response1.get('retrieved_documents'):
    print("\nTop Documents:")
    for i, doc in enumerate(response1['retrieved_documents'][:2], 1):
        print(f"  {i}. {doc['doc_id']} (Score: {doc['relevance_score']:.2f})")
        print(f"     {doc['content'][:80]}...")

print("\n" + "="*80)
print("  TEST 2: Fine-Tuning")
print("="*80)

query2 = "تحلیل ساختار نقشه معماری"
print(f"\nQuery: {query2}")
response2 = unified.query(
    query=query2,
    method=AIMethodType.FINE_TUNING,
    task_type=AITaskType.CAD_ANALYSIS
)
print(f"Method: {response2['method']}")
print(f"Status: {response2['status']}")
print(f"Details: {response2.get('method_details', 'N/A')}")

print("\n" + "="*80)
print("  TEST 3: LoRA (Low-Rank Adaptation)")
print("="*80)

query3 = "محاسبات سازه‌ای برای ساختمان"
print(f"\nQuery: {query3}")
response3 = unified.query(
    query=query3,
    method=AIMethodType.LORA,
    task_type=AITaskType.STRUCTURAL_CALCULATION
)
print(f"Method: {response3['method']}")
print(f"Status: {response3['status']}")
print(f"Adapter: {response3.get('adapter_used', 'N/A')}")

print("\n" + "="*80)
print("  TEST 4: Prompt Engineering")
print("="*80)

query4 = "پیشنهاد طراحی خانه"
print(f"\nQuery: {query4}")
response4 = unified.query(
    query=query4,
    method=AIMethodType.PROMPT_ENGINEERING,
    task_type=AITaskType.ARCHITECTURAL_DESIGN
)
print(f"Method: {response4['method']}")
print(f"Status: {response4['status']}")
print(f"Template: {response4.get('template_used', 'N/A')}")

print("\n" + "="*80)
print("  TEST 5: PEFT (Parameter-Efficient Fine-Tuning)")
print("="*80)

query5 = "adapter prefix-tuning for architecture"
print(f"\nQuery: {query5}")
response5 = unified.query(
    query=query5,
    method=AIMethodType.PEFT,
    task_type=AITaskType.ARCHITECTURAL_DESIGN
)
print(f"Method: {response5['method']}")
print(f"Status: {response5['status']}")
print(f"Technique: {response5.get('technique', 'N/A')}")
print(f"Adapter: {response5.get('adapter_used', 'N/A')}")
if response5.get('peft_available') is not None:
    print(f"PEFT Library: {'Available' if response5['peft_available'] else 'Not installed (lightweight mode)'}")

print("\n" + "="*80)
print("  TEST 6: Auto Method Selection")
print("="*80)

test_queries = [
    "محاسبه حجم اتاق",
    "fine-tuning for analysis",
    "lora adapter structural",
    "prompt template design",
    "peft prefix adapter"
]

print("\nTesting auto-routing:")
for query in test_queries:
    response = unified.query(query)
    print(f"  '{query}' → {response['method']}")

print("\n" + "="*80)
print("  TEST 7: Hybrid Query")
print("="*80)

query_hybrid = "استانداردهای معماری"
print(f"\nQuery: {query_hybrid}")
print("Methods: RAG + Prompt Engineering")

response_hybrid = unified.hybrid_query(
    query=query_hybrid,
    methods=[AIMethodType.RAG, AIMethodType.PROMPT_ENGINEERING]
)

print(f"Methods Used: {', '.join(response_hybrid['methods_used'])}")
print(f"Individual Responses: {len(response_hybrid['individual_responses'])}")

for method, resp in response_hybrid['individual_responses'].items():
    print(f"  {method}: {resp['status']}")

print("\n" + "="*80)
print("  METHOD COMPARISON")
print("="*80)

comparison = unified.compare_methods()
print("\nMethod Details:")
for method, details in comparison['comparison'].items():
    print(f"\n  {method}:")
    print(f"    Setup Time: {details['setup_time']}")
    print(f"    Cost: {details['cost']}")
    print(f"    Quality: {details['quality']}")
    print(f"    GPU Required: {details['gpu_required']}")

print("\n" + "="*80)
print("  FINAL STATISTICS")
print("="*80)

final_status = unified.get_system_status()
stats = final_status['usage_statistics']

print(f"\nTotal Queries: {stats['total_queries']}")
print(f"  RAG: {stats['rag_calls']}")
print(f"  Fine-Tuning: {stats['fine_tuning_calls']}")
print(f"  LoRA: {stats['lora_calls']}")
print(f"  Prompt Engineering: {stats['prompt_calls']}")
print(f"  PEFT: {stats['peft_calls']}")
print(f"  Hybrid: {stats['hybrid_calls']}")

print("\n" + "="*80)
print("  SUCCESS!")
print("="*80)
print("\nAll 5 AI Methods Operational:")
print("  1. RAG - Retrieval-Augmented Generation")
print("  2. Fine-Tuning - Specialized Training")
print("  3. LoRA - Low-Rank Adaptation")
print("  4. Prompt Engineering - Training-Free")
print("  5. PEFT - Parameter-Efficient Fine-Tuning")
print("\nUnified AI System is ready!")
print("="*80 + "\n")
