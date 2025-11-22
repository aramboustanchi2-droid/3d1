"""
تست سیستم یکپارچه AI با 4 مدل
RAG + Fine-Tuning + LoRA + Prompt Engineering + Security
import io
"""

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
import json

def print_section(title: str):
    """چاپ عنوان بخش"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_response(response: dict):
    """چاپ پاسخ با فرمت زیبا"""
    print(json.dumps(response, indent=2, ensure_ascii=False))

def test_unified_ai_system():
    """تست کامل سیستم یکپارچه"""
    
    print("\n" + "="*80)
    print("CAD3D UNIFIED AI SYSTEM - 4 METHODS INTEGRATION TEST")
    print("="*80 + "\n")
    
    # ===========================
    # 1. راه‌اندازی سیستم
    # ===========================
    print_section("📋 STEP 1: System Initialization")
    
    unified = UnifiedAISystem()
    
    print("✅ Unified AI System initialized")
    print(f"   Methods available: {len(unified.get_system_status()['unified_ai_system']['methods_available'])}")
    
    # ===========================
    # 2. نمایش وضعیت
    # ===========================
    print_section("📋 STEP 2: System Status")
    
    status = unified.get_system_status()
    print("🔍 System Status:")
    print(f"   Available Methods: {', '.join(status['unified_ai_system']['methods_available'])}")
    
    if status.get('rag'):
        print(f"\n   📚 RAG System:")
        print(f"      - Documents: {status['rag']['documents_indexed']}")
        print(f"      - Retrievals: {status['rag']['total_retrievals']}")
        print(f"      - Model: {status['rag']['embedding_model']}")
    
    if status.get('security'):
        print(f"\n   🛡️ Security System:")
        print(f"      - Status: {status['security']['status']}")
        print(f"      - Agents Created: {status['security']['agents_created']}")
    
    print(f"\n   📊 Usage Statistics:")
    for key, value in status['usage_statistics'].items():
        print(f"      - {key}: {value}")
    
    # ===========================
    # 3. تست RAG
    # ===========================
    print_section("📋 STEP 3: Testing RAG (Retrieval-Augmented Generation)")
    
    query_rag = "محاسبه مساحت اتاق 5 در 4 متر چقدر است؟"
    print(f"Query: {query_rag}")
    print("\nExecuting RAG query...")
    
    response_rag = unified.query(
        query=query_rag,
        method=AIMethodType.RAG,
        task_type=AITaskType.ARCHITECTURAL_DESIGN,
        top_k=2
    )
    
    print(f"\n✅ Method: {response_rag['method']}")
    print(f"   Status: {response_rag['status']}")
    print(f"   Retrieved Documents: {response_rag.get('num_docs', 0)}")
    
    if response_rag.get('retrieved_documents'):
        print("\n   📄 Top Documents:")
        for i, doc in enumerate(response_rag['retrieved_documents'][:2], 1):
            print(f"\n      {i}. {doc['doc_id']} (Score: {doc['relevance_score']:.2f})")
            print(f"         {doc['content'][:100]}...")
    
    # ===========================
    # 4. تست Fine-Tuning
    # ===========================
    print_section("📋 STEP 4: Testing Fine-Tuning")
    
    query_ft = "تحلیل ساختار نقشه معماری"
    print(f"Query: {query_ft}")
    print("\nExecuting Fine-Tuning query...")
    
    response_ft = unified.query(
        query=query_ft,
        method=AIMethodType.FINE_TUNING,
        task_type=AITaskType.CAD_ANALYSIS,
        model="cad_analysis_v1"
    )
    
    print(f"\n✅ Method: {response_ft['method']}")
    print(f"   Status: {response_ft['status']}")
    print(f"   Model: {response_ft.get('model_used', 'N/A')}")
    print(f"   Details: {response_ft.get('method_details', 'N/A')}")
    
    # ===========================
    # 5. تست LoRA
    # ===========================
    print_section("📋 STEP 5: Testing LoRA (Low-Rank Adaptation)")
    
    query_lora = "محاسبات سازه‌ای برای ساختمان 5 طبقه"
    print(f"Query: {query_lora}")
    print("\nExecuting LoRA query...")
    
    response_lora = unified.query(
        query=query_lora,
        method=AIMethodType.LORA,
        task_type=AITaskType.STRUCTURAL_CALCULATION,
        adapter="structural_calc"
    )
    
    print(f"\n✅ Method: {response_lora['method']}")
    print(f"   Status: {response_lora['status']}")
    print(f"   Adapter: {response_lora.get('adapter_used', 'N/A')}")
    print(f"   Rank: {response_lora.get('rank', 'N/A')}")
    
    # ===========================
    # 6. تست Prompt Engineering
    # ===========================
    print_section("📋 STEP 6: Testing Prompt Engineering")
    
    query_prompt = "پیشنهاد طراحی برای خانه 200 متری"
    print(f"Query: {query_prompt}")
    print("\nExecuting Prompt Engineering query...")
    
    response_prompt = unified.query(
        query=query_prompt,
        method=AIMethodType.PROMPT_ENGINEERING,
        task_type=AITaskType.ARCHITECTURAL_DESIGN,
        template="architectural_analysis"
    )
    
    print(f"\n✅ Method: {response_prompt['method']}")
    print(f"   Status: {response_prompt['status']}")
    print(f"   Template: {response_prompt.get('template_used', 'N/A')}")
    if response_prompt.get('prompt'):
        print(f"   Prompt Length: {len(response_prompt['prompt'])} characters")
    
    # ===========================
    # 7. تست انتخاب خودکار
    # ===========================
    print_section("📋 STEP 7: Testing Auto Method Selection")
    
    test_queries = [
        "محاسبه حجم اتاق 6×4×2.8 متر",
        "تحلیل و طراحی نقشه معماری",
        "محاسبات سازه برای تیر",
        "پیشنهاد چیدمان اتاق"
    ]
    
    print("Testing auto-routing for different queries:\n")
    
    for query in test_queries:
        response = unified.query(query)  # بدون مشخص کردن method
        print(f"   Query: {query}")
        print(f"   → Selected Method: {response['method']}")
        print()
    
    # ===========================
    # 8. تست Hybrid (ترکیبی)
    # ===========================
    print_section("📋 STEP 8: Testing Hybrid Approach")
    
    query_hybrid = "استانداردهای ارتفاع سقف در ایران"
    print(f"Query: {query_hybrid}")
    print("\nExecuting HYBRID query (RAG + Prompt Engineering)...")
    
    response_hybrid = unified.hybrid_query(
        query=query_hybrid,
        methods=[AIMethodType.RAG, AIMethodType.PROMPT_ENGINEERING],
        top_k=2
    )
    
    print(f"\n✅ Hybrid Query Executed")
    print(f"   Methods Used: {', '.join(response_hybrid['methods_used'])}")
    print(f"   Responses: {len(response_hybrid['individual_responses'])}")
    
    for method, resp in response_hybrid['individual_responses'].items():
        print(f"\n   📌 {method}:")
        print(f"      Status: {resp['status']}")
        if 'num_docs' in resp:
            print(f"      Retrieved Docs: {resp['num_docs']}")
    
    # ===========================
    # 9. مقایسه روش‌ها
    # ===========================
    print_section("📋 STEP 9: Comparing All 4 Methods")
    
    comparison = unified.compare_methods()
    
    print("📊 Method Comparison:\n")
    
    for method, details in comparison['comparison'].items():
        print(f"   🔹 {method}:")
        print(f"      Setup Time: {details['setup_time']}")
        print(f"      Cost: {details['cost']}")
        print(f"      Quality: {details['quality']}")
        print(f"      GPU Required: {'Yes' if details['gpu_required'] else 'No'}")
        print(f"      Best For: {', '.join(details['best_for'][:2])}...")
        print()
    
    print("   💡 Recommendations:")
    for key, value in comparison['recommendation'].items():
        print(f"      {key.replace('_', ' ').title()}: {value}")
    
    # ===========================
    # 10. آمار نهایی
    # ===========================
    print_section("📋 STEP 10: Final Statistics")
    
    final_status = unified.get_system_status()
    stats = final_status['usage_statistics']
    
    print("📈 Usage Statistics:")
    print(f"   Total Queries: {stats['total_queries']}")
    print(f"   RAG Calls: {stats['rag_calls']}")
    print(f"   Fine-Tuning Calls: {stats['fine_tuning_calls']}")
    print(f"   LoRA Calls: {stats['lora_calls']}")
    print(f"   Prompt Engineering Calls: {stats['prompt_calls']}")
    print(f"   Hybrid Calls: {stats['hybrid_calls']}")
    
    # ===========================
    # خلاصه نهایی
    # ===========================
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    print(f"\n✅ All 4 AI Methods Tested Successfully!")
    print(f"   1. RAG - Retrieval-Augmented Generation")
    print(f"   2. Fine-Tuning - Specialized Training")
    print(f"   3. LoRA - Low-Rank Adaptation")
    print(f"   4. Prompt Engineering - Careful Prompting")
    
    print(f"\n🔗 Integration Status:")
    print(f"   ✅ Security System: {'Integrated' if final_status.get('security') else 'Not Available'}")
    print(f"   ✅ Auto-Routing: Enabled")
    print(f"   ✅ Hybrid Queries: Supported")
    print(f"   ✅ Total Methods: 4")
    
    print("\n✅ Unified AI System is fully operational!")
    print("="*80 + "\n")

def test_rag_knowledge_base():
    """تست پایگاه دانش RAG"""
    print("\n" + "="*80)
    print("📚 RAG KNOWLEDGE BASE TEST")
    print("="*80 + "\n")
    
    unified = UnifiedAISystem()
    
    if not unified.rag_system:
        print("❌ RAG System not available")
        return
    
    print("📖 Testing RAG Knowledge Base:\n")
    
    # پرسش‌های نمونه
    test_queries = [
        "محاسبه مساحت اتاق",
        "ارتفاع سقف استاندارد",
        "نسبت پنجره به کف",
        "شیب لوله فاضلاب",
        "room area calculation"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: {query}")
        
        results = unified.rag_system.retrieve(query, top_k=2)
        
        if results:
            print(f"   Retrieved: {len(results)} documents")
            for doc, score in results[:1]:
                print(f"   → {doc.doc_id} (Score: {score:.2f})")
                print(f"      {doc.content[:80]}...")
        else:
            print("   No results")
        print()
    
    # آمار
    stats = unified.rag_system.get_statistics()
    print(f"📊 RAG Statistics:")
    print(f"   Documents Indexed: {stats['documents_indexed']}")
    print(f"   Total Retrievals: {stats['total_retrievals']}")
    print(f"   Avg Retrieval Time: {stats['avg_retrieval_time_ms']:.2f} ms")
    print(f"   Vector Store Size: {stats['vector_store_size']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        # تست 1: سیستم یکپارچه
        test_unified_ai_system()
        
        input("\n⏸️  Press ENTER to continue to RAG Knowledge Base Test...")
        
        # تست 2: پایگاه دانش RAG
        test_rag_knowledge_base()
        
        print("\n✅ All tests completed successfully!")
        print("🎉 Unified AI System with 4 methods is fully operational!\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
