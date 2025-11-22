#!/usr/bin/env python3
"""
KURDO-AI RAG System + Four Methods Integration Test
Tests RAG and hybrid combinations with Fine-Tuning, LoRA, and Prompt Engineering
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain
import json


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(result):
    """Pretty print result."""
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def test_rag_basics():
    """Test basic RAG functionality."""
    print_section("1. RAG SYSTEM - BASIC FUNCTIONALITY")
    
    brain = SuperAIBrain()
    
    # Check RAG statistics
    print("📊 RAG Statistics:")
    stats = brain.get_rag_statistics()
    print_result(stats)
    
    # Test retrieval
    print("\n🔍 Test Query: 'محاسبه مساحت اتاق'")
    print("-" * 60)
    
    results = brain.retrieve_knowledge(
        query="محاسبه مساحت اتاق",
        top_k=3
    )
    
    print(f"✅ Retrieved {len(results)} documents:\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"📄 Document {i} (Relevance: {score:.3f})")
        print(f"   ID: {doc.doc_id}")
        print(f"   Content: {doc.content[:100]}...")
        print(f"   Metadata: {doc.metadata}")
        print()


def test_rag_prompts():
    """Test RAG prompt generation."""
    print_section("2. RAG PROMPT GENERATION")
    
    brain = SuperAIBrain()
    
    query = "چند آجر برای دیوار 15 متری با ارتفاع 3 متر نیاز است؟"
    
    print(f"🎯 Query: {query}")
    print("\n" + "-" * 60 + "\n")
    
    # Generate RAG prompt
    prompt = brain.generate_rag_prompt(
        query=query,
        top_k=3
    )
    
    print("📝 Generated RAG Prompt:")
    print(prompt)


def test_rag_query():
    """Test complete RAG query."""
    print_section("3. COMPLETE RAG QUERY")
    
    brain = SuperAIBrain()
    
    query = "حداقل ارتفاع سقف اتاق خواب چقدر است؟"
    
    print(f"🎯 Query: {query}")
    print("\n" + "-" * 60 + "\n")
    
    response = brain.rag_query(
        query=query,
        top_k=3,
        generation_method="prompt_engineering"
    )
    
    print("📋 RAG Response:")
    print(f"\n  Query: {response['query']}")
    print(f"  Method: {response['generation_method']}")
    print(f"  Retrieved: {response['num_documents_retrieved']} documents\n")
    
    print("  📚 Retrieved Documents:")
    for i, doc in enumerate(response['retrieved_documents'], 1):
        print(f"\n  [{i}] {doc['doc_id']} (Relevance: {doc['relevance_score']:.3f})")
        print(f"      {doc['content'][:150]}...")
    
    print("\n" + "-" * 60 + "\n")
    print("  💬 Generated Prompt Preview:")
    print(f"  {response['prompt'][:300]}...")


def test_add_custom_knowledge():
    """Test adding custom knowledge."""
    print_section("4. ADDING CUSTOM KNOWLEDGE")
    
    brain = SuperAIBrain()
    
    # Add custom document
    custom_doc = """
    تخمین هزینه ساخت ساختمان مسکونی در تهران (1403):
    - اسکلت بتنی: 8-10 میلیون تومان/متر
    - اسکلت فلزی: 9-11 میلیون تومان/متر
    - نازک‌کاری (کامل): 4-5 میلیون تومان/متر
    - نما (سنگ): 2-3 میلیون تومان/متر
    جمع کل برای 100 متر: حدود 1.5-2 میلیارد تومان
    """
    
    print("📝 Adding Custom Document:")
    print(custom_doc.strip())
    print("\n" + "-" * 60 + "\n")
    
    doc = brain.add_knowledge_document(
        content=custom_doc.strip(),
        doc_id="custom_cost_001",
        metadata={"category": "cost", "year": "1403", "language": "fa"}
    )
    
    if doc:
        print(f"✅ Document Added: {doc.doc_id}")
        print(f"   Metadata: {doc.metadata}")
    
    # Test retrieval of custom document
    print("\n🔍 Testing Retrieval:")
    results = brain.retrieve_knowledge(
        query="تخمین هزینه ساخت آپارتمان",
        top_k=2
    )
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n  [{i}] {doc.doc_id} (Score: {score:.3f})")
        print(f"      {doc.content[:100]}...")


def test_hybrid_rag_prompt_engineering():
    """Test RAG + Prompt Engineering hybrid."""
    print_section("5. HYBRID: RAG + PROMPT ENGINEERING")
    
    brain = SuperAIBrain()
    
    query = "محاسبه مساحت اتاق 9×7 متر"
    
    # Few-shot examples
    examples = [
        {"input": "مساحت 5×4", "output": "20 متر مربع"},
        {"input": "مساحت 8×6", "output": "48 متر مربع"}
    ]
    
    print(f"🎯 Query: {query}")
    print(f"📚 Few-Shot Examples: {len(examples)}")
    print(f"🔍 RAG Documents: top 2")
    print("\n" + "-" * 60 + "\n")
    
    prompt = brain.hybrid_rag_prompt_engineering(
        query=query,
        few_shot_examples=examples,
        top_k=2
    )
    
    print("🚀 Hybrid Prompt (RAG + Few-Shot):")
    print(prompt[:500] + "...\n")
    
    print("✅ This combines:")
    print("  • RAG: Retrieved relevant context from knowledge base")
    print("  • Prompt Engineering: Added few-shot examples")
    print("  • Result: Best of both worlds!")


def test_all_four_methods_comparison():
    """Compare all four methods."""
    print_section("6. COMPARISON: ALL FOUR METHODS")
    
    brain = SuperAIBrain()
    
    comparison = brain.compare_all_four_methods()
    
    print("📊 Complete Comparison:")
    print_result(comparison)
    
    # Print summary table
    print("\n" + "=" * 100)
    print("QUICK COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Method':<25} {'Time':<15} {'Cost':<15} {'GPU':<15} {'Best For':<30}")
    print("-" * 100)
    
    methods_summary = [
        ("RAG", "Minutes", "$0", "No", "Knowledge bases, facts"),
        ("Prompt Engineering", "Instant", "$0", "No", "Prototyping, quick start"),
        ("LoRA", "1-3 hours", "$0", "Yes (6GB+)", "Multiple tasks, limited GPU"),
        ("Fine-Tuning", "4-10 hours", "$10-50", "Yes (40GB+)", "Production, best quality")
    ]
    
    for method, time, cost, gpu, best_for in methods_summary:
        print(f"{method:<25} {time:<15} {cost:<15} {gpu:<15} {best_for:<30}")
    
    print("=" * 100)


def test_hybrid_strategies():
    """Test different hybrid strategies."""
    print_section("7. HYBRID STRATEGIES")
    
    brain = SuperAIBrain()
    
    comparison = brain.compare_all_four_methods()
    
    print("🔄 Available Hybrid Strategies:\n")
    
    strategies = comparison.get("hybrid_strategies", {})
    
    for strategy_name, details in strategies.items():
        print(f"📌 {strategy_name.upper().replace('_', ' ')}")
        print(f"   Description: {details['description']}")
        print(f"   Best For: {details['best_for']}")
        print(f"   Setup Time: {details['setup_time']}")
        print(f"   Cost: {details['cost']}")
        print()


def test_practical_use_case():
    """Test practical architectural use case."""
    print_section("8. PRACTICAL USE CASE: ARCHITECTURAL ASSISTANT")
    
    brain = SuperAIBrain()
    
    print("🏗️  Scenario: Architectural consultant answering client questions\n")
    
    queries = [
        "عمق پی برای ساختمان 5 طبقه در تهران؟",
        "چند آجر برای دیوار 20 متری نیاز دارم؟",
        "ابعاد استاندارد پارکینگ چیست؟",
        "نسبت پنجره به کف اتاق؟"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"🔹 Query {i}: {query}")
        print("-" * 60)
        
        # Use RAG to answer
        results = brain.retrieve_knowledge(query, top_k=1)
        
        if results:
            doc, score = results[0]
            print(f"   📚 Retrieved (Relevance: {score:.3f}):")
            print(f"   {doc.content[:200]}...")
            print()
        else:
            print("   ❌ No relevant documents found\n")


def test_save_load_knowledge_base():
    """Test saving and loading knowledge base."""
    print_section("9. SAVE/LOAD KNOWLEDGE BASE")
    
    brain = SuperAIBrain()
    
    # Get current stats
    stats_before = brain.get_rag_statistics()
    print("📊 Current Knowledge Base:")
    print(f"   Documents: {stats_before.get('vector_store_size', 0)}")
    print(f"   Total Retrievals: {stats_before.get('total_retrievals', 0)}")
    
    # Save
    print("\n💾 Saving knowledge base...")
    result = brain.save_rag_knowledge_base(name="test_kb")
    print(f"   Status: {result.get('status', 'unknown')}")
    
    # Load
    print("\n📂 Loading knowledge base...")
    result = brain.load_rag_knowledge_base(name="test_kb")
    print(f"   Status: {result.get('status', 'unknown')}")
    
    # Verify
    stats_after = brain.get_rag_statistics()
    print("\n✅ Verification:")
    print(f"   Documents: {stats_after.get('vector_store_size', 0)}")
    print(f"   Match: {'Yes' if stats_before.get('vector_store_size') == stats_after.get('vector_store_size') else 'No'}")


def demo_four_methods_workflow():
    """Demonstrate complete workflow with all four methods."""
    print_section("10. COMPLETE WORKFLOW: ALL FOUR METHODS")
    
    print("""
🎯 REAL-WORLD SCENARIO: Building an Architectural AI Assistant

Phase 1: IMMEDIATE START (Day 1)
┌─────────────────────────────────────────────────────────────┐
│ Method: RAG + Prompt Engineering                            │
│ Time: 1-2 hours                                             │
│ Cost: $0                                                    │
├─────────────────────────────────────────────────────────────┤
│ • Index architectural standards (مبحث 19، استاندارد 2800)  │
│ • Add calculation formulas                                  │
│ • Create few-shot prompts                                   │
│ • Launch MVP!                                               │
└─────────────────────────────────────────────────────────────┘

Phase 2: COLLECT DATA (Weeks 1-2)
┌─────────────────────────────────────────────────────────────┐
│ • Log all user queries                                      │
│ • Collect expert responses                                  │
│ • Build dataset: 50-100 examples                           │
│ • RAG handles 80% of queries successfully                   │
└─────────────────────────────────────────────────────────────┘

Phase 3: TRAIN LoRA (Week 3)
┌─────────────────────────────────────────────────────────────┐
│ Method: RAG + LoRA                                          │
│ Time: 2-3 hours training                                    │
│ Cost: $0 (local GPU)                                        │
├─────────────────────────────────────────────────────────────┤
│ • Train LoRA adapter on collected data                      │
│ • Keep RAG for facts and standards                          │
│ • Use LoRA for reasoning and calculations                   │
│ • Accuracy improves to 90%                                  │
└─────────────────────────────────────────────────────────────┘

Phase 4: SCALE UP (Month 2)
┌─────────────────────────────────────────────────────────────┐
│ • Collect 500+ examples                                     │
│ • Multiple LoRA adapters for different tasks:               │
│   - Calculations (مساحت، حجم، آجر)                         │
│   - Standards (مبحث 19، استاندارد 2800)                   │
│   - Cost estimation                                         │
│   - Design review                                           │
└─────────────────────────────────────────────────────────────┘

Phase 5: PRODUCTION (Month 3+)
┌─────────────────────────────────────────────────────────────┐
│ Method: RAG + Fine-Tuning + LoRA + Prompt Engineering      │
│ Time: One-time 8-hour training                             │
│ Cost: $50 (cloud) or $0 (local A100)                       │
├─────────────────────────────────────────────────────────────┤
│ • Fine-tune base model on 1000+ examples                    │
│ • Keep RAG for dynamic knowledge                            │
│ • Keep LoRA adapters for specific tasks                     │
│ • Keep Prompt Engineering for edge cases                    │
│ • Accuracy: 95%+                                            │
└─────────────────────────────────────────────────────────────┘

FINAL ARCHITECTURE:
═══════════════════
                    User Query
                        ↓
        ┌───────────────┴───────────────┐
        │     Query Router              │
        │  (Classify query type)        │
        └───────────────┬───────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │           RAG                 │
        │  (Retrieve relevant context)  │
        └───────────────┬───────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │    Generation Method          │
        ├───────────────────────────────┤
        │ • Simple facts → RAG only     │
        │ • Calculations → LoRA         │
        │ • Complex → Fine-Tuned        │
        │ • Edge cases → Few-Shot       │
        └───────────────┬───────────────┘
                        ↓
                    Response

✅ BENEFITS:
• Start immediately with RAG
• Iterate quickly with LoRA
• Scale to production with Fine-Tuning
• Handle everything with combined approach
• Total cost: $0-50 (vs $500+ for traditional approach)
    """)


def interactive_menu():
    """Interactive test menu."""
    print_section("KURDO-AI RAG + FOUR METHODS - INTERACTIVE DEMO")
    
    menu = """
    Choose a test to run:
    
    1. 📚 RAG System - Basic Functionality
    2. 📝 RAG Prompt Generation
    3. 💬 Complete RAG Query
    4. ➕ Add Custom Knowledge
    5. 🔄 Hybrid: RAG + Prompt Engineering
    6. 📊 Compare All Four Methods
    7. 🎯 Hybrid Strategies
    8. 🏗️  Practical Use Case
    9. 💾 Save/Load Knowledge Base
    10. 🚀 Complete Workflow Demo
    11. 🔥 Run All Tests
    12. ❌ Exit
    
    Enter choice (1-12): """
    
    tests = {
        '1': test_rag_basics,
        '2': test_rag_prompts,
        '3': test_rag_query,
        '4': test_add_custom_knowledge,
        '5': test_hybrid_rag_prompt_engineering,
        '6': test_all_four_methods_comparison,
        '7': test_hybrid_strategies,
        '8': test_practical_use_case,
        '9': test_save_load_knowledge_base,
        '10': demo_four_methods_workflow,
    }
    
    while True:
        try:
            choice = input(menu).strip()
            
            if choice == '12':
                print("\n👋 Goodbye!")
                break
            elif choice == '11':
                print("\n🚀 Running all tests...\n")
                for test_func in tests.values():
                    try:
                        test_func()
                    except Exception as e:
                        print(f"❌ Test failed: {e}\n")
                print("\n✅ All tests complete!")
            elif choice in tests:
                try:
                    tests[choice]()
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    import traceback
                    traceback.print_exc()
            else:
                print("\n❌ Invalid choice. Please enter 1-12.\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except EOFError:
            break


def main():
    """Main entry point."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🎯 KURDO-AI: RAG + FOUR METHODS INTEGRATION TEST 🎯                      ║
║                                                                              ║
║  Four Complementary Methods:                                                 ║
║    1️⃣  RAG - Retrieval-Augmented Generation (Knowledge Base)               ║
║    2️⃣  Prompt Engineering - Zero/Few-Shot Learning                         ║
║    3️⃣  LoRA - Parameter-Efficient Fine-Tuning                              ║
║    4️⃣  Fine-Tuning - Complete Model Adaptation                             ║
║                                                                              ║
║  Hybrid Strategies:                                                          ║
║    ✅ RAG + Prompt Engineering (Instant, $0)                                ║
║    ✅ RAG + LoRA (Fast, Accurate)                                           ║
║    ✅ RAG + Fine-Tuning (Production-Grade)                                  ║
║    ✅ All Four Combined (Enterprise AI)                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--rag-basics':
            test_rag_basics()
        elif arg == '--rag-prompts':
            test_rag_prompts()
        elif arg == '--rag-query':
            test_rag_query()
        elif arg == '--add-knowledge':
            test_add_custom_knowledge()
        elif arg == '--hybrid':
            test_hybrid_rag_prompt_engineering()
        elif arg == '--compare':
            test_all_four_methods_comparison()
        elif arg == '--strategies':
            test_hybrid_strategies()
        elif arg == '--use-case':
            test_practical_use_case()
        elif arg == '--save-load':
            test_save_load_knowledge_base()
        elif arg == '--workflow':
            demo_four_methods_workflow()
        elif arg == '--all':
            print("\n🚀 Running all tests...\n")
            test_rag_basics()
            test_rag_prompts()
            test_rag_query()
            test_add_custom_knowledge()
            test_hybrid_rag_prompt_engineering()
            test_all_four_methods_comparison()
            test_hybrid_strategies()
            test_practical_use_case()
            test_save_load_knowledge_base()
            demo_four_methods_workflow()
            print("\n✅ All tests complete!")
        else:
            print(f"Unknown argument: {arg}")
            print("\nAvailable arguments:")
            print("  --rag-basics    : Test RAG basics")
            print("  --rag-prompts   : Test RAG prompts")
            print("  --rag-query     : Test RAG query")
            print("  --add-knowledge : Test adding knowledge")
            print("  --hybrid        : Test RAG + Prompt Eng")
            print("  --compare       : Compare all methods")
            print("  --strategies    : Show hybrid strategies")
            print("  --use-case      : Practical example")
            print("  --save-load     : Save/load KB")
            print("  --workflow      : Complete workflow")
            print("  --all           : Run all tests")
            print("  (no args)       : Interactive menu")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
