#!/usr/bin/env python3
"""
KURDO-AI Hybrid Training System - Test & Demo
Tests all training methods: Full Fine-Tuning, LoRA, Hybrid Auto-Selection
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain
import json


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(result):
    """Pretty print a result dictionary."""
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()


def test_recommendation_system():
    """Test the intelligent recommendation system."""
    print_section("1. INTELLIGENT TRAINING METHOD RECOMMENDATION")
    
    brain = SuperAIBrain()
    
    # Scenario 1: User with RTX 3060 (12GB)
    print("📊 Scenario 1: RTX 3060 12GB, 100 samples, local training")
    print("-" * 60)
    rec = brain.recommend_training_method(
        model_size_gb=7.0,
        dataset_size=100,
        gpu_memory_gb=12.0,
        provider="local"
    )
    print_result(rec)
    
    # Scenario 2: User with RTX 2060 (6GB)
    print("📊 Scenario 2: RTX 2060 6GB, 50 samples, local training")
    print("-" * 60)
    rec = brain.recommend_training_method(
        model_size_gb=7.0,
        dataset_size=50,
        gpu_memory_gb=6.0,
        provider="local"
    )
    print_result(rec)
    
    # Scenario 3: No GPU, wants to use OpenAI
    print("📊 Scenario 3: No GPU, OpenAI, $20 budget")
    print("-" * 60)
    rec = brain.recommend_training_method(
        model_size_gb=0,
        dataset_size=100,
        gpu_memory_gb=0,
        budget_usd=20.0,
        provider="openai"
    )
    print_result(rec)
    
    # Scenario 4: Powerful GPU (A100 40GB)
    print("📊 Scenario 4: A100 40GB, 500 samples, unlimited time")
    print("-" * 60)
    rec = brain.recommend_training_method(
        model_size_gb=7.0,
        dataset_size=500,
        gpu_memory_gb=40.0,
        provider="local"
    )
    print_result(rec)


def test_comparison_table():
    """Test comprehensive method comparison."""
    print_section("2. COMPREHENSIVE TRAINING METHODS COMPARISON")
    
    brain = SuperAIBrain()
    comparison = brain.compare_all_training_methods(
        model_name="meta-llama/Llama-2-7b-hf",
        dataset_size=100
    )
    
    print("📋 All Training Methods Overview:")
    print("-" * 60)
    print_result(comparison)
    
    # Print as a formatted table
    print("\n📊 Quick Comparison Table:")
    print("-" * 100)
    print(f"{'Method':<25} {'Time':<15} {'Cost':<20} {'GPU Required':<30}")
    print("-" * 100)
    
    for method_name, details in comparison.get("methods", {}).items():
        print(f"{method_name:<25} "
              f"{details['estimated_time']:<15} "
              f"{details['estimated_cost']:<20} "
              f"{details['gpu_required']:<30}")
    print("-" * 100)


def test_lora_vs_full_finetuning():
    """Test LoRA vs Full Fine-Tuning comparison."""
    print_section("3. LoRA vs FULL FINE-TUNING COMPARISON")
    
    brain = SuperAIBrain()
    comparison = brain.compare_training_methods(
        model_name="meta-llama/Llama-2-7b-hf"
    )
    
    print("⚖️  Detailed Technical Comparison:")
    print("-" * 60)
    print_result(comparison)


def test_brain_status():
    """Test brain status with all training modules."""
    print_section("4. KURDO-AI BRAIN STATUS")
    
    brain = SuperAIBrain()
    status = brain.get_status()
    
    print("🧠 Brain System Status:")
    print("-" * 60)
    
    # Training capabilities
    print("\n🎓 Training Capabilities:")
    print(f"  - Fine-Tuning Available: {status.get('fine_tuning', False)}")
    print(f"  - LoRA Available: {status.get('lora', False)}")
    print(f"  - Hybrid Training Available: {status.get('hybrid_training', 'Unknown')}")
    
    # External connectors
    if status.get('external_connectors'):
        print(f"\n🌐 External Connectors: {status['external_connectors']}")
    
    # Memory
    if status.get('memory_stats'):
        mem = status['memory_stats']
        print(f"\n💾 Memory: {mem.get('total_memories', 0)} items")
    
    # Councils
    print(f"\n🏛️  Active Councils: {len(status.get('councils', []))}")
    
    print("\n📊 Full Status:")
    print("-" * 60)
    print_result(status)


def test_list_adapters():
    """Test listing trained LoRA adapters."""
    print_section("5. TRAINED LoRA ADAPTERS")
    
    brain = SuperAIBrain()
    adapters = brain.list_lora_adapters()
    
    print("🎯 Available LoRA Adapters:")
    print("-" * 60)
    print_result(adapters)


def test_list_fine_tuned_models():
    """Test listing fine-tuned models."""
    print_section("6. FINE-TUNED MODELS")
    
    brain = SuperAIBrain()
    models = brain.list_fine_tuned_models()
    
    print("🎯 Fine-Tuned Models History:")
    print("-" * 60)
    print_result(models)


def interactive_menu():
    """Interactive test menu."""
    print_section("KURDO-AI HYBRID TRAINING SYSTEM - INTERACTIVE DEMO")
    
    menu = """
    Choose a test to run:
    
    1. 🤖 Intelligent Training Recommendation (4 scenarios)
    2. 📊 Comprehensive Methods Comparison Table
    3. ⚖️  LoRA vs Full Fine-Tuning (Technical Details)
    4. 🧠 KURDO-AI Brain Status
    5. 🎯 List Trained LoRA Adapters
    6. 📋 List Fine-Tuned Models
    7. 🚀 Run All Tests
    8. ❌ Exit
    
    Enter choice (1-8): """
    
    tests = {
        '1': test_recommendation_system,
        '2': test_comparison_table,
        '3': test_lora_vs_full_finetuning,
        '4': test_brain_status,
        '5': test_list_adapters,
        '6': test_list_fine_tuned_models,
    }
    
    while True:
        try:
            choice = input(menu).strip()
            
            if choice == '8':
                print("\n👋 Goodbye!")
                break
            elif choice == '7':
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
            else:
                print("\n❌ Invalid choice. Please enter 1-8.\n")
                
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
║          🤖 KURDO-AI HYBRID TRAINING SYSTEM - TEST SUITE 🤖                 ║
║                                                                              ║
║  Tests intelligent training method selection:                               ║
║    • Full Fine-Tuning (OpenAI, HuggingFace, Anthropic)                      ║
║    • LoRA (8-bit, 4-bit) - Parameter-Efficient Fine-Tuning                  ║
║    • Hybrid Auto-Selection (recommends best method)                         ║
║                                                                              ║
║  Features:                                                                   ║
║    ✅ Auto-detects GPU memory and capabilities                              ║
║    ✅ Recommends optimal training method                                    ║
║    ✅ Compares costs, time, quality trade-offs                              ║
║    ✅ Supports local (GPU) and cloud (OpenAI/Anthropic)                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--recommend':
            test_recommendation_system()
        elif arg == '--compare':
            test_comparison_table()
        elif arg == '--lora-vs-ft':
            test_lora_vs_full_finetuning()
        elif arg == '--status':
            test_brain_status()
        elif arg == '--adapters':
            test_list_adapters()
        elif arg == '--models':
            test_list_fine_tuned_models()
        elif arg == '--all':
            print("\n🚀 Running all tests...\n")
            test_recommendation_system()
            test_comparison_table()
            test_lora_vs_full_finetuning()
            test_brain_status()
            test_list_adapters()
            test_list_fine_tuned_models()
            print("\n✅ All tests complete!")
        else:
            print(f"Unknown argument: {arg}")
            print("\nAvailable arguments:")
            print("  --recommend    : Test recommendation system")
            print("  --compare      : Show comparison table")
            print("  --lora-vs-ft   : Compare LoRA vs Full FT")
            print("  --status       : Show brain status")
            print("  --adapters     : List LoRA adapters")
            print("  --models       : List fine-tuned models")
            print("  --all          : Run all tests")
            print("  (no args)      : Interactive menu")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
