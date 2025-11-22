#!/usr/bin/env python3
"""
KURDO-AI Prompt Engineering - Test & Demo
Tests training-free methods: Templates, Few-shot, Chain-of-Thought, Caching
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


def test_prompt_templates():
    """Test built-in prompt templates."""
    print_section("1. PROMPT TEMPLATES")
    
    brain = SuperAIBrain()
    
    # List all templates
    print("📋 Available Templates:")
    templates = brain.list_prompt_templates()
    for template in templates:
        print(f"  • {template}")
    
    print("\n" + "-" * 60 + "\n")
    
    # Test architectural calculation template
    print("🏗️  Example 1: Architectural Calculation")
    print("-" * 60)
    prompt = brain.use_prompt_template(
        "arch_calculation",
        task="محاسبه مساحت اتاق",
        given_values="طول: 6 متر، عرض: 4 متر",
        required_output="مساحت به متر مربع"
    )
    print(prompt)
    
    # Test code generation template
    print("\n💻 Example 2: Code Generation")
    print("-" * 60)
    prompt = brain.use_prompt_template(
        "code_generation",
        language="Python",
        task="Calculate room area",
        requirements="- Take length and width as input\n- Return area in square meters\n- Add input validation"
    )
    print(prompt)
    
    # Test design review template
    print("\n🔍 Example 3: Design Review")
    print("-" * 60)
    prompt = brain.use_prompt_template(
        "design_review",
        project_name="Residential Tower - Tehran",
        design_element="Foundation design for 10-story building",
        applicable_standards="مبحث 19، استاندارد 2800"
    )
    print(prompt)


def test_few_shot_learning():
    """Test few-shot learning."""
    print_section("2. FEW-SHOT LEARNING")
    
    brain = SuperAIBrain()
    
    # Architectural examples
    examples = [
        {
            "input": "محاسبه مساحت اتاق 5×4 متر",
            "output": "مساحت = طول × عرض = 5 × 4 = 20 متر مربع"
        },
        {
            "input": "مساحت اتاق 6×3.5 متر؟",
            "output": "مساحت = 6 × 3.5 = 21 متر مربع"
        },
        {
            "input": "Calculate area of 8m × 5m room",
            "output": "Area = length × width = 8 × 5 = 40 square meters"
        }
    ]
    
    print("📚 Training Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\n  Example {i}:")
        print(f"    Input: {ex['input']}")
        print(f"    Output: {ex['output']}")
    
    print("\n" + "-" * 60 + "\n")
    
    # New query
    new_query = "محاسبه مساحت اتاق 7.5×6 متر"
    
    print(f"🎯 New Query: {new_query}")
    print("\n" + "-" * 60 + "\n")
    
    prompt = brain.create_few_shot_prompt(
        task_description="Calculate room area in square meters. Show formula and result.",
        examples=examples,
        current_input=new_query,
        max_examples=3
    )
    
    print("📝 Generated Few-Shot Prompt:")
    print(prompt)


def test_chain_of_thought():
    """Test chain-of-thought reasoning."""
    print_section("3. CHAIN-OF-THOUGHT REASONING")
    
    brain = SuperAIBrain()
    
    # Complex problem
    problem = """
    یک ساختمان 5 طبقه با ابعاد هر طبقه 12×15 متر می‌خواهیم بسازیم.
    ارتفاع هر طبقه 3 متر است.
    چند آجر و چند تن سیمان برای ساخت دیوارهای خارجی نیاز داریم؟
    (ضخامت دیوار خارجی 30 سانتی‌متر)
    """
    
    print("🧩 Complex Problem:")
    print(problem)
    print("\n" + "-" * 60 + "\n")
    
    prompt = brain.create_chain_of_thought_prompt(
        problem=problem.strip(),
        domain="architectural engineering"
    )
    
    print("🧠 Chain-of-Thought Prompt:")
    print(prompt)


def test_cached_system_prompt():
    """Test cached system prompt (Anthropic style)."""
    print_section("4. CACHED SYSTEM PROMPT")
    
    brain = SuperAIBrain()
    
    # Training examples
    training_examples = [
        {
            "input": "محاسبه مساحت اتاق 5×4 متر",
            "output": "مساحت = 5 × 4 = 20 متر مربع"
        },
        {
            "input": "چند آجر برای دیوار 10 متری نیاز است؟",
            "output": "مساحت = 10 × 3 = 30 m²\nآجر = 30 × 60 = 1,800 عدد"
        },
        {
            "input": "حداقل ارتفاع سقف؟",
            "output": "طبق مبحث 19: حداقل 2.4 متر برای اتاق‌های اصلی"
        },
        {
            "input": "Calculate volume of 6×4×2.8m room",
            "output": "Volume = 6 × 4 × 2.8 = 67.2 cubic meters"
        },
        {
            "input": "Foundation depth for 3-story building?",
            "output": "Minimum: 1.5-2 meters below ground level, depending on soil conditions"
        }
    ]
    
    print("📚 Creating Cached Prompt with Examples:")
    print(f"  Total examples: {len(training_examples)}")
    print("\n" + "-" * 60 + "\n")
    
    cached = brain.create_cached_system_prompt(
        system_role="KURDO-AI - Expert Architectural Assistant",
        training_examples=training_examples,
        max_examples=5
    )
    
    print("✅ Cached Prompt Created:")
    print(f"  Cache ID: {cached.get('cache_id', 'N/A')}")
    print(f"  Examples cached: {cached.get('num_examples', 0)}")
    print(f"  Estimated tokens: {cached.get('estimated_tokens', 0)}")
    print(f"\n  Usage: {cached.get('usage', 'N/A')}")
    
    print("\n" + "-" * 60 + "\n")
    print("📄 Cached Content Preview (first 500 chars):")
    content = cached.get('cached_content', '')
    print(content[:500] + "..." if len(content) > 500 else content)


def test_prompt_statistics():
    """Test prompt statistics."""
    print_section("5. PROMPT ENGINEERING STATISTICS")
    
    brain = SuperAIBrain()
    
    # Use some templates first to generate stats
    brain.use_prompt_template("arch_calculation", task="test", given_values="test", required_output="test")
    brain.use_prompt_template("code_generation", language="Python", task="test", requirements="test")
    
    stats = brain.get_prompt_statistics()
    
    print("📊 Usage Statistics:")
    print_result(stats)


def test_comparison():
    """Test comparison with training methods."""
    print_section("6. PROMPT ENGINEERING vs TRAINING METHODS")
    
    brain = SuperAIBrain()
    
    comparison = brain.compare_prompt_vs_training()
    
    print("⚖️  Detailed Comparison:")
    print_result(comparison)
    
    # Print summary table
    print("\n📊 Quick Comparison:")
    print("-" * 100)
    print(f"{'Method':<25} {'Setup Time':<15} {'Cost':<15} {'GPU Required':<15} {'Best For':<30}")
    print("-" * 100)
    
    methods_data = [
        ("Prompt Engineering", "Instant", "$0", "No", "Prototyping, no data"),
        ("LoRA", "1-3 hours", "$0", "Yes (6GB+)", "Multiple tasks, limited GPU"),
        ("Fine-Tuning", "2-10 hours", "$10-50", "Yes (40GB+)", "Production, best quality")
    ]
    
    for method, time, cost, gpu, best_for in methods_data:
        print(f"{method:<25} {time:<15} {cost:<15} {gpu:<15} {best_for:<30}")
    
    print("-" * 100)


def test_all_three_methods():
    """Test and compare all three training approaches."""
    print_section("7. ALL THREE METHODS: COMPARISON DEMO")
    
    brain = SuperAIBrain()
    
    # Sample architectural data
    sample_data = [
        {"input": "محاسبه مساحت 5×4", "output": "20 متر مربع"},
        {"input": "حجم 6×4×3", "output": "72 متر مکعب"},
        {"input": "آجر برای 10 متر", "output": "1800 عدد"}
    ]
    
    print("🎯 Task: Train/Configure KURDO-AI for architectural calculations")
    print(f"📊 Sample Data: {len(sample_data)} examples")
    print("\n" + "-" * 60 + "\n")
    
    # Method 1: Prompt Engineering (instant)
    print("1️⃣  PROMPT ENGINEERING (Instant, No Training)")
    print("   ✅ Setup: Create few-shot prompt")
    print("   ⏱️  Time: 0 seconds")
    print("   💰 Cost: $0")
    print("   📝 Usage: Include examples in every API call")
    
    few_shot = brain.create_few_shot_prompt(
        task_description="Calculate architectural values",
        examples=sample_data,
        current_input="محاسبه مساحت 7×5",
        max_examples=3
    )
    print(f"\n   Sample prompt length: {len(few_shot)} characters\n")
    
    # Method 2: LoRA (fast training)
    print("2️⃣  LoRA (Fast Training)")
    print("   ✅ Setup: Train adapter on GPU")
    print("   ⏱️  Time: 1-2 hours (RTX 3060)")
    print("   💰 Cost: $0 (local)")
    print("   📝 Usage: Load adapter, then inference")
    print("   💾 Adapter size: ~50MB\n")
    
    # Method 3: Fine-Tuning (best quality)
    print("3️⃣  FULL FINE-TUNING (Best Quality)")
    print("   ✅ Setup: Full model training")
    print("   ⏱️  Time: 4-8 hours (A100)")
    print("   💰 Cost: $0 (local) or $20-50 (cloud)")
    print("   📝 Usage: Use fine-tuned model directly")
    print("   💾 Model size: ~14GB\n")
    
    print("-" * 60)
    print("\n📋 RECOMMENDATION:")
    print("  • Start with Prompt Engineering (instant)")
    print("  • Collect real usage data")
    print("  • Train LoRA if you have GPU (50-100 examples)")
    print("  • Use Fine-Tuning for production (500+ examples)")


def interactive_menu():
    """Interactive test menu."""
    print_section("KURDO-AI PROMPT ENGINEERING - INTERACTIVE DEMO")
    
    menu = """
    Choose a test to run:
    
    1. 📋 Prompt Templates (Built-in)
    2. 📚 Few-Shot Learning (No Training)
    3. 🧠 Chain-of-Thought Reasoning
    4. 💾 Cached System Prompt (Anthropic)
    5. 📊 Usage Statistics
    6. ⚖️  Comparison: Prompt vs Training
    7. 🎯 All Three Methods Demo
    8. 🚀 Run All Tests
    9. ❌ Exit
    
    Enter choice (1-9): """
    
    tests = {
        '1': test_prompt_templates,
        '2': test_few_shot_learning,
        '3': test_chain_of_thought,
        '4': test_cached_system_prompt,
        '5': test_prompt_statistics,
        '6': test_comparison,
        '7': test_all_three_methods,
    }
    
    while True:
        try:
            choice = input(menu).strip()
            
            if choice == '9':
                print("\n👋 Goodbye!")
                break
            elif choice == '8':
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
                print("\n❌ Invalid choice. Please enter 1-9.\n")
                
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
║          🎯 KURDO-AI PROMPT ENGINEERING - TEST SUITE 🎯                     ║
║                                                                              ║
║  Training-Free Methods:                                                      ║
║    • Prompt Templates - Reusable patterns                                   ║
║    • Few-Shot Learning - Learn from examples                                ║
║    • Chain-of-Thought - Complex reasoning                                   ║
║    • Cached Prompts - Cost-effective (Anthropic)                            ║
║                                                                              ║
║  Advantages:                                                                 ║
║    ✅ Instant setup (no training time)                                      ║
║    ✅ Zero cost (except inference)                                          ║
║    ✅ No GPU required                                                       ║
║    ✅ Extremely flexible                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--templates':
            test_prompt_templates()
        elif arg == '--few-shot':
            test_few_shot_learning()
        elif arg == '--cot':
            test_chain_of_thought()
        elif arg == '--cached':
            test_cached_system_prompt()
        elif arg == '--stats':
            test_prompt_statistics()
        elif arg == '--compare':
            test_comparison()
        elif arg == '--three-methods':
            test_all_three_methods()
        elif arg == '--all':
            print("\n🚀 Running all tests...\n")
            test_prompt_templates()
            test_few_shot_learning()
            test_chain_of_thought()
            test_cached_system_prompt()
            test_prompt_statistics()
            test_comparison()
            test_all_three_methods()
            print("\n✅ All tests complete!")
        else:
            print(f"Unknown argument: {arg}")
            print("\nAvailable arguments:")
            print("  --templates      : Test prompt templates")
            print("  --few-shot       : Test few-shot learning")
            print("  --cot            : Test chain-of-thought")
            print("  --cached         : Test cached prompts")
            print("  --stats          : Show statistics")
            print("  --compare        : Compare methods")
            print("  --three-methods  : Demo all three approaches")
            print("  --all            : Run all tests")
            print("  (no args)        : Interactive menu")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
