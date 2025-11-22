"""
Quick Demo - University Knowledge System
نمایش سریع سیستم یادگیری از دانشگاه‌ها

NOTE: This is a demonstration script. For production continuous operation use
`run_university_system.py` which launches the learning loop and the FastAPI dashboard.
"""

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*80)
print("  🎓 UNIVERSITY KNOWLEDGE SYSTEM - QUICK DEMO")
print("  سیستم یادگیری از دانشگاه‌های برتر دنیا")
print("="*80 + "\n")

# Import configurations
from cad3d.super_ai.university_config import UNIVERSITIES, AGENT_CONFIG

print("✅ Configuration loaded\n")

# Show universities
print("📚 AVAILABLE UNIVERSITIES:\n")
for i, (key, info) in enumerate(UNIVERSITIES.items(), 1):
    print(f"{i:2d}. {info['name']:<45} ({info['country']})")
    print(f"    Resources: {len(info['resources'])} sources")
    print(f"    Focus: {', '.join(info['focus_areas'][:3])}")
    
    # Show first resource
    first_resource = list(info['resources'].values())[0]
    print(f"    Example: {first_resource['url']}")
    print()

print("\n" + "="*80)
print("  📊 SYSTEM CAPABILITIES")
print("="*80 + "\n")

capabilities = [
    ("🌐 Web Scraping", "Extract content from university websites without API"),
    ("🤖 Smart Agents", "10 specialized agents, one per university"),
    ("🧠 RAG Integration", "Connect to existing RAG System for knowledge storage"),
    ("⏰ Auto-Update", "Scheduled daily/weekly updates in background"),
    ("💾 Smart Cache", "Reduce requests, respect rate limits"),
    ("📈 Statistics", "Track pages scraped, documents collected"),
    ("🔍 Search", "Query university knowledge semantically"),
    ("📝 Logging", "Complete logs for all operations")
]

for name, desc in capabilities:
    print(f"  {name:<20} → {desc}")

print("\n" + "="*80)
print("  🚀 QUICK START")
print("="*80 + "\n")

print("1. Initialize system:")
print("   from cad3d.super_ai.university_agents import UniversityAgentManager")
print("   from cad3d.super_ai.rag_system import RAGSystem")
print("   ")
print("   rag = RAGSystem()")
print("   manager = UniversityAgentManager(UNIVERSITIES, AGENT_CONFIG, rag)")
print()

print("2. Learn from a university:")
print("   result = manager.learn_from_specific(['MIT'])")
print()

print("3. Learn from all:")
print("   result = manager.learn_from_all()")
print()

print("4. Set up automatic updates:")
print("   from cad3d.super_ai.university_scheduler import UniversityLearningScheduler")
print("   scheduler = UniversityLearningScheduler(manager)")
print("   scheduler.setup_default_schedules()  # Top 5: daily, Next 5: weekly")
print("   scheduler.start()  # Runs in background")
print()

print("5. Run full demo:")
print("   python test_university_integration.py")
print()

print("\n" + "="*80)
print("  📖 DOCUMENTATION")
print("="*80 + "\n")

docs = [
    ("UNIVERSITY_KNOWLEDGE_SYSTEM.md", "Complete system documentation"),
    ("UNIVERSITY_SYSTEM_SUMMARY.md", "Summary and quick reference"),
    ("university_config.py", "Configuration and universities list"),
    ("university_scraper.py", "Web scraping implementation"),
    ("university_agents.py", "Agent system implementation"),
    ("university_scheduler.py", "Automatic scheduling system"),
    ("test_university_integration.py", "Integration test and demo")
]

for file, desc in docs:
    print(f"  {file:<35} - {desc}")

print("\n" + "="*80)
print("  ✅ SYSTEM READY!")
print("="*80 + "\n")

print("📌 Key Features:")
print("  ✓ 10 top universities covered")
print("  ✓ 30+ free resources (no API needed)")
print("  ✓ Automatic scraping and learning")
print("  ✓ RAG integration for semantic search")
print("  ✓ Scheduled updates (daily/weekly)")
print("  ✓ Complete Persian & English docs")
print()

print("🎯 Next Steps:")
print("  1. Install: pip install requests beautifulsoup4 schedule")
print("  2. Run: python test_university_integration.py")
print("  3. Enjoy learning from world's best universities!")
print()

print("="*80 + "\n")
