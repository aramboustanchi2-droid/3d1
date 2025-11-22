"""
Complete University System Demo with Security & Database
دمو کامل سیستم با امنیت، نظارت و پایگاه داده
"""

import io
import sys
import logging

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

from cad3d.super_ai.university_config import UNIVERSITIES, AGENT_CONFIG
from cad3d.super_ai.university_agents import UniversityAgentManager
from cad3d.super_ai.rag_system import RAGSystem
from cad3d.super_ai.agent_security import AgentSecuritySystem, SpecializationManager
from cad3d.super_ai.knowledge_database import UniversityKnowledgeDB

print("\n" + "="*80)
print("  🎓 COMPLETE UNIVERSITY KNOWLEDGE SYSTEM DEMO")
print("  سیستم کامل با امنیت، نظارت و پایگاه داده")
print("="*80 + "\n")

# ==================
# STEP 1: Initialize All Systems
# ==================
print("Step 1: Initializing Systems...")
print("-" * 80)

# Database
db = UniversityKnowledgeDB()
print("✓ Database initialized")

# Security
security = AgentSecuritySystem()
print("✓ Security system initialized")

# Specialization Manager
spec_manager = SpecializationManager()
print("✓ Specialization manager initialized")

# RAG System
rag = RAGSystem()
print("✓ RAG system initialized")

# Agent Manager (with security)
agent_manager = UniversityAgentManager(
    UNIVERSITIES,
    AGENT_CONFIG,
    rag,
    security
)
print("✓ Agent manager initialized with security\n")

# ==================
# STEP 2: Register Universities in DB
# ==================
print("Step 2: Registering Universities in Database...")
print("-" * 80)

for key, info in UNIVERSITIES.items():
    db.add_university(key, info)
    print(f"✓ Registered: {info['name']}")

print()

# ==================
# STEP 3: Show Universities with Expanded Focus Areas
# ==================
print("Step 3: Universities Overview")
print("-" * 80 + "\n")

for i, (key, info) in enumerate(UNIVERSITIES.items(), 1):
    print(f"{i:2d}. {info['name']}")
    print(f"    Country: {info['country']}")
    print(f"    Focus Areas: {', '.join(info['focus_areas'][:5])}")
    if len(info['focus_areas']) > 5:
        print(f"                 {', '.join(info['focus_areas'][5:])}")
    print()

# ==================
# STEP 4: Security System Demo
# ==================
print("\n" + "="*80)
print("  🔒 SECURITY SYSTEM DEMONSTRATION")
print("="*80 + "\n")

# Test cases
test_docs = [
    {
        "content": "This is a research paper on artificial intelligence and machine learning.",
        "metadata": {
            "university": "MIT",
            "resource": "research",
            "url": "https://dspace.mit.edu/handle/12345"
        }
    },
    {
        "content": "This document contains illegal hacking techniques.",  # Should be blocked
        "metadata": {
            "university": "MIT",
            "resource": "unknown",
            "url": "https://suspicious-site.com/hack"
        }
    },
    {
        "content": "A" * 1500000,  # Too long - should warn
        "metadata": {
            "university": "Stanford",
            "resource": "course",
            "url": "https://online.stanford.edu/course"
        }
    }
]

print("Testing document validation...\n")

for i, doc in enumerate(test_docs, 1):
    print(f"Test {i}:")
    is_valid, compliance, reason = security.validate_document(doc)
    
    icon = "✓" if is_valid else "✗"
    status = f"{icon} {compliance.value.upper()}"
    
    print(f"  Result: {status}")
    print(f"  Reason: {reason}")
    print()

# Security report
report = security.get_monitoring_report()
print("Security Summary:")
print(f"  Total Checks: {report['statistics']['total_checks']}")
print(f"  Compliant: {report['statistics']['compliant']}")
print(f"  Warnings: {report['statistics']['warnings']}")
print(f"  Violations: {report['statistics']['violations']}")
print(f"  Blocked: {report['statistics']['blocked']}")
print(f"  Compliance Rate: {report['compliance_rate']}")

# ==================
# STEP 5: Specialization Detection
# ==================
print("\n" + "="*80)
print("  📚 SPECIALIZATION DETECTION")
print("="*80 + "\n")

test_contents = [
    "This paper discusses structural analysis of reinforced concrete beams and columns in civil engineering.",
    "Business strategy and marketing management in modern organizations requires data-driven decision making.",
    "Macroeconomic policy and fiscal stimulus during economic recession periods affect GDP growth."
]

for i, content in enumerate(test_contents, 1):
    scores = spec_manager.detect_specialization(content)
    print(f"Content {i}:")
    print(f"  Text: {content[:80]}...")
    print(f"  Detected:")
    for field, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
        if score > 0:
            print(f"    - {field}: {score*100:.1f}%")
    print()

# ==================
# STEP 6: Database Statistics
# ==================
print("\n" + "="*80)
print("  💾 DATABASE STATISTICS")
print("="*80 + "\n")

summary = db.get_all_universities_summary()

print(f"Total Universities in DB: {len(summary)}")
print(f"Total Documents: {sum(u['doc_count'] or 0 for u in summary)}")
print()

print("Universities in Database:")
for uni in summary[:5]:
    print(f"  - {uni['name']}: {uni['doc_count'] or 0} documents")

# ==================
# STEP 7: Agent Status
# ==================
print("\n" + "="*80)
print("  🤖 AGENT STATUS")
print("="*80 + "\n")

statuses = agent_manager.get_all_statuses()

for status in statuses[:3]:
    print(f"Agent: {status['university']}")
    print(f"  Country: {status['country']}")
    print(f"  Documents: {status['state']['total_documents']}")
    print(f"  Pages Scraped: {status['state']['total_pages_scraped']}")
    print(f"  Compliance Score: {status['state']['compliance_score']:.1f}%")
    print(f"  Security Violations: {status['state']['security_violations']}")
    print(f"  Should Update: {status['should_update']}")
    print()

# ==================
# SUMMARY
# ==================
print("\n" + "="*80)
print("  ✅ SYSTEM FEATURES DEMONSTRATED")
print("="*80 + "\n")

features = [
    ("🔒 Security System", "Validates content, blocks violations, monitors compliance"),
    ("📊 Monitoring", "Tracks all operations, logs security events"),
    ("💾 Database", "SQLite database with structured storage"),
    ("📚 Specializations", "Detects engineering, management, economics topics"),
    ("🎓 10 Universities", "Expanded focus areas: engineering, management, economics"),
    ("🤖 Smart Agents", "Each agent has security validation and compliance tracking"),
    ("📈 Statistics", "Complete tracking of documents, pages, violations"),
    ("🌐 Dashboard Ready", "Run: streamlit run dashboard_university.py")
]

for name, desc in features:
    print(f"  {name:<25} → {desc}")

print("\n" + "="*80)
print("  📌 NEXT STEPS")
print("="*80 + "\n")

print("1. Run Dashboard:")
print("   streamlit run dashboard_university.py")
print()

print("2. Start Learning (with security):")
print("   from cad3d.super_ai.university_agents import UniversityAgentManager")
print("   manager.learn_from_specific(['MIT'])  # Security validated automatically")
print()

print("3. View Database:")
print("   sqlite3 university_cache/knowledge.db")
print("   SELECT * FROM universities;")
print()

print("4. Check Security Logs:")
print("   ls university_cache/security_logs/")
print()

print("="*80 + "\n")

print("🎉 All systems operational and ready!")
print("   - Security: Active ✓")
print("   - Monitoring: Active ✓")
print("   - Database: Ready ✓")
print("   - Specializations: Engineering, Management, Economics ✓")
print()
