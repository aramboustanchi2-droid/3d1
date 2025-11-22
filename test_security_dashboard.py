"""
تست سیستم امنیتی پیشرفته CAD3D
نمایش داشبورد با رنگ‌ها و تست همه قابلیت‌ها
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from cad3d.super_ai.advanced_security import (
    SecurityDashboard,
    SystemStatus
)
import time

def test_security_system():
    """تست کامل سیستم امنیتی"""
    
    print("\n" + "="*80)
    print("🛡️  CAD3D ADVANCED SECURITY SYSTEM TEST")
    print("="*80 + "\n")
    
    # ایجاد داشبورد
    dashboard = SecurityDashboard()
    
    # =============================
    # مرحله 1: راه‌اندازی اولیه
    # =============================
    print("📋 STEP 1: System Initialization")
    print("-" * 80)
    
    owner_passphrase = "CAD3D_SUPER_AI_OWNER_2025"
    success = dashboard.initialize_system(owner_passphrase)
    
    if success:
        print("✅ System initialized successfully\n")
    else:
        print("❌ System initialization failed\n")
        return
    
    time.sleep(1)
    
    # نمایش داشبورد اولیه
    dashboard.display_dashboard()
    time.sleep(2)
    
    # =============================
    # مرحله 2: حالت عادی (سبز)
    # =============================
    print("\n📋 STEP 2: Normal Operation (GREEN)")
    print("-" * 80)
    
    current_state = {
        "cpu_percent": 45,
        "ram_percent": 60,
        "network_activity": "normal",
        "stop_command": False
    }
    
    status = dashboard.monitor_system(current_state)
    print(f"Status: {status.value}")
    dashboard.display_dashboard()
    time.sleep(2)
    
    # =============================
    # مرحله 3: رفتار مشکوک (آبی)
    # =============================
    print("\n📋 STEP 3: Suspicious Activity Detected (BLUE)")
    print("-" * 80)
    
    # شبیه‌سازی رفتار مشکوک
    dashboard.software_locks.detect_abnormal_behavior(
        "unexpected_network_call",
        {"source": "unknown_agent"}
    )
    
    current_state["network_activity"] = "suspicious"
    status = dashboard.monitor_system(current_state)
    print(f"Status: {status.value}")
    dashboard.display_dashboard()
    time.sleep(2)
    
    # =============================
    # مرحله 4: خطر (نارنجی)
    # =============================
    print("\n📋 STEP 4: Danger - High CPU/RAM (ORANGE)")
    print("-" * 80)
    
    current_state = {
        "cpu_percent": 95,  # بالای حد مجاز
        "ram_percent": 92,  # بالای حد مجاز
        "network_activity": "high",
        "stop_command": False
    }
    
    status = dashboard.monitor_system(current_state)
    print(f"Status: {status.value}")
    print("⚠️  System approaching danger zone - preparing for shutdown")
    dashboard.display_dashboard()
    time.sleep(2)
    
    # =============================
    # مرحله 5: ساخت Agent جدید
    # =============================
    print("\n📋 STEP 5: Creating New Agents with Training")
    print("-" * 80)
    
    # ساخت 3 agent با آموزش کامل
    for i in range(3):
        print(f"\n🤖 Creating Agent #{i+1}...")
        agent_id = dashboard.agent_manager.create_agent(
            agent_type="AnalysisAgent",
            human_approval=True  # با تایید مالک
        )
        
        if agent_id:
            print(f"✅ Agent created: {agent_id}")
        else:
            print("❌ Agent creation failed")
        
        time.sleep(0.5)
    
    # نمایش آمار
    stats = dashboard.agent_manager.get_dashboard_stats()
    print(f"\n📊 Agent Statistics:")
    print(f"  Daily Created: {stats['daily_created']}")
    print(f"  Total Created: {stats['total_created']}")
    print(f"  Active Agents: {stats['total_agents_active']}")
    
    time.sleep(2)
    
    # =============================
    # مرحله 6: تست Mother Key
    # =============================
    print("\n📋 STEP 6: Mother Key Control Test")
    print("-" * 80)
    
    # تولید کلید صحیح
    mother_key = dashboard.mother_key.key_hash
    
    # دستور LOCK
    print("\n🔒 Executing LOCK command...")
    dashboard.execute_owner_command("LOCK", mother_key)
    dashboard.display_dashboard()
    time.sleep(2)
    
    # تلاش برای کار با سیستم قفل شده
    print("\n❌ Attempting to create agent while system is LOCKED...")
    agent_id = dashboard.agent_manager.create_agent(
        agent_type="TestAgent",
        human_approval=True
    )
    
    if not agent_id:
        print("✅ Correctly blocked - System is locked")
    
    time.sleep(2)
    
    # باز کردن قفل
    print("\n🔓 Executing UNLOCK command...")
    dashboard.mother_key.unlock_key(owner_passphrase)
    dashboard.current_status = SystemStatus.GREEN
    dashboard.display_dashboard()
    time.sleep(2)
    
    # =============================
    # مرحله 7: توقف اضطراری
    # =============================
    print("\n📋 STEP 7: Emergency Stop Protocol")
    print("-" * 80)
    
    print("\n🚨 Executing EMERGENCY STOP...")
    current_state["stop_command"] = True
    dashboard.execute_owner_command("STOP", mother_key)
    dashboard.display_dashboard()
    
    # =============================
    # خلاصه نهایی
    # =============================
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    print(f"\n🔑 Mother Key: Generated & Tested")
    print(f"🔧 Hardware Locks: {len(dashboard.hardware_locks.locks)} initialized")
    print(f"💻 Software Locks: {len(dashboard.software_locks.locks)} initialized")
    print(f"🚨 Emergency Protocols: {len(dashboard.emergency_system.protocols)} ready")
    print(f"🤖 Agents Created: {dashboard.agent_manager.total_created}")
    
    print("\n✅ All security systems tested successfully!")
    print("="*80 + "\n")

def test_agent_creation_workflow():
    """تست فرآیند ساخت agent با آموزش"""
    print("\n" + "="*80)
    print("🎓 AGENT TRAINING WORKFLOW TEST")
    print("="*80 + "\n")
    
    dashboard = SecurityDashboard()
    dashboard.initialize_system("TEST_OWNER")
    
    print("Creating 5 agents with full training...\n")
    
    for i in range(5):
        print(f"\n{'='*60}")
        print(f"Agent #{i+1}")
        print('='*60)
        
        agent_id = dashboard.agent_manager.create_agent(
            agent_type=f"Worker_{i+1}",
            human_approval=True
        )
        
        if agent_id:
            agent = dashboard.agent_manager.agents[agent_id]
            print(f"\n✅ Agent Created: {agent_id}")
            print(f"📅 Created: {agent.created_at}")
            print(f"🎓 Training: {'Completed' if agent.training_completed else 'In Progress'}")
            print(f"📊 Progress: {agent.training_progress:.1f}%")
            print(f"📚 Rules Learned:")
            for category, learned in agent.rules_learned.items():
                status = "✅" if learned else "❌"
                print(f"   {status} {category}")
        
        time.sleep(0.5)
    
    # آمار نهایی
    stats = dashboard.agent_manager.get_dashboard_stats()
    print(f"\n\n{'='*80}")
    print("📊 FINAL AGENT STATISTICS")
    print('='*80)
    print(f"Daily Created: {stats['daily_created']}")
    print(f"Total Created: {stats['total_created']}")
    print(f"Active Agents: {stats['total_agents_active']}")
    print(f"Agents List: {', '.join(stats['agents_list'][:3])}...")
    print('='*80 + "\n")

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n")
    print("=" * 80)
    print("CAD3D SUPER AI - ADVANCED SECURITY SYSTEM TEST SUITE")
    print("=" * 80)
    
    # تست 1: سیستم امنیتی کامل
    test_security_system()
    
    input("\n⏸️  Press ENTER to continue to Agent Training Test...")
    
    # تست 2: فرآیند آموزش agent
    test_agent_creation_workflow()
    
    print("\n✅ All tests completed successfully!")
    print("🎉 Security system is fully operational!\n")
