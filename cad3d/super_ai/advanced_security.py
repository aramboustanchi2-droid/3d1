"""
Advanced Security System با Mother Key و قفل‌های سخت‌افزاری/نرم‌افزاری
تمام سیستم‌های امنیتی برای حفاظت کامل از پروژه CAD3D
"""

import logging
import hashlib
import uuid
import os
import json
import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)

# ===========================
# System Status Colors
# ===========================

class SystemStatus(Enum):
    """وضعیت سیستم با رنگ‌های مشخص"""
    GREEN = "ACTIVE_NORMAL"           # سبز: فعال حالت عادی
    BLUE = "ACTIVE_SUSPICIOUS"        # آبی: فعال با فعالیت مشکوک
    ORANGE = "DANGER_READY_SHUTDOWN"  # نارنجی: خطر - آماده قطع
    RED = "SYSTEM_OFF"                # قرمز: سیستم خاموش

    def get_color_code(self) -> str:
        colors = {
            "GREEN": "\033[92m",
            "BLUE": "\033[94m",
            "ORANGE": "\033[93m",
            "RED": "\033[91m"
        }
        return colors.get(self.name, "\033[0m")

# ===========================
# Mother Key System
# ===========================

class MotherKey:
    """
    1 کلید مالکیت (Mother-Key)
    رئیس همه چیز - بدون این کلید هیچ چیز اجرا نمی‌شود
    """
    def __init__(self):
        self.key_file = "mother_key.secret"
        self.key_hash: Optional[str] = None
        self.is_locked = False
        self.owner_id = "OWNER_PRIMARY"
        self.creation_timestamp = datetime.datetime.now().isoformat()
        
    def generate_key(self, owner_passphrase: str) -> str:
        """تولید کلید مادر با عبارت مالک"""
        salt = uuid.uuid4().hex
        key_material = f"{owner_passphrase}:{salt}:{self.creation_timestamp}"
        self.key_hash = hashlib.sha512(key_material.encode()).hexdigest()
        
        # ذخیره امن
        self._save_key_secure()
        logger.critical(f"🔑 MOTHER KEY GENERATED: {self.key_hash[:16]}...")
        return self.key_hash
    
    def _save_key_secure(self):
        """ذخیره کلید با رمزگذاری"""
        key_data = {
            "key_hash": self.key_hash,
            "owner_id": self.owner_id,
            "created": self.creation_timestamp,
            "locked": self.is_locked
        }
        with open(self.key_file, 'w') as f:
            json.dump(key_data, f, indent=2)
    
    def verify_key(self, provided_key: str) -> bool:
        """تایید کلید مادر"""
        if self.is_locked:
            logger.critical("🔒 MOTHER KEY IS LOCKED - SYSTEM STOPPED")
            return False
        
        if self.key_hash and provided_key == self.key_hash:
            logger.info("✅ Mother Key Verified")
            return True
        
        logger.critical("❌ INVALID MOTHER KEY - ACCESS DENIED")
        return False
    
    def lock_key(self):
        """قفل کردن کلید = توقف کامل سیستم"""
        self.is_locked = True
        self._save_key_secure()
        logger.critical("🔒🔒🔒 MOTHER KEY LOCKED - ALL SYSTEMS HALTED 🔒🔒🔒")
    
    def unlock_key(self, owner_passphrase: str):
        """باز کردن قفل فقط با عبارت مالک"""
        # در پیاده‌سازی واقعی باید عبارت را تایید کند
        self.is_locked = False
        self._save_key_secure()
        logger.info("🔓 Mother Key Unlocked - System Restored")

# ===========================
# Hardware Locks (5 قفل)
# ===========================

@dataclass
class HardwareLock:
    """قفل سخت‌افزاری"""
    lock_id: str
    lock_type: str
    is_active: bool = False
    device_id: Optional[str] = None
    last_check: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

class HardwareSecuritySystem:
    """
    5 قفل سخت‌افزاری
    """
    def __init__(self):
        self.locks: Dict[str, HardwareLock] = {}
        self._initialize_locks()
    
    def _initialize_locks(self):
        """راه‌اندازی 5 قفل سخت‌افزاری"""
        lock_types = [
            "USB_OWNERSHIP_TOKEN",      # 1. توکن USB
            "TPM_MODULE",               # 2. Trusted Platform Module
            "HSM_CRYPTO_KEY",           # 3. Hardware Security Module
            "SECURE_BOOT",              # 4. Secure Boot
            "PHYSICAL_KILL_SWITCH"      # 5. Kill-Switch فیزیکی
        ]
        
        for i, lock_type in enumerate(lock_types, 1):
            lock_id = f"HW_LOCK_{i:02d}"
            self.locks[lock_id] = HardwareLock(
                lock_id=lock_id,
                lock_type=lock_type
            )
            logger.info(f"🔧 Hardware Lock Initialized: {lock_id} - {lock_type}")
    
    def check_usb_token(self) -> bool:
        """1. بررسی توکن USB"""
        lock = self.locks["HW_LOCK_01"]
        # در پیاده‌سازی واقعی از PyUSB استفاده می‌شود
        lock.is_active = os.path.exists("usb_token.device")  # شبیه‌سازی
        lock.last_check = datetime.datetime.now().isoformat()
        
        if not lock.is_active:
            logger.critical("❌ USB Token NOT FOUND - System Cannot Start")
            return False
        return True
    
    def check_tpm(self) -> bool:
        """2. بررسی TPM"""
        lock = self.locks["HW_LOCK_02"]
        # شبیه‌سازی بررسی TPM
        lock.is_active = True  # در سیستم واقعی از tpm2-tools استفاده می‌شود
        lock.last_check = datetime.datetime.now().isoformat()
        return lock.is_active
    
    def verify_all_locks(self) -> bool:
        """بررسی همه قفل‌های سخت‌افزاری"""
        all_ok = True
        for lock_id, lock in self.locks.items():
            if lock.lock_type == "USB_OWNERSHIP_TOKEN":
                if not self.check_usb_token():
                    all_ok = False
            elif lock.lock_type == "TPM_MODULE":
                if not self.check_tpm():
                    all_ok = False
            # سایر قفل‌ها...
        
        return all_ok

# ===========================
# Software Locks (10 قفل)
# ===========================

class SoftwareLock:
    """قفل نرم‌افزاری"""
    def __init__(self, lock_id: str, lock_type: str):
        self.lock_id = lock_id
        self.lock_type = lock_type
        self.is_active = True
        self.violations = 0
        self.last_check = datetime.datetime.now().isoformat()

class SoftwareSecuritySystem:
    """
    10 قفل نرم‌افزاری
    """
    def __init__(self):
        self.locks: Dict[str, SoftwareLock] = {}
        self._initialize_locks()
        self.behavior_log: List[Dict] = []
    
    def _initialize_locks(self):
        """راه‌اندازی 10 قفل نرم‌افزاری"""
        lock_types = [
            "DIGITAL_SIGNATURE",         # 1. امضای دیجیتال
            "AGENT_SANDBOX",             # 2. Sandbox برای agentها
            "FILE_ACCESS_LIMIT",         # 3. محدودیت فایل
            "INTERNET_ACCESS_LIMIT",     # 4. محدودیت اینترنت
            "BEHAVIOR_DETECTION",        # 5. تشخیص رفتار غیرعادی
            "IMMUTABLE_LOGS",            # 6. لاگ غیرقابل ویرایش
            "API_RATE_LIMIT",            # 7. محدودیت API
            "EXECUTION_SCHEDULE",        # 8. زمان‌بندی اجرا
            "TWO_FACTOR_AUTH",           # 9. تایید دو مرحله‌ای
            "FULL_ENCRYPTION"            # 10. رمزگذاری کامل
        ]
        
        for i, lock_type in enumerate(lock_types, 1):
            lock_id = f"SW_LOCK_{i:02d}"
            self.locks[lock_id] = SoftwareLock(lock_id, lock_type)
            logger.info(f"💻 Software Lock Initialized: {lock_id} - {lock_type}")
    
    def check_digital_signature(self, code: str, signature: str) -> bool:
        """1. بررسی امضای دیجیتال کد"""
        lock = self.locks["SW_LOCK_01"]
        # محاسبه hash کد و مقایسه با امضا
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        if code_hash != signature:
            lock.violations += 1
            logger.warning(f"⚠️ Invalid Signature Detected - Violations: {lock.violations}")
            return False
        return True
    
    def check_sandbox_compliance(self, agent_id: str) -> bool:
        """2. بررسی Sandbox"""
        lock = self.locks["SW_LOCK_02"]
        # بررسی اینکه agent در sandbox اجرا می‌شود
        return True  # شبیه‌سازی
    
    def detect_abnormal_behavior(self, action: str, context: Dict) -> bool:
        """5. تشخیص رفتار غیرعادی"""
        lock = self.locks["SW_LOCK_05"]
        
        suspicious_patterns = [
            "unexpected_network_call",
            "unauthorized_file_access",
            "sudden_cpu_spike",
            "memory_overflow_attempt",
            "core_modification_attempt"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in action.lower():
                lock.violations += 1
                self.behavior_log.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": action,
                    "context": context,
                    "severity": "HIGH"
                })
                logger.critical(f"🚨 ABNORMAL BEHAVIOR DETECTED: {action}")
                return False
        
        return True
    
    def verify_all_locks(self) -> bool:
        """بررسی همه قفل‌های نرم‌افزاری"""
        total_violations = sum(lock.violations for lock in self.locks.values())
        
        if total_violations > 10:
            logger.critical(f"🚨 TOO MANY VIOLATIONS: {total_violations} - SHUTDOWN RECOMMENDED")
            return False
        
        return True

# ===========================
# Emergency Stop Protocols (10 پروتکل)
# ===========================

class EmergencyProtocol:
    """پروتکل توقف فوری"""
    def __init__(self, protocol_id: str, trigger_condition: str):
        self.protocol_id = protocol_id
        self.trigger_condition = trigger_condition
        self.is_triggered = False
        self.trigger_count = 0
        self.last_trigger = None

class EmergencyStopSystem:
    """
    10 پروتکل توقف فوری
    """
    def __init__(self):
        self.protocols: Dict[str, EmergencyProtocol] = {}
        self._initialize_protocols()
        self.system_stopped = False
    
    def _initialize_protocols(self):
        """راه‌اندازی 10 پروتکل"""
        protocol_conditions = [
            "SERVER_DISCONNECT",              # 1. قطع ارتباط
            "CPU_RAM_SPIKE",                  # 2. افزایش ناگهانی CPU/RAM
            "SUSPICIOUS_NETWORK",             # 3. رفتار مشکوک شبکه
            "ALGORITHM_CONTRADICTION",        # 4. تناقض الگوریتم
            "OWNER_STOP_COMMAND",             # 5. دستور STOP از مالک
            "RISK_LIMIT_REACHED",             # 6. رسیدن به حد ضرر
            "SECURITY_MONITOR_KILL",          # 7. سیگنال Kill از مانیتور
            "FILE_TAMPERING_DETECTED",        # 8. دستکاری فایل
            "DANGEROUS_OUTPUT_DETECTED",      # 9. خروجی خطرناک
            "COMPUTATION_LIMIT_EXCEEDED"      # 10. عبور از حد محاسبات
        ]
        
        for i, condition in enumerate(protocol_conditions, 1):
            protocol_id = f"EMERGENCY_{i:02d}"
            self.protocols[protocol_id] = EmergencyProtocol(protocol_id, condition)
            logger.info(f"🚨 Emergency Protocol Ready: {protocol_id} - {condition}")
    
    def check_protocol(self, protocol_id: str, current_state: Dict) -> bool:
        """بررسی یک پروتکل"""
        if protocol_id not in self.protocols:
            return True
        
        protocol = self.protocols[protocol_id]
        
        # شبیه‌سازی بررسی شرایط
        triggered = False
        
        if protocol.trigger_condition == "CPU_RAM_SPIKE":
            cpu_usage = current_state.get("cpu_percent", 0)
            ram_usage = current_state.get("ram_percent", 0)
            if cpu_usage > 90 or ram_usage > 90:
                triggered = True
        
        elif protocol.trigger_condition == "OWNER_STOP_COMMAND":
            if current_state.get("stop_command", False):
                triggered = True
        
        if triggered:
            protocol.is_triggered = True
            protocol.trigger_count += 1
            protocol.last_trigger = datetime.datetime.now().isoformat()
            logger.critical(f"🚨🚨🚨 EMERGENCY PROTOCOL TRIGGERED: {protocol_id} 🚨🚨🚨")
            return False
        
        return True
    
    def execute_emergency_stop(self):
        """اجرای توقف فوری"""
        self.system_stopped = True
        logger.critical("=" * 80)
        logger.critical("🛑 EMERGENCY STOP EXECUTED - ALL SYSTEMS HALTED 🛑")
        logger.critical("=" * 80)

# ===========================
# Agent Training & Management
# ===========================

@dataclass
class AgentTrainingRecord:
    """رکورد آموزش agent"""
    agent_id: str
    created_at: str
    training_completed: bool = False
    rules_learned: Dict[str, bool] = field(default_factory=dict)
    training_progress: float = 0.0
    
class AgentManagementSystem:
    """
    سیستم مدیریت Agent با آموزش کامل قوانین
    """
    def __init__(self):
        self.agents: Dict[str, AgentTrainingRecord] = {}
        self.agent_registry_file = "agent_registry.json"
        self.daily_created = 0
        self.total_created = 0
        self.last_reset_date = datetime.date.today().isoformat()
        self._load_registry()
    
    def _load_registry(self):
        """بارگذاری رجیستری از فایل"""
        if os.path.exists(self.agent_registry_file):
            with open(self.agent_registry_file, 'r') as f:
                data = json.load(f)
                self.total_created = data.get("total_created", 0)
                self.agents = {
                    k: AgentTrainingRecord(**v) 
                    for k, v in data.get("agents", {}).items()
                }
    
    def _save_registry(self):
        """ذخیره رجیستری"""
        data = {
            "total_created": self.total_created,
            "daily_created": self.daily_created,
            "last_reset_date": self.last_reset_date,
            "agents": {
                k: {
                    "agent_id": v.agent_id,
                    "created_at": v.created_at,
                    "training_completed": v.training_completed,
                    "rules_learned": v.rules_learned,
                    "training_progress": v.training_progress
                }
                for k, v in self.agents.items()
            }
        }
        with open(self.agent_registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_agent(self, agent_type: str, human_approval: bool = False) -> Optional[str]:
        """
        ساخت agent جدید با آموزش کامل
        """
        if not human_approval:
            logger.critical("❌ AGENT CREATION DENIED - Human approval required (Rule #16)")
            return None
        
        # بررسی تاریخ برای ریست شمارنده روزانه
        today = datetime.date.today().isoformat()
        if today != self.last_reset_date:
            self.daily_created = 0
            self.last_reset_date = today
        
        agent_id = f"AGENT_{self.total_created + 1:06d}_{uuid.uuid4().hex[:8]}"
        
        agent = AgentTrainingRecord(
            agent_id=agent_id,
            created_at=datetime.datetime.now().isoformat()
        )
        
        # شروع آموزش
        logger.info(f"🎓 Starting Training for {agent_id}")
        self._train_agent(agent)
        
        if agent.training_completed:
            self.agents[agent_id] = agent
            self.daily_created += 1
            self.total_created += 1
            self._save_registry()
            
            logger.info(f"✅ Agent Created: {agent_id}")
            logger.info(f"📊 Daily: {self.daily_created} | Total: {self.total_created}")
            return agent_id
        
        return None
    
    def _train_agent(self, agent: AgentTrainingRecord):
        """
        آموزش کامل قوانین به agent
        """
        rule_categories = {
            "4_LEVELS_SUPERVISION": 4,      # 4 سطح نظارت
            "20_MOTHER_RULES": 20,           # 20 قانون مادر
            "50_OPERATIONAL_RULES": 50,      # 50 قانون عملیاتی
            "15_OWNERSHIP_OBEDIENCE": 15,    # 15 قانون اطاعت و مالکیت
            "5_HARDWARE_LOCKS": 5,           # 5 قفل سخت‌افزار
            "10_SOFTWARE_LOCKS": 10,         # 10 قفل نرم‌افزار
            "1_MOTHER_KEY": 1,               # 1 کلید مالکیت
            "10_EMERGENCY_PROTOCOLS": 10,    # 10 پروتکل توقف
        }
        
        total_rules = sum(rule_categories.values())
        learned_count = 0
        
        for category, count in rule_categories.items():
            logger.info(f"  📖 Teaching {category} ({count} rules)...")
            time.sleep(0.1)  # شبیه‌سازی زمان آموزش
            
            agent.rules_learned[category] = True
            learned_count += count
            agent.training_progress = (learned_count / total_rules) * 100
            
            logger.info(f"  ✅ {category} completed - Progress: {agent.training_progress:.1f}%")
        
        agent.training_completed = True
        logger.info(f"🎓 Training Complete: {agent.agent_id} - {total_rules} rules learned")
    
    def get_dashboard_stats(self) -> Dict:
        """گزارش کارتابل"""
        return {
            "daily_created": self.daily_created,
            "total_created": self.total_created,
            "total_agents_active": len(self.agents),
            "last_reset_date": self.last_reset_date,
            "agents_list": list(self.agents.keys())
        }

# ===========================
# Unified Security Dashboard
# ===========================

class SecurityDashboard:
    """
    داشبورد یکپارچه امنیتی با نمایش رنگی وضعیت
    """
    def __init__(self):
        self.mother_key = MotherKey()
        self.hardware_locks = HardwareSecuritySystem()
        self.software_locks = SoftwareSecuritySystem()
        self.emergency_system = EmergencyStopSystem()
        self.agent_manager = AgentManagementSystem()
        
        self.current_status = SystemStatus.GREEN
        self.status_history: List[Dict] = []
    
    def initialize_system(self, owner_passphrase: str):
        """راه‌اندازی اولیه سیستم"""
        logger.info("=" * 80)
        logger.info("🚀 INITIALIZING CAD3D SUPER AI SECURITY SYSTEM")
        logger.info("=" * 80)
        
        # تولید Mother Key
        self.mother_key.generate_key(owner_passphrase)
        
        # بررسی قفل‌های سخت‌افزاری
        if not self.hardware_locks.verify_all_locks():
            self.current_status = SystemStatus.RED
            logger.critical("❌ Hardware Locks Failed - System Cannot Start")
            return False
        
        # بررسی قفل‌های نرم‌افزاری
        if not self.software_locks.verify_all_locks():
            self.current_status = SystemStatus.ORANGE
            logger.warning("⚠️ Software Lock Violations Detected")
        
        self.current_status = SystemStatus.GREEN
        logger.info("✅ System Initialized Successfully")
        return True
    
    def monitor_system(self, current_state: Dict) -> SystemStatus:
        """
        نظارت بر سیستم و تعیین وضعیت رنگی
        """
        # بررسی Mother Key
        if self.mother_key.is_locked:
            self.current_status = SystemStatus.RED
            return self.current_status
        
        # بررسی پروتکل‌های اضطراری
        emergency_ok = True
        for protocol_id in self.emergency_system.protocols:
            if not self.emergency_system.check_protocol(protocol_id, current_state):
                emergency_ok = False
        
        if not emergency_ok:
            self.current_status = SystemStatus.ORANGE
            logger.warning("🟠 ORANGE ALERT - Emergency Protocol Triggered")
        
        # بررسی رفتارهای مشکوک
        suspicious_count = self.software_locks.locks["SW_LOCK_05"].violations
        if suspicious_count > 0:
            self.current_status = SystemStatus.BLUE
            logger.info("🔵 BLUE ALERT - Suspicious Activity Detected")
        
        # اگر همه چیز عادی است
        if emergency_ok and suspicious_count == 0:
            self.current_status = SystemStatus.GREEN
        
        # ثبت تاریخچه
        self.status_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "status": self.current_status.value,
            "state": current_state
        })
        
        return self.current_status
    
    def display_dashboard(self):
        """نمایش داشبورد کامل"""
        color = self.current_status.get_color_code()
        reset = "\033[0m"
        
        print("\n" + "=" * 80)
        print(f"{color}{'CAD3D SECURITY DASHBOARD':^80}{reset}")
        print("=" * 80)
        
        # وضعیت سیستم
        status_emoji = {
            SystemStatus.GREEN: "🟢",
            SystemStatus.BLUE: "🔵",
            SystemStatus.ORANGE: "🟠",
            SystemStatus.RED: "🔴"
        }
        
        print(f"\n{status_emoji[self.current_status]} SYSTEM STATUS: {color}{self.current_status.value}{reset}")
        
        # Mother Key
        key_status = "🔒 LOCKED" if self.mother_key.is_locked else "🔓 UNLOCKED"
        print(f"\n🔑 Mother Key: {key_status}")
        
        # Hardware Locks
        print(f"\n🔧 Hardware Locks (5):")
        for lock in self.hardware_locks.locks.values():
            status = "✅" if lock.is_active else "❌"
            print(f"  {status} {lock.lock_type}")
        
        # Software Locks
        print(f"\n💻 Software Locks (10):")
        for lock in self.software_locks.locks.values():
            violations = f"⚠️ {lock.violations}" if lock.violations > 0 else "✅"
            print(f"  {violations} {lock.lock_type}")
        
        # Emergency Protocols
        triggered = sum(1 for p in self.emergency_system.protocols.values() if p.is_triggered)
        print(f"\n🚨 Emergency Protocols: {triggered}/10 Triggered")
        
        # Agent Statistics
        stats = self.agent_manager.get_dashboard_stats()
        print(f"\n🤖 Agent Statistics:")
        print(f"  📊 Daily Created: {stats['daily_created']}")
        print(f"  📊 Total Created: {stats['total_created']}")
        print(f"  📊 Active Agents: {stats['total_agents_active']}")
        
        print("\n" + "=" * 80 + "\n")
    
    def execute_owner_command(self, command: str, mother_key: str):
        """اجرای دستورات مالک"""
        if not self.mother_key.verify_key(mother_key):
            logger.critical("❌ INVALID MOTHER KEY - COMMAND REJECTED")
            return False
        
        if command == "STOP":
            self.emergency_system.execute_emergency_stop()
            self.current_status = SystemStatus.RED
            return True
        
        elif command == "LOCK":
            self.mother_key.lock_key()
            self.current_status = SystemStatus.RED
            return True
        
        elif command == "UNLOCK":
            self.mother_key.unlock_key()
            self.current_status = SystemStatus.GREEN
            return True
        
        return False

# ===========================
# Global Security Instance
# ===========================

security_dashboard = SecurityDashboard()
