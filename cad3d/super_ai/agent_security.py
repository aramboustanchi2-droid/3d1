"""
Agent Security & Compliance System
سیستم امنیتی و انطباق ایجنت‌ها

اطمینان از رعایت قوانین و ضوابط توسط ایجنت‌های یادگیری
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """سطح انطباق"""
    COMPLIANT = "compliant"           # مطابق
    WARNING = "warning"               # هشدار
    VIOLATION = "violation"           # تخلف
    BLOCKED = "blocked"               # مسدود شده

class ContentCategory(Enum):
    """دسته‌بندی محتوا"""
    EDUCATIONAL = "educational"       # آموزشی
    RESEARCH = "research"             # تحقیقاتی
    TECHNICAL = "technical"           # فنی
    ADMINISTRATIVE = "administrative" # اداری
    PROHIBITED = "prohibited"         # ممنوعه

class AgentSecuritySystem:
    """
    سیستم امنیتی و نظارتی برای ایجنت‌های یادگیری
    
    مسئولیت‌ها:
    - بررسی محتوا قبل از ذخیره
    - نظارت بر رفتار ایجنت‌ها
    - اعمال قوانین و محدودیت‌ها
    - گزارش‌دهی و لاگ‌گیری
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path)
        
        # قوانین و محدودیت‌ها
        self.rules = {
            'allowed_domains': self._get_allowed_domains(),
            'prohibited_keywords': self._get_prohibited_keywords(),
            'allowed_categories': [
                ContentCategory.EDUCATIONAL,
                ContentCategory.RESEARCH,
                ContentCategory.TECHNICAL,
                ContentCategory.ADMINISTRATIVE
            ],
            'max_content_length': 1000000,  # حداکثر طول محتوا (کاراکتر)
            'required_fields': ['university', 'resource', 'url', 'content']
        }
        
        # آمار نظارتی
        self.monitoring_stats = {
            'total_checks': 0,
            'compliant': 0,
            'warnings': 0,
            'violations': 0,
            'blocked': 0
        }
        
        # لاگ‌های امنیتی
        self.security_logs = []
        self.logs_dir = Path('university_cache/security_logs')
        self.logs_dir.mkdir(exist_ok=True, parents=True)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """بارگذاری تنظیمات"""
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # تنظیمات پیش‌فرض
        return {
            'strict_mode': True,
            'auto_block_violations': True,
            'log_all_checks': False,
            'notify_on_violation': True
        }
    
    def _get_allowed_domains(self) -> Set[str]:
        """دامنه‌های مجاز برای scraping"""
        return {
            # MIT
            'ocw.mit.edu', 'dspace.mit.edu', 'csail.mit.edu',
            # Stanford
            'online.stanford.edu', 'ai.stanford.edu', 'engineering.stanford.edu',
            # Cambridge
            'repository.cam.ac.uk', 'cam.ac.uk',
            # Oxford
            'ora.ox.ac.uk', 'ox.ac.uk',
            # Berkeley
            'eecs.berkeley.edu', 'bair.berkeley.edu', 'berkeley.edu',
            # ETH Zurich
            'ethz.ch',
            # Caltech
            'caltech.edu',
            # Imperial
            'imperial.ac.uk',
            # Carnegie Mellon
            'cmu.edu',
            # TU Delft
            'tudelft.nl'
        }
    
    def _get_prohibited_keywords(self) -> Set[str]:
        """کلمات ممنوعه"""
        return {
            # محتوای نامناسب
            'illegal', 'hack', 'crack', 'pirate', 'torrent',
            # اطلاعات شخصی
            'password', 'credit card', 'ssn', 'social security',
            # محتوای خطرناک
            'weapon', 'explosive', 'malware', 'virus',
            # سیاسی/جنجالی (اختیاری)
            # می‌توانید بر اساس نیاز اضافه کنید
        }
    
    def check_url_compliance(self, url: str) -> ComplianceLevel:
        """
        بررسی انطباق URL
        
        Args:
            url: آدرس URL
        
        Returns:
            سطح انطباق
        """
        # بررسی دامنه
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # بررسی دامنه در لیست مجاز
        is_allowed = any(allowed in domain for allowed in self.rules['allowed_domains'])
        
        if not is_allowed:
            self._log_security_event(
                'url_violation',
                f"Unauthorized domain: {domain}",
                {'url': url, 'domain': domain}
            )
            return ComplianceLevel.VIOLATION
        
        return ComplianceLevel.COMPLIANT
    
    def check_content_compliance(self, content: str, metadata: Dict) -> ComplianceLevel:
        """
        بررسی انطباق محتوا
        
        Args:
            content: متن محتوا
            metadata: متادیتا
        
        Returns:
            سطح انطباق
        """
        self.monitoring_stats['total_checks'] += 1
        
        # بررسی طول محتوا
        if len(content) > self.rules['max_content_length']:
            self._log_security_event(
                'content_length_warning',
                f"Content too long: {len(content)} chars",
                metadata
            )
            self.monitoring_stats['warnings'] += 1
            return ComplianceLevel.WARNING
        
        # بررسی کلمات ممنوعه
        content_lower = content.lower()
        found_prohibited = []
        
        for keyword in self.rules['prohibited_keywords']:
            if keyword in content_lower:
                found_prohibited.append(keyword)
        
        if found_prohibited:
            self._log_security_event(
                'prohibited_content',
                f"Prohibited keywords found: {', '.join(found_prohibited)}",
                metadata
            )
            self.monitoring_stats['violations'] += 1
            
            if self.config['auto_block_violations']:
                self.monitoring_stats['blocked'] += 1
                return ComplianceLevel.BLOCKED
            
            return ComplianceLevel.VIOLATION
        
        # بررسی فیلدهای ضروری
        for field in self.rules['required_fields']:
            if field not in metadata and field != 'content':
                self._log_security_event(
                    'missing_field',
                    f"Required field missing: {field}",
                    metadata
                )
                self.monitoring_stats['warnings'] += 1
                return ComplianceLevel.WARNING
        
        # همه چیز مطابق است
        self.monitoring_stats['compliant'] += 1
        return ComplianceLevel.COMPLIANT
    
    def validate_document(self, document: Dict) -> tuple[bool, ComplianceLevel, str]:
        """
        اعتبارسنجی کامل یک سند
        
        Args:
            document: سند شامل content و metadata
        
        Returns:
            (is_valid, compliance_level, reason)
        """
        # بررسی URL
        if 'url' in document.get('metadata', {}):
            url_compliance = self.check_url_compliance(document['metadata']['url'])
            if url_compliance in [ComplianceLevel.VIOLATION, ComplianceLevel.BLOCKED]:
                return False, url_compliance, "Invalid URL domain"
        
        # بررسی محتوا
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        content_compliance = self.check_content_compliance(content, metadata)
        
        if content_compliance == ComplianceLevel.BLOCKED:
            return False, content_compliance, "Prohibited content detected"
        
        if content_compliance == ComplianceLevel.VIOLATION:
            return False, content_compliance, "Content violation"
        
        if content_compliance == ComplianceLevel.WARNING:
            # هشدار اما قابل قبول
            return True, content_compliance, "Content accepted with warnings"
        
        return True, ComplianceLevel.COMPLIANT, "Document is compliant"
    
    def _log_security_event(self, event_type: str, message: str, metadata: Dict):
        """ثبت رویداد امنیتی"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'message': message,
            'metadata': metadata
        }
        
        self.security_logs.append(event)
        
        # ذخیره در فایل
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d')}_security.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        # لاگ در console
        if self.config['notify_on_violation'] and event_type in ['prohibited_content', 'url_violation']:
            logger.warning(f"🚨 Security Event: {message}")
    
    def get_monitoring_report(self) -> Dict:
        """گزارش نظارتی"""
        total = self.monitoring_stats['total_checks']
        
        return {
            'statistics': self.monitoring_stats,
            'compliance_rate': (
                f"{100 * self.monitoring_stats['compliant'] / total:.1f}%"
                if total > 0 else "0%"
            ),
            'recent_logs': self.security_logs[-10:],  # 10 رویداد آخر
            'config': self.config
        }
    
    def get_agent_score(self, agent_name: str) -> float:
        """
        امتیاز انطباق یک ایجنت
        
        Returns:
            امتیاز بین 0 تا 100
        """
        # در نسخه‌های بعدی می‌توان آمار هر ایجنت را جداگانه نگه داشت
        total = self.monitoring_stats['total_checks']
        if total == 0:
            return 100.0
        
        compliant = self.monitoring_stats['compliant']
        warnings = self.monitoring_stats['warnings']
        violations = self.monitoring_stats['violations']
        
        score = 100 * (compliant + 0.5 * warnings) / total
        return max(0.0, min(100.0, score))


class SpecializationManager:
    """
    مدیریت تخصص‌های علمی
    
    اطمینان از جمع‌آوری اطلاعات تخصصی در رشته‌های مختلف
    """
    
    def __init__(self):
        # رشته‌های تخصصی
        self.specializations = {
            'engineering': {
                'civil': ['structural', 'geotechnical', 'transportation', 'hydraulic'],
                'mechanical': ['thermodynamics', 'fluid mechanics', 'manufacturing', 'robotics'],
                'electrical': ['power systems', 'electronics', 'signal processing', 'control'],
                'computer': ['algorithms', 'software engineering', 'AI', 'networks'],
                'chemical': ['process engineering', 'materials', 'thermodynamics'],
                'industrial': ['optimization', 'operations research', 'supply chain'],
                'architecture': ['design', 'urban planning', 'sustainable architecture']
            },
            'management': {
                'business': ['strategy', 'marketing', 'finance', 'operations'],
                'project': ['planning', 'scheduling', 'risk management', 'agile'],
                'hr': ['recruitment', 'training', 'performance', 'organizational behavior'],
                'quality': ['QA/QC', 'six sigma', 'lean', 'ISO standards']
            },
            'economics': {
                'micro': ['consumer theory', 'market structures', 'game theory'],
                'macro': ['monetary policy', 'fiscal policy', 'growth', 'unemployment'],
                'financial': ['investments', 'portfolio theory', 'derivatives', 'risk'],
                'development': ['growth models', 'poverty', 'inequality']
            }
        }
        
        # کلمات کلیدی برای هر رشته
        self.keywords = self._build_keywords()
    
    def _build_keywords(self) -> Dict[str, List[str]]:
        """ساخت لیست کلمات کلیدی"""
        keywords = {}
        
        for field, subfields in self.specializations.items():
            keywords[field] = []
            for subfield, topics in subfields.items():
                keywords[field].extend([subfield] + topics)
        
        return keywords
    
    def detect_specialization(self, content: str) -> Dict[str, float]:
        """
        تشخیص تخصص محتوا
        
        Args:
            content: متن محتوا
        
        Returns:
            دیکشنری {field: relevance_score}
        """
        content_lower = content.lower()
        scores = {}
        
        for field, field_keywords in self.keywords.items():
            count = sum(1 for keyword in field_keywords if keyword in content_lower)
            scores[field] = count / len(field_keywords) if field_keywords else 0.0
        
        return scores
    
    def get_missing_specializations(self, collected_docs: List[Dict]) -> List[str]:
        """
        رشته‌هایی که کمتر پوشش داده شده‌اند
        
        Args:
            collected_docs: لیست اسناد جمع‌آوری‌شده
        
        Returns:
            لیست رشته‌های با پوشش کم
        """
        specialization_counts = {field: 0 for field in self.specializations.keys()}
        
        for doc in collected_docs:
            content = doc.get('content', '')
            scores = self.detect_specialization(content)
            
            # اگر امتیاز بالای 0.1 داشته باشد، به آن رشته تعلق دارد
            for field, score in scores.items():
                if score > 0.1:
                    specialization_counts[field] += 1
        
        # رشته‌هایی با تعداد کم
        avg_count = sum(specialization_counts.values()) / len(specialization_counts)
        missing = [
            field for field, count in specialization_counts.items()
            if count < avg_count * 0.5
        ]
        
        return missing
