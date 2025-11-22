"""
University Security & Compliance Module
ماژول امنیت و انطباق برای ایجنت‌های دانشگاهی

وظایف:
- تعریف سیاست امنیتی دامنه‌های مجاز
- تشخیص محتوای حساس یا ممنوع
- پاکسازی و سانیتایز محتوا
- ثبت رویدادهای امنیتی در حافظه و پایگاه‌داده (در صورت ادغام)
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

SENSITIVE_PATTERNS = [
    r"api[_-]?key", r"secret", r"password", r"token", r"confidential",
    r"private", r"credential", r"ssh[- ]?key", r"-----BEGIN [A-Z ]+-----"
]
PROHIBITED_CONTENT = [
    "attack", "exploit", "sql injection", "ddos", "malware", "phishing"
]

SANITIZE_REPLACEMENTS = {
    r"[A-Za-z0-9]{32,}": "[REDACTED_HASH]",
    r"\b(?:\d{4}-\d{4}-\d{4}-\d{4})\b": "[REDACTED_CARD]"
}

@dataclass
class SecurityEvent:
    timestamp: str
    event_type: str
    detail: str
    severity: str = "info"
    url: Optional[str] = None

@dataclass
class SecurityPolicy:
    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    max_page_length: int = 1_000_000  # characters
    enable_sanitize: bool = True

    def is_domain_allowed(self, url: str) -> bool:
        for bd in self.blocked_domains:
            if bd in url:
                return False
        if not self.allowed_domains:
            return True
        return any(dom in url for dom in self.allowed_domains)

class SecurityMonitor:
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy(
            allowed_domains=[
                "mit.edu", "stanford.edu", "cam.ac.uk", "ox.ac.uk", "berkeley.edu",
                "ethz.ch", "caltech.edu", "imperial.ac.uk", "cmu.edu", "tudelft.nl"
            ]
        )
        self.events: List[SecurityEvent] = []

    def log(self, event_type: str, detail: str, severity: str = "info", url: Optional[str] = None):
        ev = SecurityEvent(
            timestamp=datetime.utcnow().isoformat(),
            event_type=event_type,
            detail=detail,
            severity=severity,
            url=url
        )
        self.events.append(ev)
        msg = f"[{severity.upper()}] {event_type}: {detail}"
        if url:
            msg += f" (url={url})"
        if severity in ("warning", "error"):
            logger.warning(msg)
        else:
            logger.info(msg)

    def domain_check(self, url: str) -> bool:
        if not self.policy.is_domain_allowed(url):
            self.log("domain_block", f"Blocked domain in URL {url}", "error", url)
            return False
        return True

    def scan_content(self, text: str, url: Optional[str] = None) -> Dict:
        findings = []
        # Sensitive patterns
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                findings.append({"pattern": pattern, "type": "sensitive"})
        # Prohibited intents
        lower_text = text.lower()
        for kw in PROHIBITED_CONTENT:
            if kw in lower_text:
                findings.append({"keyword": kw, "type": "prohibited"})
        # Oversize check
        oversize = len(text) > self.policy.max_page_length
        if findings:
            self.log("content_flag", f"Found {len(findings)} security indicators", "warning", url)
        if oversize:
            self.log("oversize", f"Page length {len(text)} exceeds limit {self.policy.max_page_length}", "warning", url)
        return {"findings": findings, "oversize": oversize}

    def sanitize(self, text: str) -> str:
        if not self.policy.enable_sanitize:
            return text
        sanitized = text
        for pattern, replacement in SANITIZE_REPLACEMENTS.items():
            sanitized = re.sub(pattern, replacement, sanitized)
        return sanitized

    def enforce(self, url: str, content: str) -> Dict:
        allowed = self.domain_check(url)
        scan = self.scan_content(content, url)
        sanitized = self.sanitize(content) if scan["findings"] else content
        return {
            "allowed": allowed,
            "scan": scan,
            "sanitized_content": sanitized
        }

    def export_events(self) -> List[Dict]:
        return [ev.__dict__ for ev in self.events]

# Singleton (optional)
security_monitor = SecurityMonitor()
