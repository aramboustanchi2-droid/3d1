"""University Monitoring Module
ماژول پایش سیستم دانشگاهی: جمع‌آوری آمار، وضعیت ایجنت‌ها و رویدادهای امنیتی.
"""
from __future__ import annotations
from typing import Dict, Any
from .university_storage import storage

def get_overview() -> Dict[str, Any]:
    return storage.get_statistics()

def get_agents() -> Dict[str, Any]:
    return {
        'agents': storage.get_agent_states()
    }

def get_security_events(limit: int = 100) -> Dict[str, Any]:
    return {
        'events': storage.get_security_events(limit)
    }

def get_universities() -> Dict[str, Any]:
    return {
        'universities': storage.get_universities()
    }

def get_documents(university_key: str | None = None, limit: int = 50) -> Dict[str, Any]:
    return {
        'documents': storage.get_documents(university_key, limit)
    }
