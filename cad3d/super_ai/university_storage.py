"""
University Storage Layer (SQLite)
ماژول ذخیره‌سازی پایگاه‌داده برای سیستم دانشگاهی

جداول:
- universities(id, key, name, country, rank)
- resources(id, university_id, resource_key, url, type, description)
- pages(id, resource_id, url, title, length, scraped_at)
- documents(id, university_id, resource_key, title, url, content, created_at)
- agent_state(id, university_key, last_update, total_documents, total_pages_scraped)
- security_events(id, timestamp, event_type, detail, severity, url)

ویژگی‌ها:
- ایجاد خودکار Schema
- توابع درج و جستجو
- استفاده از sqlite3 (بدون وابستگی جدید)
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

DB_FILE = Path("university_knowledge.db")
_lock = threading.Lock()

SCHEMA = {
    "universities": """
        CREATE TABLE IF NOT EXISTS universities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            name TEXT,
            country TEXT,
            rank INTEGER
        )
    """,
    "resources": """
        CREATE TABLE IF NOT EXISTS resources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            university_id INTEGER,
            resource_key TEXT,
            url TEXT,
            type TEXT,
            description TEXT,
            FOREIGN KEY(university_id) REFERENCES universities(id)
        )
    """,
    "pages": """
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_id INTEGER,
            url TEXT,
            title TEXT,
            length INTEGER,
            scraped_at TEXT,
            FOREIGN KEY(resource_id) REFERENCES resources(id)
        )
    """,
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            university_id INTEGER,
            resource_key TEXT,
            title TEXT,
            url TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY(university_id) REFERENCES universities(id)
        )
    """,
    "agent_state": """
        CREATE TABLE IF NOT EXISTS agent_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            university_key TEXT UNIQUE,
            last_update TEXT,
            total_documents INTEGER,
            total_pages_scraped INTEGER
        )
    """,
    "security_events": """
        CREATE TABLE IF NOT EXISTS security_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            event_type TEXT,
            detail TEXT,
            severity TEXT,
            url TEXT
        )
    """
}

class UniversityStorage:
    def __init__(self, db_path: Path = DB_FILE):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            for name, ddl in SCHEMA.items():
                cur.execute(ddl)
            conn.commit()
            conn.close()

    # Universities
    def upsert_university(self, key: str, info: Dict):
        with _lock:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute("SELECT id FROM universities WHERE key=?", (key,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE universities SET name=?, country=?, rank=? WHERE id=?",
                            (info['name'], info['country'], info.get('rank'), row[0]))
                uni_id = row[0]
            else:
                cur.execute("INSERT INTO universities(key, name, country, rank) VALUES(?,?,?,?)",
                            (key, info['name'], info['country'], info.get('rank')))
                uni_id = cur.lastrowid
            conn.commit(); conn.close()
            return uni_id

    def insert_resource(self, university_id: int, rkey: str, rinfo: Dict):
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("INSERT INTO resources(university_id, resource_key, url, type, description) VALUES(?,?,?,?,?)",
                        (university_id, rkey, rinfo['url'], rinfo.get('type'), rinfo.get('description')))
            rid = cur.lastrowid
            conn.commit(); conn.close()
            return rid

    def insert_page(self, resource_id: int, page: Dict):
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("INSERT INTO pages(resource_id, url, title, length, scraped_at) VALUES(?,?,?,?,?)",
                        (resource_id, page.get('url'), page.get('title'), page.get('length'), datetime.utcnow().isoformat()))
            conn.commit(); conn.close()

    def insert_document(self, university_id: int, resource_key: str, doc: Dict):
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("INSERT INTO documents(university_id, resource_key, title, url, content, created_at) VALUES(?,?,?,?,?,?)",
                        (university_id, resource_key, doc.get('title'), doc.get('url'), doc.get('content'), datetime.utcnow().isoformat()))
            conn.commit(); conn.close()

    def get_resource_id(self, university_id: int, resource_key: str) -> Optional[int]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT id FROM resources WHERE university_id=? AND resource_key=?", (university_id, resource_key))
            row = cur.fetchone(); conn.close()
            return row[0] if row else None

    def upsert_agent_state(self, university_key: str, state: Dict):
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT id FROM agent_state WHERE university_key=?", (university_key,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE agent_state SET last_update=?, total_documents=?, total_pages_scraped=? WHERE id=?",
                            (state.get('last_update'), state.get('total_documents'), state.get('total_pages_scraped'), row[0]))
            else:
                cur.execute("INSERT INTO agent_state(university_key, last_update, total_documents, total_pages_scraped) VALUES(?,?,?,?)",
                            (university_key, state.get('last_update'), state.get('total_documents'), state.get('total_pages_scraped')))
            conn.commit(); conn.close()

    def insert_security_events(self, events: List[Dict]):
        if not events:
            return
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.executemany("INSERT INTO security_events(timestamp, event_type, detail, severity, url) VALUES(?,?,?,?,?)",
                            [(e.get('timestamp'), e.get('event_type'), e.get('detail'), e.get('severity'), e.get('url')) for e in events])
            conn.commit(); conn.close()

    # Query helpers
    def get_statistics(self) -> Dict[str, Any]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM universities"); unis = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM resources"); resources = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM pages"); pages = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents"); documents = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM security_events"); sec_events = cur.fetchone()[0]
            conn.close()
        return {
            'universities': unis,
            'resources': resources,
            'pages': pages,
            'documents': documents,
            'security_events': sec_events
        }

    def get_agent_states(self) -> List[Dict[str, Any]]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT university_key, last_update, total_documents, total_pages_scraped FROM agent_state")
            rows = cur.fetchall(); conn.close()
        return [
            {
                'university_key': r[0],
                'last_update': r[1],
                'total_documents': r[2],
                'total_pages_scraped': r[3]
            } for r in rows
        ]

    def get_security_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT timestamp, event_type, detail, severity, url FROM security_events ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall(); conn.close()
        return [
            {
                'timestamp': r[0],
                'event_type': r[1],
                'detail': r[2],
                'severity': r[3],
                'url': r[4]
            } for r in rows
        ]

    def get_universities(self) -> List[Dict[str, Any]]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            cur.execute("SELECT key, name, country, rank FROM universities")
            rows = cur.fetchall(); conn.close()
        return [
            {
                'key': r[0],
                'name': r[1],
                'country': r[2],
                'rank': r[3]
            } for r in rows
        ]

    def get_documents(self, university_key: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with _lock:
            conn = self._connect(); cur = conn.cursor()
            if university_key:
                cur.execute("SELECT d.title, d.url, d.created_at, d.resource_key, u.name FROM documents d JOIN universities u ON d.university_id=u.id WHERE u.key=? ORDER BY d.id DESC LIMIT ?", (university_key, limit))
            else:
                cur.execute("SELECT d.title, d.url, d.created_at, d.resource_key, u.name FROM documents d JOIN universities u ON d.university_id=u.id ORDER BY d.id DESC LIMIT ?", (limit,))
            rows = cur.fetchall(); conn.close()
        return [
            {
                'title': r[0],
                'url': r[1],
                'created_at': r[2],
                'resource_key': r[3],
                'university': r[4]
            } for r in rows
        ]

# Singleton
storage = UniversityStorage()
