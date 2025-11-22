"""
University Knowledge Database
پایگاه داده کامل اطلاعات دانشگاهی

ذخیره‌سازی ساختاریافته تمام اطلاعات جمع‌آوری‌شده
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UniversityKnowledgeDB:
    """
    پایگاه داده SQLite برای ذخیره اطلاعات دانشگاهی
    
    جداول:
    - universities: اطلاعات دانشگاه‌ها
    - documents: اسناد جمع‌آوری‌شده
    - scraping_sessions: جلسات scraping
    - security_events: رویدادهای امنیتی
    - specialization_stats: آمار تخصص‌ها
    """
    
    def __init__(self, db_path: str = "university_cache/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # برای دسترسی ستونی
        
        self._create_tables()
        logger.info(f"✓ Database initialized: {self.db_path}")
    
    def _create_tables(self):
        """ایجاد جداول"""
        cursor = self.conn.cursor()
        
        # جدول دانشگاه‌ها
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS universities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                country TEXT,
                rank INTEGER,
                focus_areas TEXT,  -- JSON array
                total_documents INTEGER DEFAULT 0,
                total_pages_scraped INTEGER DEFAULT 0,
                last_update TEXT,
                compliance_score REAL DEFAULT 100.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول اسناد
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                university_key TEXT NOT NULL,
                resource TEXT NOT NULL,
                url TEXT,
                title TEXT,
                content TEXT,
                content_length INTEGER,
                metadata TEXT,  -- JSON
                specializations TEXT,  -- JSON array
                compliance_level TEXT,
                scraped_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (university_key) REFERENCES universities(key)
            )
        """)
        
        # جدول جلسات scraping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraping_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                university_key TEXT NOT NULL,
                pages_scraped INTEGER DEFAULT 0,
                documents_collected INTEGER DEFAULT 0,
                documents_accepted INTEGER DEFAULT 0,
                documents_rejected INTEGER DEFAULT 0,
                duration_seconds REAL,
                status TEXT,  -- success/failed
                error_message TEXT,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (university_key) REFERENCES universities(key)
            )
        """)
        
        # جدول رویدادهای امنیتی
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                university_key TEXT,
                event_type TEXT NOT NULL,
                severity TEXT,  -- info/warning/violation/blocked
                message TEXT,
                metadata TEXT,  -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول آمار تخصص‌ها
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS specialization_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                field TEXT NOT NULL,
                subfield TEXT,
                document_count INTEGER DEFAULT 0,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(field, subfield)
            )
        """)
        
        # ایندکس‌ها برای سرعت
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_university ON documents(university_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_scraped ON documents(scraped_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_university ON scraping_sessions(university_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_security_university ON security_events(university_key)")
        
        self.conn.commit()
    
    def add_university(self, key: str, info: Dict) -> int:
        """افزودن یا به‌روزرسانی دانشگاه"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO universities (key, name, country, rank, focus_areas)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                name = excluded.name,
                country = excluded.country,
                rank = excluded.rank,
                focus_areas = excluded.focus_areas
        """, (
            key,
            info['name'],
            info['country'],
            info['rank'],
            json.dumps(info['focus_areas'])
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def add_document(self, doc: Dict, university_key: str, compliance_level: str) -> int:
        """افزودن سند"""
        cursor = self.conn.cursor()
        
        metadata = doc.get('metadata', {})
        
        cursor.execute("""
            INSERT INTO documents (
                university_key, resource, url, title, content,
                content_length, metadata, compliance_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            university_key,
            metadata.get('resource', 'unknown'),
            metadata.get('url', ''),
            doc.get('title', 'Untitled'),
            doc.get('content', ''),
            len(doc.get('content', '')),
            json.dumps(metadata),
            compliance_level
        ))
        
        # به‌روزرسانی شمارنده دانشگاه
        cursor.execute("""
            UPDATE universities
            SET total_documents = total_documents + 1
            WHERE key = ?
        """, (university_key,))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def start_scraping_session(self, university_key: str) -> int:
        """شروع جلسه scraping"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO scraping_sessions (university_key, started_at)
            VALUES (?, ?)
        """, (university_key, datetime.now().isoformat()))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def complete_scraping_session(
        self,
        session_id: int,
        pages: int,
        collected: int,
        accepted: int,
        rejected: int,
        duration: float,
        status: str,
        error: Optional[str] = None
    ):
        """اتمام جلسه scraping"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE scraping_sessions SET
                pages_scraped = ?,
                documents_collected = ?,
                documents_accepted = ?,
                documents_rejected = ?,
                duration_seconds = ?,
                status = ?,
                error_message = ?,
                completed_at = ?
            WHERE id = ?
        """, (
            pages, collected, accepted, rejected,
            duration, status, error,
            datetime.now().isoformat(),
            session_id
        ))
        
        self.conn.commit()
    
    def log_security_event(self, university_key: str, event_type: str, severity: str, message: str, metadata: Dict):
        """ثبت رویداد امنیتی"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO security_events (university_key, event_type, severity, message, metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            university_key, event_type, severity, message, json.dumps(metadata)
        ))
        
        self.conn.commit()
    
    def update_specialization_stats(self, field: str, subfield: str, count: int = 1):
        """به‌روزرسانی آمار تخصص"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO specialization_stats (field, subfield, document_count)
            VALUES (?, ?, ?)
            ON CONFLICT(field, subfield) DO UPDATE SET
                document_count = document_count + ?,
                last_updated = CURRENT_TIMESTAMP
        """, (field, subfield, count, count))
        
        self.conn.commit()
    
    def get_university_stats(self, university_key: str) -> Dict:
        """آمار یک دانشگاه"""
        cursor = self.conn.cursor()
        
        # اطلاعات اصلی
        cursor.execute("""
            SELECT * FROM universities WHERE key = ?
        """, (university_key,))
        
        uni = cursor.fetchone()
        if not uni:
            return {}
        
        # آمار اسناد
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(content_length) as total_chars,
                AVG(content_length) as avg_chars
            FROM documents WHERE university_key = ?
        """, (university_key,))
        doc_stats = cursor.fetchone()
        
        # آمار جلسات
        cursor.execute("""
            SELECT 
                COUNT(*) as sessions,
                SUM(pages_scraped) as total_pages,
                AVG(duration_seconds) as avg_duration
            FROM scraping_sessions WHERE university_key = ?
        """, (university_key,))
        session_stats = cursor.fetchone()
        
        return {
            'university': dict(uni),
            'documents': dict(doc_stats),
            'sessions': dict(session_stats)
        }
    
    def get_all_universities_summary(self) -> List[Dict]:
        """خلاصه همه دانشگاه‌ها"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                u.*,
                COUNT(d.id) as doc_count,
                SUM(d.content_length) as total_content
            FROM universities u
            LEFT JOIN documents d ON u.key = d.university_key
            GROUP BY u.id
            ORDER BY u.rank
        """)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_specialization_coverage(self) -> Dict:
        """پوشش تخصص‌ها"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT field, SUM(document_count) as total
            FROM specialization_stats
            GROUP BY field
            ORDER BY total DESC
        """)
        
        return {row['field']: row['total'] for row in cursor.fetchall()}
    
    def get_security_summary(self) -> Dict:
        """خلاصه امنیتی"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                severity,
                COUNT(*) as count
            FROM security_events
            GROUP BY severity
        """)
        
        severity_counts = {row['severity']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT COUNT(*) as total FROM security_events
        """)
        total = cursor.fetchone()['total']
        
        return {
            'total_events': total,
            'by_severity': severity_counts
        }
    
    def close(self):
        """بستن اتصال"""
        self.conn.close()
