"""
University Learning Agents - ایجنت‌های یادگیری دانشگاهی

ایجنت‌های تخصصی برای یادگیری مداوم از دانشگاه‌های برتر
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path

from .university_scraper import UniversityResourceCollector
from .rag_system import RAGSystem
from .university_storage import storage
from .university_security import security_monitor
try:
    from .agent_security import AgentSecuritySystem, SpecializationManager, ComplianceLevel
except ImportError:
    # Fallback minimal security system if agent_security is absent
    class ComplianceLevel:
        OK = "ok"
        WARNING = "warning"
        REJECT = "reject"
    class AgentSecuritySystem:
        def validate_document(self, doc):
            return True, ComplianceLevel.OK, "ok"
        def get_agent_score(self, key):
            return 100.0
    class SpecializationManager:
        pass

logger = logging.getLogger(__name__)

class UniversityAgent:
    """
    ایجنت تخصصی برای یک دانشگاه
    
    مسئولیت‌ها:
    - جمع‌آوری دوره‌ای محتوا
    - استخراج و پردازش اطلاعات
    - به‌روزرسانی پایگاه دانش RAG
    - ردیابی تغییرات و محتوای جدید
    """
    
    def __init__(
        self,
        university_key: str,
        university_info: Dict,
        config: Dict,
        rag_system: Optional[RAGSystem] = None,
        security_system: Optional[AgentSecuritySystem] = None
    ):
        self.university_key = university_key
        self.university_info = university_info
        self.config = config
        self.rag_system = rag_system
        self.security_system = security_system or AgentSecuritySystem()
        
        self.state = {
            'last_update': None,
            'total_documents': 0,
            'total_pages_scraped': 0,
            'last_successful_scrape': None,
            'errors': [],
            'security_violations': 0,
            'compliance_score': 100.0
        }
        
        # State file
        self.state_file = Path(config['storage']['cache_dir']) / f"{university_key}_state.json"
        self.load_state()
    
    def load_state(self):
        """بارگذاری وضعیت از فایل"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                self.state = json.load(f)
    
    def save_state(self):
        """ذخیره وضعیت در فایل"""
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
    
    def should_update(self) -> bool:
        """
        بررسی نیاز به به‌روزرسانی
        
        Returns:
            True اگر زمان به‌روزرسانی رسیده باشد
        """
        if self.state['last_update'] is None:
            return True
        
        last_update = datetime.fromisoformat(self.state['last_update'])
        update_frequency = self.config['learning']['update_frequency']
        
        if update_frequency == 'daily':
            return datetime.now() - last_update > timedelta(days=1)
        elif update_frequency == 'weekly':
            return datetime.now() - last_update > timedelta(weeks=1)
        elif update_frequency == 'monthly':
            return datetime.now() - last_update > timedelta(days=30)
        
        return False
    
    def collect_content(self, collector: UniversityResourceCollector) -> Dict:
        """
        جمع‌آوری محتوا از دانشگاه
        
        Args:
            collector: شیء جمع‌آوری‌کننده
        
        Returns:
            داده‌های جمع‌آوری‌شده
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Agent: {self.university_info['name']}")
        logger.info(f"{'='*80}")
        
        try:
            max_pages = self.config['learning']['max_documents_per_session']
            data = collector.collect_from_university(self.university_key, max_pages)
            
            self.state['last_successful_scrape'] = datetime.now().isoformat()
            self.state['total_pages_scraped'] += data.get('total_pages', 0)
            
            return data
            
        except Exception as e:
            error_msg = f"Error collecting content: {str(e)}"
            logger.error(error_msg)
            self.state['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
            return {}
    
    def process_content(self, content: Dict) -> List[Dict]:
        """
        پردازش محتوای جمع‌آوری‌شده
        
        Args:
            content: داده‌های خام
        
        Returns:
            لیست اسناد پردازش‌شده برای RAG
        """
        documents = []
        
        for resource_key, resource_data in content.get('resources', {}).items():
            # صفحه اصلی
            main_page = resource_data.get('main_page')
            if main_page:
                doc = self._create_document(
                    main_page,
                    resource_key,
                    resource_data['info']['description']
                )
                documents.append(doc)
            
            # صفحات فرعی
            for sub_page in resource_data.get('sub_pages', []):
                doc = self._create_document(
                    sub_page,
                    resource_key,
                    f"Sub-page from {resource_key}"
                )
                documents.append(doc)
        
        return documents
    
    def _create_document(self, page_data: Dict, resource_key: str, description: str) -> Dict:
        """ایجاد سند برای RAG"""
        return {
            'content': page_data.get('text', ''),
            'title': page_data.get('title', 'Untitled'),
            'url': page_data.get('url', ''),
            'metadata': {
                'university': self.university_info['name'],
                'university_key': self.university_key,
                'resource': resource_key,
                'description': description,
                'focus_areas': self.university_info['focus_areas'],
                'country': self.university_info['country'],
                'scraped_at': datetime.now().isoformat()
            }
        }
    
    def update_rag_system(self, documents: List[Dict]) -> int:
        """
        به‌روزرسانی سیستم RAG با اسناد جدید (با بررسی امنیتی)
        
        Args:
            documents: لیست اسناد
        
        Returns:
            تعداد اسناد اضافه‌شده
        """
        if not self.rag_system:
            logger.warning("RAG system not available")
            return 0
        
        added = 0
        violations = 0
        
        for doc in documents:
            try:
                # بررسی امنیتی
                is_valid, compliance, reason = self.security_system.validate_document(doc)
                
                if not is_valid:
                    violations += 1
                    logger.warning(f"  ⚠️  Document rejected: {reason}")
                    continue
                
                if compliance == ComplianceLevel.WARNING:
                    logger.info(f"  ⚠️  Document accepted with warning: {reason}")
                
                # اضافه به RAG
                self.rag_system.add_document(
                    doc['content'],
                    doc['metadata']
                )
                # Persist document into database if university id is set
                if hasattr(self, 'uni_id'):
                    try:
                        storage.insert_document(self.uni_id, doc['metadata']['resource'], {
                            'title': doc['title'],
                            'url': doc['url'],
                            'content': doc['content']
                        })
                    except Exception as db_e:
                        logger.error(f"Document DB insert failed: {db_e}")
                added += 1
                
            except Exception as e:
                logger.error(f"Error adding document: {e}")
        
        # به‌روزرسانی آمار امنیتی
        self.state['security_violations'] += violations
        self.state['compliance_score'] = self.security_system.get_agent_score(self.university_key)
        
        logger.info(f"  Added {added} documents to RAG system")
        if violations > 0:
            logger.warning(f"  ⚠️  {violations} documents rejected due to security")
        
        return added
    
    def learn(self, collector: UniversityResourceCollector) -> Dict:
        """
        فرآیند کامل یادگیری
        
        1. جمع‌آوری محتوا
        2. پردازش
        3. به‌روزرسانی RAG
        4. ذخیره وضعیت
        
        Returns:
            آمار یادگیری
        """
        logger.info(f"\n🎓 Learning from: {self.university_info['name']}")
        
        # جمع‌آوری
        content = self.collect_content(collector)
        if not content:
            return {'status': 'failed', 'reason': 'No content collected'}
        
        # پردازش
        documents = self.process_content(content)
        logger.info(f"  Processed {len(documents)} documents")
        
        # به‌روزرسانی RAG
        added = self.update_rag_system(documents)
        
        # به‌روزرسانی وضعیت
        self.state['last_update'] = datetime.now().isoformat()
        self.state['total_documents'] += added
        self.save_state()

        # Persist agent state & security events
        storage.upsert_agent_state(self.university_key, {
            'last_update': self.state['last_update'],
            'total_documents': self.state['total_documents'],
            'total_pages_scraped': self.state['total_pages_scraped']
        })
        storage.insert_security_events(security_monitor.export_events())
        
        stats = {
            'status': 'success',
            'university': self.university_info['name'],
            'pages_scraped': content.get('total_pages', 0),
            'documents_processed': len(documents),
            'documents_added_to_rag': added,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"  ✓ Learning complete: {added} documents added")
        return stats
    
    def get_status(self) -> Dict:
        """وضعیت فعلی ایجنت"""
        return {
            'university': self.university_info['name'],
            'country': self.university_info['country'],
            'focus_areas': self.university_info['focus_areas'],
            'state': self.state,
            'should_update': self.should_update()
        }


class UniversityAgentManager:
    """
    مدیریت تمام ایجنت‌های دانشگاهی
    """
    
    def __init__(
        self,
        universities: Dict,
        config: Dict,
        rag_system: Optional[RAGSystem] = None,
        security_system: Optional[AgentSecuritySystem] = None
    ):
        self.universities = universities
        self.config = config
        self.rag_system = rag_system
        self.security_system = security_system or AgentSecuritySystem()
        self.specialization_manager = SpecializationManager()
        
        # ایجاد ایجنت برای هر دانشگاه
        self.agents = {}
        for key, info in universities.items():
            # Persist university & resources metadata into DB
            uni_id = storage.upsert_university(key, info)
            for rkey, rinfo in info.get('resources', {}).items():
                storage.insert_resource(uni_id, rkey, rinfo)
            self.agents[key] = UniversityAgent(
                key, info, config, rag_system, self.security_system
            )
            self.agents[key].uni_id = uni_id  # attach DB id for persistence
        
        logger.info(f"✓ Initialized {len(self.agents)} university agents with security")
    
    def learn_from_all(self) -> Dict:
        """
        یادگیری از همه دانشگاه‌ها
        
        Returns:
            آمار کلی یادگیری
        """
        collector = UniversityResourceCollector(self.universities, self.config)
        
        results = []
        for agent_key, agent in self.agents.items():
            if agent.should_update():
                result = agent.learn(collector)
                results.append(result)
            else:
                logger.info(f"  Skipping {agent.university_info['name']} (not due for update)")
        
        # آمار کلی
        stats = {
            'total_agents': len(self.agents),
            'agents_updated': len(results),
            'total_pages_scraped': sum(r.get('pages_scraped', 0) for r in results),
            'total_documents_added': sum(r.get('documents_added_to_rag', 0) for r in results),
            'results': results
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Learning Summary:")
        logger.info(f"  Agents updated: {stats['agents_updated']}/{stats['total_agents']}")
        logger.info(f"  Pages scraped: {stats['total_pages_scraped']}")
        logger.info(f"  Documents added to RAG: {stats['total_documents_added']}")
        logger.info(f"{'='*80}\n")
        
        return stats
    
    def learn_from_specific(self, university_keys: List[str]) -> Dict:
        """
        یادگیری از دانشگاه‌های خاص
        
        Args:
            university_keys: لیست کلیدهای دانشگاه
        
        Returns:
            آمار یادگیری
        """
        collector = UniversityResourceCollector(self.universities, self.config)
        
        results = []
        for key in university_keys:
            if key in self.agents:
                result = self.agents[key].learn(collector)
                results.append(result)
        
        return {
            'agents_updated': len(results),
            'results': results
        }
    
    def get_all_statuses(self) -> List[Dict]:
        """وضعیت همه ایجنت‌ها"""
        return [agent.get_status() for agent in self.agents.values()]
    
    def get_statistics(self) -> Dict:
        """آمار کلی سیستم"""
        statuses = self.get_all_statuses()
        
        return {
            'total_agents': len(self.agents),
            'agents_needing_update': sum(1 for s in statuses if s['should_update']),
            'total_documents_collected': sum(s['state']['total_documents'] for s in statuses),
            'total_pages_scraped': sum(s['state']['total_pages_scraped'] for s in statuses),
            'by_university': {
                s['university']: {
                    'country': s['country'],
                    'focus_areas': s['focus_areas'],
                    'documents': s['state']['total_documents'],
                    'pages': s['state']['total_pages_scraped'],
                    'last_update': s['state']['last_update']
                }
                for s in statuses
            }
        }
