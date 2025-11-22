"""
University Learning Scheduler - زمان‌بند یادگیری خودکار

سیستم زمان‌بندی برای به‌روزرسانی خودکار دانش از دانشگاه‌ها
"""

import schedule
import time
import logging
from datetime import datetime
from typing import Callable, Dict
from pathlib import Path
import json
import threading

from .university_agents import UniversityAgentManager
from .university_config import UNIVERSITIES, AGENT_CONFIG

logger = logging.getLogger(__name__)

class UniversityLearningScheduler:
    """
    Scheduler برای یادگیری خودکار و دوره‌ای
    
    ویژگی‌ها:
    - یادگیری روزانه/هفتگی/ماهانه
    - اولویت‌بندی دانشگاه‌ها
    - مدیریت خطا و تلاش مجدد
    - گزارش‌دهی
    """
    
    def __init__(self, agent_manager: UniversityAgentManager):
        self.agent_manager = agent_manager
        self.is_running = False
        self.scheduler_thread = None
        
        # Schedule configuration
        self.schedules = {
            'daily': [],      # لیست دانشگاه‌هایی که روزانه چک می‌شوند
            'weekly': [],     # هفتگی
            'monthly': []     # ماهانه
        }
        
        # Logs
        self.logs_dir = Path(AGENT_CONFIG['storage']['cache_dir']) / 'scheduler_logs'
        self.logs_dir.mkdir(exist_ok=True, parents=True)
    
    def add_university_schedule(
        self,
        university_key: str,
        frequency: str = 'daily',
        time_of_day: str = '02:00'
    ):
        """
        افزودن زمان‌بندی برای یک دانشگاه
        
        Args:
            university_key: کلید دانشگاه
            frequency: تناوب ('daily', 'weekly', 'monthly')
            time_of_day: ساعت اجرا (مثلا '02:00')
        """
        if frequency not in self.schedules:
            logger.error(f"Invalid frequency: {frequency}")
            return
        
        self.schedules[frequency].append({
            'university_key': university_key,
            'time': time_of_day
        })
        
        logger.info(f"✓ Scheduled {university_key} for {frequency} updates at {time_of_day}")
    
    def setup_default_schedules(self):
        """تنظیم زمان‌بندی پیش‌فرض برای همه دانشگاه‌ها"""
        
        # Top 5: Daily updates
        top_5 = ["MIT", "Stanford", "Cambridge", "Oxford", "Berkeley"]
        for uni in top_5:
            self.add_university_schedule(uni, 'daily', '02:00')
        
        # Next 5: Weekly updates
        next_5 = ["ETH_Zurich", "Caltech", "Imperial", "Carnegie_Mellon", "TU_Delft"]
        for uni in next_5:
            self.add_university_schedule(uni, 'weekly', '03:00')
        
        logger.info(f"\n✓ Default schedules configured:")
        logger.info(f"  Daily: {len(self.schedules['daily'])} universities")
        logger.info(f"  Weekly: {len(self.schedules['weekly'])} universities")
    
    def _learning_job(self, university_keys: list):
        """
        Job برای یادگیری
        
        Args:
            university_keys: لیست دانشگاه‌ها
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Scheduled Learning Job: {datetime.now()}")
        logger.info(f"Universities: {', '.join(university_keys)}")
        logger.info(f"{'='*80}\n")
        
        try:
            result = self.agent_manager.learn_from_specific(university_keys)
            
            # ذخیره لاگ
            self._save_log({
                'timestamp': datetime.now().isoformat(),
                'universities': university_keys,
                'status': 'success',
                'result': result
            })
            
            logger.info(f"\n✓ Learning job completed")
            logger.info(f"  Agents updated: {result['agents_updated']}")
            logger.info(f"  Total documents: {sum(r.get('documents_added_to_rag', 0) for r in result['results'])}")
            
        except Exception as e:
            logger.error(f"✗ Learning job failed: {e}")
            
            # ذخیره لاگ خطا
            self._save_log({
                'timestamp': datetime.now().isoformat(),
                'universities': university_keys,
                'status': 'failed',
                'error': str(e)
            })
    
    def _save_log(self, log_data: Dict):
        """ذخیره لاگ"""
        log_file = self.logs_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    def start(self):
        """شروع scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        logger.info("\n" + "="*80)
        logger.info("  STARTING UNIVERSITY LEARNING SCHEDULER")
        logger.info("="*80 + "\n")
        
        # تنظیم job‌ها
        
        # Daily jobs
        for item in self.schedules['daily']:
            schedule.every().day.at(item['time']).do(
                self._learning_job,
                [item['university_key']]
            )
            logger.info(f"  Daily: {item['university_key']} at {item['time']}")
        
        # Weekly jobs (Sundays)
        for item in self.schedules['weekly']:
            schedule.every().sunday.at(item['time']).do(
                self._learning_job,
                [item['university_key']]
            )
            logger.info(f"  Weekly: {item['university_key']} on Sundays at {item['time']}")
        
        # Monthly jobs (1st of each month)
        for item in self.schedules['monthly']:
            schedule.every().month.at(item['time']).do(
                self._learning_job,
                [item['university_key']]
            )
            logger.info(f"  Monthly: {item['university_key']} on 1st at {item['time']}")
        
        logger.info(f"\n✓ Scheduler configured")
        logger.info(f"  Total jobs: {len(schedule.get_jobs())}")
        
        # شروع thread
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"✓ Scheduler started in background thread\n")
    
    def _run_scheduler(self):
        """Loop اصلی scheduler"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # بررسی هر دقیقه
    
    def stop(self):
        """توقف scheduler"""
        logger.info("\nStopping scheduler...")
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        schedule.clear()
        logger.info("✓ Scheduler stopped\n")
    
    def run_now(self, university_keys: list):
        """اجرای فوری یک job"""
        logger.info(f"\nRunning immediate learning job...")
        self._learning_job(university_keys)
    
    def get_next_runs(self) -> list:
        """زمان اجرای بعدی job‌ها"""
        jobs = schedule.get_jobs()
        
        next_runs = []
        for job in jobs:
            next_runs.append({
                'job': str(job.job_func),
                'next_run': job.next_run.isoformat() if job.next_run else None
            })
        
        return next_runs
    
    def get_logs(self, limit: int = 10) -> list:
        """دریافت آخرین لاگ‌ها"""
        log_files = sorted(self.logs_dir.glob('*.json'), reverse=True)[:limit]
        
        logs = []
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs.append(json.load(f))
        
        return logs


def demo_scheduler():
    """دمو scheduler"""
    from .rag_system import RAGSystem
    
    logger.info("\n" + "="*80)
    logger.info("  SCHEDULER DEMO")
    logger.info("="*80 + "\n")
    
    # Initialize
    rag_system = RAGSystem()
    agent_manager = UniversityAgentManager(UNIVERSITIES, AGENT_CONFIG, rag_system)
    scheduler = UniversityLearningScheduler(agent_manager)
    
    # تنظیم زمان‌بندی پیش‌فرض
    scheduler.setup_default_schedules()
    
    # نمایش job‌های بعدی
    logger.info("\nNext scheduled runs:")
    next_runs = scheduler.get_next_runs()
    for run in next_runs[:5]:
        logger.info(f"  {run['job']}: {run['next_run']}")
    
    # اجرای فوری یک job برای تست
    logger.info("\n\nRunning immediate test job (MIT)...")
    scheduler.run_now(['MIT'])
    
    # نمایش لاگ‌ها
    logger.info("\n\nRecent logs:")
    logs = scheduler.get_logs(limit=3)
    for log in logs:
        logger.info(f"\n  {log['timestamp']}:")
        logger.info(f"    Status: {log['status']}")
        logger.info(f"    Universities: {', '.join(log['universities'])}")
    
    logger.info("\n" + "="*80)
    logger.info("  SCHEDULER DEMO COMPLETE")
    logger.info("="*80 + "\n")
    
    logger.info("To run scheduler continuously:")
    logger.info("  scheduler.start()")
    logger.info("  # Runs in background thread")
    logger.info("  # Press Ctrl+C to stop")
    logger.info("")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    demo_scheduler()
