"""Production Runner for University Knowledge System
راه‌انداز عملیاتی سیستم دانشگاهی: اجرای دوره‌ای فرآیند یادگیری و سرویس داشبورد.
"""
from __future__ import annotations
import os
import time
import threading
import uvicorn
from cad3d.super_ai.university_agents import UniversityAgentManager
from cad3d.super_ai.university_scraper import UniversityResourceCollector  # not directly used but ensures deps
from cad3d.super_ai.university_dashboard import app as dashboard_app
from cad3d.super_ai import university_config  # assumed existing
from cad3d.super_ai.rag_system import RAGSystem

UPDATE_INTERVAL = int(os.getenv("UNIV_UPDATE_INTERVAL", "3600"))  # seconds
RUN_ONCE = os.getenv("UNIV_RUN_ONCE", "0") == "1"
DASHBOARD_PORT = int(os.getenv("UNIV_DASHBOARD_PORT", "8081"))
HOST = os.getenv("UNIV_DASHBOARD_HOST", "0.0.0.0")

rag_system = RAGSystem()
manager = UniversityAgentManager(university_config.UNIVERSITIES, university_config.CONFIG, rag_system)

def learning_loop():
    while True:
        try:
            manager.learn_from_all()
        except Exception as e:
            print(f"[LearningLoop] Error: {e}
")
        if RUN_ONCE:
            break
        time.sleep(UPDATE_INTERVAL)

def start_dashboard():
    uvicorn.run(dashboard_app, host=HOST, port=DASHBOARD_PORT, log_level="info")

if __name__ == "__main__":
    # Run dashboard in separate thread
    t_dashboard = threading.Thread(target=start_dashboard, daemon=True)
    t_dashboard.start()

    # Start learning loop
    learning_loop()
