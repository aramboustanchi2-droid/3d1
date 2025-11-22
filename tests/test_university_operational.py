import os
import pytest
from cad3d.super_ai import university_config
from cad3d.super_ai.university_agents import UniversityAgentManager
from cad3d.super_ai.rag_system import RAGSystem
from cad3d.super_ai import university_monitor

def test_learning_and_storage(tmp_path, monkeypatch):
    # Use temp cache dir
    monkeypatch.setenv('UNIV_RUN_ONCE', '1')
    cfg = university_config.CONFIG.copy()
    cfg['storage'] = cfg.get('storage', {}).copy()
    cfg['storage']['cache_dir'] = str(tmp_path)
    rag = RAGSystem()
    mgr = UniversityAgentManager({'MIT': university_config.UNIVERSITIES['MIT']}, cfg, rag)
    stats = mgr.learn_from_all()
    assert stats['agents_updated'] >= 0
    overview = university_monitor.get_overview()
    assert 'universities' in overview and 'documents' in overview

def test_monitor_endpoints():
    overview = university_monitor.get_overview()
    agents = university_monitor.get_agents()
    assert 'agents' in agents
    assert set(['universities','resources','pages','documents','security_events']).issubset(set(overview.keys()))
