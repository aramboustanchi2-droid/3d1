"""
Unified Backend Health & Capability Test
Tests initialization of all configured AI model database backends.
Run:
    python test_all_backends.py
Optional environment variables for real services:
    POSTGRES_URL, MYSQL_URL, REDIS_URL, MONGO_URL
Falls back to safe defaults / mock modes if not provided.
"""
from __future__ import annotations
import os
from cad3d.super_ai.ai_db_factory import create_ai_database

RESULTS = []


def safe_init(name: str, fn):
    try:
        inst = fn()
        # Lightweight call (ping or statistics)
        stats = {}
        if hasattr(inst, 'get_statistics'):
            stats = inst.get_statistics()
        elif hasattr(inst, 'ping') and inst.ping():
            stats = {'ping': True}
        RESULTS.append((name, True, stats))
    except Exception as e:
        RESULTS.append((name, False, str(e)))


def main():
    print("\n=== AI Multi-Backend Initialization Test ===\n")

    # SQLite
    safe_init('sqlite', lambda: create_ai_database(backend='sqlite', db_path='ai_models.db'))

    # PostgreSQL
    pg_url = os.environ.get('POSTGRES_URL')
    if pg_url:
        safe_init('postgresql', lambda: create_ai_database(backend='postgresql', connection_string=pg_url))
    else:
        RESULTS.append(('postgresql', False, 'POSTGRES_URL not set'))

    # MySQL
    mysql_url = os.environ.get('MYSQL_URL')
    if mysql_url:
        safe_init('mysql', lambda: create_ai_database(backend='mysql', connection_string=mysql_url))
    else:
        RESULTS.append(('mysql', False, 'MYSQL_URL not set'))

    # Redis
    redis_url = os.environ.get('REDIS_URL')
    if redis_url:
        safe_init('redis', lambda: create_ai_database(backend='redis', connection_string=redis_url))
    else:
        # Attempt default host
        safe_init('redis', lambda: create_ai_database(backend='redis', redis_host='localhost', redis_port=6379))

    # ChromaDB
    safe_init('chromadb', lambda: create_ai_database(backend='chromadb', persist_directory='chromadb_health'))

    # FAISS
    safe_init('faiss', lambda: create_ai_database(backend='faiss', index_path='faiss_health', dimension=384))

    # MongoDB
    mongo_url = os.environ.get('MONGO_URL')
    if mongo_url:
        safe_init('mongodb', lambda: create_ai_database(backend='mongodb', connection_string=mongo_url))
    else:
        safe_init('mongodb', lambda: create_ai_database(backend='mongodb', mongo_use_mock=True))

    print("\n--- Results ---")
    for name, ok, info in RESULTS:
        status = '✅ OK' if ok else '❌ FAIL'
        print(f"{name:<12} {status}  {info}")

    success = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\nSummary: {success}/{total} backends initialized successfully")

    # Write summary file
    with open('backend_health_summary.json', 'w', encoding='utf-8') as f:
        import json
        json.dump({
            'results': [
                {'backend': n, 'ok': o, 'info': i} for n, o, i in RESULTS
            ]
        }, f, indent=2, ensure_ascii=False)
    print("Saved: backend_health_summary.json")


if __name__ == '__main__':
    main()
