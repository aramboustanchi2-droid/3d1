"""
AI Model Database Factory
کارخانه پایگاه داده برای انتخاب SQLite/PostgreSQL/MySQL/Redis/ChromaDB/FAISS

Usage:
    # SQLite (default)
    db = create_ai_database()
    
    # PostgreSQL
    db = create_ai_database(backend='postgresql', 
                            connection_string='postgresql://user:pass@localhost/aimodels')
    
    # MySQL
    db = create_ai_database(backend='mysql',
                            connection_string='mysql+pymysql://user:pass@localhost/aimodels')
    
    # Redis
    db = create_ai_database(backend='redis', redis_host='localhost', redis_port=6379)
    
    # ChromaDB (vector database)
    db = create_ai_database(backend='chromadb', persist_directory='chromadb_data')
    
    # FAISS (vector search)
    db = create_ai_database(backend='faiss', index_path='faiss_data', dimension=384)
    
    # From config file
    db = create_ai_database_from_config('ai_db_config.yaml')
"""
from __future__ import annotations
from typing import Literal, Optional, Union
from pathlib import Path
import yaml
import json


def create_ai_database(
    backend: Literal['sqlite', 'postgresql', 'mysql', 'redis', 'chromadb', 'faiss', 'mongodb'] = 'sqlite',
    connection_string: Optional[str] = None,
    db_path: str = 'ai_models.db',
    echo: bool = False,
    redis_host: str = 'localhost',
    redis_port: int = 6379,
    redis_db: int = 0,
    redis_password: Optional[str] = None,
    persist_directory: str = 'chromadb_data',
    index_path: str = 'faiss_data',
    dimension: int = 384,
    mongo_host: str = 'localhost',
    mongo_port: int = 27017,
    mongo_database: str = 'aidb',
    mongo_user: Optional[str] = None,
    mongo_password: Optional[str] = None,
    mongo_use_mock: bool = False
):
    """
    ایجاد پایگاه داده AI با backend دلخواه
    
    Args:
        backend: نوع پایگاه داده ('sqlite', 'postgresql', 'mysql', 'redis', 'chromadb', 'faiss', 'mongodb')
        connection_string: connection string برای SQL databases یا Redis
        db_path: مسیر فایل برای SQLite
        echo: نمایش SQL queries
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number (0-15)
        redis_password: Redis password
        persist_directory: ChromaDB persistence directory
        index_path: FAISS index storage path
        dimension: Vector embedding dimension (for FAISS)
        mongo_*: تنظیمات MongoDB (هاست، پورت، دیتابیس، کاربر، پسورد، حالت mock)
        
    Returns:
        نمونه از کلاس پایگاه داده مربوطه
    """
    if backend == 'sqlite':
        from .ai_model_database import AIModelDatabase
        return AIModelDatabase(db_path=db_path)
    
    elif backend in ('postgresql', 'mysql'):
        if not connection_string:
            raise ValueError(f"{backend} نیاز به connection_string دارد")
        
        from .ai_model_database_sql import AIModelDatabaseSQL
        return AIModelDatabaseSQL(connection_string=connection_string, echo=echo)
    
    elif backend == 'redis':
        from .ai_model_database_redis import AIModelDatabaseRedis
        
        # Parse connection string if provided (redis://[:password@]host:port/db)
        if connection_string:
            import re
            match = re.match(r'redis://(?::(.+)@)?([^:]+):(\d+)/(\d+)', connection_string)
            if match:
                redis_password = match.group(1) or redis_password
                redis_host = match.group(2)
                redis_port = int(match.group(3))
                redis_db = int(match.group(4))
        
        return AIModelDatabaseRedis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password
        )
    
    elif backend == 'chromadb':
        from .ai_model_database_vector import AIModelDatabaseChroma
        return AIModelDatabaseChroma(persist_directory=persist_directory)
    
    elif backend == 'faiss':
        from .ai_model_database_vector import AIModelDatabaseFAISS
        return AIModelDatabaseFAISS(index_path=index_path, dimension=dimension)

    elif backend == 'mongodb':
        from .ai_model_database_mongo import AIModelDatabaseMongo
        if connection_string:
            return AIModelDatabaseMongo(connection_string=connection_string, use_mock=mongo_use_mock)
        return AIModelDatabaseMongo(
            host=mongo_host,
            port=mongo_port,
            database=mongo_database,
            username=mongo_user,
            password=mongo_password,
            use_mock=mongo_use_mock
        )
    
    else:
        raise ValueError(
            f"Backend پشتیبانی نمی‌شود: {backend}. "
            f"باید یکی از: sqlite, postgresql, mysql, redis, chromadb, faiss, mongodb"
        )


def create_ai_database_from_config(config_path: str):
    """
    ایجاد پایگاه داده از فایل تنظیمات
    
    Args:
        config_path: مسیر فایل YAML یا JSON
        
    Returns:
        نمونه از کلاس پایگاه داده
        
    Example config (YAML):
        backend: postgresql
        connection:
          host: localhost
          port: 5432
          database: aimodels
          user: postgres
          password: secret
        pool_size: 5
        echo: false
        
    Example config (JSON):
        {
          "backend": "mysql",
          "connection": {
            "host": "localhost",
            "port": 3306,
            "database": "aimodels",
            "user": "root",
            "password": "secret"
          },
          "pool_size": 5,
          "echo": false
        }
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"فایل تنظیمات پیدا نشد: {config_path}")
    
    # Load config
    if config_path.suffix in ('.yaml', '.yml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError("فایل تنظیمات باید YAML یا JSON باشد")
    
    backend = config.get('backend', 'sqlite').lower()
    echo = config.get('echo', False)
    
    if backend == 'sqlite':
        db_path = config.get('db_path', 'ai_models.db')
        # Make absolute path relative to project root (E:\3d)
        if not Path(db_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent  # e:\3d
            db_path = str(project_root / db_path)
        return create_ai_database(backend='sqlite', db_path=db_path)
    
    elif backend in ('postgresql', 'mysql'):
        conn = config.get('connection', {})
        
        if 'connection_string' in config:
            # Direct connection string
            connection_string = config['connection_string']
        else:
            # Build from connection dict
            host = conn.get('host', 'localhost')
            port = conn.get('port', 5432 if backend == 'postgresql' else 3306)
            database = conn.get('database', 'aimodels')
            user = conn.get('user', '')
            password = conn.get('password', '')
            
            if backend == 'postgresql':
                connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
            else:  # mysql
                connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
        
        return create_ai_database(
            backend=backend,
            connection_string=connection_string,
            echo=echo
        )
    
    elif backend == 'redis':
        conn = config.get('connection', {})
        
        if 'connection_string' in config:
            # Direct connection string (redis://[:password@]host:port/db)
            connection_string = config['connection_string']
            return create_ai_database(backend='redis', connection_string=connection_string)
        else:
            # Build from connection dict
            host = conn.get('host', 'localhost')
            port = conn.get('port', 6379)
            db = conn.get('db', 0)
            password = conn.get('password', None)
            
            return create_ai_database(
                backend='redis',
                redis_host=host,
                redis_port=port,
                redis_db=db,
                redis_password=password
            )
    
    elif backend == 'chromadb':
        persist_directory = config.get('persist_directory', 'chromadb_data')
        # Make absolute path
        if not Path(persist_directory).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            persist_directory = str(project_root / persist_directory)
        
        return create_ai_database(backend='chromadb', persist_directory=persist_directory)
    
    elif backend == 'faiss':
        index_path = config.get('index_path', 'faiss_data')
        dimension = config.get('dimension', 384)
        
        # Make absolute path
        if not Path(index_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            index_path = str(project_root / index_path)
        
        return create_ai_database(backend='faiss', index_path=index_path, dimension=dimension)

    elif backend == 'mongodb':
        conn = config.get('connection', {})
        use_mock = config.get('use_mock', False)
        if 'connection_string' in config:
            return create_ai_database(
                backend='mongodb',
                connection_string=config['connection_string'],
                mongo_use_mock=use_mock
            )
        host = conn.get('host', 'localhost')
        port = conn.get('port', 27017)
        database = conn.get('database', 'aidb')
        user = conn.get('user')
        password = conn.get('password')
        return create_ai_database(
            backend='mongodb',
            mongo_host=host,
            mongo_port=port,
            mongo_database=database,
            mongo_user=user,
            mongo_password=password,
            mongo_use_mock=use_mock
        )
    
    else:
        raise ValueError(f"Backend پشتیبانی نمی‌شود: {backend}")


def get_default_database():
    """
    دریافت پایگاه داده پیش‌فرض
    
    ترتیب جستجو:
    1. فایل تنظیمات ai_db_config.yaml در پوشه پروژه
    2. متغیر محیطی AI_DB_CONFIG
    3. SQLite پیش‌فرض (ai_models.db)
    
    Returns:
        نمونه از کلاس پایگاه داده
    """
    import os
    
    # Check for config file in project root
    project_root = Path(__file__).parent.parent.parent  # e:\3d
    config_file = project_root / 'ai_db_config.yaml'
    
    if config_file.exists():
        return create_ai_database_from_config(str(config_file))
    
    # Check environment variable
    config_path = os.environ.get('AI_DB_CONFIG')
    if config_path and Path(config_path).exists():
        return create_ai_database_from_config(config_path)
    
    # Default: SQLite
    db_path = str(project_root / 'ai_models.db')
    return create_ai_database(backend='sqlite', db_path=db_path)


# Convenience singleton
_default_db = None

def ai_db():
    """دریافت نمونه singleton از پایگاه داده پیش‌فرض"""
    global _default_db
    if _default_db is None:
        _default_db = get_default_database()
    return _default_db
