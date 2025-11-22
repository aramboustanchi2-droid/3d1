"""
MongoDB Change Streams - Real-time Data Change Monitoring
نظارت بر تغییرات داده‌ها به‌صورت Real-time در MongoDB

Purpose:
- Monitor insert/update/delete operations on MongoDB collections in real-time
- Perfect for:
  * Live dashboards (show new predictions as they arrive)
  * Event-driven architectures (trigger actions on data changes)
  * Replication/sync to other systems
  * Audit logging and compliance
  * Cache invalidation

Features:
- Watch specific collections or entire database
- Filter changes by operation type (insert, update, delete, replace)
- Resume capability (resume_token) for fault tolerance
- Async/threaded execution for non-blocking monitoring
- Callback-based or queue-based event handling
- Full document support (get complete document after change)

Requirements:
- MongoDB 3.6+ (Change Streams feature)
- Replica Set or Sharded Cluster (NOT standalone MongoDB)
- For local development: use mongomock or run MongoDB replica set

Installation:
    pip install pymongo

Usage:
    # Watch all changes on predictions collection
    python -m cad3d.super_ai.mongodb_change_streams --collection predictions
    
    # Watch specific operations
    python -m cad3d.super_ai.mongodb_change_streams --collection models --operations insert update
    
    # Custom callback
    from cad3d.super_ai.mongodb_change_streams import MongoChangeStreams
    
    def on_change(change):
        print(f"New prediction: {change['fullDocument']}")
    
    watcher = MongoChangeStreams(db, callback=on_change)
    watcher.watch_collection('predictions')
"""
from __future__ import annotations
import os
import time
import argparse
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from queue import Queue
from threading import Thread, Event

try:
    import pymongo
    from pymongo.errors import PyMongoError
except ImportError:
    pymongo = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MongoChangeStreams:
    """
    MongoDB Change Streams Wrapper
    
    Monitors real-time changes (insert/update/delete) on MongoDB collections.
    """
    
    def __init__(
        self,
        db_instance,
        callback: Optional[Callable[[Dict], None]] = None,
        use_queue: bool = False,
        queue_size: int = 1000
    ):
        """
        Args:
            db_instance: AIModelDatabaseMongo instance (or raw pymongo.database.Database)
            callback: Function to call on each change event (change_dict)
            use_queue: If True, put changes in a queue instead of callback
            queue_size: Maximum queue size (if use_queue=True)
        """
        # Get pymongo database
        if hasattr(db_instance, '_db'):
            self.db = db_instance._db
        else:
            self.db = db_instance
        
        self.callback = callback or self._default_callback
        self.use_queue = use_queue
        self.change_queue: Queue = Queue(maxsize=queue_size) if use_queue else None
        
        # Control flags
        self._stop_event = Event()
        self._threads: List[Thread] = []
        
        # Statistics
        self.stats = {
            'total_changes': 0,
            'inserts': 0,
            'updates': 0,
            'deletes': 0,
            'replaces': 0,
            'errors': 0
        }
    
    def _default_callback(self, change: Dict):
        """Default callback: just log the change"""
        op = change.get('operationType', 'unknown')
        ns = change.get('ns', {})
        coll = ns.get('coll', 'unknown')
        doc_id = change.get('documentKey', {}).get('_id', 'unknown')
        
        logger.info(f"Change detected: {op} on {coll} (doc _id: {doc_id})")
    
    def _process_change(self, change: Dict):
        """Process a single change event"""
        try:
            op = change.get('operationType', 'unknown')
            self.stats['total_changes'] += 1
            
            if op == 'insert':
                self.stats['inserts'] += 1
            elif op == 'update':
                self.stats['updates'] += 1
            elif op == 'delete':
                self.stats['deletes'] += 1
            elif op == 'replace':
                self.stats['replaces'] += 1
            
            # Deliver event
            if self.use_queue:
                try:
                    self.change_queue.put(change, block=False)
                except:
                    logger.warning("Change queue full, dropping event")
            else:
                self.callback(change)
        
        except Exception as e:
            logger.error(f"Error processing change: {e}")
            self.stats['errors'] += 1
    
    def watch_collection(
        self,
        collection_name: str,
        operation_types: Optional[List[str]] = None,
        full_document: str = 'updateLookup',
        resume_after: Optional[Dict] = None
    ):
        """
        Watch changes on a specific collection (blocking call)
        
        Args:
            collection_name: Collection to watch ('models', 'predictions', etc.)
            operation_types: Filter by operations ['insert', 'update', 'delete', 'replace']
            full_document: 'updateLookup' to get full doc on updates, 'default' for delta only
            resume_after: Resume token for fault tolerance (from previous watch)
        """
        logger.info(f"Starting change stream on collection: {collection_name}")
        
        # Build pipeline filter
        pipeline = []
        if operation_types:
            pipeline.append({'$match': {'operationType': {'$in': operation_types}}})
        
        collection = self.db[collection_name]
        
        try:
            options = {'full_document': full_document}
            if resume_after:
                options['resume_after'] = resume_after
            
            with collection.watch(pipeline, **options) as stream:
                logger.info(f"Watching {collection_name}... (Press Ctrl+C to stop)")
                
                for change in stream:
                    if self._stop_event.is_set():
                        logger.info(f"Stop signal received for {collection_name}")
                        break
                    
                    self._process_change(change)
                    
                    # Store resume token for fault tolerance
                    resume_token = stream.resume_token
        
        except PyMongoError as e:
            logger.error(f"MongoDB error: {e}")
            self.stats['errors'] += 1
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.stats['errors'] += 1
    
    def watch_collection_async(
        self,
        collection_name: str,
        operation_types: Optional[List[str]] = None,
        full_document: str = 'updateLookup'
    ) -> Thread:
        """
        Watch collection in background thread (non-blocking)
        
        Returns:
            Thread object (call .join() to wait, or just let it run)
        """
        thread = Thread(
            target=self.watch_collection,
            args=(collection_name, operation_types, full_document),
            daemon=True
        )
        thread.start()
        self._threads.append(thread)
        return thread
    
    def watch_multiple_collections(
        self,
        collection_names: List[str],
        operation_types: Optional[List[str]] = None,
        full_document: str = 'updateLookup'
    ):
        """
        Watch multiple collections simultaneously (each in separate thread)
        
        Args:
            collection_names: List of collections to watch
            operation_types: Filter by operations
            full_document: Full document mode
        """
        logger.info(f"Starting {len(collection_names)} change streams...")
        
        for coll_name in collection_names:
            self.watch_collection_async(coll_name, operation_types, full_document)
        
        # Wait for all threads
        try:
            for thread in self._threads:
                thread.join()
        except KeyboardInterrupt:
            logger.info("Stopping all watchers...")
            self.stop()
    
    def watch_database(
        self,
        operation_types: Optional[List[str]] = None,
        full_document: str = 'updateLookup'
    ):
        """
        Watch all collections in the database (blocking call)
        
        Args:
            operation_types: Filter by operations
            full_document: Full document mode
        """
        logger.info("Starting change stream on entire database")
        
        pipeline = []
        if operation_types:
            pipeline.append({'$match': {'operationType': {'$in': operation_types}}})
        
        try:
            with self.db.watch(pipeline, full_document=full_document) as stream:
                logger.info("Watching entire database... (Press Ctrl+C to stop)")
                
                for change in stream:
                    if self._stop_event.is_set():
                        logger.info("Stop signal received")
                        break
                    
                    self._process_change(change)
        
        except PyMongoError as e:
            logger.error(f"MongoDB error: {e}")
            self.stats['errors'] += 1
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.stats['errors'] += 1
    
    def stop(self):
        """Stop all watchers"""
        logger.info("Stopping change stream watchers...")
        self._stop_event.set()
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics of processed changes"""
        return self.stats.copy()
    
    def get_changes_from_queue(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get next change from queue (if use_queue=True)
        
        Args:
            timeout: Block for N seconds, None = block forever, 0 = non-blocking
        
        Returns:
            Change dict or None if queue empty
        """
        if not self.use_queue or not self.change_queue:
            raise RuntimeError("Queue mode not enabled (use_queue=False)")
        
        try:
            return self.change_queue.get(timeout=timeout)
        except:
            return None


def main():
    parser = argparse.ArgumentParser(description='MongoDB Change Streams Monitor')
    parser.add_argument('--mongo-url', type=str, default=None, help='MongoDB connection URL')
    parser.add_argument('--database', type=str, default='aidb', help='Database name (default: aidb)')
    parser.add_argument('--collection', type=str, default=None, help='Collection to watch (if not specified, watches all)')
    parser.add_argument('--operations', nargs='+', default=None, help='Filter operations: insert update delete replace')
    parser.add_argument('--full-document', type=str, default='updateLookup', choices=['default', 'updateLookup'], help='Full document mode')
    args = parser.parse_args()
    
    # Get connection string
    mongo_url = args.mongo_url or os.getenv('MONGO_URL')
    
    if not mongo_url:
        logger.error("MongoDB URL not provided. Set MONGO_URL environment variable or use --mongo-url")
        logger.info("\nExample:")
        logger.info("  $env:MONGO_URL='mongodb://localhost:27017'")
        logger.info("  python -m cad3d.super_ai.mongodb_change_streams --collection predictions")
        return 1
    
    try:
        # Import AI DB factory
        from cad3d.super_ai.ai_db_factory import create_ai_database
        
        logger.info(f"Connecting to MongoDB: {mongo_url[:30]}...")
        db = create_ai_database(backend='mongodb', connection_string=mongo_url, database=args.database)
        
        # Create change stream watcher
        watcher = MongoChangeStreams(db)
        
        # Watch collection or entire database
        if args.collection:
            watcher.watch_collection(
                collection_name=args.collection,
                operation_types=args.operations,
                full_document=args.full_document
            )
        else:
            watcher.watch_database(
                operation_types=args.operations,
                full_document=args.full_document
            )
        
        # Print statistics
        logger.info("\nFinal Statistics:")
        stats = watcher.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        return 0
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Ensure you have installed: pip install pymongo")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
