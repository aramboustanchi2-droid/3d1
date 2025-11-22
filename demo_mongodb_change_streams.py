"""
MongoDB Change Streams - Demo Script
نمایش عملکرد Change Streams در MongoDB

This script demonstrates:
1. Start watching predictions collection in background
2. Insert sample predictions
3. Show real-time change notifications
4. Display statistics

Run:
    python demo_mongodb_change_streams.py

Requirements:
    - MongoDB server running (replica set or sharded cluster)
    - MONGO_URL environment variable set
    
Note: MongoDB Change Streams require replica set. For local testing:
    1. Start MongoDB as replica set: mongod --replSet rs0
    2. Initialize: mongo --eval "rs.initiate()"
    3. Set MONGO_URL: $env:MONGO_URL='mongodb://localhost:27017/?replicaSet=rs0'
"""
import os
import time
from datetime import datetime
from cad3d.super_ai.ai_db_factory import create_ai_database
from cad3d.super_ai.mongodb_change_streams import MongoChangeStreams


def custom_callback(change):
    """Custom callback to handle change events"""
    op = change.get('operationType', 'unknown')
    ns = change.get('ns', {})
    coll = ns.get('coll', 'unknown')
    
    print(f"\n🔔 Change Event:")
    print(f"   Operation: {op}")
    print(f"   Collection: {coll}")
    
    if op == 'insert':
        doc = change.get('fullDocument', {})
        print(f"   New Document ID: {doc.get('id', 'N/A')}")
        if coll == 'predictions':
            print(f"   Output: {doc.get('output', 'N/A')}")
            print(f"   Confidence: {doc.get('confidence', 'N/A')}")
    
    elif op == 'update':
        doc = change.get('fullDocument', {})
        updated_fields = change.get('updateDescription', {}).get('updatedFields', {})
        print(f"   Document ID: {doc.get('id', 'N/A') if doc else 'N/A'}")
        print(f"   Updated Fields: {list(updated_fields.keys())}")
    
    elif op == 'delete':
        doc_key = change.get('documentKey', {})
        print(f"   Deleted Document: {doc_key}")
    
    print()


def populate_with_changes(db, delay: float = 1.0):
    """Add data with delays to trigger change events"""
    print("\n[Data Generator] Starting to add sample data...\n")
    
    # Create model
    model_id = db.create_model(
        name="TestModel-ChangeStream",
        description="Model for change stream demo",
        architecture="ResNet50",
        framework="PyTorch",
        task_type="classification"
    )
    print(f"[Data Generator] ✓ Model created (ID: {model_id})")
    time.sleep(delay)
    
    # Create version
    version_id = db.create_model_version(
        model_id=model_id,
        version="1.0.0",
        checkpoint_path="/models/test_v1.pth",
        config={"lr": 0.001}
    )
    print(f"[Data Generator] ✓ Version created (ID: {version_id})")
    time.sleep(delay)
    
    # Create predictions
    for i in range(5):
        db.log_prediction(
            model_version_id=version_id,
            input_data={"image_id": i},
            output_data={"class": f"class_{i % 3}", "score": 0.8 + i * 0.02},
            confidence=0.85 + i * 0.01,
            inference_time_ms=10.0 + i
        )
        print(f"[Data Generator] ✓ Prediction {i+1} logged")
        time.sleep(delay)
    
    print("\n[Data Generator] All data added!\n")


def main():
    print("\n" + "="*70)
    print("MongoDB Change Streams Demo")
    print("="*70)
    
    # Check environment
    mongo_url = os.getenv('MONGO_URL')
    
    if not mongo_url:
        print("\n❌ Error: MONGO_URL not set")
        print("\nSet environment variable:")
        print("  $env:MONGO_URL='mongodb://localhost:27017/?replicaSet=rs0'")
        print("\n⚠️  Note: Change Streams require MongoDB Replica Set")
        print("\nSetup MongoDB Replica Set (local):")
        print("  1. Start MongoDB: mongod --replSet rs0 --port 27017")
        print("  2. Initialize: mongo --eval \"rs.initiate()\"")
        print("  3. Verify: mongo --eval \"rs.status()\"")
        return 1
    
    try:
        # Connect to MongoDB
        print(f"\nConnecting to MongoDB: {mongo_url[:50]}...")
        db = create_ai_database(
            backend='mongodb',
            connection_string=mongo_url,
            database='aidb_changestream_demo'
        )
        print("✓ MongoDB connected\n")
        
        # Verify replica set (Change Streams requirement)
        try:
            # Check if replica set is configured
            server_info = db._db.client.server_info()
            print(f"MongoDB Version: {server_info.get('version', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Warning: {e}")
        
        print("\n" + "="*70)
        print("[1] Setting up Change Stream Watcher")
        print("="*70)
        
        # Create watcher with custom callback
        watcher = MongoChangeStreams(db, callback=custom_callback)
        
        # Start watching predictions collection in background
        print("\nStarting watcher on 'predictions' collection...")
        watcher_thread = watcher.watch_collection_async(
            collection_name='predictions',
            operation_types=['insert', 'update', 'delete'],
            full_document='updateLookup'
        )
        
        print("✓ Watcher started (running in background)\n")
        time.sleep(1)  # Give watcher time to initialize
        
        print("\n" + "="*70)
        print("[2] Generating Sample Data (will trigger change events)")
        print("="*70)
        
        # Add data (this will trigger change events)
        populate_with_changes(db, delay=1.5)
        
        # Wait a bit to ensure all changes are captured
        print("\n[Main] Waiting for change events to be processed...")
        time.sleep(3)
        
        # Stop watcher
        print("\n[Main] Stopping watcher...")
        watcher.stop()
        watcher_thread.join(timeout=2)
        
        # Show statistics
        print("\n" + "="*70)
        print("Statistics")
        print("="*70)
        
        stats = watcher.get_stats()
        print(f"\nChange Events Captured:")
        print(f"  Total: {stats['total_changes']}")
        print(f"  Inserts: {stats['inserts']}")
        print(f"  Updates: {stats['updates']}")
        print(f"  Deletes: {stats['deletes']}")
        print(f"  Errors: {stats['errors']}")
        
        # Verify data in database
        print(f"\nDatabase State:")
        db_stats = db.get_statistics()
        print(f"  Models: {db_stats['models']}")
        print(f"  Versions: {db_stats['model_versions']}")
        print(f"  Predictions: {db_stats['predictions']}")
        
        print("\n✅ Demo completed successfully!")
        print("\nNote: Change Streams enable real-time data monitoring")
        print("      Perfect for dashboards, event-driven architectures, and sync systems")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        # Check if it's a replica set error
        error_str = str(e).lower()
        if 'repl' in error_str or 'replica' in error_str or 'not master' in error_str:
            print("\n💡 Hint: Change Streams require MongoDB Replica Set")
            print("\nQuick Setup (local development):")
            print("  # Stop MongoDB if running")
            print("  mongod --replSet rs0 --port 27017 --dbpath C:\\data\\db")
            print("  # In another terminal:")
            print("  mongo --eval \"rs.initiate()\"")
            print("  # Wait 10 seconds, then:")
            print("  mongo --eval \"rs.status()\"")
            print("\n  $env:MONGO_URL='mongodb://localhost:27017/?replicaSet=rs0'")
        
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
