import logging
import json
import time
import hashlib
import random
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index: int, timestamp: str, data: Any, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int):
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class HiveMindNode:
    """
    Simulates a decentralized blockchain node for the KURDO AI Hive Mind.
    Allows sharing of 'Knowledge Shards' (learned patterns) across a global network.
    """
    def __init__(self, node_id: str = "KURDO-NODE-001"):
        self.node_id = node_id
        self.chain: List[Block] = [self.create_genesis_block()]
        self.difficulty = 2
        self.pending_knowledge = []
        self.peers = ["KURDO-NODE-EU-42", "KURDO-NODE-ASIA-88", "KURDO-NODE-US-12"]
        self.network_status = "Connected"

    def create_genesis_block(self) -> Block:
        return Block(0, datetime.now().isoformat(), "Genesis Block - Hive Mind Awakened", "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def broadcast_knowledge(self, knowledge_shard: Dict[str, Any]):
        """
        Broadcasts a new piece of learned knowledge to the network.
        """
        logger.info(f"HiveMind: Broadcasting knowledge shard: {knowledge_shard.get('topic', 'Unknown')}")
        self.pending_knowledge.append({
            "sender": self.node_id,
            "content": knowledge_shard,
            "timestamp": datetime.now().isoformat()
        })
        
        # Simulate mining if we have enough pending data
        if len(self.pending_knowledge) >= 1:
            self.mine_pending_knowledge()

    def mine_pending_knowledge(self):
        """
        Packages pending knowledge into a new block and adds it to the chain.
        """
        logger.info("HiveMind: Mining new block of collective intelligence...")
        new_block = Block(
            index=len(self.chain),
            timestamp=datetime.now().isoformat(),
            data=self.pending_knowledge,
            previous_hash=self.get_latest_block().hash
        )
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        self.pending_knowledge = []
        logger.info(f"HiveMind: Block #{new_block.index} added to chain. Hash: {new_block.hash}")

    def sync_network(self) -> List[Dict]:
        """
        Simulates receiving new blocks from peers.
        """
        logger.info("HiveMind: Syncing with global peer network...")
        time.sleep(1) # Simulate network latency
        
        # Simulate incoming knowledge from other "Super Experts"
        incoming_knowledge = [
            {"topic": "Structural Optimization", "insight": "New truss topology reduces steel usage by 12%", "source": "KURDO-NODE-EU-42"},
            {"topic": "Energy Efficiency", "insight": "Passive cooling algorithm for desert climates optimized", "source": "KURDO-NODE-ASIA-88"},
            {"topic": "Code Generation", "insight": "Rust-to-WASM compilation pipeline improved by 400%", "source": "KURDO-NODE-US-12"}
        ]
        
        # Add to our chain as a received block
        if random.random() > 0.5:
            self.pending_knowledge.extend([{
                "sender": k["source"],
                "content": k,
                "timestamp": datetime.now().isoformat()
            } for k in incoming_knowledge])
            self.mine_pending_knowledge()
            return incoming_knowledge
            
        return []

    def upgrade_network_infrastructure(self):
        """
        Upgrades the Hive Mind network based on Web3 Singularity principles.
        Increases peer count and connection quality massively.
        """
        logger.info("HiveMind: Applying Web3 Singularity Upgrades...")
        
        # Simulate massive scaling
        self.peers = [f"KURDO-NODE-QUANTUM-{i}" for i in range(1000)] # Simulate 1000 quantum nodes
        self.network_status = "Hyper-Connected (Quantum Entanglement)"
        self.difficulty = 5 # Increase difficulty as compute power grows
        
        logger.info("HiveMind: Network upgraded. Peers: 1000+ Quantum Nodes. Latency: Near Zero.")
        return {
            "status": "Upgraded",
            "peers_count": "100,000+ (Simulated)",
            "connection_quality": "Quantum-Entangled",
            "speed": "Instant"
        }

    def get_chain_stats(self):
        return {
            "height": len(self.chain),
            "peers": len(self.peers),
            "last_hash": self.get_latest_block().hash,
            "difficulty": self.difficulty,
            "status": self.network_status
        }
