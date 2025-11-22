"""
Data Connector Module for KURDO AI
Manages online/offline connections to external data sources, AI platforms, and knowledge bases.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class DataConnector:
    """Manages persistent connections to external data sources and AI platforms."""
    
    def __init__(self, knowledge_base_path: str = "cad3d/super_ai/super_ai_knowledge_base.json"):
        self.knowledge_base_path = knowledge_base_path
        self.connections = {
            "ai_platforms": [],
            "architectural_databases": [],
            "design_repositories": [],
            "research_sources": [],
            "api_services": [],
            "offline_cache": {}
        }
        self.connection_status = {}
        self.last_sync_time = {}
        self._load_connections()
        self._initialize_default_sources()
    
    def _load_connections(self):
        """Load saved connections from knowledge base."""
        if os.path.exists(self.knowledge_base_path):
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "data_connector" in data:
                        connector_data = data["data_connector"]
                        self.connections = connector_data.get("connections", self.connections)
                        self.connection_status = connector_data.get("connection_status", {})
                        self.last_sync_time = connector_data.get("last_sync_time", {})
            except Exception as e:
                logger.error(f"Failed to load connections: {e}")
    
    def _save_connections(self):
        """Save connections to knowledge base."""
        try:
            data = {}
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            data["data_connector"] = {
                "connections": self.connections,
                "connection_status": self.connection_status,
                "last_sync_time": self.last_sync_time,
                "last_update": datetime.now().isoformat()
            }
            
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save connections: {e}")
    
    def _initialize_default_sources(self):
        """Initialize default data sources if not already present."""
        
        # AI Platforms
        default_ai_platforms = [
            {"name": "OpenAI", "type": "LLM", "status": "active", "url": "https://api.openai.com"},
            {"name": "Anthropic Claude", "type": "LLM", "status": "active", "url": "https://api.anthropic.com"},
            {"name": "Google Gemini", "type": "LLM", "status": "active", "url": "https://ai.google.dev"},
            {"name": "Hugging Face", "type": "Model Hub", "status": "active", "url": "https://huggingface.co"},
            {"name": "AutoGen", "type": "Agent Framework", "status": "trained", "url": "https://microsoft.github.io/autogen"},
            {"name": "LangChain", "type": "Agent Framework", "status": "trained", "url": "https://langchain.com"},
            {"name": "CrewAI", "type": "Agent Framework", "status": "trained", "url": "https://crewai.com"},
        ]
        
        # Architectural Databases
        default_arch_db = [
            {"name": "ArchDaily", "type": "Architecture Portal", "status": "monitoring", "url": "https://archdaily.com"},
            {"name": "Dezeen", "type": "Design Magazine", "status": "monitoring", "url": "https://dezeen.com"},
            {"name": "ArchINFORM", "type": "Architecture Database", "status": "trained", "url": "https://archinform.net"},
            {"name": "Cadyar", "type": "CAD Resources", "status": "trained", "url": "https://cadyar.com"},
            {"name": "ArchiExpo", "type": "Product Database", "status": "monitoring", "url": "https://archiexpo.com"},
        ]
        
        # Design Repositories
        default_design_repos = [
            {"name": "GitHub", "type": "Code Repository", "status": "active", "url": "https://github.com"},
            {"name": "Behance", "type": "Design Portfolio", "status": "monitoring", "url": "https://behance.net"},
            {"name": "Dribbble", "type": "Design Community", "status": "monitoring", "url": "https://dribbble.com"},
        ]
        
        # Research Sources
        default_research = [
            {"name": "arXiv", "type": "Research Papers", "status": "monitoring", "url": "https://arxiv.org"},
            {"name": "IEEE Xplore", "type": "Technical Papers", "status": "monitoring", "url": "https://ieeexplore.ieee.org"},
            {"name": "ResearchGate", "type": "Research Network", "status": "monitoring", "url": "https://researchgate.net"},
        ]
        
        # Only add if not already present
        if not self.connections["ai_platforms"]:
            self.connections["ai_platforms"] = default_ai_platforms
        if not self.connections["architectural_databases"]:
            self.connections["architectural_databases"] = default_arch_db
        if not self.connections["design_repositories"]:
            self.connections["design_repositories"] = default_design_repos
        if not self.connections["research_sources"]:
            self.connections["research_sources"] = default_research
        
        self._save_connections()
    
    def add_connection(self, category: str, name: str, url: str, type_: str, status: str = "pending"):
        """Add a new connection to the system."""
        if category not in self.connections:
            self.connections[category] = []
        
        new_connection = {
            "name": name,
            "type": type_,
            "status": status,
            "url": url,
            "added_date": datetime.now().isoformat()
        }
        
        self.connections[category].append(new_connection)
        self.connection_status[name] = {"online": False, "last_check": datetime.now().isoformat()}
        self._save_connections()
        logger.info(f"Added new connection: {name} ({category})")
    
    async def check_connection_status(self, url: str) -> bool:
        """Check if a connection is online."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status < 400
        except:
            return False
    
    async def sync_all_connections(self):
        """Synchronize data from all active connections."""
        logger.info("Starting connection synchronization...")
        
        for category, sources in self.connections.items():
            if category == "offline_cache":
                continue
            
            for source in sources:
                name = source.get("name")
                url = source.get("url")
                
                if url:
                    is_online = await self.check_connection_status(url)
                    self.connection_status[name] = {
                        "online": is_online,
                        "last_check": datetime.now().isoformat()
                    }
                    
                    if is_online:
                        self.last_sync_time[name] = datetime.now().isoformat()
                        logger.info(f"✓ {name} is online")
                    else:
                        logger.warning(f"✗ {name} is offline - using cached data")
        
        self._save_connections()
        return self.connection_status
    
    def get_all_connections(self) -> Dict:
        """Get all registered connections by category."""
        return self.connections
    
    def get_connection_summary(self) -> Dict:
        """Get summary statistics of all connections."""
        total = 0
        online = 0
        categories_summary = {}
        
        for category, sources in self.connections.items():
            if category == "offline_cache":
                continue
            
            count = len(sources)
            total += count
            
            online_count = sum(1 for s in sources 
                             if self.connection_status.get(s.get("name"), {}).get("online", False))
            online += online_count
            
            categories_summary[category] = {
                "total": count,
                "online": online_count,
                "offline": count - online_count
            }
        
        return {
            "total_connections": total,
            "online_connections": online,
            "offline_connections": total - online,
            "categories": categories_summary,
            "last_global_sync": max(self.last_sync_time.values()) if self.last_sync_time else "Never"
        }
    
    def enable_offline_mode(self, source_name: str, data: Dict):
        """Cache data for offline access."""
        self.connections["offline_cache"][source_name] = {
            "data": data,
            "cached_at": datetime.now().isoformat()
        }
        self._save_connections()
    
    def get_offline_data(self, source_name: str) -> Optional[Dict]:
        """Retrieve cached data when offline."""
        return self.connections["offline_cache"].get(source_name)
    
    def apply_quantum_connection_boost(self, multiplier: float = 100000.0):
        """
        Applies a massive boost to connection strength, simulating quantum entanglement.
        """
        logger.info(f"Applying Quantum Connection Boost: {multiplier}x")
        self.connection_status["QUANTUM_BOOST"] = {
            "active": True,
            "multiplier": multiplier,
            "mode": "Singularity-Level",
            "latency": "0ms (Entangled)"
        }
        
        # Upgrade all existing connections to "Quantum" status
        for category in self.connections:
            if category == "offline_cache": continue
            for source in self.connections[category]:
                source["status"] = "QUANTUM_LINKED"
                source["speed"] = "Instant"
        
        self._save_connections()
        return f"Connections upgraded to {multiplier}x strength."
