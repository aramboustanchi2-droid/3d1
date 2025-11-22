from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import collections
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0
    source: str = "unknown"

class MemoryLayer(ABC):
    @abstractmethod
    def store(self, item: MemoryItem):
        pass

    @abstractmethod
    def retrieve(self, query: Any) -> List[MemoryItem]:
        pass

class WorkingMemory(MemoryLayer):
    """
    Ultra-fast, low capacity memory for immediate processing.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items = collections.deque(maxlen=capacity)

    def store(self, item: MemoryItem):
        self.items.append(item)
        logger.debug(f"Stored in Working Memory: {str(item.content)[:30]}...")

    def retrieve(self, query: Any) -> List[MemoryItem]:
        # Simple linear search
        return [i for i in self.items if str(query).lower() in str(i.content).lower()]

    def get_context_window(self) -> List[Any]:
        return [i.content for i in self.items]

class ShortTermMemory(MemoryLayer):
    """
    Recent history, session-based memory.
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.items = [] 

    def store(self, item: MemoryItem):
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items.pop(0) # FIFO

    def retrieve(self, query: Any) -> List[MemoryItem]:
        return [i for i in self.items if str(query).lower() in str(i.content).lower()]

class LongTermMemory(MemoryLayer):
    """
    Persistent storage for high-value information.
    """
    def __init__(self):
        self.storage = [] # Placeholder for database/vector store

    def store(self, item: MemoryItem):
        self.storage.append(item)
        # TODO: Implement persistence (JSON/SQLite/VectorDB)
        logger.info(f"Persisted to LongTerm Memory: {str(item.content)[:30]}...")

    def retrieve(self, query: Any) -> List[MemoryItem]:
        # Placeholder for semantic search
        return [i for i in self.storage if str(query).lower() in str(i.content).lower()]

class SuperAIMemory:
    def __init__(self):
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    def add_context(self, data: Any, importance: float = 0.5, source: str = "system"):
        item = MemoryItem(content=data, importance=importance, source=source)
        
        # Always goes to working memory
        self.working.store(item)
        
        # Promotion logic
        # If it's important enough, it goes to short term
        if importance > 0.3:
            self.short_term.store(item)
            
        # If it's critical, it goes to long term
        if importance > 0.8:
            self.long_term.store(item)

    def recall(self, query: str) -> Dict[str, List[MemoryItem]]:
        return {
            "working": self.working.retrieve(query),
            "short_term": self.short_term.retrieve(query),
            "long_term": self.long_term.retrieve(query)
        }
    
    def get_current_context(self) -> List[Any]:
        return self.working.get_context_window()
