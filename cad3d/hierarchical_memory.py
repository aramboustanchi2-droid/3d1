"""Hierarchical memory utilities for the CAD3D universal engineer system.

This module adds *non-breaking* helpers around the existing project to
organize memory into multiple layers:

- Feature-level memory (visual/low-level features summary)
- Graph-level memory (CADGraph snapshots and index)
- Semantic-level memory (model decisions and analysis reports)
- Learning-history memory (training runs and metrics)
- Config memory (engineering rules and system config)

All classes are lightweight facades over simple JSON/dir structures so that
existing code paths remain unchanged. Other modules can *optionally* use these
helpers to persist and query information in a structured way.

Enhanced with:
- LRU caching for fast repeated access
- Query systems for advanced filtering
- Memory consolidation for space optimization
- Usage analytics for monitoring
- Lifecycle management for cleanup
"""

from __future__ import annotations

import json
import os
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from .cad_graph import CADGraph


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_ROOT = Path("memory")
DEFAULT_CACHE_SIZE = 100
DEFAULT_TTL_SECONDS = 3600  # 1 hour


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _hash_key(*args: Any) -> str:
    """Generate a hash key from arguments."""
    key_str = str(args)
    return hashlib.md5(key_str.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LRU Cache with TTL
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    value: Any
    timestamp: float
    access_count: int = 0


class LRUCache:
    """LRU cache with time-to-live support."""

    def __init__(self, max_size: int = DEFAULT_CACHE_SIZE, ttl: int = DEFAULT_TTL_SECONDS):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            del self.cache[key]
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.access_count += 1
        self.hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)

        self.cache[key] = CacheEntry(
            value=value,
            timestamp=time.time()
        )

    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        if key in self.cache:
            del self.cache[key]

    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }


# ---------------------------------------------------------------------------
# Feature-level memory (vision & low-level features)
# ---------------------------------------------------------------------------

@dataclass
class FeatureMemoryRecord:
    project_id: str
    input_id: str
    model_id: str
    embedding_dim: int
    stats: Dict[str, float]
    source_file: Optional[str] = None
    timestamp: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if not data.get("timestamp"):
            data["timestamp"] = _now_iso()
        return data
    
    def matches_filter(self, **filters) -> bool:
        """Check if record matches given filters."""
        for key, value in filters.items():
            if key == "tags":
                if not any(tag in self.tags for tag in value):
                    return False
            elif hasattr(self, key):
                if getattr(self, key) != value:
                    return False
        return True


class FeatureLevelMemory:
    """Persists lightweight summaries of feature embeddings.

    This does **not** store raw tensors; it stores small JSON descriptors so
    that higher-level systems can reason about the distribution of features
    without loading large arrays.
    
    Enhanced with caching and query capabilities.
    """

    def __init__(
        self, 
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ) -> None:
        self.root = root / "feature_level"
        _ensure_dir(self.root)
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None
        self.stats_file = self.root / "_stats.json"
        self._load_stats()

    def _load_stats(self) -> None:
        """Load memory usage statistics."""
        if self.stats_file.exists():
            with self.stats_file.open("r", encoding="utf-8") as f:
                self.stats = json.load(f)
        else:
            self.stats = {
                "total_records": 0,
                "total_size_mb": 0.0,
                "last_update": _now_iso(),
                "projects": {}
            }

    def _update_stats(self, project_id: str, size_bytes: int) -> None:
        """Update statistics."""
        self.stats["total_records"] += 1
        self.stats["total_size_mb"] += size_bytes / (1024 ** 2)
        self.stats["last_update"] = _now_iso()
        
        if project_id not in self.stats["projects"]:
            self.stats["projects"][project_id] = {"count": 0, "size_mb": 0.0}
        
        self.stats["projects"][project_id]["count"] += 1
        self.stats["projects"][project_id]["size_mb"] += size_bytes / (1024 ** 2)
        
        with self.stats_file.open("w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

    def save_record(self, record: FeatureMemoryRecord) -> Path:
        """Save a feature record and return the created path."""
        project_dir = self.root / record.project_id
        _ensure_dir(project_dir)
        filename = f"feature_{record.input_id}_{record.model_id}.json"
        path = project_dir / filename
        
        data = record.to_dict()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        # Update cache
        if self.cache:
            cache_key = _hash_key(record.project_id, record.input_id, record.model_id)
            self.cache.set(cache_key, record)
        
        # Update stats
        self._update_stats(record.project_id, path.stat().st_size)
        
        return path

    def load_record(
        self, 
        project_id: str, 
        input_id: str, 
        model_id: str
    ) -> Optional[FeatureMemoryRecord]:
        """Load a feature record with caching."""
        cache_key = _hash_key(project_id, input_id, model_id)
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Load from disk
        filename = f"feature_{input_id}_{model_id}.json"
        path = self.root / project_id / filename
        
        if not path.exists():
            return None
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        record = FeatureMemoryRecord(**data)
        
        # Update cache
        if self.cache:
            self.cache.set(cache_key, record)
        
        return record

    def query_records(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[FeatureMemoryRecord]:
        """Query records with filters."""
        results = []
        
        # Determine search path
        if project_id:
            project_dirs = [self.root / project_id]
        else:
            project_dirs = [d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("_")]
        
        for project_dir in project_dirs:
            if not project_dir.exists():
                continue
                
            for file_path in project_dir.glob("feature_*.json"):
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                record = FeatureMemoryRecord(**data)
                
                # Apply filters
                if model_id and record.model_id != model_id:
                    continue
                if tags and not any(tag in record.tags for tag in tags):
                    continue
                
                results.append(record)
                
                if len(results) >= limit:
                    return results
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = dict(self.stats)
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        return stats

    def consolidate(self, older_than_days: int = 30) -> Dict[str, int]:
        """Consolidate old records into archive.
        
        Returns count of archived records.
        """
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        archive_dir = self.root / "_archive"
        _ensure_dir(archive_dir)
        
        archived = 0
        
        for project_dir in self.root.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("_"):
                continue
            
            for file_path in project_dir.glob("feature_*.json"):
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mtime < cutoff:
                    # Move to archive
                    archive_project = archive_dir / project_dir.name
                    _ensure_dir(archive_project)
                    
                    new_path = archive_project / file_path.name
                    file_path.rename(new_path)
                    archived += 1
        
        return {"archived_count": archived}


# ---------------------------------------------------------------------------
# Graph-level memory (CADGraph snapshots and index)
# ---------------------------------------------------------------------------

@dataclass
class GraphIndexEntry:
    project_id: str
    version: str
    file: str
    created_at: str
    analyzers: List[str]
    notes: str = ""
    size_mb: float = 0.0
    node_count: int = 0
    edge_count: int = 0
    tags: List[str] = field(default_factory=list)


class GraphLevelMemory:
    """Manages persisted CADGraph snapshots and a simple project index.

    Graphs themselves are stored via ``CADGraph.save_to_json``; this class
    only coordinates directory layout and an index file per project.
    
    Enhanced with versioning, diff tracking, and query support.
    """

    def __init__(
        self, 
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ) -> None:
        self.root = root / "graph_level"
        self.projects_dir = self.root / "projects"
        _ensure_dir(self.projects_dir)
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None

    def _project_index_path(self, project_id: str) -> Path:
        return self.projects_dir / f"{project_id}_index.json"

    def save_graph(
        self,
        project_id: str,
        version: str,
        graph: CADGraph,
        analyzers: Optional[List[str]] = None,
        notes: str = "",
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Persist a CADGraph snapshot and update the project index.

        This is a thin wrapper; existing code can still call
        ``graph.save_to_json`` directly. Using this helper simply adds
        bookkeeping on top.
        """
        project_dir = self.projects_dir / project_id
        _ensure_dir(project_dir)

        graph_filename = f"graph_{project_id}_{version}.json"
        graph_path = project_dir / graph_filename
        graph.save_to_json(str(graph_path))
        
        # Get graph stats
        node_count = len(graph.components)
        edge_count = len(graph.dependencies)
        size_mb = graph_path.stat().st_size / (1024 ** 2)

        entry = GraphIndexEntry(
            project_id=project_id,
            version=version,
            file=str(graph_path.relative_to(self.root)),
            created_at=_now_iso(),
            analyzers=analyzers or [],
            notes=notes,
            size_mb=size_mb,
            node_count=node_count,
            edge_count=edge_count,
            tags=tags or []
        )
        self._update_index(entry)
        
        # Cache the graph
        if self.cache:
            cache_key = _hash_key(project_id, version)
            self.cache.set(cache_key, graph)
        
        return graph_path

    def load_graph(
        self, 
        project_id: str, 
        version: str
    ) -> Optional[CADGraph]:
        """Load a graph with caching."""
        cache_key = _hash_key(project_id, version)
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Load from disk
        graph_filename = f"graph_{project_id}_{version}.json"
        graph_path = self.projects_dir / project_id / graph_filename
        
        if not graph_path.exists():
            return None
        
        graph = CADGraph.load_from_json(str(graph_path))
        
        # Update cache
        if self.cache:
            self.cache.set(cache_key, graph)
        
        return graph

    def _update_index(self, entry: GraphIndexEntry) -> None:
        index_path = self._project_index_path(entry.project_id)
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"project_id": entry.project_id, "graphs": []}

        data.setdefault("graphs", []).append(asdict(entry))

        _ensure_dir(index_path.parent)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_graphs(self, project_id: str) -> List[Dict[str, Any]]:
        """Return all index entries for a project (possibly empty)."""
        index_path = self._project_index_path(project_id)
        if not index_path.exists():
            return []
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("graphs", [])

    def get_latest_version(self, project_id: str) -> Optional[str]:
        """Get the latest version for a project."""
        graphs = self.list_graphs(project_id)
        if not graphs:
            return None
        
        # Sort by created_at
        sorted_graphs = sorted(graphs, key=lambda x: x["created_at"], reverse=True)
        return sorted_graphs[0]["version"]

    def compare_versions(
        self,
        project_id: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare two graph versions."""
        graph1 = self.load_graph(project_id, version1)
        graph2 = self.load_graph(project_id, version2)
        
        if not graph1 or not graph2:
            return {"error": "One or both versions not found"}
        
        return {
            "version1": version1,
            "version2": version2,
            "nodes_added": len(graph2.components) - len(graph1.components),
            "edges_added": len(graph2.dependencies) - len(graph1.dependencies),
            "node_count_v1": len(graph1.components),
            "node_count_v2": len(graph2.components),
            "edge_count_v1": len(graph1.dependencies),
            "edge_count_v2": len(graph2.dependencies)
        }

    def cleanup_old_versions(
        self,
        project_id: str,
        keep_latest: int = 5
    ) -> Dict[str, int]:
        """Keep only the latest N versions, archive the rest."""
        graphs = self.list_graphs(project_id)
        
        if len(graphs) <= keep_latest:
            return {"removed": 0}
        
        # Sort by created_at
        sorted_graphs = sorted(graphs, key=lambda x: x["created_at"], reverse=True)
        to_remove = sorted_graphs[keep_latest:]
        
        archive_dir = self.root / "_archive" / project_id
        _ensure_dir(archive_dir)
        
        removed = 0
        for graph_entry in to_remove:
            file_path = self.root / graph_entry["file"]
            if file_path.exists():
                archive_path = archive_dir / file_path.name
                file_path.rename(archive_path)
                removed += 1
        
        # Update index
        data = {"project_id": project_id, "graphs": sorted_graphs[:keep_latest]}
        index_path = self._project_index_path(project_id)
        with index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        return {"removed": removed}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        total_projects = 0
        total_graphs = 0
        total_size_mb = 0.0
        
        for project_dir in self.projects_dir.iterdir():
            if project_dir.is_file() and project_dir.suffix == ".json":
                total_projects += 1
                with project_dir.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    graphs = data.get("graphs", [])
                    total_graphs += len(graphs)
                    total_size_mb += sum(g.get("size_mb", 0.0) for g in graphs)
        
        stats = {
            "total_projects": total_projects,
            "total_graphs": total_graphs,
            "total_size_mb": total_size_mb
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


# ---------------------------------------------------------------------------
# Semantic-level memory (decisions and analyses)
# ---------------------------------------------------------------------------

class SemanticLevelMemory:
    """Stores model decisions and analysis summaries.

    Each decision file is a small JSON document linking a graph snapshot,
    a model version, and the predictions/warnings produced.
    
    Enhanced with search, filtering, and analytics.
    """

    def __init__(
        self, 
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ) -> None:
        self.root = root / "semantic_level" / "decisions"
        _ensure_dir(self.root)
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None

    def save_decision(
        self,
        project_id: str,
        graph_version: str,
        model_id: str,
        predictions: List[Dict[str, Any]],
        global_warnings: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Path:
        """Save a decision record and return its path."""
        project_dir = self.root / project_id
        _ensure_dir(project_dir)
        timestamp = _now_iso().replace(":", "_")
        filename = f"decision_{graph_version}_{model_id}_{timestamp}.json"
        path = project_dir / filename

        payload: Dict[str, Any] = {
            "project_id": project_id,
            "graph_version": graph_version,
            "model_id": model_id,
            "timestamp": _now_iso(),
            "predictions": predictions,
            "global_warnings": global_warnings or [],
            "confidence_score": confidence_score,
            "tags": tags or [],
            "prediction_count": len(predictions),
        }
        if extra:
            payload["extra"] = extra

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        # Cache
        if self.cache:
            cache_key = _hash_key(project_id, graph_version, model_id, timestamp)
            self.cache.set(cache_key, payload)
        
        return path

    def query_decisions(
        self,
        project_id: Optional[str] = None,
        model_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query decisions with filters."""
        results = []
        
        # Determine search path
        if project_id:
            project_dirs = [self.root / project_id]
        else:
            project_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        
        for project_dir in project_dirs:
            if not project_dir.exists():
                continue
            
            for file_path in sorted(project_dir.glob("decision_*.json"), reverse=True):
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Apply filters
                if model_id and data.get("model_id") != model_id:
                    continue
                if min_confidence and data.get("confidence_score", 0.0) < min_confidence:
                    continue
                if tags and not any(tag in data.get("tags", []) for tag in tags):
                    continue
                
                results.append(data)
                
                if len(results) >= limit:
                    return results
        
        return results

    def get_decision_trends(
        self,
        project_id: str,
        model_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Analyze decision trends over time."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        decisions = self.query_decisions(
            project_id=project_id,
            model_id=model_id,
            limit=1000
        )
        
        # Filter by time
        recent_decisions = [
            d for d in decisions
            if datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")) > cutoff
        ]
        
        if not recent_decisions:
            return {"error": "No recent decisions found"}
        
        # Calculate trends
        avg_predictions = sum(d.get("prediction_count", 0) for d in recent_decisions) / len(recent_decisions)
        avg_warnings = sum(len(d.get("global_warnings", [])) for d in recent_decisions) / len(recent_decisions)
        
        confidence_scores = [d.get("confidence_score", 0.0) for d in recent_decisions if d.get("confidence_score")]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "project_id": project_id,
            "model_id": model_id,
            "period_days": days,
            "total_decisions": len(recent_decisions),
            "avg_predictions_per_decision": avg_predictions,
            "avg_warnings_per_decision": avg_warnings,
            "avg_confidence_score": avg_confidence,
            "decision_frequency": len(recent_decisions) / days
        }

    def consolidate_decisions(
        self,
        project_id: str,
        older_than_days: int = 90
    ) -> Dict[str, int]:
        """Archive old decisions."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        archive_dir = self.root.parent / "_archive" / "decisions" / project_id
        _ensure_dir(archive_dir)
        
        project_dir = self.root / project_id
        if not project_dir.exists():
            return {"archived": 0}
        
        archived = 0
        for file_path in project_dir.glob("decision_*.json"):
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            if mtime < cutoff:
                new_path = archive_dir / file_path.name
                file_path.rename(new_path)
                archived += 1
        
        return {"archived": archived}

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        total_decisions = 0
        total_projects = 0
        
        for project_dir in self.root.iterdir():
            if project_dir.is_dir():
                total_projects += 1
                total_decisions += len(list(project_dir.glob("decision_*.json")))
        
        stats = {
            "total_projects": total_projects,
            "total_decisions": total_decisions
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


# ---------------------------------------------------------------------------
# Learning-history memory (training runs)
# ---------------------------------------------------------------------------

class LearningHistoryMemory:
    """Tracks training runs, metrics and associated checkpoints.
    
    Enhanced with experiment tracking, comparison, and best-model selection.
    """

    def __init__(
        self, 
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ) -> None:
        self.root = root / "learning_history"
        _ensure_dir(self.root)
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None
        self.leaderboard_path = self.root / "_leaderboard.json"

    def save_run(
        self, 
        run_id: str, 
        payload: Dict[str, Any],
        update_leaderboard: bool = True
    ) -> Path:
        """Persist a training run description under the given run_id."""
        payload = dict(payload)
        payload.setdefault("run_id", run_id)
        payload.setdefault("timestamp", _now_iso())

        path = self.root / f"training_run_{run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        # Cache
        if self.cache:
            cache_key = _hash_key(run_id)
            self.cache.set(cache_key, payload)
        
        # Update leaderboard
        if update_leaderboard and "metrics" in payload:
            self._update_leaderboard(run_id, payload)
        
        return path

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load a training run with caching."""
        cache_key = _hash_key(run_id)
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Load from disk
        path = self.root / f"training_run_{run_id}.json"
        if not path.exists():
            return None
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update cache
        if self.cache:
            self.cache.set(cache_key, data)
        
        return data

    def _update_leaderboard(self, run_id: str, payload: Dict[str, Any]) -> None:
        """Update the leaderboard with run metrics."""
        if self.leaderboard_path.exists():
            with self.leaderboard_path.open("r", encoding="utf-8") as f:
                leaderboard = json.load(f)
        else:
            leaderboard = {"runs": []}
        
        # Extract key metrics
        metrics = payload.get("metrics", {})
        entry = {
            "run_id": run_id,
            "timestamp": payload.get("timestamp"),
            "model_type": payload.get("model_type", "unknown"),
            "metrics": metrics,
            "checkpoint_path": payload.get("checkpoint_path"),
        }
        
        # Add to leaderboard
        leaderboard["runs"].append(entry)
        
        # Sort by primary metric (e.g., val_loss or mAP)
        primary_metric = metrics.get("val_loss") or metrics.get("mAP") or 0.0
        entry["primary_metric"] = primary_metric
        
        leaderboard["runs"].sort(key=lambda x: x.get("primary_metric", 0.0), reverse=True)
        
        # Keep top 100
        leaderboard["runs"] = leaderboard["runs"][:100]
        
        with self.leaderboard_path.open("w", encoding="utf-8") as f:
            json.dump(leaderboard, f, indent=2)

    def get_leaderboard(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top K runs from leaderboard."""
        if not self.leaderboard_path.exists():
            return []
        
        with self.leaderboard_path.open("r", encoding="utf-8") as f:
            leaderboard = json.load(f)
        
        return leaderboard.get("runs", [])[:top_k]

    def get_best_run(
        self,
        metric: str = "val_loss",
        mode: str = "min"
    ) -> Optional[Dict[str, Any]]:
        """Find the best run based on a metric."""
        runs = []
        
        for file_path in self.root.glob("training_run_*.json"):
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            if "metrics" in data and metric in data["metrics"]:
                runs.append(data)
        
        if not runs:
            return None
        
        # Sort by metric
        reverse = mode == "max"
        runs.sort(key=lambda x: x["metrics"].get(metric, float('inf') if mode == "min" else float('-inf')), reverse=reverse)
        
        return runs[0]

    def compare_runs(
        self,
        run_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple training runs."""
        runs_data = []
        
        for run_id in run_ids:
            run = self.load_run(run_id)
            if run:
                runs_data.append(run)
        
        if not runs_data:
            return {"error": "No runs found"}
        
        # Collect all unique metrics
        all_metrics = set()
        for run in runs_data:
            all_metrics.update(run.get("metrics", {}).keys())
        
        comparison = {
            "run_ids": run_ids,
            "metrics": {}
        }
        
        for metric in all_metrics:
            values = [run.get("metrics", {}).get(metric) for run in runs_data]
            comparison["metrics"][metric] = {
                "values": values,
                "best_run": run_ids[values.index(min(values))] if all(v is not None for v in values) else None
            }
        
        return comparison

    def query_runs(
        self,
        model_type: Optional[str] = None,
        after_date: Optional[str] = None,
        min_metric: Optional[Dict[str, float]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query training runs with filters."""
        results = []
        
        for file_path in sorted(self.root.glob("training_run_*.json"), reverse=True):
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Apply filters
            if model_type and data.get("model_type") != model_type:
                continue
            
            if after_date:
                run_date = data.get("timestamp", "")
                if run_date < after_date:
                    continue
            
            if min_metric:
                metrics = data.get("metrics", {})
                skip = False
                for metric_name, min_val in min_metric.items():
                    if metrics.get(metric_name, 0.0) < min_val:
                        skip = True
                        break
                if skip:
                    continue
            
            results.append(data)
            
            if len(results) >= limit:
                break
        
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get learning history statistics."""
        total_runs = len(list(self.root.glob("training_run_*.json")))
        
        # Aggregate metrics
        all_metrics = []
        model_types = set()
        
        for file_path in self.root.glob("training_run_*.json"):
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if "metrics" in data:
                    all_metrics.append(data["metrics"])
                if "model_type" in data:
                    model_types.add(data["model_type"])
        
        stats = {
            "total_runs": total_runs,
            "unique_model_types": len(model_types),
            "model_types": list(model_types)
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


# ---------------------------------------------------------------------------
# Config memory (engineering rules & system config)
# ---------------------------------------------------------------------------

class ConfigMemory:
    """Simple loader/saver for engineering rules and system configs.

    This does not replace existing config mechanisms (like ``config.py`` or
    environment variables); it merely offers a structured place to store
    richer engineering rule sets as JSON files.
    
    Enhanced with versioning, validation, and inheritance.
    """

    def __init__(
        self, 
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ) -> None:
        self.root = root / "config"
        self.rules_dir = self.root / "engineering_rules"
        self.system_dir = self.root / "system"
        _ensure_dir(self.rules_dir)
        _ensure_dir(self.system_dir)
        self.cache = LRUCache(max_size=cache_size) if enable_cache else None

    def save_rules(
        self, 
        name: str, 
        payload: Dict[str, Any],
        version: Optional[str] = None
    ) -> Path:
        """Save engineering rules with optional versioning."""
        if version:
            filename = f"{name}_v{version}.json"
        else:
            filename = f"{name}.json"
        
        path = self.rules_dir / filename
        
        # Add metadata
        payload["_metadata"] = {
            "name": name,
            "version": version,
            "updated_at": _now_iso()
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        # Cache
        if self.cache:
            cache_key = _hash_key("rules", name, version or "latest")
            self.cache.set(cache_key, payload)
        
        return path

    def load_rules(
        self, 
        name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load engineering rules with caching."""
        cache_key = _hash_key("rules", name, version or "latest")
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Determine path
        if version:
            path = self.rules_dir / f"{name}_v{version}.json"
        else:
            path = self.rules_dir / f"{name}.json"
        
        if not path.exists():
            return None
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Cache
        if self.cache:
            self.cache.set(cache_key, data)
        
        return data

    def list_rule_versions(self, name: str) -> List[str]:
        """List all versions of a rule."""
        versions = []
        
        for file_path in self.rules_dir.glob(f"{name}_v*.json"):
            # Extract version from filename
            version = file_path.stem.split("_v")[-1]
            versions.append(version)
        
        # Check for non-versioned file
        if (self.rules_dir / f"{name}.json").exists():
            versions.append("latest")
        
        return sorted(versions)

    def save_system_config(
        self, 
        name: str, 
        payload: Dict[str, Any]
    ) -> Path:
        """Save system configuration."""
        path = self.system_dir / f"{name}.json"
        
        # Add metadata
        payload["_metadata"] = {
            "name": name,
            "updated_at": _now_iso()
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        
        # Cache
        if self.cache:
            cache_key = _hash_key("system", name)
            self.cache.set(cache_key, payload)
        
        return path

    def load_system_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load system configuration with caching."""
        cache_key = _hash_key("system", name)
        
        # Check cache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        path = self.system_dir / f"{name}.json"
        if not path.exists():
            return None
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Cache
        if self.cache:
            self.cache.set(cache_key, data)
        
        return data

    def merge_configs(
        self,
        base_config: Dict[str, Any],
        override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two configurations (deep merge)."""
        result = dict(base_config)
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result

    def validate_config(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Simple config validation against schema.
        
        Returns (is_valid, errors).
        """
        errors = []
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Check types
        for field, expected_type in schema.get("types", {}).items():
            if field in config:
                actual_type = type(config[field]).__name__
                if actual_type != expected_type:
                    errors.append(f"Field '{field}' should be {expected_type}, got {actual_type}")
        
        return len(errors) == 0, errors

    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        rules_count = len(list(self.rules_dir.glob("*.json")))
        system_count = len(list(self.system_dir.glob("*.json")))
        
        stats = {
            "rules_count": rules_count,
            "system_configs_count": system_count
        }
        
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        return stats


__all__ = [
    "FeatureMemoryRecord",
    "FeatureLevelMemory",
    "GraphIndexEntry",
    "GraphLevelMemory",
    "SemanticLevelMemory",
    "LearningHistoryMemory",
    "ConfigMemory",
    "LRUCache",
    "CacheEntry",
    "HierarchicalMemoryManager",
]


# ---------------------------------------------------------------------------
# Unified Memory Manager
# ---------------------------------------------------------------------------

class HierarchicalMemoryManager:
    """Unified manager for all memory levels.
    
    Provides a single interface to access and manage all memory layers
    with centralized caching, monitoring, and maintenance operations.
    """

    def __init__(
        self,
        root: Path = DEFAULT_MEMORY_ROOT,
        enable_cache: bool = True,
        cache_size: int = DEFAULT_CACHE_SIZE
    ):
        """Initialize all memory levels."""
        self.root = root
        _ensure_dir(root)
        
        # Initialize all memory layers
        self.feature = FeatureLevelMemory(root, enable_cache, cache_size)
        self.graph = GraphLevelMemory(root, enable_cache, cache_size)
        self.semantic = SemanticLevelMemory(root, enable_cache, cache_size)
        self.learning = LearningHistoryMemory(root, enable_cache, cache_size)
        self.config = ConfigMemory(root, enable_cache, cache_size)
        
        self.layers = {
            "feature": self.feature,
            "graph": self.graph,
            "semantic": self.semantic,
            "learning": self.learning,
            "config": self.config
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics from all memory levels."""
        stats = {
            "root": str(self.root),
            "layers": {}
        }
        
        for name, layer in self.layers.items():
            stats["layers"][name] = layer.get_stats()
        
        # Calculate total size
        total_size_mb = 0.0
        for root_dir, dirs, files in os.walk(self.root):
            for file in files:
                file_path = Path(root_dir) / file
                total_size_mb += file_path.stat().st_size / (1024 ** 2)
        
        stats["total_size_mb"] = total_size_mb
        
        return stats
    
    def clear_all_caches(self) -> Dict[str, str]:
        """Clear caches in all memory levels."""
        results = {}
        
        for name, layer in self.layers.items():
            if hasattr(layer, 'cache') and layer.cache:
                layer.cache.clear()
                results[name] = "cleared"
            else:
                results[name] = "no_cache"
        
        return results
    
    def consolidate_all(
        self,
        feature_days: int = 30,
        semantic_days: int = 90,
        graph_keep_versions: int = 5
    ) -> Dict[str, Any]:
        """Run consolidation on all memory levels."""
        results = {}
        
        # Feature level
        if hasattr(self.feature, 'consolidate'):
            results["feature"] = self.feature.consolidate(older_than_days=feature_days)
        
        # Semantic level
        if hasattr(self.semantic, 'consolidate_decisions'):
            # Consolidate for all projects
            semantic_results = {}
            for project_dir in (self.root / "semantic_level" / "decisions").iterdir():
                if project_dir.is_dir():
                    project_id = project_dir.name
                    semantic_results[project_id] = self.semantic.consolidate_decisions(
                        project_id, 
                        older_than_days=semantic_days
                    )
            results["semantic"] = semantic_results
        
        # Graph level - cleanup old versions
        if hasattr(self.graph, 'cleanup_old_versions'):
            graph_results = {}
            for index_file in (self.root / "graph_level" / "projects").glob("*_index.json"):
                project_id = index_file.stem.replace("_index", "")
                graph_results[project_id] = self.graph.cleanup_old_versions(
                    project_id,
                    keep_latest=graph_keep_versions
                )
            results["graph"] = graph_results
        
        return results
    
    def export_summary(self, output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Export a comprehensive summary of all memory."""
        summary = {
            "generated_at": _now_iso(),
            "root": str(self.root),
            "stats": self.get_global_stats()
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all memory levels."""
        health = {
            "status": "healthy",
            "timestamp": _now_iso(),
            "issues": []
        }
        
        # Check if directories exist
        for name, layer in self.layers.items():
            if not layer.root.exists():
                health["issues"].append(f"{name} layer root missing: {layer.root}")
                health["status"] = "degraded"
        
        # Check cache hit rates
        for name, layer in self.layers.items():
            if hasattr(layer, 'cache') and layer.cache:
                cache_stats = layer.cache.get_stats()
                hit_rate = cache_stats.get("hit_rate", 0.0)
                
                if hit_rate < 0.3 and cache_stats["total_requests"] > 100:
                    health["issues"].append(
                        f"{name} layer has low cache hit rate: {hit_rate:.2%}"
                    )
                    health["status"] = "warning"
        
        # Check disk usage
        stats = self.get_global_stats()
        total_size_mb = stats.get("total_size_mb", 0.0)
        
        if total_size_mb > 1000:  # More than 1GB
            health["issues"].append(
                f"Total memory size is large: {total_size_mb:.2f} MB"
            )
            if health["status"] == "healthy":
                health["status"] = "warning"
        
        return health
    
    def query_across_layers(
        self,
        project_id: str,
        include_features: bool = True,
        include_graphs: bool = True,
        include_decisions: bool = True,
        include_training: bool = True
    ) -> Dict[str, Any]:
        """Query data across multiple memory layers for a project."""
        result = {
            "project_id": project_id,
            "timestamp": _now_iso()
        }
        
        # Feature records
        if include_features:
            result["features"] = self.feature.query_records(
                project_id=project_id,
                limit=50
            )
        
        # Graph versions
        if include_graphs:
            result["graphs"] = self.graph.list_graphs(project_id)
        
        # Decisions
        if include_decisions:
            result["decisions"] = self.semantic.query_decisions(
                project_id=project_id,
                limit=50
            )
        
        # Training runs
        if include_training:
            result["training_runs"] = self.learning.query_runs(
                limit=50
            )
        
        return result
    
    def backup(self, backup_path: Path) -> Dict[str, Any]:
        """Backup all memory to specified path."""
        import shutil
        
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = _now_iso().replace(":", "_")
        backup_dir = backup_path / f"memory_backup_{timestamp}"
        
        # Copy entire memory directory
        shutil.copytree(self.root, backup_dir)
        
        return {
            "backup_path": str(backup_dir),
            "timestamp": _now_iso(),
            "size_mb": sum(
                f.stat().st_size 
                for f in backup_dir.rglob("*") 
                if f.is_file()
            ) / (1024 ** 2)
        }
    
    def restore(self, backup_path: Path) -> Dict[str, Any]:
        """Restore memory from backup."""
        import shutil
        
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            return {"error": "Backup path does not exist"}
        
        # Clear current memory
        if self.root.exists():
            shutil.rmtree(self.root)
        
        # Copy backup
        shutil.copytree(backup_path, self.root)
        
        # Clear all caches
        self.clear_all_caches()
        
        return {
            "restored_from": str(backup_path),
            "timestamp": _now_iso(),
            "status": "success"
        }
