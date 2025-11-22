"""
PEFT System Manager

Provides a lightweight manager around Parameter-Efficient Fine-Tuning techniques
(e.g., Prefix-Tuning, P-Tuning, IA3, (Q)LoRA) to complement existing methods.

This module is designed to work even without the external 'peft' package.
If 'peft' is available, it will expose additional capability flags in status.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    name: str
    technique: str  # e.g., 'prefix_tuning', 'p_tuning', 'ia3', 'qlora', 'adalora', 'lora'
    path: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: str = datetime.now().isoformat()
    loaded: bool = False


class PEFTManager:
    def __init__(self, device: str = "auto"):
        self.device = device
        self.techniques: List[str] = [
            "prefix_tuning",
            "p_tuning",
            "ia3",
            "adalora",
            "qlora",
            "lora"
        ]
        self.adapters: Dict[str, AdapterInfo] = {}
        self.active_adapter: Optional[str] = None

        # Optional dependency detection
        try:
            import peft  # type: ignore
            self._has_peft = True
            self._peft_version = getattr(peft, "__version__", "unknown")
            logger.info(f"PEFT library detected: v{self._peft_version}")
        except Exception:
            self._has_peft = False
            self._peft_version = None
            logger.info("PEFT library not installed; running in lightweight mode")

        # Register a default demo adapter
        self.register_adapter(
            name="architectural_prefix",
            technique="prefix_tuning",
            config={"virtual_tokens": 16, "domain": "architecture"}
        )

    def list_techniques(self) -> List[str]:
        return list(self.techniques)

    def register_adapter(self, name: str, technique: str, path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> AdapterInfo:
        if technique not in self.techniques:
            raise ValueError(f"Unknown PEFT technique: {technique}")
        info = AdapterInfo(name=name, technique=technique, path=path, config=config or {})
        self.adapters[name] = info
        logger.info(f"Registered PEFT adapter '{name}' with technique '{technique}'")
        return info

    def load_adapter(self, name: str) -> Dict[str, Any]:
        if name not in self.adapters:
            return {"status": "error", "message": f"Adapter not found: {name}"}
        # Simulate load
        for a in self.adapters.values():
            a.loaded = False
        self.adapters[name].loaded = True
        self.active_adapter = name
        return {"status": "success", "active_adapter": name}

    def unload_adapter(self) -> Dict[str, Any]:
        if self.active_adapter and self.active_adapter in self.adapters:
            self.adapters[self.active_adapter].loaded = False
        self.active_adapter = None
        return {"status": "success", "active_adapter": None}

    def apply(self, query: str, task_type: Optional[str] = None, adapter: Optional[str] = None, technique: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # Determine adapter
        adapter_name = adapter or self.active_adapter or next(iter(self.adapters), None)
        if not adapter_name:
            return {"status": "error", "message": "No adapters registered"}
        if adapter_name not in self.adapters:
            return {"status": "error", "message": f"Adapter not found: {adapter_name}"}

        # Load if needed
        if self.active_adapter != adapter_name:
            self.load_adapter(adapter_name)

        info = self.adapters[adapter_name]
        used_tech = technique or info.technique

        # Simulated application — in real integration, we'd wrap base model forward with PEFT modules
        note = (
            f"Applied PEFT ({used_tech}) adapter '{adapter_name}'"
            + (f" on task '{task_type}'" if task_type else "")
        )

        return {
            "status": "success",
            "technique": used_tech,
            "adapter": adapter_name,
            "note": note,
            "peft_available": self._has_peft,
            "peft_version": self._peft_version
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "peft_available": self._has_peft,
            "peft_version": self._peft_version,
            "device": self.device,
            "active_adapter": self.active_adapter,
            "adapters": [asdict(a) for a in self.adapters.values()],
            "techniques": self.list_techniques()
        }

    def compare_with_others(self) -> Dict[str, Any]:
        return {
            "peft": {
                "type": "Parameter-Efficient Fine-Tuning",
                "setup_time": "Minutes to Hours",
                "cost": "Low ($0-$50)",
                "gpu_required": False,
                "quality": "Very Good",
                "flexibility": "High (multiple techniques)",
                "best_for": [
                    "Adapting base models quickly",
                    "Limited compute environments",
                    "Multiple domain adapters",
                    "Updating behavior without full retrain"
                ],
                "techniques": self.list_techniques()
            }
        }
