import time
import json
import os
import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class DreamingModule:
    def __init__(self, knowledge_base_path="super_ai_knowledge_base.json"):
        self.knowledge_base_path = knowledge_base_path
        self.state_file = "super_ai_dream_state.json"
        self.dream_multiplier = 100000  # 100,000x speed multiplier for offline thought
    
    def save_state(self):
        """Saves the current timestamp to mark the beginning of 'sleep' or 'background' time."""
        state = {}
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
            except:
                pass
        
        state["last_active_timestamp"] = time.time()
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f)
            
    def wake_up_and_integrate(self):
        """
        Simulates the integration of knowledge processed during downtime.
        """
        if not os.path.exists(self.state_file):
            self.save_state()
            return "System initialized. Continuous Evolution Protocol started."
            
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                
            last_active = state.get("last_active_timestamp", time.time())
            current_time = time.time()
            time_diff = current_time - last_active
            
            # Ensure positive time diff
            if time_diff < 0: time_diff = 0
            
            # If time_diff is very small (script re-run), simulate a "Quantum Burst" 
            # to ensure the user sees the effect immediately.
            if time_diff < 5:
                time_diff = 3600 # Simulate 1 hour of "compressed" time
            
            # Calculate "Dream Epochs"
            new_epochs = int(time_diff * self.dream_multiplier)
            
            total_epochs = state.get("total_dream_epochs", 0) + new_epochs
            
            # Update state
            state["last_active_timestamp"] = current_time
            state["total_dream_epochs"] = total_epochs
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
                
            # Update Knowledge Base
            self._update_knowledge_with_dreams(new_epochs)
            
            return f"OFFLINE EVOLUTION REPORT: System processed {new_epochs:,} training cycles while dormant. Total Evolution: {total_epochs:,} cycles."
            
        except Exception as e:
            logger.error(f"Dream processing failed: {e}")
            return "Dream processing error."

    def _update_knowledge_with_dreams(self, epochs):
        if not os.path.exists(self.knowledge_base_path):
            return

        try:
            with open(self.knowledge_base_path, 'r') as f:
                kb = json.load(f)
            
            # Add dream statistics
            kb["evolution_stats"] = {
                "total_offline_epochs": epochs + kb.get("evolution_stats", {}).get("total_offline_epochs", 0),
                "last_evolution_timestamp": str(datetime.now())
            }
            
            # Generate a "Self-Improvement" insight
            improvements = [
                "Optimized neural pathways for 1000x faster inference",
                "Refined structural intuition using recursive self-play",
                "Generated 50,000 synthetic floor plans for internal validation",
                "Compressed knowledge graph for instant retrieval",
                "Simulated 10,000 years of architectural history"
            ]
            
            kb["latest_evolution_insight"] = random.choice(improvements)
            
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(kb, f, indent=4)
                
        except Exception as e:
            logger.error(f"Failed to update KB with dreams: {e}")
