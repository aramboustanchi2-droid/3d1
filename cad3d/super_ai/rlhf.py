import json
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReinforcementLearningModule:
    """
    Implements Reinforcement Learning from Human Feedback (RLHF).
    Allows the 'Supreme Leader' to critique outputs, which adjusts the system's
    internal reward model and policy weights.
    """
    def __init__(self, storage_path=None):
        if storage_path is None:
            # Default to cad3d/super_ai/rlhf_data.json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.storage_path = os.path.join(base_dir, "rlhf_data.json")
        else:
            self.storage_path = storage_path
            
        self.feedback_history = []
        # Simulated Reward Model Weights
        self.reward_weights = {
            "creativity": 1.0,
            "efficiency": 1.0,
            "safety": 1.0,
            "user_satisfaction": 1.0,
            "obedience": 1.0
        }
        self.load_history()

    def load_history(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_history = data.get("history", [])
                    self.reward_weights = data.get("weights", self.reward_weights)
            except Exception as e:
                logger.error(f"Failed to load RLHF history: {e}")

    def save_history(self):
        try:
            data = {
                "history": self.feedback_history,
                "weights": self.reward_weights,
                "last_update": datetime.now().isoformat()
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save RLHF history: {e}")

    def submit_feedback(self, input_context, model_output, feedback_score, feedback_text=None, category="general"):
        """
        Submit human feedback for a specific model output.
        feedback_score: -1.0 (Bad) to 1.0 (Good)
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": input_context,
            "output": model_output,
            "score": feedback_score,
            "critique": feedback_text,
            "category": category
        }
        self.feedback_history.append(entry)
        
        # Immediate "Learning" (Policy Update Simulation)
        self._update_weights(feedback_score, category)
        
        self.save_history()
        
        log_msg = f"RLHF Update: Score {feedback_score} received. "
        if feedback_text:
            log_msg += f"Critique: '{feedback_text}'. "
        log_msg += "Reward Model adjusted."
        logger.info(log_msg)
        
        return {
            "status": "success",
            "message": "Feedback assimilated. Neural weights updated.",
            "new_weights": self.reward_weights
        }

    def _update_weights(self, score, category):
        """
        Simulates updating the internal reward model based on feedback.
        This represents the 'PPO' (Proximal Policy Optimization) step in RLHF.
        """
        learning_rate = 0.05
        
        # Update General Satisfaction
        self.reward_weights["user_satisfaction"] += learning_rate * score
        
        # Update Obedience (Admin feedback is law)
        self.reward_weights["obedience"] += learning_rate * abs(score) # Any feedback increases attention
        
        # Update Specific Category
        if category in self.reward_weights:
            self.reward_weights[category] += learning_rate * score
        elif category == "design":
            self.reward_weights["creativity"] += learning_rate * score
        elif category == "code":
            self.reward_weights["efficiency"] += learning_rate * score

        # Normalize weights to keep them stable
        # (Simple normalization to keep average around 1.0)
        avg_weight = sum(self.reward_weights.values()) / len(self.reward_weights)
        if avg_weight > 0:
            for k in self.reward_weights:
                self.reward_weights[k] /= avg_weight

    def get_stats(self):
        return {
            "total_feedback_samples": len(self.feedback_history),
            "current_weights": self.reward_weights,
            "last_feedback": self.feedback_history[-1] if self.feedback_history else None
        }
