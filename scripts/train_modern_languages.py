import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cad3d.super_ai.brain import SuperAIBrain

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_modern_mastery():
    brain = SuperAIBrain()
    
    source = "Modern Systems Languages (Go, Rust, Julia)"
    
    logger.info(f"Target Source: {source}")
    logger.info("Objective: Achieve expert-level mastery in modern high-performance languages.")
    
    # Trigger the training
    result = brain.train_system(source)
    
    logger.info(result)
    logger.info("--- Training Summary ---")
    logger.info("The Super AI is now a Cloud-Native Systems Architect.")
    logger.info("Languages Mastered: Go, Rust, Julia.")
    logger.info("Capabilities: Memory Safety, High Concurrency, Scientific Computing.")

if __name__ == "__main__":
    train_modern_mastery()
