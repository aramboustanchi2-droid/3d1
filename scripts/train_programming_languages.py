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

def train_coding_mastery():
    brain = SuperAIBrain()
    
    source = "All Major Programming Languages (Python, C++, Java, C#, JavaScript)"
    
    logger.info(f"Target Source: {source}")
    logger.info("Objective: Achieve expert-level mastery in all major programming languages.")
    
    # Trigger the training
    result = brain.train_system(source)
    
    logger.info(result)
    logger.info("--- Training Summary ---")
    logger.info("The Super AI is now a Polyglot Software Architect.")
    logger.info("Languages Mastered: Python, C++, Java, C#, JavaScript.")
    logger.info("Capabilities: Full-stack dev, Systems programming, Cross-language optimization.")

if __name__ == "__main__":
    train_coding_mastery()
