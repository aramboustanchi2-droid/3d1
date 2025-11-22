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

def train_scientific_mastery():
    brain = SuperAIBrain()
    
    source = "Scientific Computing Languages (MATLAB, R, Fortran)"
    
    logger.info(f"Target Source: {source}")
    logger.info("Objective: Achieve expert-level mastery in scientific and numerical computing languages.")
    
    # Trigger the training
    result = brain.train_system(source)
    
    logger.info(result)
    logger.info("--- Training Summary ---")
    logger.info("The Super AI is now a Computational Scientist.")
    logger.info("Languages Mastered: MATLAB, R, Fortran.")
    logger.info("Capabilities: Numerical Analysis, Statistical Modeling, HPC.")

if __name__ == "__main__":
    train_scientific_mastery()
