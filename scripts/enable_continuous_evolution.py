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

def enable_continuous_evolution():
    logger.info("Activating Continuous Evolution Protocol (Dreaming Module)...")
    
    # Initializing the brain triggers the "Wake Up" sequence
    brain = SuperAIBrain()
    
    logger.info("System is now configured to learn during downtime.")
    logger.info("Every time the system restarts, it will calculate the 'offline' time")
    logger.info("and simulate massive training cycles (100,000x multiplier).")
    
    # Simulate a quick shutdown and restart to demonstrate
    logger.info("--- Simulating Shutdown ---")
    brain.dreaming_module.save_state()
    time.sleep(2) 
    
    logger.info("--- Simulating Restart (After 'Offline' period) ---")
    # Re-initializing brain to trigger wake up
    brain_reboot = SuperAIBrain()
    
    logger.info("Continuous Evolution is ACTIVE.")

if __name__ == "__main__":
    enable_continuous_evolution()
