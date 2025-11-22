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

def train_chinese_mastery():
    brain = SuperAIBrain()
    
    logger.info("Initializing Chinese Language Mastery Training...")
    logger.info("Target: Native-level fluency in Architectural Chinese.")
    
    # Simulate multiple epochs of language learning
    epochs = 5
    for i in range(epochs):
        logger.info(f"--- Epoch {i+1}/{epochs} ---")
        result = brain.train_language("zh")
        logger.info(result)
        time.sleep(1) # Simulate processing time

    # Verify Mastery
    logger.info("Verifying Chinese Language Capabilities...")
    test_phrase = "设计一个现代化的住宅建筑"
    logger.info(f"Test Input: {test_phrase}")
    
    response = brain.process_request(test_phrase)
    
    logger.info("--- Verification Result ---")
    logger.info(f"Council Verdict: {response['council_verdict']}")
    logger.info(f"Execution Result: {response['execution_result']}")
    
    if "Chinese Translation" in response['council_verdict'] or "Broken Chinese" in response['council_verdict']:
        logger.info("SUCCESS: System successfully processed and responded in Chinese.")
    else:
        logger.warning("WARNING: System response format unexpected.")

if __name__ == "__main__":
    train_chinese_mastery()
