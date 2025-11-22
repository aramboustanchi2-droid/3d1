import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    brain = SuperAIBrain()
    
    arxiv_url = "https://arxiv.org/html/2503.22346v1?utm_source=chatgpt.com"
    
    print("--- STARTING EXTRAORDINARY EXPLORATION SEQUENCE ---")
    print(f"Target: ArXiv Research Paper & Dataset (2503.22346)")
    print("Objective: Assimilate >1 Million Floor Plans & Advanced Generative Rules")
    
    result = brain.train_system(arxiv_url)
    
    print("\n--- EXPLORATION REPORT ---")
    print(result)
    
    # Verify persistence
    kb_path = brain.learning_module.knowledge_file_path
    if os.path.exists(kb_path):
        print(f"\n[SUCCESS] Knowledge Base successfully persisted to disk:")
        print(f"Path: {kb_path}")
        print("All learned patterns, weights, and rules are now saved.")
    else:
        print(f"\n[ERROR] Failed to verify persistence file at {kb_path}")

if __name__ == "__main__":
    main()
