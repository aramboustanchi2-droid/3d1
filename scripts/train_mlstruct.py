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
    
    mlstruct_url = "https://github.com/MLSTRUCT/MLSTRUCT-FP?utm_source=chatgpt.com"
    
    print("--- STARTING STRUCTURAL INTELLIGENCE UPGRADE ---")
    print(f"Target: MLSTRUCT-FP Dataset")
    print("Objective: Learn Multi-Unit Layouts & Wall/Slab Segmentation (JSON)")
    
    result = brain.train_system(mlstruct_url)
    
    print("\n--- UPGRADE REPORT ---")
    print(result)
    
    # Verify persistence
    kb_path = brain.learning_module.knowledge_file_path
    if os.path.exists(kb_path):
        print(f"\n[SUCCESS] Knowledge Base updated and saved:")
        print(f"Path: {kb_path}")
    else:
        print(f"\n[ERROR] Failed to verify persistence file at {kb_path}")

if __name__ == "__main__":
    main()
