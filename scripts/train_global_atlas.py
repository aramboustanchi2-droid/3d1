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
    
    atlas_url = "https://arxiv.org/abs/2506.04106?utm_source=chatgpt.com"
    
    print("--- STARTING PLANETARY URBAN TRAINING ---")
    print(f"Target: GlobalBuildingAtlas Dataset")
    print("Objective: Master City-Scale Modeling, LoD1 Generation, and Global Urban Morphology")
    
    result = brain.train_system(atlas_url)
    
    print("\n--- TRAINING REPORT ---")
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
