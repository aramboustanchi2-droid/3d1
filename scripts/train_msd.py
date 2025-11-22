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
    
    msd_url = "https://www.researchgate.net/publication/382271831_MSD_A_Benchmark_Dataset_for_Floor_Plan_Generation_of_Building_Complexes?utm_source=chatgpt.com"
    
    print("--- STARTING SWISS PRECISION TRAINING ---")
    print(f"Target: MSD (Modified Swiss Dwellings) Dataset")
    print("Objective: Learn Complex Building Topologies & High-Standard Housing Rules")
    print("Scope: 5,300+ Buildings, 18,900+ Apartments")
    
    result = brain.train_system(msd_url)
    
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
