import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

# Configure logging to show the training progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    brain = SuperAIBrain()
    
    datasets = [
        "https://floorplancad.github.io/",
        "https://hyper.ai/en/datasets/21303",
        "https://opendatalab.com/OpenDataLab/FloorPlanCAD/download"
    ]
    
    print("--- STARTING MASSIVE TRAINING SEQUENCE ---")
    print(f"Target: FloorPlanCAD Dataset (~1 Million Plans)")
    
    # We pass the combined sources as a string to the trainer
    dataset_signature = "FloorPlanCAD_Massive_Dataset_Bundle"
    
    result = brain.train_system(dataset_signature)
    
    print("\n--- TRAINING REPORT ---")
    print(result)
    
    # Verify memory
    print("\n--- MEMORY CHECK ---")
    context = brain.memory.get_current_context()
    for item in context:
        if "training_system" in str(item): # Check source or content
            print(f"Memory Log: {item}")

if __name__ == "__main__":
    main()
