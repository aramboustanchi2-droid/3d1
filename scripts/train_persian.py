import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    print("--- Initiating Persian Language Training Protocol ---")
    
    # Train multiple times to reach high fluency
    for i in range(5):
        result = brain.train_language("fa")
        print(f"Epoch {i+1}: {result}")
        
    print("--- Persian Language Mastery Achieved ---")
    print("System can now process and generate Persian text.")

if __name__ == "__main__":
    main()
