import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    autoagent_url = "https://arxiv.org/abs/2502.05957?utm_source=chatgpt.com (AutoAgent)"
    
    print(f"--- Sending Super AI to learn from: {autoagent_url} ---")
    result = brain.train_system(autoagent_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
