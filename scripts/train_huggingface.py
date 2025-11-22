import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    hf_url = "https://www.debutinfotech.com/blog/frameworks-for-ai-agent-development?utm_source=chatgpt.com (Hugging Face Transformers + Agents)"
    
    print(f"--- Sending Super AI to learn from: {hf_url} ---")
    result = brain.train_system(hf_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
