import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    sk_url = "https://www.linkedin.com/pulse/top-5-frameworks-building-ai-agents-2025-sahil-malhotra-wmisc?utm_source=chatgpt.com (Microsoft Semantic Kernel)"
    
    print(f"--- Sending Super AI to learn from: {sk_url} ---")
    result = brain.train_system(sk_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
