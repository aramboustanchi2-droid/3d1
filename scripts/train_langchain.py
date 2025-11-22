import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    langchain_url = "https://medium.com/@elisowski/top-ai-agent-frameworks-in-2025-9bcedab2e239?utm_source=chatgpt.com (LangChain)"
    
    print(f"--- Sending Super AI to learn from: {langchain_url} ---")
    result = brain.train_system(langchain_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
