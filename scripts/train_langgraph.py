import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    langgraph_url = "https://www.linkedin.com/pulse/top-5-frameworks-building-ai-agents-2025-sahil-malhotra-wmisc?utm_source=chatgpt.com (LangGraph)"
    
    print(f"--- Sending Super AI to learn from: {langgraph_url} ---")
    result = brain.train_system(langgraph_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
