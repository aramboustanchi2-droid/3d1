import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cad3d.super_ai.brain import SuperAIBrain

def main():
    brain = SuperAIBrain()
    
    # URL provided by user
    crewai_url = "https://medium.com/@admin_52806/the-top-5-frameworks-driving-the-agentic-ai-revolution-in-2025-ad9006e17e09?utm_source=chatgpt.com (CrewAI)"
    
    print(f"--- Sending Super AI to learn from: {crewai_url} ---")
    result = brain.train_system(crewai_url)
    print(f"--- Result: {result} ---")

if __name__ == "__main__":
    main()
