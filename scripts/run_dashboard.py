import os
import subprocess
import sys

def run_dashboard():
    dashboard_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cad3d', 'dashboard', 'app.py')
    print(f"Launching Dashboard from: {dashboard_path}")
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly", "pandas"])
    
    # Run streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_dashboard()
