"""
Main Application Launcher.
Launches FastAPI Backend, mounts static Dashboard, and opens web browser directly to the dashboard.
"""

import os
import sys
import time
import webbrowser
import threading
import uvicorn
from fastapi.staticfiles import StaticFiles

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.main import app

# Mount static frontend directory
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/dashboard", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"[Launcher] Web Dashboard available at: http://127.0.0.1:8000/dashboard/index.html")


def open_browser():
    """Opens browser directly to dashboard after 1.5 second delay."""
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000/")


if __name__ == "__main__":
    print("[Launcher] Starting Real-Time Multilingual Translation & Audio Dubbing Server...")
    print("[Launcher] Navigating directly to Dashboard at http://127.0.0.1:8000/")
    
    # Launch browser thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run(app, host="127.0.0.1", port=8000)
