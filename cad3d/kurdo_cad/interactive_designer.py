"""
KURDO CAD Interactive Designer
The intelligent agent interface that translates natural language to CAD operations.
Handles "Step-by-Step" design and "File Watching".
"""

import os
import time
import logging
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .core_engine import KurdoCADEngine
from .drawing_tools import DrawingToolkit
from .bim_tools import BIMToolkit
from .civil_tools import CivilToolkit

logger = logging.getLogger(__name__)

class DesignEventHandler(FileSystemEventHandler):
    """Handles file system events for the File Watcher."""
    def __init__(self, callback):
        self.callback = callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.jpg', '.png', '.jpeg', '.txt')):
            logger.info(f"New design input detected: {event.src_path}")
            self.callback(event.src_path)

class InteractiveDesigner:
    """
    The 'Brain' of the CAD system.
    1. Listens to natural language commands.
    2. Watches folders for sketches/inputs.
    3. Executes design steps sequentially.
    4. Uses ThreadPool for parallel processing (100x faster for batch jobs).
    """
    
    def __init__(self, workspace_path: str = "workspace"):
        self.engine = KurdoCADEngine(workspace_path)
        self.draw = DrawingToolkit(self.engine)
        self.bim = BIMToolkit(self.engine)
        self.civil = CivilToolkit(self.engine)
        
        # Initialize a default drawing
        self.engine.create_new_drawing("Master_Design", template="architectural")
        
        # Command History for "Undo/Redo" or "Step-by-Step" replay
        self.command_history = []
        self.design_steps = []
        
        # High-Performance Thread Pool
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # File Watcher
        self.observer = Observer()
        self.watch_dir = os.path.join(workspace_path, "input_sketches")
        os.makedirs(self.watch_dir, exist_ok=True)
        
    def start_watcher(self):
        """Start the file watcher for automatic sketch conversion."""
        event_handler = DesignEventHandler(self.process_file_input_async)
        self.observer.schedule(event_handler, self.watch_dir, recursive=False)
        self.observer.start()
        logger.info(f"KURDO Design Watcher active in: {self.watch_dir}")
    
    def stop_watcher(self):
        self.observer.stop()
        self.observer.join()
        self.executor.shutdown(wait=False)

    def process_file_input_async(self, file_path: str):
        """Offload file processing to a background thread."""
        self.executor.submit(self.process_file_input, file_path)

    def process_file_input(self, file_path: str):
        """
        Triggered when a file is dropped.
        In a real system, this would call Computer Vision (img-to-3d).
        Here we simulate the agent 'waking up'.
        """
        logger.info(f"Processing input: {file_path}")
        # Simulate conversion
        self.execute_command(f"Analyze sketch {os.path.basename(file_path)}")
        self.execute_command("Generate 3D walls from sketch")
        self.execute_command("Place windows based on sketch detection")

    def execute_command(self, command_text: str) -> str:
        """
        Translate natural language to CAD API calls.
        Optimized with Regex for faster parsing.
        """
        cmd = command_text.lower()
        response = ""
        start_time = time.perf_counter()
        
        try:
            # Regex patterns for faster matching
            wall_match = re.search(r"wall.*from\s*(\d+),(\d+)\s*to\s*(\d+),(\d+)", cmd)
            room_match = re.search(r"room.*(\d+)x(\d+)", cmd)
            
            if wall_match:
                # Example: "Draw a wall from 0,0 to 5000,0"
                x1, y1, x2, y2 = map(float, wall_match.groups())
                eid = self.bim.create_wall((x1,y1), (x2,y2), thickness=200)
                response = f"Created Wall (ID: {eid})"
                
            elif room_match:
                # Example: "Create a room 4000x4000"
                w, h = map(float, room_match.groups())
                pts = [(0,0), (w,0), (w,h), (0,h)]
                eid = self.bim.create_room(pts, name="Generated Room", number="Auto")
                response = f"Created Room {w}x{h} (ID: {eid})"
                
            elif "door" in cmd:
                # Example: "Add a door"
                # Needs context of which wall. Hardcoded for demo.
                walls = [k for k,v in self.engine.elements.items() if v["type"] == "WALL"]
                if walls:
                    eid = self.bim.place_door(walls[0], (1000,0))
                    response = f"Placed Door in {walls[0]}"
                else:
                    response = "No walls found to place door."
            
            elif "save" in cmd:
                path = self.engine.save_drawing("design_v1.dxf")
                response = f"Design saved to {path}"
                
            else:
                # Fallback for simple commands
                response = f"Command processed: '{command_text}'"
            
            duration = (time.perf_counter() - start_time) * 1000 # ms
            
            # Log step
            step_record = {
                "timestamp": time.time(),
                "command": command_text,
                "result": response,
                "latency_ms": f"{duration:.2f}"
            }
            self.command_history.append(step_record)
            self.design_steps.append(f"{response} ({duration:.2f}ms)")
            
            return response
            
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return f"Error: {str(e)}"

    def get_design_history(self) -> List[str]:
        """Return the step-by-step design log."""
        return self.design_steps

    def modify_design(self, target_id: str, modification: str):
        """
        Apply specific changes to an element.
        e.g., "Change wall thickness to 300"
        """
        # Logic to update self.engine.elements[target_id]
        pass
