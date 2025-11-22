import json
import os
import time
from datetime import datetime

class CentralCouncil:
    def __init__(self):
        self.council_members = [
            "Supreme Leader (User)",
            "System Architect",
            "Security Chief",
            "Data Overseer",
            "Evolution Manager",
            "Resource Allocator",
            "Interface Liaison"
        ]
        self.active_directives = []
        self.system_log = []
        self.state_file = os.path.join(os.path.dirname(__file__), "central_council_state.json")
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.active_directives = data.get("active_directives", [])
                    self.system_log = data.get("system_log", [])
            except:
                pass

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({
                "active_directives": self.active_directives,
                "system_log": self.system_log
            }, f, indent=4)

    def process_command(self, command, user_role="admin"):
        """
        Process a command from the Supreme Leader (User).
        """
        if user_role != "admin":
            return "⛔ Access Denied: Only the Supreme Leader can issue commands to the Central Council."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] COMMAND RECEIVED: {command}"
        self.system_log.append(log_entry)

        # Simulate hierarchical propagation
        response = self._propagate_command(command)
        
        self.save_state()
        return response

    def _propagate_command(self, command):
        """
        Simulate the command flowing down the hierarchy.
        """
        steps = []
        steps.append("1. 👂 Council Hearing: Command received and logged.")
        steps.append("2. 🗳️ Deliberation: Representatives from all sectors (CAD, AI, Web, Mobile) acknowledged.")
        
        # Simple keyword analysis for simulation
        if "update" in command.lower():
            steps.append("3. ⚡ Action: Triggering System Update Protocols...")
            self.active_directives.append({"type": "UPDATE", "desc": command, "status": "In Progress"})
        elif "deploy" in command.lower():
            steps.append("3. 🚀 Action: Mobilizing Agent Swarm for Deployment...")
            self.active_directives.append({"type": "DEPLOY", "desc": command, "status": "In Progress"})
        elif "stop" in command.lower():
            steps.append("3. 🛑 Action: Emergency Halt Signal Broadcasted.")
            self.active_directives.append({"type": "STOP", "desc": command, "status": "Active"})
        else:
            steps.append("3. 📢 Action: General Directive Broadcasted to all subsystems.")
            self.active_directives.append({"type": "GENERAL", "desc": command, "status": "Active"})

        steps.append("4. ✅ Execution: Directive accepted by System Core.")
        
        return "\n".join(steps)

    def get_status(self):
        return {
            "members": self.council_members,
            "active_directives_count": len(self.active_directives),
            "last_log": self.system_log[-1] if self.system_log else "No activity."
        }
