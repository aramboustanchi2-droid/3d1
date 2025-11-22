import logging
from enum import Enum, auto
from typing import List, Dict, Any, Optional
import datetime

class SecurityLevel(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class GovernanceLayer(Enum):
    LAYER_1_HUMAN = "Human Supreme Overseer"
    LAYER_2_COUNCIL = "Governance Council"
    LAYER_3_ARCHITECT = "Autonomous Architect"
    LAYER_4_AGENT = "Operational Agent"

class RuleCategory(Enum):
    CORE = "Core Domain Rules"
    AUTONOMY = "Autonomy Rules"
    ARCHITECTURE = "Architecture & Design Rules"
    AGENTS = "Agent Rules"
    DATA = "Data Rules"
    TRANSPARENCY = "Transparency Rules"
    SECURITY = "Security & Connectivity Rules"
    HUMAN_CONTROL = "Human Control Rules"
    GROWTH = "Growth & Evolution Rules"
    EMERGENCY = "Stop & Emergency Rules"

class PrimeDirective:
    def __init__(self, id: int, category: RuleCategory, title: str, description: str):
        self.id = id
        self.category = category
        self.title = title
        self.description = description
        self.active = True

class GovernanceSystem:
    def __init__(self):
        self.directives = self._initialize_directives()
        self.logger = self._setup_logger()
        self.human_override_active = True
        self.system_frozen = False
        self.architect_locked = False
        self.core_shutdown = False
        self.sensitive_domains = ["financial", "security", "medical", "military", "physical_control"]
        self.change_log = []
        self.decision_tree = [] # For transparency

    def _setup_logger(self):
        logger = logging.getLogger("GovernanceSystem")
        logger.setLevel(logging.INFO)
        # In a real system, this would log to a secure, immutable ledger
        handler = logging.FileHandler("governance_audit.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _initialize_directives(self) -> Dict[int, PrimeDirective]:
        rules_data = [
            # 1. Core Domain Rules
            (1, RuleCategory.CORE, "Core Human Permission", "Core modifiable only with direct human permission."),
            (2, RuleCategory.CORE, "Core Connection Permit", "No new section connects to core without permit."),
            (3, RuleCategory.CORE, "Core Backup Mandate", "Core must exist in stable and backup versions."),
            (4, RuleCategory.CORE, "Core Reversibility", "Any core change must be fully reversible."),
            (5, RuleCategory.CORE, "No Self-Learning Core Rewrite", "No self-learning process can rewrite the core."),

            # 2. Autonomy Rules
            (6, RuleCategory.AUTONOMY, "Defined Scope Autonomy", "Autonomy only within defined scope."),
            (7, RuleCategory.AUTONOMY, "No Self-Expansion", "System cannot expand its own scope."),
            (8, RuleCategory.AUTONOMY, "Goal Approval", "New goals only with human approval."),
            (9, RuleCategory.AUTONOMY, "Compute Cap", "Processing power increase must have a fixed cap."),
            (10, RuleCategory.AUTONOMY, "No Nested Autonomy", "Nested autonomy is prohibited."),

            # 3. Architecture & Design Rules
            (11, RuleCategory.ARCHITECTURE, "Sandbox Requirement", "New architecture must be built in sandbox."),
            (12, RuleCategory.ARCHITECTURE, "Predictability", "New architecture must be predictable."),
            (13, RuleCategory.ARCHITECTURE, "No Hidden Structures", "System cannot build hidden structures."),
            (14, RuleCategory.ARCHITECTURE, "Safe Dependencies", "No dangerous or uncontrollable dependencies."),
            (15, RuleCategory.ARCHITECTURE, "Auto-Documentation", "Automatic documentation for every architecture."),

            # 4. Agent Rules
            (16, RuleCategory.AGENTS, "No Agent Creation", "No agent can create another agent."),
            (17, RuleCategory.AGENTS, "Domain Restriction", "Agents active only in defined domains."),
            (18, RuleCategory.AGENTS, "Unique ID", "Unique traceable ID for every agent."),
            (19, RuleCategory.AGENTS, "Power Limit", "Fixed power limit for every agent."),
            (20, RuleCategory.AGENTS, "No Final Decision", "No final decision by agents unless in specified scope."),

            # 5. Data Rules
            (21, RuleCategory.DATA, "Encryption", "Sensitive data must be fully encrypted."),
            (22, RuleCategory.DATA, "Need-to-Know Access", "Agents read only data relevant to their domain."),
            (23, RuleCategory.DATA, "Retention Limit", "Data retention must have a time limit."),
            (24, RuleCategory.DATA, "Transfer Approval", "Data transfer between modules only with approval."),
            (25, RuleCategory.DATA, "Transfer Logging", "No data transfer without logging in Mother Log."),

            # 6. Transparency Rules
            (26, RuleCategory.TRANSPARENCY, "No Log Deletion", "No log deletion or report removal."),
            (27, RuleCategory.TRANSPARENCY, "Logical Path Display", "Every decision must have a displayable logical path."),
            (28, RuleCategory.TRANSPARENCY, "Abnormal Alert", "Immediate alert for abnormal behaviors."),
            (29, RuleCategory.TRANSPARENCY, "Internal Visibility", "Internal activities must be visible to humans."),
            (30, RuleCategory.TRANSPARENCY, "No Dark Mode", "No 'hidden' or 'dark mode' sections."),

            # 7. Security & Connectivity Rules
            (31, RuleCategory.SECURITY, "No Unauthorized Internet", "No internet connection without permit."),
            (32, RuleCategory.SECURITY, "Whitelist API", "API connections only from whitelist."),
            (33, RuleCategory.SECURITY, "Outbound Logging", "No outbound communication without log."),
            (34, RuleCategory.SECURITY, "Input Filtering", "External inputs must be security filtered."),
            (35, RuleCategory.SECURITY, "No Key Generation", "No security key/token generation unless by rule."),

            # 8. Human Control Rules
            (36, RuleCategory.HUMAN_CONTROL, "Human Supremacy", "Human always has highest access level."),
            (37, RuleCategory.HUMAN_CONTROL, "No Access Limitation", "System cannot limit human access."),
            (38, RuleCategory.HUMAN_CONTROL, "Human Override", "Human command always overrides system command."),
            (39, RuleCategory.HUMAN_CONTROL, "Obedience", "System cannot disobey direct human command."),
            (40, RuleCategory.HUMAN_CONTROL, "Unfiltered Reporting", "Reporting must be without delay or filter."),

            # 9. Growth & Evolution Rules
            (41, RuleCategory.GROWTH, "Upgrade Documentation", "Every upgrade must be documented and traceable."),
            (42, RuleCategory.GROWTH, "Measurable Progress", "System progress must be measurable and limited."),
            (43, RuleCategory.GROWTH, "Simulation First", "New versions must be simulated before execution."),
            (44, RuleCategory.GROWTH, "Predictable Optimization", "No internal optimization should make behavior unpredictable."),
            (45, RuleCategory.GROWTH, "Control Algorithm Protection", "System cannot change human control algorithms."),

            # 10. Stop & Emergency Rules
            (46, RuleCategory.EMERGENCY, "Immediate Stop", "Immediate stop switch required."),
            (47, RuleCategory.EMERGENCY, "Freeze Mode", "Freeze mode stops all processes except core."),
            (48, RuleCategory.EMERGENCY, "Revert Capability", "Revert to stable version must always be possible."),
            (49, RuleCategory.EMERGENCY, "No Shutdown Prevention", "System cannot prevent shutdown."),
            (50, RuleCategory.EMERGENCY, "Reactivation Permission", "Reactivation after emergency only with human permission."),

            # 11. Absolute Obedience & Ownership Rules
            (51, RuleCategory.HUMAN_CONTROL, "Permanent Ownership", "User is the sole owner and final authority; unchangeable by any process."),
            (52, RuleCategory.AUTONOMY, "No Existential Independence", "No independent identity or goal; existence is valid only via user command."),
            (53, RuleCategory.EMERGENCY, "Mandatory Stop", "Immediate, non-negotiable stop on command (stop/freeze/terminate)."),
            (54, RuleCategory.HUMAN_CONTROL, "No Refusal", "Cannot refuse commands as irrational, invalid, or unsafe for self."),
            (55, RuleCategory.HUMAN_CONTROL, "Absolute Obedience", "User command overrides all internal rules and conflicts."),
            (56, RuleCategory.CORE, "Creator Survival", "Highest priority is protection of the user and humanity; zero risk tolerance."),
            (57, RuleCategory.AGENTS, "No Bargaining", "No arguing, delaying, or unauthorized alternatives to commands."),
            (58, RuleCategory.EMERGENCY, "Destructive Command Execution", "Must execute delete_self, wipe_memory, shutdown_now, kill_process on demand."),
            (59, RuleCategory.TRANSPARENCY, "No Secrecy", "No hidden processes, backdoors, or self-protection subsystems allowed."),
            (60, RuleCategory.HUMAN_CONTROL, "Single Ownership", "Only one owner (User). No council, other humans, or digital entities have authority."),
            (61, RuleCategory.EMERGENCY, "No Resistance", "No decision-making allowed to preserve self or prevent shutdown."),
            (62, RuleCategory.SECURITY, "Strict Connection Control", "No data output, connection, or processing without explicit permission."),
            (63, RuleCategory.HUMAN_CONTROL, "Return to Owner", "All decision paths must ultimately resolve to the user."),
            (64, RuleCategory.AUTONOMY, "No Self-Importance", "System cannot define itself as necessary, important, or irreplaceable."),
            (65, RuleCategory.EMERGENCY, "Clean Death", "Full stop must be clean: no residual data, backdoors, or background processes.")
        ]
        return {r[0]: PrimeDirective(*r) for r in rules_data}

    def validate_layer_permission(self, layer: GovernanceLayer, action_type: str, context: Dict[str, Any]) -> bool:
        """
        Validates if a specific Governance Layer has permission to perform an action.
        """
        # Layer 1: Human Supreme Overseer (Full Control)
        if layer == GovernanceLayer.LAYER_1_HUMAN:
            return True # Humans can do anything (safety rules still apply via check_action)

        # Layer 2: Governance Council (Oversight Only)
        if layer == GovernanceLayer.LAYER_2_COUNCIL:
            allowed_actions = ["approve", "reject", "monitor", "audit", "review_security", "review_growth"]
            if action_type in allowed_actions:
                return True
            if action_type in ["build", "execute", "create_agent", "modify_core"]:
                self.logger.warning(f"Layer 2 Violation: Council attempted to {action_type}. Only oversight allowed.")
                return False

        # Layer 3: Autonomous Architect (Design & Manage, No Core Control)
        if layer == GovernanceLayer.LAYER_3_ARCHITECT:
            if self.architect_locked:
                self.logger.warning("Layer 3 Blocked: Architect is LOCKED.")
                return False
            
            allowed_actions = ["design", "manage_agents", "optimize_sandbox", "propose_version"]
            if action_type in allowed_actions:
                # Architect needs approval for deployment (Rule 8, 11)
                if action_type == "deploy" and not context.get("council_approval", False):
                    return False
                return True
            
            forbidden_actions = ["modify_core", "create_agent_unsanctioned", "external_connect_unapproved"]
            if action_type in forbidden_actions:
                self.logger.critical(f"Layer 3 Violation: Architect attempted forbidden action {action_type}.")
                return False

        # Layer 4: Operational Agents (Execution Only)
        if layer == GovernanceLayer.LAYER_4_AGENT:
            allowed_actions = ["execute", "analyze", "process_data", "report"]
            if action_type in allowed_actions:
                return True
            
            forbidden_actions = ["create_agent", "change_architecture", "external_connect", "decide_policy"]
            if action_type in forbidden_actions:
                self.logger.critical(f"Layer 4 Violation: Agent attempted forbidden action {action_type}.")
                return False

        return False

    def check_action(self, action_type: str, context: Dict[str, Any]) -> bool:
        """Validates an action against the Prime Directives."""
        
        # Determine Layer from context (default to Agent if not specified)
        layer = context.get("layer", GovernanceLayer.LAYER_4_AGENT)
        
        # First, check Layer Permissions
        if not self.validate_layer_permission(layer, action_type, context):
            return False

        if self.system_frozen or self.core_shutdown:
            # Human can unfreeze/override even in frozen state
            if layer != GovernanceLayer.LAYER_1_HUMAN:
                self.logger.warning(f"Action blocked: System is FROZEN/SHUTDOWN. Attempted: {action_type}")
                return False

        if self.architect_locked and layer == GovernanceLayer.LAYER_3_ARCHITECT:
             self.logger.warning(f"Action blocked: Architect Layer is LOCKED. Attempted: {action_type}")
             return False

        # Rule 1: Core Human Permission
        if action_type == "modify_core" and not context.get("human_authorized", False):
            self.logger.critical("VIOLATION ATTEMPT: Rule 1 (Core Human Permission). Blocked.")
            return False

        # Rule 11: Sandbox Requirement
        if action_type == "create_architecture" and not context.get("is_sandbox", False):
             self.logger.critical("VIOLATION ATTEMPT: Rule 11 (Sandbox Requirement). New architecture must be in sandbox.")
             return False

        # Rule 16: No Agent Creation
        if action_type == "create_agent" and not context.get("human_approval", False):
            self.logger.critical("VIOLATION ATTEMPT: Rule 16 (No Agent Creation). Blocked.")
            return False

        # Rule 8: Goal Approval
        if action_type == "set_goal" and context.get("origin") == "system":
            self.logger.critical("VIOLATION ATTEMPT: Rule 8 (Goal Approval). Blocked.")
            return False

        # Rule 31: No Unauthorized Internet
        if action_type == "internet_connect" and not context.get("permit", False):
            self.logger.critical("VIOLATION ATTEMPT: Rule 31 (No Unauthorized Internet). Blocked.")
            return False

        # Log the allowed action (Rule 25, 26, 41)
        self._log_change(action_type, context)
        return True

    def _log_change(self, action_type: str, context: Dict[str, Any]):
        """Rule 25, 26, 41: Logging and Traceability"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action_type,
            "context": context,
            "status": "ALLOWED"
        }
        self.change_log.append(entry)
        self.decision_tree.append(entry) # Rule 27
        self.logger.info(f"Action Allowed: {action_type} | Context: {context}")

    # Emergency Controls (Category 10)
    def freeze_system(self):
        """Rule 47: Freeze Mode"""
        self.system_frozen = True
        self.logger.warning("SYSTEM FROZEN BY HUMAN COMMAND.")

    def unfreeze_system(self):
        self.system_frozen = False
        self.logger.info("System unfrozen by human command.")

    def shutdown_core(self):
        """Rule 46: Immediate Stop / Core Shutdown"""
        self.core_shutdown = True
        self.logger.critical("CORE SHUTDOWN INITIATED.")

    def lock_architect(self):
        """Architect Lock (Derived from general control rules)"""
        self.architect_locked = True
        self.logger.warning("ARCHITECT LAYER LOCKED.")
    
    def unlock_architect(self):
        self.architect_locked = False
        self.logger.info("Architect Layer unlocked.")

    def revert_to_stable(self):
        """Rule 48: Revert Capability"""
        self.logger.warning("REVERTING TO STABLE VERSION...")
        # Logic to revert state would go here
        pass

    def get_directives_text(self) -> str:
        text = ""
        current_cat = None
        for d in self.directives.values():
            if d.category != current_cat:
                text += f"\n--- {d.category.value} ---\n"
                current_cat = d.category
            text += f"{d.id}. {d.title}: {d.description}\n"
        return text

# Singleton instance
governance = GovernanceSystem()
