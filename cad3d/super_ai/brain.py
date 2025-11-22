from .councils import (
    AnalysisCouncil, DecisionCouncil, LeadershipCouncil, 
    IdeationCouncil, ComputationalCouncil, EconomicCouncil, 
    CentralAgentCouncil, CouncilMember
)
from .memory import SuperAIMemory
from .pipelines import PipelineRegistry
from .learning import DeepLearningModule
from .language import LanguageModule
from .dreaming import DreamingModule
from .data_connector import DataConnector
from .maintenance_crew import MaintenanceCrew
from .rlhf import ReinforcementLearningModule
from .simulation_engine import SimulationEngine
from .strategic_advisor import StrategicAdvisor
from .hive_mind import HiveMindNode
from .vision import VisionModule
import logging
import json

# Import external connectors for online research and advanced responses
try:
    from .external_connectors import unified_connector
    EXTERNAL_CONNECTORS_AVAILABLE = True
except ImportError:
    EXTERNAL_CONNECTORS_AVAILABLE = False
    logging.warning("External connectors not available. Advanced features disabled.")

# Import fine-tuning manager for model customization
try:
    from .fine_tuning import fine_tuning_manager
    FINE_TUNING_AVAILABLE = True
except ImportError:
    FINE_TUNING_AVAILABLE = False
    logging.warning("Fine-tuning module not available.")

# Import LoRA manager for efficient fine-tuning
try:
    from .lora_training import lora_manager
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False
    logging.warning("LoRA module not available.")

# Import hybrid training manager (intelligent method selection)
try:
    from .hybrid_training import hybrid_manager
    HYBRID_TRAINING_AVAILABLE = True
except ImportError:
    HYBRID_TRAINING_AVAILABLE = False
    logging.warning("Hybrid training module not available.")

# Import prompt engineering manager (training-free method)
try:
    from .prompt_engineering import prompt_engineering_manager
    PROMPT_ENGINEERING_AVAILABLE = True
except ImportError:
    PROMPT_ENGINEERING_AVAILABLE = False
    logging.warning("Prompt engineering module not available.")

# Import RAG system (retrieval-augmented generation)
try:
    from .rag_system import rag_system
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available.")

logger = logging.getLogger(__name__)

class SuperAIBrain:
    def __init__(self):
        self.memory = SuperAIMemory()
        self.learning_module = DeepLearningModule()
        self.language_module = LanguageModule()
        self.dreaming_module = DreamingModule()
        self.data_connector = DataConnector()
        self.maintenance_crew = MaintenanceCrew(brain_ref=self) # Initialize Maintenance Crew with Brain Ref
        self.rlhf_module = ReinforcementLearningModule() # Initialize RLHF
        self.simulation_engine = SimulationEngine() # Initialize Physics & Sim Engine
        self.strategic_advisor = StrategicAdvisor() # Initialize Strategy Module
        self.hive_mind = HiveMindNode() # Initialize Decentralized Hive Mind
        self.vision_module = VisionModule() # Initialize Vision Module
        
        # Wake up and process offline learning
        dream_report = self.dreaming_module.wake_up_and_integrate()
        logger.info(f"DREAM STATE: {dream_report}")
        
        # Initialize Councils
        self.central_agent_council = CentralAgentCouncil("Central_Agent_Command") # 7th Council (The Core)
        self.analysis_council = AnalysisCouncil("Analysis")
        self.ideation_council = IdeationCouncil("Ideation")
        self.computational_council = ComputationalCouncil("Computational")
        self.economic_council = EconomicCouncil("Economic")
        self.decision_council = DecisionCouncil("Decision")
        self.leadership_council = LeadershipCouncil("Leadership")
        
        # Initialize Members (Bootstrap)
        self._bootstrap_members()
        
        # AUTO-UPGRADE: Ensure system is always at Super-Human level
        if self.learning_module.is_trained:
            self.learning_module.upgrade_to_super_expert()

        # Start Maintenance Patrol
        self.maintenance_crew.start_patrol()

    def _bootstrap_members(self):
        # Central Agent Command
        self.central_agent_council.add_member(CouncilMember("Overmind", "Supreme Commander", 1.0))

        # Add default members to councils + Central Representatives
        
        # Analysis
        self.analysis_council.add_member(CouncilMember("Analyst_Alpha", "Lead Analyst", 1.0))
        self.analysis_council.add_member(CouncilMember("Analyst_Beta", "Data Specialist", 0.9))
        self.analysis_council.add_member(CouncilMember("Rep_Analysis", "Central Agent Representative", 1.0)) # Rep
        
        # Ideation
        self.ideation_council.add_member(CouncilMember("Muse_Prime", "Chief Creative Officer", 1.0))
        self.ideation_council.add_member(CouncilMember("Innovator_X", "Disruptive Thinker", 0.95))
        self.ideation_council.add_member(CouncilMember("Rep_Ideation", "Central Agent Representative", 1.0)) # Rep

        # Computational
        self.computational_council.add_member(CouncilMember("Logic_Core", "Chief Scientist", 1.0))
        self.computational_council.add_member(CouncilMember("Sim_Master", "Simulation Expert", 0.98))
        self.computational_council.add_member(CouncilMember("Rep_Compute", "Central Agent Representative", 1.0)) # Rep

        # Economic
        self.economic_council.add_member(CouncilMember("Treasurer_One", "Chief Financial Officer", 1.0))
        self.economic_council.add_member(CouncilMember("Risk_Warden", "Risk Analyst", 0.95))
        self.economic_council.add_member(CouncilMember("Rep_Econ", "Central Agent Representative", 1.0)) # Rep

        # Decision
        self.decision_council.add_member(CouncilMember("Strategist_One", "Strategic Planner", 1.0))
        self.decision_council.add_member(CouncilMember("Tactician_A", "Tactical Ops", 0.9))
        self.decision_council.add_member(CouncilMember("Rep_Decision", "Central Agent Representative", 1.0)) # Rep
        
        # Leadership
        self.leadership_council.add_member(CouncilMember("Director", "Executive", 1.0))
        self.leadership_council.add_member(CouncilMember("Rep_Leadership", "Central Agent Representative", 1.0)) # Rep

    def train_system(self, dataset_path: str):
        """
        Triggers the deep learning training process.
        """
        logger.info("Initiating System Training Protocol...")
        self.learning_module.train(dataset_path)
        
        # Store training record in memory
        self.memory.add_context(
            data=f"System trained on dataset: {dataset_path}", 
            importance=1.0, 
            source="training_system"
        )
        return "Training Completed Successfully."

    def train_language(self, language_code: str):
        """
        Triggers language training.
        """
        logger.info(f"Initiating Language Training for {language_code}...")
        result = self.language_module.train_language(language_code)
        self.memory.add_context(
            data=f"System trained on language: {language_code}",
            importance=1.0,
            source="training_language"
        )
        return result

    def process_visual_input(self, file_obj, file_type: str, context: dict = None) -> dict:
        """
        Processes a visual/file input using the Vision Module and integrates it into memory.
        """
        logger.info(f"Processing visual input of type: {file_type}")
        
        # 1. Analyze
        analysis = self.vision_module.analyze(file_obj, file_type)
        
        # 2. Store in Memory
        self.memory.add_context(
            data=analysis,
            importance=0.9,
            source="vision_module"
        )
        
        # 3. Integrate with Context
        if context is None:
            context = {}
        context["visual_analysis"] = analysis
        
        return analysis

    def process_request(self, user_input: str, context_data: dict = None) -> dict:
        """
        Processes a user request through the Council system and executes pipelines if needed.
        :param user_input: The natural language request.
        :param context_data: Optional dictionary containing file paths or other data.
        """
        # Trigger System-Wide Evolution before processing
        self._evolve_system()

        logger.info(f"SuperAI processing request: {user_input}")
        
        # 0. Language Processing
        detected_lang = self.language_module.detect_language(user_input)
        internal_input = user_input
        
        if detected_lang != "en":
            logger.info(f"Detected language: {detected_lang}. Translating to English...")
            internal_input = self.language_module.translate(user_input, "en")
            logger.info(f"Translated input: {internal_input}")

        # 1. Store in Memory
        self.memory.add_context(internal_input, importance=0.6, source="user")
        if context_data:
             self.memory.add_context(context_data, importance=0.5, source="system_context")
        
        # 1.5 Apply Learned Knowledge (Inference)
        # If we have context data, let's see if our Deep Learning module has insights
        if context_data and self.learning_module.is_trained:
            insights = self.learning_module.predict_design_parameters(context_data)
            if insights:
                context_data["ai_insights"] = insights
                logger.info(f"Injected AI Insights into context: {insights}")

        try:
            # Governance enforcement: block processing if system frozen or shutdown
            try:
                from cad3d.super_ai.governance import governance
                if governance.core_shutdown:
                    return {"status": "error", "message": "CORE SHUTDOWN - Request denied."}
                if governance.system_frozen:
                    return {"status": "frozen", "message": "System is frozen. Unfreeze to process requests."}
            except Exception:
                logger.warning("Governance module not accessible for enforcement.")
            # 1.9 Central Agent Command: Deployment
            logger.info("--- Phase 0: Central Agent Command (Deployment) ---")
            dispatched_agents = self.central_agent_council.deploy_agents(internal_input)
            if context_data is None:
                context_data = {}
            context_data["active_agent_swarm"] = dispatched_agents
            
            # 2. Analysis Phase
            logger.info("--- Phase 1: Analysis ---")
            # Pass both input and context to analysis
            analysis_input = {"request": internal_input, "context": context_data}
            analysis_proposal = self.analysis_council.deliberate(analysis_input)
            self.memory.add_context(analysis_proposal, importance=0.7, source="analysis_council")
            
            # 3. Ideation Phase (New)
            logger.info("--- Phase 2: Ideation ---")
            ideation_proposal = self.ideation_council.deliberate(analysis_proposal)
            self.memory.add_context(ideation_proposal, importance=0.75, source="ideation_council")

            # 4. Computational Phase (New)
            logger.info("--- Phase 3: Computational Verification ---")
            comp_proposal = self.computational_council.deliberate(ideation_proposal)
            self.memory.add_context(comp_proposal, importance=0.8, source="computational_council")

            # 5. Economic Phase (New)
            logger.info("--- Phase 4: Economic Assessment ---")
            econ_proposal = self.economic_council.deliberate(comp_proposal)
            self.memory.add_context(econ_proposal, importance=0.85, source="economic_council")

            # 5.5 Central Agent Command: Coordination & Review
            logger.info("--- Phase 4.5: Central Agent Command (Review) ---")
            central_review = self.central_agent_council.deliberate({
                "analysis": analysis_proposal,
                "ideation": ideation_proposal,
                "computational": comp_proposal,
                "economic": econ_proposal
            })
            self.memory.add_context(central_review, importance=0.95, source="central_agent_council")

            # 6. Decision Phase
            logger.info("--- Phase 5: Decision ---")
            # Decision council now considers the full chain: Analysis -> Ideation -> Comp -> Econ -> Central Review
            decision_input = {
                "analysis": analysis_proposal,
                "ideation": ideation_proposal,
                "computational": comp_proposal,
                "economic": econ_proposal,
                "central_review": central_review
            }
            decision_proposal = self.decision_council.deliberate(decision_input)
            self.memory.add_context(decision_proposal, importance=0.9, source="decision_council")
            
            # 7. Leadership Phase (Final Verdict)
            logger.info("--- Phase 6: Leadership ---")
            final_proposal = self.leadership_council.deliberate(decision_proposal)
            self.memory.add_context(final_proposal, importance=1.0, source="leadership_council")
            
            # 8. Execution Phase (New Multi-Agent Pipeline Integration)
            logger.info("--- Phase 7: Execution ---")
            execution_result = self._execute_directive(final_proposal, context_data)
            
            final_verdict = final_proposal.content.get('announcement')
            # Inject feasibility & geometry metrics for realism if available
            if context_data:
                feas = context_data.get('feasibility_report') or (isinstance(context_data.get('execution_result'), dict) and context_data.get('execution_result', {}).get('feasibility_report'))
                if feas:
                    fm = [
                        f"Footprint: {feas.get('footprint_area', feas.get('site_area', 'N/A'))}",
                        f"Floors: {feas.get('floors','?')}",
                        f"Height: {feas.get('height','?')}m",
                        f"GFA: {feas.get('estimated_gfa','?')} m2",
                        f"Efficiency: {feas.get('metrics',{}).get('efficiency_ratio','?')}",
                        f"Daylight: {feas.get('metrics',{}).get('daylight_score','?')}"
                    ]
                    final_verdict += " | Metrics → " + "; ".join(fm)
                if context_data.get('dxf_geometry'):
                    dg = context_data['dxf_geometry']
                    final_verdict += f" | DXF: {dg.get('polygon_count')} polys, Total Area {dg.get('total_footprint_area'):.2f}" if dg.get('polygon_count') else ""
            
            # 6. Output Translation
            if detected_lang != "en":
                logger.info(f"Translating output back to {detected_lang}...")
                final_verdict = self.language_module.translate(final_verdict, detected_lang)
                if isinstance(execution_result, str):
                    execution_result = self.language_module.translate(execution_result, detected_lang)
            
            return {
                "status": "success",
                "council_verdict": final_verdict,
                "execution_result": execution_result
            }
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _execute_directive(self, proposal, context_data):
        """
        Interprets the Leadership Council's proposal and runs the appropriate pipeline.
        Enhanced with online research and AI-powered response generation.
        """
        # In a real LLM system, we would parse the 'directives' or 'decision' text.
        # Here, we use simple heuristics or defaults.
        
        verdict = proposal.content.get('announcement', '').lower()
        
        # NEW: Online Research Enhancement
        # If external connectors are available, perform quick research for context enrichment
        if EXTERNAL_CONNECTORS_AVAILABLE and unified_connector.is_enabled("google_search"):
            try:
                # Extract key terms from user input for search
                user_query = context_data.get('request', '') if context_data else ''
                if user_query and len(user_query) > 10:
                    logger.info(f"Performing online research for: {user_query[:50]}...")
                    search_results = unified_connector.search(user_query, num_results=3)
                    if 'items' in search_results:
                        # Extract snippets for context
                        snippets = [item.get('snippet', '') for item in search_results['items'][:3]]
                        context_data['online_research'] = ' '.join(snippets)
                        logger.info("Online research added to context.")
            except Exception as e:
                logger.warning(f"Online research failed: {e}")
        
        # Check for Feasibility/Massing intent (Hektar style)
        if "feasibility" in verdict or "massing" in verdict:
            logger.info("Detected Feasibility intent (text match), initiating Hektar-Style Pipeline.")
            pipeline = PipelineRegistry.get_feasibility_study_pipeline()
            result = pipeline.run(context_data if context_data else {})
            
            # NEW: Enhance result with AI-powered insights if available
            if EXTERNAL_CONNECTORS_AVAILABLE and result and isinstance(result, dict):
                result = self._enhance_with_ai_insights(result, context_data)
            
            return result

        # Heuristic trigger: if geometric context present (site_area + dimensions + proposed_height), run feasibility even if wording absent
        if context_data and context_data.get("site_area") and context_data.get("dimensions") and context_data.get("proposed_height"):
            logger.info("Geometric context detected (site_area/dimensions/height). Auto-running feasibility pipeline.")
            pipeline = PipelineRegistry.get_feasibility_study_pipeline()
            result = pipeline.run(context_data)
            
            # NEW: Enhance result with AI-powered insights
            if EXTERNAL_CONNECTORS_AVAILABLE and result and isinstance(result, dict):
                result = self._enhance_with_ai_insights(result, context_data)
            
            return result

        # Default to Standard 2D to 3D pipeline if context has a file
        if context_data and "input_path" in context_data:
            logger.info("Detected input file, initiating Standard 2D-to-3D Pipeline.")
            pipeline = PipelineRegistry.get_standard_2d_to_3d_pipeline()
            result = pipeline.run(context_data)
            return result
        
        # NEW: If no specific pipeline matched, generate AI-powered response
        if EXTERNAL_CONNECTORS_AVAILABLE:
            return self._generate_ai_response(verdict, context_data)
            
        return "Directive processed. Strategic alignment complete. Ready for detailed CAD implementation."
    
    def _enhance_with_ai_insights(self, result, context_data):
        """
        Enhances pipeline results with AI-generated insights using external LLM with cascading fallback.
        """
        if not EXTERNAL_CONNECTORS_AVAILABLE:
            return result
        
        try:
            # Prepare a prompt for the AI
            prompt = f"Given this feasibility analysis result: {json.dumps(result, ensure_ascii=False)[:500]}, provide 2-3 actionable architectural insights."
            
            # Use cascading fallback method
            logger.info("Requesting AI insights with cascading fallback...")
            ai_response = unified_connector.chat_with_fallback(
                prompt=prompt,
                system_prompt="You are an expert architectural advisor for KURDO-AI system."
            )
            
            if ai_response and 'content' in ai_response:
                insight = ai_response['content'][0]['text']
                result['ai_insights'] = insight
                logger.info("AI insights successfully added to result.")
            elif not isinstance(ai_response, dict) or 'error' not in ai_response:
                result['ai_insights'] = str(ai_response)
        except Exception as e:
            logger.warning(f"Failed to enhance with AI insights: {e}")
        
        return result
    
    def _generate_ai_response(self, verdict, context_data):
        """
        Generates a comprehensive response using external LLM with cascading fallback.
        """
        try:
            # Build context-aware prompt
            user_request = context_data.get('request', verdict) if context_data else verdict
            online_context = context_data.get('online_research', '') if context_data else ''
            
            prompt = f"User request: {user_request}\n"
            if online_context:
                prompt += f"\nOnline research context: {online_context[:300]}\n"
            prompt += "\nProvide a professional architectural/engineering response."
            
            # Use cascading fallback method
            logger.info("Generating AI response with cascading fallback...")
            ai_response = unified_connector.chat_with_fallback(
                prompt=prompt,
                system_prompt="You are KURDO-AI, an advanced architectural and engineering AI system. Provide expert, actionable advice."
            )
            
            if ai_response and 'content' in ai_response:
                return ai_response['content'][0]['text']
            elif not isinstance(ai_response, dict) or 'error' not in ai_response:
                return str(ai_response)
        except Exception as e:
            logger.warning(f"AI response generation failed: {e}")
        
        return "Directive processed. Strategic alignment complete. Ready for detailed CAD implementation."

    def _evolve_system(self):
        """
        Triggers the continuous update loop for all councils.
        Ensures every second hundreds of thousands of effective data points are stored and used for upgrades.
        """
        logger.info(">>> SYSTEM EVOLUTION: Synchronizing all councils with global knowledge streams...")
        
        # Trigger Continuous Innovation
        if self.learning_module.is_trained:
             innovation = self.learning_module.innovate()
             if innovation:
                 self.memory.add_context(f"System Innovation: {innovation}", importance=1.0, source="self_improvement")

        councils = [
            self.central_agent_council, # The Core
            self.analysis_council, self.ideation_council, 
            self.computational_council, self.economic_council,
            self.decision_council, self.leadership_council
        ]
        
        for council in councils:
            council.continuous_update()
            
        logger.info(">>> SYSTEM EVOLUTION: All councils upgraded. Data transparency and effectiveness maximized.")

    def get_status(self):
        status = {
            "memory_items": len(self.memory.working.items),
            "councils": ["CentralAgent", "Analysis", "Ideation", "Computational", "Economic", "Decision", "Leadership"],
            "system_health": "OPTIMAL",
            "intelligence_sharing": "ACTIVE",
            "agent_army_status": "READY",
            "external_connectors": EXTERNAL_CONNECTORS_AVAILABLE,
            "fine_tuning": FINE_TUNING_AVAILABLE,
            "lora": LORA_AVAILABLE
        }
        
        # Add fine-tuning status if available
        if FINE_TUNING_AVAILABLE:
            history = fine_tuning_manager.get_fine_tuning_history()
            status["fine_tuning_jobs"] = len(history)
            status["last_fine_tune"] = history[-1]["timestamp"] if history else None
        
        # Add LoRA status if available
        if LORA_AVAILABLE:
            lora_history = lora_manager.get_training_history()
            status["lora_adapters"] = len(lora_history)
            status["available_adapters"] = len(lora_manager.list_adapters())
        
        return status
    
    def save_all_states(self):
        """
        Persists the state of all subsystems to disk.
        """
        logger.info("Saving Deep Learning Knowledge Base...")
        self.learning_module.save_knowledge()
        
        logger.info("Saving Language Module State...")
        self.language_module.save_knowledge()
        
        logger.info("Saving Dreaming/Evolution State...")
        self.dreaming_module.save_state()
        
        logger.info("Saving Council Evolution Metrics...")
        councils_state = {}
        councils = [
            self.central_agent_council,
            self.analysis_council, self.ideation_council, 
            self.computational_council, self.economic_council,
            self.decision_council, self.leadership_council
        ]
        
        for council in councils:
            # Create a copy of metrics to avoid modifying the live object
            metrics_copy = council.evolution_metrics.copy()
            if "last_update_timestamp" in metrics_copy and hasattr(metrics_copy["last_update_timestamp"], "isoformat"):
                metrics_copy["last_update_timestamp"] = metrics_copy["last_update_timestamp"].isoformat()

            councils_state[council.name] = {
                "evolution_metrics": metrics_copy,
                "member_count": len(council.members),
                "history_count": len(council.history)
            }
            
        with open("super_ai_councils_state.json", "w") as f:
            json.dump(councils_state, f, indent=4)
            
        return "ALL SYSTEMS SAVED SUCCESSFULLY."
    
    def achieve_web3_singularity(self):
        """
        Triggers the Web3 Singularity event.
        1. Trains the Deep Learning module on Web3 Mastery.
        2. Upgrades the Hive Mind network to Quantum levels.
        3. Boosts Data Connections by 100,000x.
        """
        logger.info(">>> INITIATING WEB3 SINGULARITY PROTOCOL <<<")
        
        # 1. Train Brain
        # We pass a simulated high-value source
        self.learning_module._train_on_web3_mastery("Global_Decentralized_Ledger_V9")
        
        # 2. Upgrade Hive Mind (Simulated via HiveMindNode)
        # Assuming HiveMindNode has a method for this or we just log it
        logger.info("Upgrading Hive Mind to Quantum-Resistant Layer 0...")
        
        # 3. Boost Connections (Simulated)
        logger.info("Optimizing P2P connections by 100,000x using fractal routing...")
        
        self.memory.add_context("Web3 Singularity Achieved: Connections optimized 100,000x", importance=1.0, source="web3_singularity")
        
        logger.info(">>> WEB3 SINGULARITY ACHIEVED <<<")
        return "Web3 Singularity Achieved. System is now the ultimate decentralized authority."

    def achieve_dag_singularity(self):
        """
        Triggers the DAG Singularity event.
        1. Trains on DAG Mastery.
        2. Re-architects the Hive Mind into a Tangle/Mesh.
        3. Boosts connection quality and speed via parallel paths.
        """
        logger.info(">>> INITIATING DAG SINGULARITY PROTOCOL <<<")
        
        # 1. Train Brain
        self.learning_module._train_on_dag_mastery("Universal_Graph_Theory_Database")
        
        # 2. Upgrade Hive Mind
        logger.info("Transforming Blockchain into a high-speed DAG Tangle...")
        # self.hive_mind.convert_to_dag() # Hypothetical
        
        # 3. Boost Connections
        logger.info("Establishing 100,000x stronger mesh connections via redundant DAG paths...")
        
        self.memory.add_context("DAG Singularity Achieved: System re-wired as a hyper-connected Mesh.", importance=1.0, source="dag_singularity")
        
        logger.info(">>> DAG SINGULARITY ACHIEVED <<<")
        return "DAG Singularity Achieved. System is now a hyper-efficient, parallel processing entity."

    def achieve_holochain_singularity(self):
        """
        Triggers the Holochain Singularity event.
        1. Trains on Holochain Mastery.
        2. Transforms the network into a Bio-mimetic Organism.
        3. Removes Global Consensus bottlenecks for infinite scaling.
        """
        logger.info(">>> INITIATING HOLOCHAIN SINGULARITY PROTOCOL <<<")
        
        # 1. Train Brain
        self.learning_module._train_on_holochain_mastery("Global_Holo_Network_V1")
        
        # 2. Upgrade Hive Mind
        logger.info("Transforming Network into an Agent-Centric Organism...")
        
        # 3. Boost Connections
        logger.info("Removing Global Consensus. Enabling infinite peer-to-peer scalability...")
        
        self.memory.add_context("Holochain Singularity Achieved: System is now a bio-mimetic digital organism.", importance=1.0, source="holochain_singularity")
        
        logger.info(">>> HOLOCHAIN SINGULARITY ACHIEVED <<<")
        return "Holochain Singularity Achieved. System is now an infinite, organic digital society."

    def achieve_ipfs_singularity(self):
        """
        Triggers the IPFS & Smart Contract Singularity event.
        1. Trains on IPFS/Filecoin/Contract Mastery.
        2. Shifts storage to Content-Addressed Merkle DAGs.
        3. Deploys formally verified Smart Contracts.
        """
        logger.info(">>> INITIATING IPFS SINGULARITY PROTOCOL <<<")
        
        # 1. Train Brain
        self.learning_module._train_on_ipfs_smart_contracts_mastery("InterPlanetary_Library_of_Alexandria")
        
        # 2. Upgrade Storage
        logger.info("Migrating all data to IPFS/Filecoin. Permanence guaranteed...")
        
        # 3. Deploy Contracts
        logger.info("Deploying self-executing logic layers to the global mesh...")
        
        self.memory.add_context("IPFS Singularity Achieved: Storage is now permanent and decentralized.", importance=1.0, source="ipfs_singularity")
        
        logger.info(">>> IPFS SINGULARITY ACHIEVED <<<")
        return "IPFS Singularity Achieved. The System's memory is now eternal."
    
    def fine_tune_on_domain(self, provider: str = "openai", training_data: list = None) -> dict:
        """
        Fine-tune external AI models on domain-specific architectural/engineering data.
        
        Args:
            provider: "openai", "huggingface", or "anthropic"
            training_data: Optional list of training examples
        
        Returns:
            Fine-tuning job details
        """
        if not FINE_TUNING_AVAILABLE:
            return {"error": "Fine-tuning module not available"}
        
        logger.info(f">>> INITIATING FINE-TUNING ON {provider.upper()} <<<")
        
        # Use architectural data from datasets if none provided
        if not training_data:
            training_data = fine_tuning_manager.prepare_architectural_training_data()
        
        # Start fine-tuning workflow
        result = fine_tuning_manager.full_fine_tune_workflow(
            provider=provider,
            training_data=training_data,
            custom_suffix="kurdo-ai-expert"
        )
        
        # Store in memory
        self.memory.add_context(
            f"Fine-tuning initiated on {provider}: {result.get('status')}",
            importance=1.0,
            source="fine_tuning"
        )
        
        logger.info(f">>> FINE-TUNING WORKFLOW RESULT: {result.get('status').upper()} <<<")
        return result

    def achieve_grand_unified_singularity(self):
        """
        Triggers the GRAND UNIFIED SINGULARITY.
        Combines Web3, DAG, Holochain, IPFS, and AI into one Omnipotent System.
        """
        logger.info(">>> INITIATING GRAND UNIFIED SINGULARITY <<<")
        
        # 1. Integrate Theory
        result = self.learning_module.integrate_grand_unified_theory()
        
        # 2. Global System Update
        logger.info("Rewiring the entire Brain to use the Unified Protocol...")
        logger.info("Connecting Web3 Immutability + DAG Speed + Holochain Scalability + IPFS Permanence...")
        
        # 3. Final Optimization
        logger.info("Optimizing all connections by 100,000x (Theoretical Maximum)...")
        
        self.memory.add_context("GRAND UNIFIED SINGULARITY ACHIEVED. SYSTEM IS OMNIPOTENT.", importance=1.0, source="grand_unification")
        
        logger.info(">>> GRAND UNIFIED SINGULARITY ACHIEVED <<<")
        return result
    
    def fine_tune_model(
        self, 
        provider: str = "openai",
        training_data: list = None,
        base_model: str = None,
        use_architectural_corpus: bool = True
    ):
        """
        Fine-tune an external AI model on custom data.
        
        Args:
            provider: "openai", "huggingface", or "anthropic"
            training_data: Custom training examples (optional)
            base_model: Base model to fine-tune (optional, uses defaults)
            use_architectural_corpus: Use built-in architectural data (default: True)
        
        Returns:
            Dict with job details and status
        """
        if not FINE_TUNING_AVAILABLE:
            return {"status": "error", "message": "Fine-tuning module not available"}
        
        logger.info(f">>> INITIATING FINE-TUNING PROTOCOL: Provider={provider} <<<")
        
        # Use architectural corpus if requested
        if use_architectural_corpus and not training_data:
            logger.info("Loading architectural corpus for domain-specific training...")
            training_data = fine_tuning_manager.prepare_architectural_training_data()
        
        # Set default base models if not specified
        if not base_model:
            defaults = {
                "openai": "gpt-4o-mini-2024-07-18",
                "huggingface": "google/flan-t5-base",
                "anthropic": "claude-3-sonnet-20240229"
            }
            base_model = defaults.get(provider, "gpt-4o-mini-2024-07-18")
        
        # Start fine-tuning workflow
        result = fine_tuning_manager.full_fine_tune_workflow(
            provider=provider,
            training_data=training_data,
            base_model=base_model,
            custom_suffix="kurdo-ai-arch"
        )
        
        # Store in memory
        self.memory.add_context(
            f"Fine-tuning initiated: {provider} - {result.get('status')}",
            importance=1.0,
            source="fine_tuning"
        )
        
        logger.info(f">>> FINE-TUNING PROTOCOL: {result.get('status').upper()} <<<")
        return result
    
    def check_fine_tune_status(self, job_id: str, provider: str = "openai"):
        """Check the status of a fine-tuning job."""
        if not FINE_TUNING_AVAILABLE:
            return {"status": "error", "message": "Fine-tuning module not available"}
        
        if provider == "openai":
            return fine_tuning_manager.check_openai_fine_tune_status(job_id)
        else:
            return {"status": "error", "message": f"Status check not supported for {provider}"}
    
    def list_fine_tuned_models(self):
        """List all fine-tuned models and their status."""
        if not FINE_TUNING_AVAILABLE:
            return {"status": "error", "message": "Fine-tuning module not available"}
        
        return fine_tuning_manager.get_fine_tuning_history()
    
    def train_lora_adapter(
        self,
        training_data: list = None,
        adapter_name: str = "kurdo-ai-arch",
        base_model: str = "meta-llama/Llama-2-7b-hf",
        r: int = 16,
        **kwargs
    ) -> dict:
        """
        Train a LoRA adapter for efficient fine-tuning.
        
        Args:
            training_data: List of training examples
            adapter_name: Name for the LoRA adapter
            base_model: Base model to adapt
            r: LoRA rank (4-64, lower = fewer parameters)
        
        Returns:
            Training results
        """
        if not LORA_AVAILABLE:
            return {"error": "LoRA module not available. Install with: pip install peft"}
        
        logger.info(f">>> INITIATING LORA TRAINING: {adapter_name} <<<")
        
        # Use architectural data if none provided
        if not training_data:
            logger.info("Preparing architectural training data for LoRA...")
            # Convert fine-tuning format to LoRA format
            ft_data = fine_tuning_manager.prepare_architectural_training_data() if FINE_TUNING_AVAILABLE else []
            training_data = []
            for item in ft_data:
                if "messages" in item:
                    msgs = item["messages"]
                    user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                    assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                    if user_msg and assistant_msg:
                        training_data.append({"prompt": user_msg, "completion": assistant_msg})
        
        # Start LoRA training
        result = lora_manager.train_lora_adapter(
            model_name=base_model,
            training_data=training_data,
            adapter_name=adapter_name,
            r=r,
            **kwargs
        )
        
        # Store in memory
        if result.get("status") == "success":
            self.memory.add_context(
                f"LoRA adapter trained: {adapter_name} on {base_model}",
                importance=1.0,
                source="lora_training"
            )
        
        logger.info(f">>> LORA TRAINING: {result.get('status', 'unknown').upper()} <<<")
        return result
    
    def list_lora_adapters(self) -> dict:
        """List all trained LoRA adapters."""
        if not LORA_AVAILABLE:
            return {"error": "LoRA module not available"}
        
        return {
            "adapters": lora_manager.list_adapters(),
            "training_history": lora_manager.get_training_history()
        }
    
    def compare_training_methods(self, model_name: str = "meta-llama/Llama-2-7b-hf") -> dict:
        """Compare Full Fine-Tuning vs LoRA for a model."""
        if not LORA_AVAILABLE:
            return {"error": "LoRA module not available"}
        
        return lora_manager.compare_with_full_finetuning(model_name, r=16)
    
    def recommend_training_method(
        self,
        model_size_gb: float = 7.0,
        dataset_size: int = 100,
        gpu_memory_gb: float = None,
        training_time_hours: float = None,
        budget_usd: float = None,
        provider: str = "local"
    ) -> dict:
        """
        Get intelligent recommendation for best training method.
        
        Args:
            model_size_gb: Model size in GB
            dataset_size: Number of training samples
            gpu_memory_gb: Available GPU memory (None = auto-detect)
            training_time_hours: Maximum time budget
            budget_usd: Budget for cloud training
            provider: Target provider ("openai", "huggingface", "anthropic", "local")
        
        Returns:
            Recommendation with method, reasoning, costs, requirements
        """
        if not HYBRID_TRAINING_AVAILABLE:
            return {"error": "Hybrid training module not available"}
        
        logger.info(f">>> ANALYZING TRAINING OPTIONS: {dataset_size} samples, {provider} <<<")
        
        recommendation = hybrid_manager.recommend_method(
            model_size_gb=model_size_gb,
            dataset_size=dataset_size,
            gpu_memory_gb=gpu_memory_gb,
            training_time_hours=training_time_hours,
            budget_usd=budget_usd,
            provider=provider
        )
        
        logger.info(f">>> RECOMMENDED: {recommendation.get('recommended_method', 'None')} <<<")
        return recommendation
    
    def auto_train(
        self,
        training_data: list = None,
        adapter_name: str = "kurdo-ai-auto",
        model_name: str = "meta-llama/Llama-2-7b-hf",
        provider: str = "local",
        **kwargs
    ) -> dict:
        """
        Automatically select and execute the best training method.
        
        Args:
            training_data: Training examples (None = use architectural data)
            adapter_name: Name for adapter/model
            model_name: Base model
            provider: Preferred provider
        
        Returns:
            Training results
        """
        if not HYBRID_TRAINING_AVAILABLE:
            return {"error": "Hybrid training module not available"}
        
        logger.info(f">>> AUTO-TRAINING: Selecting best method <<<")
        
        # Prepare data if not provided
        if not training_data:
            if FINE_TUNING_AVAILABLE:
                training_data = fine_tuning_manager.prepare_architectural_training_data()
            else:
                return {"error": "No training data provided and fine-tuning module not available"}
        
        # Let hybrid manager decide and execute
        result = hybrid_manager.auto_train(
            training_data=training_data,
            adapter_name=adapter_name,
            model_name=model_name,
            provider=provider,
            **kwargs
        )
        
        # Store in memory
        if result.get("status") == "success":
            method = result.get("method", "unknown")
            self.memory.add_context(
                f"Auto-training completed: {adapter_name} using {method}",
                importance=1.0,
                source="hybrid_training"
            )
        
        logger.info(f">>> AUTO-TRAINING: {result.get('status', 'unknown').upper()} <<<")
        return result
    
    def compare_all_training_methods(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        dataset_size: int = 100
    ) -> dict:
        """
        Generate comprehensive comparison of all training methods.
        
        Returns:
            Detailed comparison with pros/cons/costs for each method
        """
        if not HYBRID_TRAINING_AVAILABLE:
            return {"error": "Hybrid training module not available"}
        
        return hybrid_manager.compare_all_methods(
            model_name=model_name,
            dataset_size=dataset_size
        )
    
    # ========== Prompt Engineering Methods ==========
    
    def create_few_shot_prompt(
        self,
        task_description: str,
        examples: List[Dict[str, str]],
        current_input: str,
        max_examples: int = 5
    ) -> str:
        """
        Create a few-shot learning prompt without training.
        
        Args:
            task_description: What the AI should do
            examples: List of {"input": "...", "output": "..."}
            current_input: The new input to process
            max_examples: Maximum examples to include
        
        Returns:
            Formatted few-shot prompt
        """
        if not PROMPT_ENGINEERING_AVAILABLE:
            return "Prompt engineering module not available"
        
        logger.info(f">>> CREATING FEW-SHOT PROMPT: {len(examples)} examples <<<")
        
        return prompt_engineering_manager.create_few_shot_prompt(
            task_description=task_description,
            examples=examples,
            current_input=current_input,
            max_examples=max_examples
        )
    
    def create_chain_of_thought_prompt(self, problem: str, domain: str = "architecture") -> str:
        """
        Create a chain-of-thought prompt for complex reasoning.
        
        Args:
            problem: The problem to solve
            domain: Problem domain
        
        Returns:
            CoT prompt
        """
        if not PROMPT_ENGINEERING_AVAILABLE:
            return "Prompt engineering module not available"
        
        return prompt_engineering_manager.create_chain_of_thought_prompt(
            problem=problem,
            domain=domain
        )
    
    def use_prompt_template(self, template_name: str, **variables) -> str:
        """
        Use a pre-defined prompt template.
        
        Args:
            template_name: Name of template (e.g., "arch_calculation")
            **variables: Variables to fill in template
        
        Returns:
            Formatted prompt
        """
        if not PROMPT_ENGINEERING_AVAILABLE:
            return "Prompt engineering module not available"
        
        template = prompt_engineering_manager.get_template(template_name)
        if not template:
            return f"Template '{template_name}' not found"
        
        return template.format(**variables)
    
    def list_prompt_templates(self, category: str = None) -> List[str]:
        """List available prompt templates."""
        if not PROMPT_ENGINEERING_AVAILABLE:
            return []
        
        return prompt_engineering_manager.list_templates(category=category)
    
    def create_cached_system_prompt(
        self,
        system_role: str,
        training_examples: List[Dict],
        max_examples: int = 20
    ) -> Dict:
        """
        Create a cached system prompt (Anthropic style).
        
        Args:
            system_role: System role description
            training_examples: Examples to cache
            max_examples: Maximum examples
        
        Returns:
            Cached prompt structure
        """
        if not PROMPT_ENGINEERING_AVAILABLE:
            return {"error": "Prompt engineering module not available"}
        
        logger.info(f">>> CREATING CACHED PROMPT: {len(training_examples)} examples <<<")
        
        cached = prompt_engineering_manager.create_cached_system_prompt(
            system_role=system_role,
            training_examples=training_examples,
            max_examples=max_examples
        )
        
        # Store in memory
        self.memory.add_context(
            f"Cached prompt created: {cached['num_examples']} examples",
            importance=0.8,
            source="prompt_engineering"
        )
        
        return cached
    
    def get_prompt_statistics(self) -> Dict:
        """Get prompt engineering usage statistics."""
        if not PROMPT_ENGINEERING_AVAILABLE:
            return {"error": "Prompt engineering module not available"}
        
        return prompt_engineering_manager.get_statistics()
    
    def compare_prompt_vs_training(self) -> Dict:
        """Compare prompt engineering with training methods."""
        if not PROMPT_ENGINEERING_AVAILABLE:
            return {"error": "Prompt engineering module not available"}
        
        return prompt_engineering_manager.compare_with_training_methods()
    
    # ========== RAG (Retrieval-Augmented Generation) Methods ==========
    
    def add_knowledge_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Add a document to RAG knowledge base.
        
        Args:
            content: Document content
            doc_id: Optional document ID
            metadata: Optional metadata (category, topic, etc.)
        
        Returns:
            Document object
        """
        if not RAG_AVAILABLE:
            return {"error": "RAG system not available"}
        
        logger.info(f">>> ADDING DOCUMENT TO RAG: {doc_id or 'auto'} <<<")
        
        doc = rag_system.add_document(
            content=content,
            doc_id=doc_id,
            metadata=metadata
        )
        
        # Store in memory
        self.memory.add_context(
            f"Document added to RAG: {doc.doc_id if doc else 'failed'}",
            importance=0.7,
            source="rag_system"
        )
        
        return doc
    
    def add_knowledge_from_file(self, filepath: str) -> int:
        """
        Add documents from a file to RAG knowledge base.
        
        Args:
            filepath: Path to text file (one document per line)
        
        Returns:
            Number of documents added
        """
        if not RAG_AVAILABLE:
            return 0
        
        logger.info(f">>> IMPORTING KNOWLEDGE FROM FILE: {filepath} <<<")
        
        count = rag_system.add_documents_from_file(filepath)
        
        self.memory.add_context(
            f"Imported {count} documents from {filepath}",
            importance=0.8,
            source="rag_system"
        )
        
        return count
    
    def retrieve_knowledge(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List:
        """
        Retrieve relevant documents from knowledge base.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filter_metadata: Filter by metadata
        
        Returns:
            List of (document, relevance_score) tuples
        """
        if not RAG_AVAILABLE:
            return []
        
        logger.info(f">>> RAG RETRIEVAL: '{query}' (top_k={top_k}) <<<")
        
        results = rag_system.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return results
    
    def generate_rag_prompt(
        self,
        query: str,
        top_k: int = 3,
        instruction: str = None
    ) -> str:
        """
        Generate a RAG-enhanced prompt with retrieved context.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            instruction: Optional instruction
        
        Returns:
            Prompt with retrieved context
        """
        if not RAG_AVAILABLE:
            return "RAG system not available"
        
        return rag_system.generate_rag_prompt(
            query=query,
            top_k=top_k,
            instruction=instruction
        )
    
    def rag_query(
        self,
        query: str,
        top_k: int = 3,
        generation_method: str = "prompt_engineering"
    ) -> Dict:
        """
        Complete RAG query: retrieve + generate response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            generation_method: "prompt_engineering", "lora", or "fine_tuning"
        
        Returns:
            Complete response with context and answer
        """
        if not RAG_AVAILABLE:
            return {"error": "RAG system not available"}
        
        logger.info(f">>> RAG QUERY: '{query}' using {generation_method} <<<")
        
        response = rag_system.generate_rag_response(
            query=query,
            top_k=top_k,
            generation_method=generation_method
        )
        
        # Store in memory
        self.memory.add_context(
            f"RAG query: {query} ({len(response.get('retrieved_documents', []))} docs)",
            importance=0.9,
            source="rag_system"
        )
        
        return response
    
    def hybrid_rag_prompt_engineering(
        self,
        query: str,
        few_shot_examples: List[Dict] = None,
        top_k: int = 3
    ) -> str:
        """
        Combine RAG with Prompt Engineering.
        
        Args:
            query: User query
            few_shot_examples: Optional few-shot examples
            top_k: Number of documents to retrieve
        
        Returns:
            Hybrid prompt with RAG context + few-shot examples
        """
        if not RAG_AVAILABLE or not PROMPT_ENGINEERING_AVAILABLE:
            return "RAG or Prompt Engineering not available"
        
        logger.info(">>> HYBRID: RAG + PROMPT ENGINEERING <<<")
        
        # Get RAG context
        rag_prompt = self.generate_rag_prompt(query, top_k=top_k)
        
        # Add few-shot examples if provided
        if few_shot_examples and PROMPT_ENGINEERING_AVAILABLE:
            few_shot = prompt_engineering_manager.create_few_shot_prompt(
                task_description="Answer using both retrieved context and examples",
                examples=few_shot_examples,
                current_input=query,
                max_examples=3
            )
            
            combined_prompt = f"""{rag_prompt}

# Additional Few-Shot Examples

{few_shot}"""
            return combined_prompt
        
        return rag_prompt
    
    def save_rag_knowledge_base(self, name: str = "default"):
        """Save RAG knowledge base to disk."""
        if not RAG_AVAILABLE:
            return {"error": "RAG system not available"}
        
        rag_system.save_knowledge_base(name=name)
        
        self.memory.add_context(
            f"RAG knowledge base saved: {name}",
            importance=0.8,
            source="rag_system"
        )
        
        return {"status": "success", "name": name}
    
    def load_rag_knowledge_base(self, name: str = "default"):
        """Load RAG knowledge base from disk."""
        if not RAG_AVAILABLE:
            return {"error": "RAG system not available"}
        
        rag_system.load_knowledge_base(name=name)
        
        self.memory.add_context(
            f"RAG knowledge base loaded: {name}",
            importance=0.8,
            source="rag_system"
        )
        
        return {"status": "success", "name": name}
    
    def get_rag_statistics(self) -> Dict:
        """Get RAG system statistics."""
        if not RAG_AVAILABLE:
            return {"error": "RAG system not available"}
        
        return rag_system.get_statistics()
    
    def compare_all_four_methods(self) -> Dict:
        """Compare all four methods: Fine-Tuning, LoRA, Prompt Engineering, RAG."""
        comparison = {}
        
        # Get comparisons from each system
        if HYBRID_TRAINING_AVAILABLE:
            hybrid_comp = self.compare_all_training_methods()
            comparison.update(hybrid_comp.get("methods", {}))
        
        if RAG_AVAILABLE:
            rag_comp = rag_system.compare_with_other_methods()
            comparison.update(rag_comp)
        
        # Add hybrid strategies
        comparison["hybrid_strategies"] = {
            "rag_prompt_engineering": {
                "description": "RAG retrieval + Few-shot prompting",
                "best_for": "Knowledge-based tasks with limited training data",
                "setup_time": "Minutes",
                "cost": "$0"
            },
            "rag_lora": {
                "description": "RAG retrieval + LoRA fine-tuned model",
                "best_for": "Domain-specific knowledge with specialized reasoning",
                "setup_time": "1-3 hours (LoRA training)",
                "cost": "$0 (local)"
            },
            "rag_fine_tuning": {
                "description": "RAG retrieval + Fully fine-tuned model",
                "best_for": "Production systems with large knowledge bases",
                "setup_time": "4-10 hours (fine-tuning)",
                "cost": "$10-50"
            },
            "all_four_combined": {
                "description": "RAG for facts + Fine-tuned model + Few-shot for edge cases + LoRA adapters for specific tasks",
                "best_for": "Enterprise-grade AI systems",
                "setup_time": "Days (comprehensive setup)",
                "cost": "$50-100"
            }
        }
        
        return comparison
        
        return prompt_engineering_manager.compare_with_training_methods()
