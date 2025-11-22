from typing import List, Dict, Any, Optional
import logging
from .agents import (
    Agent, 
    InputParserAgent, 
    ArchitecturalAnalyzerAgent, 
    EngineeringValidatorAgent, 
    ModelingConfigurationAgent, 
    OutputGeneratorAgent,
    FeasibilityAgent
)

logger = logging.getLogger(__name__)

class Pipeline:
    """
    Represents a linear sequence of agents working on a shared context.
    """
    def __init__(self, name: str, description: str, agents: List[Agent]):
        self.name = name
        self.description = description
        self.agents = agents

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the pipeline.
        """
        logger.info(f"Starting Pipeline: {self.name}")
        context = initial_context.copy()
        
        for agent in self.agents:
            try:
                logger.info(f"Pipeline Step: {agent.name} ({agent.role})")
                context = agent.execute(context)
            except Exception as e:
                logger.error(f"Pipeline failed at agent {agent.name}: {e}")
                context["error"] = str(e)
                context["failed_step"] = agent.name
                break
                
        logger.info(f"Pipeline {self.name} completed.")
        return context

class PipelineRegistry:
    """
    Factory and registry for available pipelines.
    """
    @staticmethod
    def get_standard_2d_to_3d_pipeline() -> Pipeline:
        return Pipeline(
            name="Standard_2D_to_3D",
            description="Converts 2D DXF/DWG plans to 3D models with validation.",
            agents=[
                InputParserAgent(),
                ArchitecturalAnalyzerAgent(),
                EngineeringValidatorAgent(),
                ModelingConfigurationAgent(),
                OutputGeneratorAgent()
            ]
        )

    @staticmethod
    def get_quick_preview_pipeline() -> Pipeline:
        return Pipeline(
            name="Quick_Preview",
            description="Fast conversion without deep analysis.",
            agents=[
                InputParserAgent(),
                ModelingConfigurationAgent(),
                OutputGeneratorAgent()
            ]
        )

    @staticmethod
    def get_feasibility_study_pipeline() -> Pipeline:
        return Pipeline(
            name="Hektar_Style_Feasibility",
            description="Generative design and massing study for site analysis.",
            agents=[
                InputParserAgent(),
                FeasibilityAgent(),
                OutputGeneratorAgent()
            ]
        )
