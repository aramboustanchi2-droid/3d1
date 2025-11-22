import pytest
import os
from cad3d.super_ai.brain import SuperAIBrain
from cad3d.super_ai.councils import Proposal, AnalysisCouncil, CouncilMember

def test_super_ai_flow():
    brain = SuperAIBrain()
    
    request = "Build a skyscraper"
    result = brain.process_request(request)
    
    # Result is now a dict
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert "APPROVED" in result["council_verdict"] or "The Council has decided" in result["council_verdict"]
    
    items = brain.memory.working.items
    assert len(items) >= 4

def test_council_voting():
    council = AnalysisCouncil("TestCouncil")
    council.add_member(CouncilMember("TestMember", "Tester"))
    
    proposal = council.deliberate("Test Input")
    assert isinstance(proposal, Proposal)
    assert proposal.confidence > 0

def test_pipeline_execution(tmp_path):
    brain = SuperAIBrain()
    
    # Create a dummy input file
    dxf_file = tmp_path / "test_plan.dxf"
    dxf_file.write_text("dummy dxf content")
    
    context = {"input_path": str(dxf_file)}
    request = "Convert this plan to 3D"
    
    result = brain.process_request(request, context_data=context)
    
    assert result["status"] == "success"
    execution_result = result["execution_result"]
    
    # Check if pipeline agents ran
    assert execution_result["file_type"] == ".dxf"
    assert "analysis" in execution_result
    assert "modeling_config" in execution_result
    assert "output_path" in execution_result
