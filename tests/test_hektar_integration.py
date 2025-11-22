import pytest
from cad3d.super_ai.brain import SuperAIBrain

def test_hektar_learning_integration(tmp_path):
    brain = SuperAIBrain()
    
    # Simulate a request for feasibility study
    request = "Perform a feasibility study and massing for this site."
    
    # Create a dummy site input
    site_file = tmp_path / "site_boundary.dxf"
    site_file.write_text("dummy site data")
    
    context = {
        "input_path": str(site_file),
        "site_area": 5000.0
    }
    
    # We need to mock the Council's decision to ensure it triggers the right pipeline
    # Since we don't have a real LLM, we can rely on the Brain's heuristic 
    # which checks the *proposal* content.
    # However, the current Council implementation is hardcoded to return "The Council has decided: Execute Plan A".
    # To make this test pass without changing the hardcoded Council logic too much,
    # we might need to update the Council logic to be slightly more dynamic or mock it.
    
    # Let's update the DecisionCouncil in the test to return a "feasibility" decision
    # This simulates the "learning" where the council now knows about feasibility.
    
    original_deliberate = brain.leadership_council.deliberate
    
    def mock_leadership_deliberate(input_data):
        from cad3d.super_ai.councils import Proposal
        return Proposal(
            id="mock_id",
            content={
                "announcement": "The Council has decided: Proceed with Feasibility Study and Massing.",
                "directives": ["Run Hektar Pipeline"]
            },
            source="Leadership",
            confidence=1.0
        )
        
    brain.leadership_council.deliberate = mock_leadership_deliberate
    
    result = brain.process_request(request, context_data=context)
    
    assert result["status"] == "success"
    execution = result["execution_result"]
    
    # Check if FeasibilityAgent ran
    assert "feasibility_report" in execution
    assert execution["feasibility_report"]["site_area"] == 5000.0
    assert "recommended_massing" in execution["feasibility_report"]
