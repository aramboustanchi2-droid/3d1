import pytest
from cad3d.super_ai.brain import SuperAIBrain

def test_deep_learning_workflow(tmp_path):
    brain = SuperAIBrain()
    
    # 1. Train the system (Simulated)
    dataset_path = "dummy/path/to/dataset"
    train_result = brain.train_system(dataset_path)
    assert "Training Completed" in train_result
    assert brain.learning_module.is_trained
    
    # 2. Run a request that benefits from learning
    request = "Perform feasibility study"
    
    # Create a dummy site file so InputParser doesn't fail
    site_file = tmp_path / "site.dxf"
    site_file.write_text("dummy content")
    
    context = {
        "site_area": 2000.0,
        "input_path": str(site_file) # Use absolute path
    }
    
    # Mock Leadership to force Feasibility Pipeline
    original_deliberate = brain.leadership_council.deliberate
    def mock_leadership_deliberate(input_data):
        from cad3d.super_ai.councils import Proposal
        return Proposal(
            id="mock_id",
            content={
                "announcement": "Proceed with Feasibility Study.",
                "directives": ["Run Hektar Pipeline"]
            },
            source="Leadership",
            confidence=1.0
        )
    brain.leadership_council.deliberate = mock_leadership_deliberate
    
    result = brain.process_request(request, context_data=context)
    
    assert result["status"] == "success"
    execution = result["execution_result"]
    
    # Check if AI insights were used
    report = execution["feasibility_report"]
    assert report["ai_enhanced"] is True
    assert report["recommended_massing"]["source"] == "AI_Model"
    # Check if it used the learned "perimeter_block_with_courtyard" typology
    assert "perimeter_block" in report["recommended_massing"]["type"]
