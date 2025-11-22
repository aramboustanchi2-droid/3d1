import os, json, time, sys
from datetime import datetime

# Ensure project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from cad3d.super_ai.brain import SuperAIBrain
from cad3d.super_ai.governance import governance
from cad3d.super_ai import councils as councils_mod

APP_PATH = os.path.join(ROOT, 'cad3d', 'dashboard', 'app.py')
STATUS_REPORT_PATH = os.path.join(ROOT, 'FINAL_STATUS_REPORT.json')

brain = SuperAIBrain()

# 1. Health & Maintenance
health_report = []
if hasattr(brain, 'maintenance_crew'):
    try:
        health_report = brain.maintenance_crew.get_report()
    except Exception as e:
        health_report = [{"name":"ERROR","role":"maintenance","status":"failed","health":0,"logs":[str(e)]}]

# 2. Persist state (councils + modules)
save_result = brain.save_all_states()

# Save each council individually
council_save_results = {}
for c in [
    brain.central_agent_council,
    brain.analysis_council,
    brain.ideation_council,
    brain.computational_council,
    brain.economic_council,
    brain.decision_council,
    brain.leadership_council,
]:
    try:
        c.save_state()
        council_save_results[c.name] = "ok"
    except Exception as e:
        council_save_results[c.name] = f"error:{e}"  

# 3. Geometry & feasibility realism checks
with open(APP_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    app_code = f.read()
geometry_realism = all(k in app_code for k in ["build_prism_mesh", "detect_polygon_issues", "dxf_geometry"])
# Feasibility realism: verify agent implementation lives in agents.py and metrics keys present
agents_path = os.path.join(ROOT, 'cad3d', 'super_ai', 'agents.py')
try:
    with open(agents_path, 'r', encoding='utf-8', errors='ignore') as af:
        agents_code = af.read()
except FileNotFoundError:
    agents_code = ''
feasibility_realism = all(k in agents_code for k in ["class FeasibilityAgent", "efficiency_ratio", "daylight_score", "structural_risk"])

# 4. Governance state
gov_state = {
    "system_frozen": governance.system_frozen,
    "core_shutdown": governance.core_shutdown,
    "architect_locked": governance.architect_locked
}

# 5. Feasibility pipeline test
feas_test_context = {
    "site_area": 2500.0,
    "massing_shape": "rect",
    "dimensions": [50.0, 50.0],
    "proposed_height": 64.0
}
feas_response = brain.process_request("Generate building massing feasibility", context_data=feas_test_context)
exec_obj = feas_response.get("execution_result")
feas_report = exec_obj.get("feasibility_report") if isinstance(exec_obj, dict) else None

# 6. Artifact presence verification
artifacts = {}
artifact_files = [
    "super_ai_councils_state.json",
    "cad3d/super_ai/super_ai_knowledge_base.json",
]
# Add council state files
for cname in ["central_agent_command","analysis","ideation","computational","economic","decision","leadership"]:
    artifact_files.append(f"council_{cname}_state.json")

for af in artifact_files:
    ap = os.path.join(ROOT, af) if not af.startswith('cad3d') else os.path.join(ROOT, af)
    exists = os.path.exists(ap)
    size = os.path.getsize(ap) if exists else 0
    artifacts[af] = {"exists": exists, "size": size}

# 7. Maintenance health summary
maintenance_health_summary = {
    "agents": len(health_report),
    "issues_detected": sum(1 for a in health_report if a.get('status') in ['Issue Detected','Fixing']),
    "avg_health": round(sum(a.get('health',0) for a in health_report)/len(health_report),2) if health_report else 0
}

# 8. Consolidated status JSON
data = {
    "timestamp": datetime.utcnow().isoformat()+"Z",
    "save_result": save_result,
    "council_save_results": council_save_results,
    "geometry_realism": geometry_realism,
    "feasibility_realism": feasibility_realism,
    "governance_state": gov_state,
    "feasibility_test_available": feas_report is not None,
    "feasibility_test_summary": feas_report,
    "maintenance_health": maintenance_health_summary,
    "artifacts": artifacts,
    "notes": {
        "geometry_realism_requires_dxf": "DXF upload triggers polygon extraction path in app.py.",
        "governance_enforced": "Requests are blocked if frozen or shutdown.",
        "feasibility_pipeline_auto": "Runs when geometric context is present even if wording absent."
    }
}

with open(STATUS_REPORT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print("STATUS REPORT WRITTEN:", STATUS_REPORT_PATH)
print(json.dumps(data, indent=2)[:800] + "...\n(truncated)")
