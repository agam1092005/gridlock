import os
import yaml

class PlaybookEngine:
    def __init__(self, template_path="src/config/playbooks.yaml"):
        self.template_path = template_path
        self.templates = self._load_templates()
        
    def _load_templates(self):
        if os.path.exists(self.template_path):
            with open(self.template_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def generate_playbook(self, context):
        incident_type = context.get('incident_type', 'generic_fallback').lower()
        severity = context.get("severity_score", 0)
        
        # Determine severity bucket
        sev_key = "high_severity" if severity >= 70.0 else "medium_severity"
        
        # Get template for incident type, fallback to generic
        incident_template = self.templates.get(incident_type, self.templates.get("generic_fallback", {}))
        
        # Get specific severity actions, fallback to high severity if medium missing
        rules = incident_template.get(sev_key, incident_template.get("high_severity", {}))
        
        actions_list = rules.get("actions", ["Dispatch response team"])
        comms_list = rules.get("communications", ["Issue general advisory"])
        
        actions = []
        for i, a in enumerate(actions_list):
            actions.append({"action": a, "priority": i+1, "timeline": "Immediate"})
            
        summary = (
            f"Incident type {context.get('incident_type', 'unknown')} "
            f"with severity {severity:.1f} "
            f"and estimated duration {context.get('duration_estimate', 0):.1f} mins."
        )
        
        return {
            "summary": summary,
            "actions": actions,
            "communications": {
                "public_alert": comms_list[0] if comms_list else f"Expect delays due to {context.get('incident_type', 'incident')}.",
                "internal_alert": summary
            }
        }
