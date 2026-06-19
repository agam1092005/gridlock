import os
import yaml


class PlaybookEngine:
    def __init__(self, template_path="src/config/playbooks.yaml"):
        self.template_path = template_path
        self.templates = self._load_templates()

    def _load_templates(self):
        if os.path.exists(self.template_path):
            with open(self.template_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def generate_playbook(self, context):
        incident_type = context.get("incident_type", "generic_fallback").lower()
        severity = context.get("severity_score", 0)

        # Determine severity bucket
        if severity >= 70.0:
            sev_key = "high_severity"
        elif severity >= 40.0:
            sev_key = "medium_severity"
        else:
            sev_key = "low_severity"

        # Get template for incident type, fallback to generic
        incident_template = self.templates.get(
            incident_type, self.templates.get("generic_fallback", {})
        )

        # Get specific severity actions, fallback to high severity if key is missing
        rules = incident_template.get(sev_key, incident_template.get("high_severity", {}))

        actions_list = rules.get("actions", ["Dispatch response team"])
        comms_list = rules.get("communications", ["Issue general advisory"])

        # Extract resource configurations
        manpower = rules.get("manpower", "1 Traffic Constable (Monitor only)")
        barricading = rules.get("barricading", "None required")
        diversion = rules.get("diversion", "No diversion needed.")
        special_equipment = rules.get("special_equipment", "No special equipment needed")
        agency_sync = rules.get("agency_sync", "BTP internal handling only")
        public_advisory = rules.get("public_advisory", "Standard traffic monitoring")

        actions = []
        for i, a in enumerate(actions_list):
            actions.append({"action": a, "priority": i + 1, "timeline": "Immediate"})

        summary = (
            f"Incident type {context.get('incident_type', 'unknown')} "
            f"with severity {severity:.1f} "
            f"and estimated duration {context.get('duration_estimate', 0):.1f} mins."
        )

        return {
            "summary": summary,
            "severity_bucket": sev_key,
            "manpower": manpower,
            "barricading": barricading,
            "diversion": diversion,
            "special_equipment": special_equipment,
            "agency_sync": agency_sync,
            "public_advisory": public_advisory,
            "actions": actions,
            "communications": {
                "public_alert": comms_list[0]
                if comms_list
                else f"Expect delays due to {context.get('incident_type', 'incident')}.",
                "internal_alert": summary,
            },
        }
