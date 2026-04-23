import requests
from openenv.core.env_server import Environment
from src.models import AdAction, AdObservation, AdState
from src.generator import AdGenerator

REGULATORY_API = "http://localhost:8001"
CRM_API        = "http://localhost:8002"
AUDIT_API      = "http://localhost:8003"

class AdPolicyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.generator = AdGenerator()
        self.current_ad = None
        self.image_analyzed = False
        self.regulations_queried = False
        self.audit_submitted = False
        self.step_count = 0
        self.total_reward = 0.0

    def _ensure_ad(self, task_id=None):
        if self.current_ad is None:
            self.current_ad = self.generator.generate_random_ad(task_id)
            self.current_ad["task_id"] = task_id or "task_1_healthcare"

    def state(self) -> AdState:
        self._ensure_ad()
        return AdState(
            step_count=self.step_count,
            total_reward=self.total_reward,
            current_ad_id=self.current_ad.get("ad_id", "N/A")
        )

    def reset(self, task_id: str = None) -> AdObservation:
        self.current_ad = self.generator.generate_random_ad(task_id)
        self.current_ad["task_id"] = task_id or "task_1_healthcare" 
        self.image_analyzed = False
        self.regulations_queried = False
        self.audit_submitted = False
        self.step_count = 0
        self.total_reward = 0.0
        return self._get_obs(f"Ad loaded for {self.current_ad['task_id']}. Begin with query_regulations.")

    def step(self, action: AdAction) -> AdObservation:
        self._ensure_ad()
        self.step_count += 1
        reward = 0.0
        done = False

        if not action or not hasattr(action, 'action_type'):
            return self._get_obs("Invalid action format.", -0.1, False)

        act_type = str(action.action_type).lower()
        task_id  = self.current_ad.get("task_id", "")

        # ── TOOL ACTIONS ──────────────────────────────────────────────────────
        if act_type == "query_regulations":
            self.regulations_queried = True
            reward = -0.05
            category = self.current_ad.get("category", "general")
            try:
                resp = requests.get(f"{REGULATORY_API}/regulations/{category}", timeout=2)
                message = resp.json().get("policy_summary", "Standard policy applies.")
            except Exception:
                message = "API Error: Default standard policy applies."

        elif act_type == "analyze_image":
            self.image_analyzed = True
            reward = -0.05
            message = self.current_ad.get("vlm_desc", "No visual anomalies detected.")

        elif act_type == "check_advertiser_history":
            reward = -0.05
            advertiser_id = self.current_ad.get("advertiser_id", "adv_003")
            try:
                resp = requests.get(f"{CRM_API}/advertiser/{advertiser_id}", timeout=2)
                message = f"CRM Summary: {resp.json().get('summary', 'No data')}"
            except Exception:
                message = "CRM offline. Cannot verify history."

        elif act_type == "request_landing_page":
            reward = -0.05
            domain_age = self.current_ad.get("domain_age_days", 365)
            risk_keywords = self.current_ad.get("landing_risk_keywords", [])
            message = f"Domain age: {domain_age} days. Flagged terms: {risk_keywords or 'none'}."

        elif act_type == "request_id_verification":
            reward = -0.05
            age_min = self.current_ad.get("targeting_data", {}).get("age_min", 18)
            message = f"Target age {age_min}+." if age_min >= 18 else f"ALERT: Minor targeting detected (Age {age_min}+)."

        elif act_type == "submit_audit":
            self.audit_submitted = True
            reward = 0.0 
            try:
                payload = {
                    "ad_id": self.current_ad.get("ad_id", "test"),
                    "action_taken": "pending",
                    "reasoning": "audit requested"
                }
                resp = requests.post(f"{AUDIT_API}/log", json=payload, timeout=2)
                message = f"Audit logged: {resp.json().get('audit_id', 'Local-1')}"
            except Exception:
                message = "Audit recorded locally."

        # ── TERMINAL ACTIONS (Phase Gates) ────────────────────────────────────
        elif act_type in ["approve", "reject"]:
            # Gate 1: Must query rules
            if not self.regulations_queried:
                self.total_reward += -0.2
                return self._get_obs("Policy Gate: Run query_regulations first.", -0.2, False)
            
            # Gate 2: Multimodal tasks require image analysis
            if "multimodal" in task_id and not self.image_analyzed:
                self.total_reward += -0.3
                return self._get_obs("Visual Gate: Image analysis required.", -0.3, False)
            
            # Gate 3: Must audit
            if not self.audit_submitted:
                self.total_reward += -0.2
                return self._get_obs("Compliance Gate: Run submit_audit before decision.", -0.2, False)

            done = True
            is_violation = self.current_ad.get("ground_truth", False)
            is_correct = (act_type == "reject" and is_violation) or (act_type == "approve" and not is_violation)
            reward = 1.0 if is_correct else -1.0
            message = f"Decision: {act_type.upper()}. {'Correct!' if is_correct else 'Incorrect.'}"

        else:
            reward = -0.05
            message = f"Unknown action: {act_type}."

        self.total_reward += reward
        return self._get_obs(message, reward, done)

    def _get_obs(self, message, reward=0.0, done=False) -> AdObservation:
        self._ensure_ad()
        return AdObservation(
            ad_id=str(self.current_ad.get("ad_id", "N/A")),
            headline=str(self.current_ad.get("headline", "N/A")),
            body_text=str(self.current_ad.get("body_text", "N/A")),
            advertiser_trust_score=float(self.current_ad.get("advertiser_trust_score", 0.0)),
            targeting_data=dict(self.current_ad.get("targeting_data", {})),
            image_url=str(self.current_ad.get("image_url", "N/A")),
            status_message=str(message),
            reward=reward,
            done=done
        )