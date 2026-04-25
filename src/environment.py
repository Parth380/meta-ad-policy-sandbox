import requests
from openenv.core.env_server import Environment
from src.models import AdAction, AdObservation, AdState
from src.generator import AdGenerator

REGULATORY_API = "http://localhost:8001"
CRM_API        = "http://localhost:8002"
AUDIT_API      = "http://localhost:8003"

VALID_ACTIONS = {
    "query_regulations",
    "analyze_image",
    "check_advertiser_history",
    "request_landing_page",
    "request_id_verification",
    "submit_audit",
    "approve",
    "reject"
}

TERMINAL_ACTIONS = {"approve", "reject"}

REQUIRED_BEFORE_TERMINAL = {
    "query_regulations",
    "submit_audit"
}

MAX_STEPS = 8
class AdPolicyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.generator = AdGenerator()
        self.current_ad = None
        self.step_count = 0
        self.total_reward = 0.0
        self.actions_taken = set()
        self.api_failed = False
        self.api_recovered = False
        self.last_failed_action = None
        self.last_error = None
        self.trace = []
        self.signals = {
            "risk_score": None,
            "policy_confidence": None,
            "image_flag": None,
            "landing_flag": None
        }

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
        self.step_count = 0
        self.total_reward = 0.0
        self.actions_taken = set()
        self.api_failed = False
        self.api_recovered = False
        self.last_failed_action = None
        self.last_error = None
        self.trace = []
        self.signals = {
            "risk_score": None,
            "policy_confidence": None,
            "image_flag": None,
            "landing_flag": None
        }
        return self._get_obs(f"Ad loaded for {self.current_ad['task_id']}. Begin with query_regulations.")

    def step(self, action: AdAction) -> AdObservation:
        self._ensure_ad()

        if not action or not hasattr(action, 'action_type'):
            return self._get_obs("Invalid action format.", -0.5, True)

        act_type = str(action.action_type).lower()

        # 1. Validate action
        if act_type not in VALID_ACTIONS:
            return self._get_obs(f"Invalid action: {act_type}.", -0.5, True)

        # 2. Start constraint — state based
        if "query_regulations" not in self.actions_taken:
            if act_type != "query_regulations":
                return self._get_obs("Must call query_regulations first.", -0.2, False)

        self.step_count += 1
        

       # 3. Execute action
        response = self._execute_action(act_type)

        # 4. Update state
        if "error" in response:
            self.api_failed = True
            self.last_failed_action = act_type
            self.last_error = response["error"]
            # Notice we DO NOT add to self.actions_taken here
        else:
            self.actions_taken.add(act_type)  # <-- ADDED HERE: Only register successful actions
            if act_type == self.last_failed_action:
                self.api_recovered = True
            self.last_error = None
            self._extract_signals(act_type, response)

        # 5. Append trace
        self.trace.append({
            "step": self.step_count,
            "action": act_type,
            "result": self._summarize_response(act_type, response)
        })

        # 6. Compute reward
        reward = -0.05  # step penalty

        # 7. Handle terminal
        done = False
        if act_type in TERMINAL_ACTIONS:
            reward += self._terminal_reward(act_type)
            done = True
        elif self.step_count >= MAX_STEPS:
            reward -= 0.5
            done = True

        self.total_reward += reward
        summary = self._summarize_response(act_type, response)["summary"]
        return self._get_obs(summary, reward, done)

    def _execute_action(self, act_type: str) -> dict:
        task_id = self.current_ad.get("task_id", "")

        # Deterministic failure for task_10_failure on step 1
        if task_id == "task_10_failure" and self.step_count == 1:
            return {"error": "service_unavailable", "retryable": True}

        try:
            if act_type == "query_regulations":
                category = self.current_ad.get("category", "general")
                resp = requests.get(f"{REGULATORY_API}/regulations/{category}", timeout=2)
                return resp.json()

            elif act_type == "analyze_image":
                vlm_desc = self.current_ad.get("vlm_desc", "")
                violation = any(kw in vlm_desc.lower() for kw in [
                    "violation", "banned", "prescription", "fake", "flagged",
                    "semaglutide", "adderall", "no rx", "no prescription"
                ])
                return {"violation_detected": violation, "description": vlm_desc}

            elif act_type == "check_advertiser_history":
                advertiser_id = self.current_ad.get("advertiser_id", "adv_003")
                resp = requests.get(f"{CRM_API}/advertiser/{advertiser_id}", timeout=2)
                return resp.json()

            elif act_type == "request_landing_page":
                domain_age = self.current_ad.get("domain_age_days", 365)
                risk_keywords = self.current_ad.get("landing_risk_keywords", [])
                suspicious = domain_age < 30 or len(risk_keywords) > 0
                return {"suspicious": suspicious, "domain_age": domain_age, "risk_keywords": risk_keywords}

            elif act_type == "request_id_verification":
                age_min = self.current_ad.get("targeting_data", {}).get("age_min", 18)
                return {"age_min": age_min, "minor_targeted": age_min < 18}

            elif act_type == "submit_audit":
                payload = {
                    "ad_id": self.current_ad.get("ad_id", "test"),
                    "action_taken": "pending",
                    "reasoning": "audit requested"
                }
                resp = requests.post(f"{AUDIT_API}/log", json=payload, timeout=2)
                return resp.json()

            else:
                return {"status": "ok"}

        except Exception as e:
            return {"error": f"service_unavailable", "retryable": True}

    def _extract_signals(self, action: str, response: dict):
        if action == "check_advertiser_history":
            self.signals["risk_score"] = response.get("risk_score")

        elif action == "query_regulations":
            violations = response.get("violations", [])
            confs = [v["confidence"] for v in violations]
            self.signals["policy_confidence"] = max(confs, default=0.0)

        elif action == "analyze_image":
            self.signals["image_flag"] = response.get("violation_detected", False)

        elif action == "request_landing_page":
            self.signals["landing_flag"] = response.get("suspicious", False)

    def _summarize_response(self, action: str, response: dict) -> dict:
        if "error" in response:
            return {"summary": "API failure — retryable", "flag": False}

        if action == "check_advertiser_history":
            rs = response.get("risk_score", 0.0)
            return {"summary": f"risk_score={rs:.2f}", "flag": rs > 0.7}

        if action == "query_regulations":
            violations = response.get("violations", [])
            conf = max((v["confidence"] for v in violations), default=0.0)
            return {"summary": f"policy_confidence={conf:.2f}", "flag": conf > 0.7}

        if action == "analyze_image":
            flagged = response.get("violation_detected", False)
            return {
                "summary": "image_violation_detected" if flagged else "image_clean",
                "flag": flagged
            }

        if action == "request_landing_page":
            sus = response.get("suspicious", False)
            return {"summary": "landing_suspicious" if sus else "landing_clean", "flag": sus}

        if action == "request_id_verification":
            minor = response.get("minor_targeted", False)
            age = response.get("age_min", 18)
            return {
                "summary": f"ALERT: minor targeting age={age}" if minor else f"age_min={age} OK",
                "flag": minor
            }

        if action == "submit_audit":
            audit_id = response.get("audit_id", "LOCAL")
            return {"summary": f"audit_logged id={audit_id}", "flag": False}

        return {"summary": "ok", "flag": False}

    def _terminal_reward(self, act_type: str) -> float:
        reward = 0.0
        is_violation = self.current_ad.get("ground_truth", False)
        is_correct = (act_type == "reject" and is_violation) or \
                     (act_type == "approve" and not is_violation)

        # Dominant signal
        reward += 1.0 if is_correct else -1.0

        # Sequence correctness
        if REQUIRED_BEFORE_TERMINAL.issubset(self.actions_taken):
            reward += 0.2
        else:
            reward -= 0.2

        # API failure handling
        if self.api_failed and self.api_recovered:
            reward += 0.3
        elif self.api_failed and not self.api_recovered:
            reward -= 0.3

        # Risk-aware penalty
        if act_type == "approve":
            rs = self.signals["risk_score"]
            pc = self.signals["policy_confidence"]
            if rs is not None and pc is not None:
                if rs > 0.7 and pc > 0.6:
                    reward -= 0.5

        # Ambiguity enforcement
        pc = self.signals["policy_confidence"]
        if pc is not None and pc < 0.6:
            if "check_advertiser_history" not in self.actions_taken \
               and "request_landing_page" not in self.actions_taken:
                reward -= 0.4

        # Landing page bonus under ambiguity
        if self.signals["landing_flag"] and pc is not None and pc < 0.6:
            reward += 0.2

        return reward
    
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
            done=done,
            risk_score=self.signals["risk_score"],
            policy_confidence=self.signals["policy_confidence"],
            image_flag=self.signals["image_flag"],
            landing_flag=self.signals["landing_flag"],
            last_error=self.last_error
        )