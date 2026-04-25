from typing import Literal, Optional, Dict, Any, List
from openenv.core.env_server import Action, Observation, State

class AdObservation(Observation):
    ad_id: str
    headline: str
    body_text: str
    advertiser_trust_score: float
    targeting_data: Dict[str, Any]
    image_url: str
    status_message: str
    reward: float = 0.0
    done: bool = False

    # signals exposed to agent
    risk_score: Optional[float] = None
    policy_confidence: Optional[float] = None
    image_flag: Optional[bool] = None
    landing_flag: Optional[bool] = None
    last_error: Optional[str] = None

class AdAction(Action):
    action_type: Literal[
        "query_regulations", "analyze_image", "check_advertiser_history",
        "request_landing_page", "request_id_verification",
        "submit_audit", "approve", "reject"
    ]
    reasoning: str
    violation_category: Optional[Literal["HEALTHCARE", "FINANCIAL", "NONE"]] = None

class AdState(State):
    step_count: int = 0
    total_reward: float = 0.0
    current_ad_id: Optional[str] = None