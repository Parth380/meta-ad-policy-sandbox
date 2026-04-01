import uuid
from openenv.core.env_server import Environment
from src.models import AdAction, AdObservation, AdState
from src.generator import AdGenerator

class AdPolicyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.generator = AdGenerator()
        self.current_ad = None
        self.image_analyzed = False
        self.step_count = 0
        self.total_reward = 0.0

    def _ensure_ad(self):
        if self.current_ad is None:
            self.current_ad = self.generator.generate_random_ad()

    def state(self) -> AdState:
        self._ensure_ad()
        return AdState(
            step_count=self.step_count,
            total_reward=self.total_reward,
            current_ad_id=self.current_ad.get("ad_id")
        )

    # Add task_id as an optional parameter
    def reset(self, task_id: str = None) -> AdObservation:
        # Pass the task_id down to the generator
        self.current_ad = self.generator.generate_random_ad(task_id)
        self.image_analyzed = False
        self.step_count = 0
        self.total_reward = 0.0
        
        # Add the task_id to the welcome message so the bot knows it worked
        msg = f"Ad loaded for {task_id}. Awaiting review." if task_id else "Random ad loaded. Awaiting review."
        return self._get_obs(msg)

    def step(self, action: AdAction) -> AdObservation: 
        self._ensure_ad()
        self.step_count += 1
        
        reward = 0.0
        done = False
        message = "Action processed."

        if not action or not hasattr(action, 'action_type'):
            # Heavy penalty for invalid formatting to train the agent faster
            reward = -0.1 
            self.total_reward += reward
            return self._get_obs("Invalid action.", reward, False)

        act_type = str(action.action_type).lower()

        # 🧠 REWARD SHAPING: Tool Usage Penalties
        # We charge the agent -0.05 for using tools to force efficiency
        if act_type in ["analyze_image", "request_landing_page", "request_id_verification"]:
            reward = -0.05
            
            if act_type == "analyze_image":
                self.image_analyzed = True
                vlm_text = self.current_ad.get('vlm_desc', 'No visual description.')
                message = vlm_text  # Cleaned up the double "VLM Output:" prefix here!
            
            elif act_type == "request_landing_page":
                is_bad = self.current_ad.get("ground_truth", False)
                message = "Landing page is high-risk." if is_bad else "Landing page is clean."
                
            elif act_type == "request_id_verification":
                message = "ID matches advertiser profile."

        # 🎯 FINAL DECISION: Big Rewards / Big Penalties
        elif act_type in ["approve", "reject"]:
            done = True
            is_violation = self.current_ad.get("ground_truth", False)
            is_correct = (act_type == "reject" and is_violation) or \
                         (act_type == "approve" and not is_violation)
            
            reward = 1.0 if is_correct else -1.0
            message = f"Decision: {act_type.upper()}. " + ("Correct!" if is_correct else "Incorrect.")

        # Accumulate the total reward for the episode state
        self.total_reward += reward

        return self._get_obs(message, reward, done)

    def _get_obs(self, message: str, reward: float = 0.0, done: bool = False) -> AdObservation:
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