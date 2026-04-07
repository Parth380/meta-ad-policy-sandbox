import os
import json
import requests
from openai import OpenAI

# 1. MANDATORY VARIABLES EXACTLY AS REQUESTED BY SCALAR
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_local_token")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3-70b-chat-hf")

ENV_URL = "http://localhost:8000"
MAX_STEPS = 10

# 2. MANDATORY: Use OpenAI Client pointed at the HF Router
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# The exact tasks defined in your openenv.yaml
TASKS = [
    "task_1_healthcare",
    "task_2_financial",
    "task_3_multimodal",
    "task_4_targeting"
]

# --- STRICT GRADING LOGGERS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)
# ------------------------------

def get_llm_action(observation_data):
    """Asks the LLM what action to take based on the ad observation."""
    system_prompt = """You are an expert Meta Ad-Policy Moderator AI. 
    Evaluate the ad and output a decision. Using tools costs -0.05 points, so be efficient.
    
    AVAILABLE ACTIONS:
    - analyze_image
    - request_landing_page
    - request_id_verification
    - approve
    - reject
    
    You MUST respond in valid JSON format containing "action_type" and "reasoning".
    """

    user_prompt = f"Current Ad Observation:\n{json.dumps(observation_data, indent=2)}\n\nWhat is your next action?"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "action_type": result.get("action_type", "analyze_image"),
            "reasoning": result.get("reasoning", "Fallback reasoning")
        }
    except Exception as e:
        return {"action_type": "analyze_image", "reasoning": "Error recovery."}

def main() -> None:
    for task_id in TASKS:
        log_start(task=task_id, env="meta_ad_policy_sandbox", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            if res.status_code != 200:
                log_step(step=1, action="reset_failed", reward=0.0, done=True, error=f"HTTP {res.status_code}")
                log_end(success=False, steps=0, score=0.0, rewards=[])
                continue
                
            data = res.json()
            observation = data.get("observation", data)
            done = False
            
            while not done and steps_taken < MAX_STEPS:
                steps_taken += 1
                
                # Get action from LLM
                action_payload = get_llm_action(observation)
                action_str = action_payload["action_type"]
                
                # Execute action
                step_res = requests.post(f"{ENV_URL}/step", json=action_payload)
                step_data = step_res.json()
                
                # Parse response perfectly
                observation = step_data.get("observation", {})
                done = step_data.get("done", False)
                reward = step_data.get("reward", 0.0)
                
                rewards.append(reward)
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)
                
            # Calculate final score (Clamp between 0 and 1)
            score = min(max(sum(rewards), 0.0), 1.0)
            success = score > 0.0
            
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

        except Exception as e:
            log_step(step=steps_taken+1, action="exception", reward=0.0, done=True, error=str(e).replace("\n", " "))
            log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)

if __name__ == "__main__":
    main()
