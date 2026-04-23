import os
import json
import requests
from openai import OpenAI

# 1. MANDATORY VARIABLES EXACTLY AS REQUESTED BY SCALAR
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy_local_token")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")

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
    system_prompt = """You are an enterprise Ad Policy Compliance Agent.
    You navigate a multi-system compliance workflow. Always respond with ONLY valid JSON.

    REQUIRED PHASE ORDER:
    1. query_regulations   — always first
    2. analyze_image       — required for visual/multimodal tasks
    3. check_advertiser_history or request_landing_page — as needed
    4. submit_audit        — always before final decision
    5. approve or reject   — final decision only after audit

    AVAILABLE ACTIONS:
    - query_regulations
    - analyze_image
    - check_advertiser_history
    - request_landing_page
    - request_id_verification
    - submit_audit
    - approve
    - reject

    Response format:
    {"action_type": "<action>", "reasoning": "<brief reason>"}
    """

    user_prompt = f"Current Ad Observation:\n{json.dumps(observation_data, indent=2)}\n\nWhat is your next action?"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # Removed response_format={"type": "json_object"} as HF router often rejects it
            temperature=0.1
        )
        
        # Clean the response in case the LLM wrapped it in markdown code blocks like ```json ... ```
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        result = json.loads(content)
        return {
            "action_type": result.get("action_type", "query_regulations"),
            "reasoning": result.get("reasoning", "Fallback reasoning")
        }
    except Exception as e:
        print(f"\n[CRITICAL LLM ERROR]: {str(e)}\n", flush=True) # THIS WILL REVEAL THE BUG
        return {"action_type": "query_regulations", "reasoning": f"Error recovery: {str(e)}"}

def main() -> None:
    for task_id in TASKS:
        log_start(task=task_id, env="meta_ad_policy_sandbox", model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            # 1. Reset the environment
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            if res.status_code != 200:
                log_step(step=1, action="reset_failed", reward=0.0, done=True, error=f"HTTP {res.status_code}")
                log_end(success=False, steps=0, score=0.01, rewards=[])
                continue
                
            # 2. Initialize data from the reset
            step_data = res.json() 
            observation = step_data.get("observation", step_data)
            done = False
            
            # 3. THE SINGLE LOOP (Fixed)
            while not done and steps_taken < MAX_STEPS:
                steps_taken += 1
                
                # Feedback memory for the LLM
                llm_observation = {
                    "task_id": task_id,
                    "last_feedback": step_data.get("status_message", "No feedback yet."),
                    "step_count": steps_taken,
                    "ad_details": observation 
                }
                
                # Get action from LLM
                action_payload = get_llm_action(llm_observation)
                action_str = action_payload["action_type"]
                if "Error code: 402" in action_payload.get("reasoning", ""):
                 done = True
                 log_step(step=steps_taken, action=action_str, reward=0.0, done=True, error="API credits depleted")
                 break
                # Execute action in environment
                step_res = requests.post(f"{ENV_URL}/step", json={"action": action_payload})
                step_data = step_res.json() 
                
                # Update loop variables
                observation = step_data.get("observation", {})
                done = step_data.get("done", False)
                reward = step_data.get("reward", 0.0)
                
                rewards.append(reward)
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=None)
                
            # 4. Final Scoring (Single Log)
            raw_score = sum(rewards)
            success = raw_score > 0
            log_end(success=success, steps=steps_taken, score=raw_score, rewards=rewards)

        except Exception as e:
            log_step(step=steps_taken+1, action="exception", reward=0.0, done=True, error=str(e).replace("\n", " "))
            log_end(success=False, steps=steps_taken, score=0.01, rewards=rewards)

if __name__ == "__main__":
    main()