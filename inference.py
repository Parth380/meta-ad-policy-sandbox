import os
import json
import requests
from openai import OpenAI

# 1. 🚨 MANDATORY VARIABLES EXACTLY AS REQUESTED BY SCALAR
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
        print(f"⚠️ LLM Call Failed: {e}. Defaulting to safe fallback.")
        return {"action_type": "analyze_image", "reasoning": "Error recovery."}

def main() -> None:
    print("🚀 Starting Meta Ad-Policy Automated Inference...")
    total_score = 0.0

    for task_id in TASKS:
        print(f"\n--- 🎬 Starting Episode: {task_id} ---")
        
        try:
            res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            if res.status_code != 200:
                print(f"❌ Env connection failed. Check if Docker is running on port 8000.")
                return
        except requests.exceptions.ConnectionError:
            print(f"❌ Env connection refused. Is your OpenEnv Docker container running?")
            return
            
        observation = res.json()
        done = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS:
            step_count += 1
            print(f"  Step {step_count} | Status: {observation.get('status_message', 'No status')}")
            
            action_payload = get_llm_action(observation)
            print(f"  🤖 Agent Action: {action_payload['action_type'].upper()}")
            
            step_res = requests.post(f"{ENV_URL}/step", json=action_payload)
            step_data = step_res.json()
            
            # Extract from the OpenEnv schema
            observation = step_data.get("observation", step_data)
            done = observation.get("done", False)
            reward = observation.get("reward", 0.0)
            
            if done:
                print(f"  ✅ Episode Finished! Final Step Reward: {reward}")
                total_score += reward

    print(f"\n🎉 Evaluation Complete! Total Agent Score: {total_score} / {len(TASKS)}")

if __name__ == "__main__":
    main()