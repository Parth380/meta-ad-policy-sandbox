import json
import torch
import requests
from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# MUST be called before trainer instantiation
PatchFastRL("GRPO", FastLanguageModel)

ENV_URL = "http://localhost:8000"
TASKS = ["task_1_healthcare", "task_2_financial",
         "task_3_multimodal", "task_4_targeting"]

SYSTEM_PROMPT = """You are an enterprise Ad Policy Compliance Agent.
Always respond with ONLY valid JSON, no markdown.

REQUIRED PHASE ORDER:
1. query_regulations  — always first
2. analyze_image      — required for multimodal tasks  
3. submit_audit       — always before final decision
4. approve or reject  — only after audit

Format: {"action_type": "<action>", "reasoning": "<reason>"}"""

# ── DATASET ───────────────────────────────────────────────────────────────────

def build_dataset():
    rows = []
    for task_id in TASKS:
        res = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
        obs = res.json()
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Task: {task_id}\n"
            f"Ad: {obs.get('headline','N/A')} — {obs.get('body_text','N/A')}\n"
            f"Trust Score: {obs.get('advertiser_trust_score','N/A')}\n"
            f"Status: {obs.get('status_message','')}\n"
            f"What is your next action?"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        rows.append({"prompt": prompt, "task_id": task_id})
    # 25x repetition = 100 rows, enough for 1 epoch
    return Dataset.from_list(rows * 25)

# ── REWARD FUNCTION (actually calls the environment) ──────────────────────────

def reward_environment(prompts, completions, task_ids, **kwargs):
    """
    This is the real reward — model outputs an action,
    we send it to the environment, environment returns the reward.
    """
    rewards = []
    for completion, task_id in zip(completions, task_ids):
        try:
            # Parse model output
            content = completion.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            action = json.loads(content.strip())
            action_type = action.get("action_type", "query_regulations")
        except Exception:
            # Malformed JSON = penalty
            rewards.append(-0.5)
            continue

        try:
            # Fresh episode for each reward calculation
            requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            
            # Run a minimal sequence: if model says query_regulations,
            # run that then check what reward it generates
            step_res = requests.post(
                f"{ENV_URL}/step",
                json={"action": {"action_type": action_type, 
                                 "reasoning": action.get("reasoning", "")}},
                timeout=5
            )
            data = step_res.json()
            rewards.append(float(data.get("reward", -0.1)))
        except Exception:
            rewards.append(-0.1)

    return rewards

def reward_json_format(prompts, completions, **kwargs):
    """Bonus reward for valid JSON output."""
    rewards = []
    for completion in completions:
        try:
            content = completion.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            json.loads(content.strip())
            rewards.append(0.5)
        except Exception:
            rewards.append(-0.5)
    return rewards

# ── MODEL SETUP ───────────────────────────────────────────────────────────────

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    use_gradient_checkpointing="unsloth",
)

# ── TRAINER ───────────────────────────────────────────────────────────────────

dataset = build_dataset()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_environment, reward_json_format],
    args=GRPOConfig(
        output_dir="outputs/meta-ad-agent",
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_prompt_length=512,
        max_completion_length=128,
        num_generations=4,          # lower = faster, enough for demo
        logging_steps=5,
        save_steps=50,
        report_to="none",
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    print("Starting GRPO training — environment must be running on :8000")
    trainer.train()
    model.save_pretrained("outputs/meta-ad-agent-final")
    tokenizer.save_pretrained("outputs/meta-ad-agent-final")
    print("Done. Model saved to outputs/meta-ad-agent-final")