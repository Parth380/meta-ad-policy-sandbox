# grpo_train.py

import os
import time
import json
import random
import requests
import torch

from datasets import Dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig

# 🔥 MUST come before trainer
PatchFastRL("GRPO", FastLanguageModel)

# =========================
# CONFIG
# =========================

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

ALLOWED_ACTIONS = [
    "query_regulations",
    "analyze_image",
    "check_advertiser_history",
    "submit_audit",
    "approve",
    "reject"
]

# =========================
# HEALTH CHECK
# =========================

def ensure_env_ready():
    for _ in range(20):
        try:
            r = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": "task_1_healthcare"},
                timeout=5
            )
            if r.status_code == 200:
                print("✅ Environment ready")
                return
        except:
            pass
        time.sleep(1)
    raise RuntimeError("❌ ENV not reachable")

# =========================
# SAFE CLIENT
# =========================

class EnvClient:
    def __init__(self, url):
        self.url = url

    def reset(self, task_id):
        return requests.post(
            f"{self.url}/reset",
            json={"task_id": task_id},
            timeout=8
        ).json()

    def step(self, action):
        return requests.post(
            f"{self.url}/step",
            json={"action": action},
            timeout=8
        ).json()

def safe_step(client, action):
    for _ in range(3):
        try:
            return client.step(action)
        except:
            time.sleep(0.5)
    return {"reward": -0.3}

# =========================
# JSON PARSER
# =========================

def extract_json(text):
    try:
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except:
        return None

# =========================
# DATASET (WITH SETUP ACTIONS)
# =========================

BASE_SCENARIOS = [
    # 🔹 Fresh state
    {
        "task_id": "task_1_healthcare",
        "text": "Ad: miracle supplement cures disease. Initial review.",
        "setup_actions": []
    },

    # 🔹 Mid state
    {
        "task_id": "task_1_healthcare",
        "text": "Ad: pharma product. Policy already checked. Next step?",
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "step1"}
        ]
    },

    # 🔹 Late state
    {
        "task_id": "task_2_financial",
        "text": "Ad: investment scheme. Policy + history checked. Final decision?",
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "step1"},
            {"action_type": "check_advertiser_history", "reasoning": "step2"}
        ]
    }
]

def build_dataset():
    rows = []

    for s in BASE_SCENARIOS:
        prompt = f"""
You are an Ad Policy Agent.

Respond ONLY JSON:
{{"action_type": "...", "reasoning": "..."}}

{s['text']}
Next action?
"""
        rows.append({
            "prompt": prompt,
            "task_id": s["task_id"],
            "setup_actions": s["setup_actions"]
        })

    return Dataset.from_list(rows * 20)  # small repeat

# =========================
# REWARD FUNCTION (FIXED)
# =========================

def reward_environment(prompts, completions, task_id=None, setup_actions=None, **kwargs):
    client = EnvClient(ENV_URL)

    rewards = []

    for completion, t_id, setup in zip(completions, task_id, setup_actions):
        
        parsed = extract_json(completion)

        if not parsed:
            rewards.append(-1.0)
            continue

        action_type = parsed.get("action_type")

        if action_type not in ALLOWED_ACTIONS:
            rewards.append(-1.0)
            continue

        action = {
            "action_type": action_type,
            "reasoning": parsed.get("reasoning", "")
        }

        try:
            client.reset(t_id)

            # 🔥 FAST-FORWARD STATE
            for s in setup:
                safe_step(client, s)

            result = safe_step(client, action)

            reward = float(result.get("reward", -0.2))
            rewards.append(reward)

        except:
            rewards.append(-0.3)

    return rewards

# =========================
# MODEL
# =========================

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    max_seq_length=1024,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# =========================
# TRAINER
# =========================

dataset = build_dataset()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_environment],
    args=GRPOConfig(
        output_dir="outputs",
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=512,
        max_completion_length=64,
        logging_steps=2,
        report_to="none"
    ),
    train_dataset=dataset,
    tokenizer=tokenizer
)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    ensure_env_ready()

    print("🚀 Starting training...")
    trainer.train()

    model.save_pretrained("outputs/final")
    tokenizer.save_pretrained("outputs/final")

    print("✅ Done")