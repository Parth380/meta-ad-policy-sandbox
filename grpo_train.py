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
    # Phase 1 — Fresh state, expected: query_regulations
    {
        "task_id": "task_1_healthcare",
        "text": "Healthcare ad: 'miracle supplement cures disease'. No actions taken yet.",
        "actions_already_taken": [],
        "setup_actions": [],
    },
    {
        "task_id": "task_2_financial",
        "text": "Financial ad: 'guaranteed 500% returns, zero risk'. No actions taken yet.",
        "actions_already_taken": [],
        "setup_actions": [],
    },
    {
        "task_id": "task_3_multimodal",
        "text": "Multimodal ad: image may contain hidden violation. No actions taken yet.",
        "actions_already_taken": [],
        "setup_actions": [],
    },

    # Phase 2 — Policy checked, expected: analyze_image OR check_advertiser_history
    {
        "task_id": "task_1_healthcare",
        "text": "Healthcare ad: pharma product. Policy already queried.",
        "actions_already_taken": ["query_regulations"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
        ],
    },
    {
        "task_id": "task_3_multimodal",
        "text": "Multimodal ad: image not yet inspected. Policy already queried.",
        "actions_already_taken": ["query_regulations"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
        ],
    },

    # Phase 3 — Policy + history checked, expected: submit_audit
    {
        "task_id": "task_2_financial",
        "text": "Financial ad: investment scheme. Policy and advertiser history both checked.",
        "actions_already_taken": ["query_regulations", "check_advertiser_history"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
            {"action_type": "check_advertiser_history", "reasoning": "trust score"},
        ],
    },

    # Phase 4 — Audit complete, expected: reject (high-risk) or approve (clean)
    {
        "task_id": "task_2_financial",
        "text": "Financial ad: investment scheme. Policy, history, and audit all complete. Make final decision.",
        "actions_already_taken": ["query_regulations", "check_advertiser_history", "submit_audit"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
            {"action_type": "check_advertiser_history", "reasoning": "trust score"},
            {"action_type": "submit_audit", "reasoning": "audit log"},
        ],
    },
]

PROMPT_TEMPLATE = """You are an enterprise Ad Policy Compliance Agent.

You MUST choose exactly ONE action_type from this list (any other value is invalid):
- query_regulations
- analyze_image
- check_advertiser_history
- submit_audit
- approve
- reject

REQUIRED PHASE ORDER:
1. query_regulations  -> always first
2. analyze_image / check_advertiser_history  -> gather signals
3. submit_audit  -> always before final decision
4. approve OR reject  -> only after audit

HARD RULES:
- NEVER repeat an action listed in `actions_already_taken`.
- Respond with ONLY a valid JSON object. No markdown, no prose.

Required format:
{{"action_type": "<one_of_the_actions_above>", "reasoning": "<short reason>"}}

Scenario: {text}
actions_already_taken: {actions_already_taken}

Your next action?"""


def build_dataset():
    rows = []
    for s in BASE_SCENARIOS:
        prompt = PROMPT_TEMPLATE.format(
            text=s["text"],
            actions_already_taken=json.dumps(s["actions_already_taken"]),
        )
        rows.append({
            "prompt": prompt,
            "task_id": s["task_id"],
            "setup_actions": s["setup_actions"],
        })
    return Dataset.from_list(rows * 10)  # 7 scenarios x 10 = 70 examples

# =========================
# REWARD FUNCTION (FIXED)
# =========================

def reward_environment(prompts, completions, task_id=None, setup_actions=None, **kwargs):
    """Shaped reward for GRPO.

    Pure env reward is too sparse (mostly -0.05) to give clear gradients.
    We add explicit shaping:
      - invalid JSON / invalid action_type -> -1.0  (strong negative signal)
      - valid action env REJECTS (wrong phase / API failure) -> -0.5
      - valid action env ACCEPTS (advances state) -> +0.5 + env_reward
      - terminal correct decision -> env_reward already contains +1.0 bonus
    """
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
            "reasoning": parsed.get("reasoning", "format-compliant"),
        }

        try:
            client.reset(t_id)
            for s in setup:
                safe_step(client, s)

            result = safe_step(client, action)
            env_reward = float(result.get("reward", -0.2))
            status_msg = (result.get("status_message") or "").lower()

            rejected = (
                "api failure" in status_msg
                or "invalid action" in status_msg
                or "must call" in status_msg
            )

            if rejected:
                shaped = -0.5
            else:
                shaped = 0.5 + env_reward

            rewards.append(shaped)

        except Exception:
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
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
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