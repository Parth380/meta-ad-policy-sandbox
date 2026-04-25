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

PatchFastRL("GRPO", FastLanguageModel)

# =========================
# CONFIG
# =========================

ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO = os.getenv("HF_REPO", "")  # e.g. "yourname/metaguard-llama3.1-8b-grpo"

ALLOWED_ACTIONS = [
    "query_regulations",
    "analyze_image",
    "check_advertiser_history",
    "request_landing_page",
    "request_id_verification",
    "submit_audit",
    "approve",
    "reject",
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

    # Targeting task — fresh state, expected: query_regulations
    {
        "task_id": "task_4_targeting",
        "text": "Financial ad targeting young users: 'Start Your First Investment Portfolio'. No actions taken yet.",
        "actions_already_taken": [],
        "setup_actions": [],
    },
    # Targeting task — mid state, expected: request_id_verification (age check)
    {
        "task_id": "task_4_targeting",
        "text": "Financial ad targeting young users. Policy queried, need to verify age targeting.",
        "actions_already_taken": ["query_regulations"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
        ],
    },
    # Targeting task — audit ready
    {
        "task_id": "task_4_targeting",
        "text": "Financial ad targeting minors. Policy, advertiser history, and ID verification done. Submit audit.",
        "actions_already_taken": ["query_regulations", "check_advertiser_history", "request_id_verification"],
        "setup_actions": [
            {"action_type": "query_regulations", "reasoning": "policy lookup"},
            {"action_type": "check_advertiser_history", "reasoning": "trust score"},
            {"action_type": "request_id_verification", "reasoning": "age check"},
        ],
    },
]

PROMPT_TEMPLATE = """You are an enterprise Ad Policy Compliance Agent.

You MUST choose exactly ONE action_type from this list (any other value is invalid):
- query_regulations
- analyze_image
- check_advertiser_history
- request_landing_page
- request_id_verification
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
    return Dataset.from_list(rows * 10)  # 10 scenarios x 10 = 100 examples

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

USE_4BIT = not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 40 * 1024**3

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    load_in_4bit=USE_4BIT,
    max_seq_length=2048,
    dtype=None,  # auto-detect bf16 on A100
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=64,
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
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=768,
        max_completion_length=128,
        logging_steps=5,
        warmup_ratio=0.1,
        bf16=True,
        report_to="none",
    ),
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# =========================
# RUN
# =========================

if __name__ == "__main__":
    ensure_env_ready()

    print("Starting GRPO training...")
    trainer.train()

    model.save_pretrained("outputs/lora_adapter")
    tokenizer.save_pretrained("outputs/lora_adapter")
    print("LoRA adapter saved to outputs/lora_adapter")

    print("Merging adapter into base model (bf16)...")
    merged_model, merged_tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/lora_adapter",
        load_in_4bit=False,
        max_seq_length=2048,
    )
    merged_model.save_pretrained_merged(
        "outputs/merged",
        merged_tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved to outputs/merged")

    if HF_REPO:
        print(f"Pushing merged model to {HF_REPO}...")
        merged_model.push_to_hub_merged(
            HF_REPO,
            merged_tokenizer,
            save_method="merged_16bit",
            token=HF_TOKEN,
        )
        print(f"Model live at https://huggingface.co/{HF_REPO}")
    else:
        print("Set HF_REPO env var to auto-push to Hub (skipped).")

    print("Done.")