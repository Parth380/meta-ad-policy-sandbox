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

# #region agent log
import pathlib as _pl
_DLOG = _pl.Path("debug-851b5f.log")
def _dlog(hyp, loc, msg, data=None):
    import time as _t
    entry = json.dumps({"sessionId":"851b5f","hypothesisId":hyp,"location":loc,"message":msg,"data":data or {},"timestamp":int(_t.time()*1000)})
    with open(_DLOG, "a") as f: f.write(entry + "\n")
    print(f"[DBG:{hyp}] {msg} {data or ''}", flush=True)
# #endregion

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
    # #region agent log
    _dlog("B", "grpo_train.py:ensure_env_ready", "Checking env", {"ENV_URL": ENV_URL})
    # #endregion
    for i in range(20):
        try:
            r = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": "task_1_healthcare"},
                timeout=5
            )
            if r.status_code == 200:
                # #region agent log
                _dlog("B", "grpo_train.py:ensure_env_ready", "Env ready", {"attempt": i+1, "status": r.status_code})
                # #endregion
                print("✅ Environment ready")
                return
        except Exception as e:
            # #region agent log
            if i == 0: _dlog("B", "grpo_train.py:ensure_env_ready", "Env connection failed", {"attempt": i+1, "error": str(e)[:200]})
            # #endregion
            pass
        time.sleep(1)
    # #region agent log
    _dlog("B", "grpo_train.py:ensure_env_ready", "ENV UNREACHABLE after 20 attempts", {})
    # #endregion
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

_reward_call_count = [0]

def reward_environment(prompts, completions, task_id=None, setup_actions=None, **kwargs):
    """Shaped reward for GRPO."""
    _reward_call_count[0] += 1
    _call = _reward_call_count[0]
    # #region agent log
    _dlog("C", "grpo_train.py:reward_env", f"reward call #{_call}", {
        "n_prompts": len(prompts) if prompts else 0,
        "n_completions": len(completions) if completions else 0,
        "completions_type": type(completions).__name__,
        "first_completion_type": type(completions[0]).__name__ if completions else "N/A",
        "first_completion_preview": str(completions[0])[:150] if completions else "N/A",
        "task_id_is_none": task_id is None,
        "setup_actions_is_none": setup_actions is None,
        "kwargs_keys": list(kwargs.keys()),
    })
    # #endregion

    client = EnvClient(ENV_URL)
    rewards = []

    if task_id is None or setup_actions is None:
        # #region agent log
        _dlog("D", "grpo_train.py:reward_env", "task_id or setup_actions is None — returning -1 for all", {"call": _call})
        # #endregion
        return [-1.0] * len(completions)

    for idx, (completion, t_id, setup) in enumerate(zip(completions, task_id, setup_actions)):
        parsed = extract_json(completion)
        # #region agent log
        if _call <= 3: _dlog("D", "grpo_train.py:reward_loop", f"call#{_call} item#{idx}", {"parsed_ok": parsed is not None, "action": parsed.get("action_type") if parsed else None, "raw_preview": str(completion)[:120], "task_id": t_id})
        # #endregion
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

if torch.cuda.is_available():
    _props = torch.cuda.get_device_properties(0)
    _vram = _props.total_memory
    _name = _props.name
    _cc = (_props.major, _props.minor)  # compute capability
    print(f"GPU: {_name}  VRAM: {_vram / 1024**3:.1f} GB  Compute: {_cc[0]}.{_cc[1]}")
else:
    _vram = 0
    _name = "CPU"
    _cc = (0, 0)

USE_4BIT = _vram < 40 * 1024**3   # T4 (15 GB), L4 (24 GB) → 4-bit; A100 (80 GB) → full
USE_BF16 = _cc >= (8, 0) and not USE_4BIT  # bf16 only when full-precision; 4-bit LoRA uses fp16 internally

# #region agent log
_dlog("A", "grpo_train.py:gpu_detect", "GPU config resolved", {"name":_name,"vram_gb":round(_vram/1024**3,1),"cc":list(_cc),"USE_4BIT":USE_4BIT,"USE_BF16":USE_BF16})
# #endregion

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    load_in_4bit=USE_4BIT,
    max_seq_length=2048,
    dtype=torch.float16 if USE_4BIT else None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16 if USE_4BIT else 32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=32 if USE_4BIT else 64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# =========================
# TRAINER
# =========================

dataset = build_dataset()

# #region agent log
_dlog("A", "grpo_train.py:trainer_init", "Creating GRPOTrainer", {"USE_4BIT":USE_4BIT,"USE_BF16":USE_BF16,"epochs":1 if USE_4BIT else 3,"batch":1 if USE_4BIT else 2,"gens":2 if USE_4BIT else 4,"dataset_len":len(dataset)})
# #endregion

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_environment],
    args=GRPOConfig(
        output_dir="outputs",
        learning_rate=2e-5,
        num_train_epochs=1 if USE_4BIT else 3,
        per_device_train_batch_size=1 if USE_4BIT else 2,
        gradient_accumulation_steps=2 if USE_4BIT else 4,
        num_generations=2 if USE_4BIT else 4,
        max_prompt_length=768,
        max_completion_length=128,
        logging_steps=3 if USE_4BIT else 5,
        warmup_steps=5 if USE_4BIT else 10,
        bf16=USE_BF16,
        fp16=not USE_BF16,
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

    # #region agent log
    _dlog("E", "grpo_train.py:train_start", "About to call trainer.train()", {"gpu_mem_allocated_gb": round(torch.cuda.memory_allocated()/1024**3, 2) if torch.cuda.is_available() else 0})
    # #endregion
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