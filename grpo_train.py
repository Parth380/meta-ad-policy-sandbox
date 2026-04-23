import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
from src.environment import AdPolicyEnvironment

# 1. Load Model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 1024,
    load_in_4bit = True,
)

# 2. Define Reward Functions
def reward_compliance(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        # Check if the model called the necessary tools in order
        if "query_regulations" in completion and "submit_audit" in completion:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

def reward_json_format(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            import json
            json.loads(completion)
            rewards.append(1.0)
        except:
            rewards.append(0.0)
    return rewards

# 3. Configure Trainer
training_args = GRPOConfig(
    output_dir = "outputs/meta-ad-agent",
    learning_rate = 5e-6,
    num_train_epochs = 1,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    max_prompt_length = 512,
    max_completion_length = 512,
    num_generations = 8, # Number of variations to compare
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_compliance, reward_json_format],
    args = training_args,
    train_dataset = [], # We will stream data from your AdGenerator here
    tokenizer = tokenizer,
)

# 4. Start Training
# trainer.train()