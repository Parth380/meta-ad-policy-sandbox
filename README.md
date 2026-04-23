````markdown
# MetaGuard: Enterprise Ad-Policy RL Sandbox

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](#)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-microservices-009688)](#)
[![RL](https://img.shields.io/badge/Training-GRPO-orange)](#)

MetaGuard is an OpenEnv-compatible reinforcement learning environment built for enterprise policy decision-making. It simulates a realistic ad-review workflow where an agent must gather context, inspect policy constraints, validate advertiser history, log its decision trail, and take a final moderation action.

The goal is not simple classification. The goal is procedural compliance under uncertainty.

---

## Why this project exists

Most moderation demos stop at “approve” or “reject.” Real systems do not work that way.

A production moderation workflow usually needs:
- policy lookup before judgment
- account and advertiser risk context
- audit logging for traceability
- support for multimodal and adversarial inputs
- stepwise compliance with a strict operating procedure

MetaGuard models that workflow as a reinforcement learning environment, so an agent is rewarded not just for the final answer, but for following the correct enterprise process.

---

## Core idea

The environment forces the agent to behave like a policy operator inside a controlled moderation stack:

1. retrieve policy constraints  
2. inspect the content  
3. check advertiser history  
4. write an audit log  
5. take a terminal decision  

Skipping steps, violating the sequence, or ignoring context results in penalties.

---

## System architecture

```mermaid
flowchart LR
    A[Agent / Policy Model] -->|reset / step| B[Environment Hub]
    B --> C[Regulatory Service]
    B --> D[Advertiser CRM Service]
    B --> E[Audit Service]
    B --> F[Scenario Generator]
    B -->|observation + reward| A
````

### Services

**Environment Hub**
Coordinates the episode lifecycle, enforces step order, applies rewards, and exposes the OpenEnv-style interface.

**Regulatory Service**
Returns policy constraints, sensitive categories, and risk rules for a given task.

**Advertiser CRM Service**
Stores advertiser history, trust level, and prior violations.

**Audit Service**
Persists the moderation trace and final decision record.

**Scenario Generator**
Creates varied tasks and adversarial edge cases so the policy does not overfit to a narrow pattern.

---

## Action space

The environment uses a structured action space designed around real moderation work.

### Required workflow actions

* `query_regulations` — fetch policy constraints
* `analyze_image` — inspect visual content when the task includes media
* `check_advertiser_history` — retrieve account risk context
* `submit_audit` — store the decision trail before final action

### Terminal actions

* `approve`
* `reject`

The environment penalizes invalid ordering, skipped steps, premature terminal actions, and unsupported decisions.

---

## Reward design

Rewards reflect enterprise correctness, not just outcome guessing:

* positive reward for correct terminal decision
* positive reward for following required procedural steps
* bonus for complete audit logging
* penalty for skipping mandatory steps
* penalty for invalid actions
* penalty for inconsistent decisions

---

## Training with GRPO

MetaGuard supports policy optimization using **GRPO (Group Relative Policy Optimization)**.

### Why GRPO

* no separate critic model required
* works well with relative reward comparisons
* suited for structured decision tasks
* integrates cleanly with environment-driven feedback

### Why Unsloth

* reduced VRAM usage
* faster fine-tuning cycles
* practical for 7B–8B models on limited hardware

### Training loop

1. sample tasks
2. run policy in environment
3. compute reward from compliance + outcome
4. update policy with GRPO
5. repeat across task families

---

## Task families

* **Healthcare claims** — unapproved medical claims, pharma violations
* **Financial claims** — predatory offers, misleading returns
* **Multimodal traps** — violations hidden in images
* **Targeting violations** — illegal demographic targeting

These scenarios test both policy understanding and procedural discipline.

---

## What makes this different

MetaGuard is not a classifier.

It simulates a real moderation workflow with:

* tool usage
* stateful decision making
* policy retrieval
* advertiser context
* auditability
* adversarial task generation
* RL-based optimization

---

## Local setup

### Install

```bash
pip install -e .
pip install -r requirements.txt
```

### Run services

```bash
python apps/regulatory_api.py
python apps/crm_api.py
python apps/audit_api.py
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Train / Inference

```bash
python grpo_train.py
python inference.py
```

### Validate

```bash
./validate.sh <YOUR_HF_SPACE_URL> .
```

---

## Repository structure

```text
.
├── apps/                 # microservices
├── server/               # environment hub
├── src/                  # environment + logic
├── grpo_train.py         # training
├── inference.py          # evaluation
├── validate.sh           # validation script
└── README.md
```

---

## Implementation notes

* strict step sequence enforced
* terminal actions gated by compliance steps
* audit logs must be structured
* reproducibility from clean setup is required
* Docker build must be standard and functional

---

## Suggested demo flow

1. show a complex policy case
2. agent calls services in correct order
3. audit log is generated
4. final decision is made
5. reward explains correctness

---

## Future improvements

* stronger multimodal reasoning
* richer policy graphs
* improved adversarial generation
* better evaluation metrics
* expanded agent compatibility

---

## License

Add your license here.

```
```
