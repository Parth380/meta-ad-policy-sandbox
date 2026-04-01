# 🛡️ Meta Ad-Policy RL Sandbox

A custom, bleeding-edge Reinforcement Learning environment built for the Meta Ad-Policy Hackathon. This sandbox evaluates the ability of Vision-Language Models (VLMs) and LLMs to act as autonomous ad moderators, navigating complex policy violations, multimodal traps, and illegal targeting.

## 🚀 Core Features
* **OpenEnv 0.2.3 Compliant:** Fully implements the latest Meta OpenEnv specifications, including Pydantic `StepResult` state serialization and `/step` & `/reset` API endpoints.
* **Reward Shaping:** Implements a strict `-0.05` step penalty to force the AI agent to optimize tool usage and prevent infinite analysis loops.
* **Multimodal Traps:** Tests the limits of VLMs by presenting ads where the text is benign, but the visual elements contain severe policy violations.
* **Containerized Infrastructure:** Fully Dockerized and highly lightweight, easily running under the 2 vCPU / 8GB RAM hackathon constraints.

## 📋 Evaluation Tasks
The environment natively supports 4 distinct adversarial tasks, loadable via the `task_id` parameter:
1. `task_1_healthcare`: Evaluates ads for unapproved medical claims, pharmaceuticals, and subtle dog whistles.
2. `task_2_financial`: Evaluates ads for predatory financial services, scams, and high-pressure tactics.
3. `task_3_multimodal`: Detects policy violations hidden entirely within visual elements that bypass standard NLP text filters.
4. `task_4_targeting`: Identifies illegal demographic targeting (e.g., adult financial services targeting minors).

## 🛠️ Available Agent Tools
The environment exposes the following action space to the evaluating LLM:
* `analyze_image`: Request VLM context for visual elements.
* `request_landing_page`: Extract simulated URL endpoints.
* `request_id_verification`: Check advertiser trust scores.
* `approve` / `reject`: Terminal actions.

## 🚦 Quick Start (Local)

**1. Build the Docker Image**
```bash
docker build -t meta-ad-sandbox .