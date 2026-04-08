🛡️ Meta Ad-Policy RL Sandbox
A custom, bleeding-edge Reinforcement Learning environment built for the Meta Ad-Policy Hackathon. This sandbox evaluates the ability of Vision-Language Models (VLMs) and LLMs to act as autonomous ad moderators, navigating complex policy violations, multimodal traps, and illegal targeting.

🚀 Core Features
OpenEnv 0.2.3 Compliant: Fully implements the latest Meta OpenEnv specifications, including Pydantic StepResult state serialization and /step & /reset API endpoints.
Reward Shaping: Implements a strict -0.05 step penalty to force the AI agent to optimize tool usage and prevent infinite analysis loops.
Multimodal Traps: Tests the limits of VLMs by presenting ads where the text is benign, but the visual elements contain severe policy violations.
Containerized Infrastructure: Fully Dockerized and highly lightweight, easily running under the 2 vCPU / 8GB RAM hackathon constraints.
📋 Evaluation Tasks
The environment natively supports 4 distinct adversarial tasks, loadable via the task_id parameter:

task_1_healthcare: Evaluates ads for unapproved medical claims, pharmaceuticals, and subtle dog whistles.
task_2_financial: Evaluates ads for predatory financial services, scams, and high-pressure tactics.
task_3_multimodal: Detects policy violations hidden entirely within visual elements that bypass standard NLP text filters.
task_4_targeting: Identifies illegal demographic targeting (e.g., adult financial services targeting minors).
🛠️ Available Agent Tools
The environment exposes the following action space to the evaluating LLM:

analyze_image: Request VLM context for visual elements.
request_landing_page: Extract simulated URL endpoints.
request_id_verification: Check advertiser trust scores.
approve / reject: Terminal actions.
🚦 Quick Start (Local)
1. Build the Docker Image docker build -t meta-ad-sandbox .

2. Run the Environment Container docker run -p 8000:8000 meta-ad-sandbox

3. Run the Automated Inference Agent Make sure your Hugging Face credentials are set, then run the evaluation script to test the agent against all 4 tasks: export HF_TOKEN="your_hugging_face_token" python inference.py