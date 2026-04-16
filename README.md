# MCTS-Guided Reasoning Trace Synthesis for LLM Self-Improvement

A research project that combines **Monte Carlo Tree Search (MCTS)** data generation with **GRPO reinforcement learning** to create a self-improving loop for small language models on mathematical reasoning.

> **The core idea**: MCTS generates high-quality reasoning traces annotated with step-level quality signals (Q-values). Those signals then fuel GRPO training — creating a feedback loop where each round of search produces better training data, and each round of training produces a better search policy.

This bridges two approaches that have not previously been combined for math reasoning:
- **rStar-Math** (MCTS + SFT) — uses tree search to generate data, trains via supervised fine-tuning
- **DeepSeek-R1** (GRPO, no MCTS) — uses reinforcement learning, but generates training data via simple sampling

**Target**: Qwen2.5-Math-1.5B on a single A100 80GB, trained on GSM8K + MATH, evaluated on GSM8K test and MATH-500.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                   Self-Improvement Loop                      │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  MCTS    │───▶│  GRPO    │───▶│  MCTS    │  (repeat)    │
│  │  Phase   │    │  Phase   │    │  Phase   │              │
│  └──────────┘    └──────────┘    └──────────┘              │
│                                                             │
│  Round 0 policy → MCTS generates traces + Q-values         │
│  Q-values become process rewards → GRPO trains Round 1      │
│  Round 1 policy → better MCTS exploration → ...            │
└─────────────────────────────────────────────────────────────┘
```

**MCTS Phase**: For each training problem, run tree search using the current model via vLLM. Each node is a reasoning step (natural language + Python code). Code is executed in a sandbox; only steps with valid code are kept. Q-values are backpropagated from terminal nodes (+1 correct, -1 incorrect). Top-2 trajectories per problem are saved.

**GRPO Phase**: Train using TRL's GRPOTrainer with three reward signals:
1. **Accuracy reward** — did the model get the right answer? (weight 1.0)
2. **Format reward** — did it use proper reasoning structure? (weight 0.1)
3. **Q-value reward** — bonus based on MCTS-derived problem difficulty (weight 0.5) ← *novel contribution*

---

## Project Structure

```
MCTS_LLM/
├── src/
│   ├── mcts/           # MCTS engine (node, UCT selection, expansion, simulation, backprop)
│   ├── sandbox/        # Safe Python code execution with timeout + pattern checks
│   ├── inference/      # Async vLLM client + Qwen2.5-Math prompt builder
│   ├── rewards/        # Accuracy, format, and Q-value reward functions
│   ├── training/       # GRPO runner (TRL + QLoRA), SFT baseline, data loader
│   ├── evaluation/     # Evaluator (vLLM batch inference on GSM8K/MATH-500)
│   ├── pipeline/       # Self-improvement loop + ablation runner
│   └── data/           # Dataset download, preprocessing, curriculum
├── scripts/            # Entry-point scripts (run_mcts, run_grpo, run_eval, etc.)
├── configs/            # YAML configs for MCTS, GRPO, SFT, eval, vLLM
├── setup/                # AWS setup scripts and instance guide
├── tests/              # Unit + integration tests (77 tests, all passing)
└── docs/               # Detailed implementation plan
```

---

## What You Need to Fill In

Before running anything, you need credentials for three services. Copy `.env.example` to `.env` and fill in these values:

```bash
cp .env.example .env
```

Open `.env` and set the following:

### 1. HuggingFace Token — **Required**
Needed to download `Qwen/Qwen2.5-Math-1.5B` from HuggingFace.

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get it at: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** → role: **Read**

### 2. Weights & Biases API Key — **Required** (for experiment tracking)
All training metrics, rewards, and completions are logged to wandb automatically.

```
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=mcts-grpo
```

Get it at: [wandb.ai/settings](https://wandb.ai/settings) → **API keys** → copy the key

### 3. AWS S3 Bucket — **Required for AWS** (not needed for local runs)
Used to sync checkpoints and MCTS traces between training phases, and to survive spot instance interruptions.

```
S3_BUCKET=your-bucket-name
AWS_DEFAULT_REGION=us-east-1
```

Create a bucket: `aws s3 mb s3://your-bucket-name`

---

## Running on AWS (Step-by-Step)

### Step 1 — Launch an EC2 Instance

Recommended instance: **`p4de.24xlarge`** (1× A100 80GB, the target for this project).
For development/testing: **`g5.2xlarge`** (1× A10G 24GB, much cheaper).

In the AWS console:
1. Go to **EC2 → Launch Instance**
2. Search for AMI: `Deep Learning AMI GPU PyTorch 2` (Ubuntu 22.04)
3. Select instance type: `p4de.24xlarge` (or `g5.2xlarge` for dev)
4. Add storage:
   - Root volume: **200 GB** gp3
   - Additional volume: **500 GB** gp3 (for data and checkpoints)
5. Configure security group: allow **SSH (port 22)** from your IP
6. Launch with your key pair

> **Cost tip**: Use **Spot Instances** — typically 60-70% cheaper. The training scripts handle spot interruptions automatically by syncing to S3.

### Step 2 — Connect and Clone

```bash
ssh -i your-key.pem ubuntu@<your-instance-ip>
git clone https://github.com/your-username/MCTS_LLM.git
cd MCTS_LLM
```

### Step 3 — Set Your Credentials

```bash
cp .env.example .env
nano .env   # Fill in HF_TOKEN, WANDB_API_KEY, S3_BUCKET
```

### Step 4 — Run Setup

This installs all dependencies, authenticates HuggingFace and wandb, and creates the required directories.

```bash
bash setup/setup.sh
source .venv/bin/activate
```

The script will print a GPU check at the end. You should see something like:
```
CUDA available: True
GPU count: 1
GPU: NVIDIA A100-SXM4-80GB
```

### Step 5 — Download and Preprocess Data

```bash
make download-data
python -c "from src.data.preprocess import preprocess_all; preprocess_all('data/raw', 'data/processed')"
```

This downloads GSM8K (7,473 train / 1,319 test) and MATH (7,500 train / 5,000 test) from HuggingFace and preprocesses them into a unified format. Takes ~5 minutes.

### Step 6 — Run the Full Pipeline

Use **tmux** so your session survives SSH disconnects:

```bash
tmux new -s training
```

Inside tmux, split into two panes (`Ctrl+B %`):

**Pane 1 — Start the vLLM server** (needed for MCTS data generation):
```bash
source .venv/bin/activate
make vllm-server
```

Wait until you see `INFO: Application startup complete`. Then switch to pane 2 (`Ctrl+B →`):

**Pane 2 — Run the self-improvement loop**:
```bash
source .venv/bin/activate
make loop
```

This runs all 3 rounds automatically:
- **Round 0**: MCTS on 15K problems (~25 hrs) → GRPO training (~3-5 days) → eval
- **Round 1**: MCTS with improved model → GRPO → eval
- **Round 2**: Final round → eval

Results are logged to wandb and saved to `outputs/results_log.json`.

### Step 7 — Monitor Progress

```bash
# GPU usage
nvtop

# CPU/memory
htop

# Training logs
tail -f outputs/logs/training.log

# Check wandb
# Open https://wandb.ai/your-username/mcts-grpo in your browser
```

---

## Running Individual Phases

If you want to run phases separately (e.g., run MCTS one day, train the next):

```bash
# Phase 1: Generate MCTS traces only
python scripts/run_mcts.py --dataset data/processed/train_combined \
    --output-dir data/mcts_traces/round_0 \
    --max-problems 1000   # Start with a small subset

# Phase 2: Train with GRPO (uses MCTS Q-value rewards if traces exist)
python scripts/run_grpo.py \
    --mcts-traces data/mcts_traces/round_0 \
    --output-dir outputs/checkpoints \
    --round 0

# Phase 3: Evaluate a checkpoint
python scripts/run_eval.py \
    --model outputs/checkpoints/grpo_round_0/final \
    --output-dir outputs/eval_results

# Run ablation experiments
python scripts/run_ablations.py \
    --mcts-traces data/mcts_traces/round_0
```

---

## Running on AWS with Spot Instances

The spot training script handles automatic checkpointing to S3 on interruption:

```bash
# On your instance:
bash setup/spot_training.sh 0   # Run round 0
bash setup/spot_training.sh 1   # Run round 1 (downloads round 0 results from S3 first)
```

If your spot instance is interrupted mid-training, just launch a new instance and re-run the same command — it will resume from the last S3 checkpoint.

---

## Running Tests Locally

```bash
conda activate mctsllm   # or: source .venv/bin/activate
pytest tests/ -v
```

Tests that don't require a GPU (77 tests) run fully offline. The one GPU-dependent test is automatically skipped if no CUDA device is found.

---

## Configuration

All hyperparameters live in `configs/`. Key settings:

| File | What to change |
|------|---------------|
| `configs/mcts_config.yaml` | `num_rollouts` (8 default), `num_candidates` (4), `max_depth` (10) |
| `configs/grpo_config.yaml` | `num_generations` (8), `learning_rate` (1e-5), `num_train_epochs` (1) |
| `configs/grpo_config.yaml` → `rewards` | Weights for accuracy (1.0), format (0.1), Q-value (0.5) |
| `configs/vllm_config.yaml` | `gpu_memory_utilization` (0.9 for MCTS, 0.3 for GRPO colocate) |

---

## Ablation Experiments

Five experiments are configured to isolate the contribution of each component:

| Experiment | What it tests |
|-----------|---------------|
| `mcts_grpo_full` | Full system (MCTS data + GRPO + Q-value reward) |
| `mcts_grpo_no_qvalue` | Does the Q-value reward add value? |
| `mcts_sft` | Does GRPO beat SFT given the same MCTS data? |
| `grpo_only` | Does MCTS data improve GRPO vs. online sampling? |
| `base_model` | Zero-shot baseline (no training) |

```bash
python scripts/run_ablations.py --mcts-traces data/mcts_traces/round_0
```

Results are saved to `outputs/ablations/comparison.json` and printed as a table.

---

## References

- **rStar-Math** (Microsoft Research, arXiv:2501.04519) — MCTS + SFT for math reasoning
- **DeepSeek-R1** (arXiv:2501.12948) — GRPO for reasoning via pure RL
- **DeepSeekMath** (arXiv:2402.03300) — GRPO algorithm
- **OmegaPRM** (arXiv:2406.06592) — Automated process supervision via MCTS
- **"Let's Verify Step by Step"** (arXiv:2305.20050) — Process reward models
