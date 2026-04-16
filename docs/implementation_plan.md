# MCTS-Guided Reasoning Trace Synthesis for LLM Self-Improvement

## Context

This project combines two powerful paradigms that have not yet been merged for math reasoning:
- **rStar-Math's MCTS-based data generation** (uses only supervised fine-tuning)
- **DeepSeek-R1's GRPO-based reinforcement learning** (uses only online sampling, not tree search)

The gap: No published work as of April 2026 combines MCTS-generated training data with GRPO specifically for mathematical reasoning. The combination is compelling because MCTS provides what GRPO lacks (structured, step-verified trajectories with Q-value annotations) and GRPO provides what SFT lacks (a training objective that directly optimizes for reward maximization rather than imitating a static dataset).

**Target**: Qwen2.5-Math-1.5B on a single A100 80GB, trained on 10-20K problems from GSM8K + MATH, evaluated on GSM8K test (1K) and MATH-500.

---

## 1. Project Structure

```
MCTS_LLM/
|-- pyproject.toml
|-- Makefile
|-- .env.example
|-- .gitignore
|
|-- configs/
|   |-- mcts_config.yaml
|   |-- grpo_config.yaml
|   |-- sft_config.yaml
|   |-- eval_config.yaml
|   |-- vllm_config.yaml
|
|-- src/
|   |-- __init__.py
|   |-- mcts/
|   |   |-- __init__.py
|   |   |-- node.py                    # MCTSNode dataclass
|   |   |-- tree.py                    # MCTSTree orchestrator (main loop)
|   |   |-- selection.py               # UCT selection policy
|   |   |-- expansion.py               # Candidate generation + code filtering
|   |   |-- simulation.py              # Rollout to terminal state
|   |   |-- backpropagation.py         # Reward backpropagation + Q-value updates
|   |   |-- extract.py                 # Extract top-k trajectories with Q-values
|   |
|   |-- sandbox/
|   |   |-- __init__.py
|   |   |-- executor.py                # Sandboxed Python code execution (subprocess)
|   |
|   |-- inference/
|   |   |-- __init__.py
|   |   |-- vllm_client.py             # Async vLLM OpenAI-compatible API client
|   |   |-- prompt_builder.py          # Prompt formatting for Qwen2.5-Math
|   |
|   |-- rewards/
|   |   |-- __init__.py
|   |   |-- accuracy.py                # Answer extraction + symbolic equivalence (math_verify)
|   |   |-- format_reward.py           # Think/answer tag format checking
|   |   |-- qvalue_reward.py           # Q-value process reward for GRPO (novel)
|   |   |-- answer_extraction.py       # Extract \boxed{} and #### answers
|   |
|   |-- training/
|   |   |-- __init__.py
|   |   |-- grpo_runner.py             # GRPO training pipeline (TRL GRPOTrainer)
|   |   |-- sft_runner.py              # SFT baseline training pipeline
|   |   |-- data_loader.py             # Dataset loading and formatting for TRL
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- download.py                # Download GSM8K, MATH, MATH-500
|   |   |-- preprocess.py              # Normalize, format, create prompt-only datasets
|   |   |-- mcts_dataset.py            # Convert MCTS trajectories to training data
|   |   |-- curriculum.py              # MCTS-based difficulty sorting
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- evaluator.py               # Run eval on GSM8K test / MATH-500
|   |   |-- metrics.py                 # Accuracy, pass@k, per-level breakdown
|   |
|   |-- pipeline/
|   |   |-- __init__.py
|   |   |-- self_improvement_loop.py   # Orchestrate MCTS -> GRPO -> MCTS rounds
|   |   |-- ablation_runner.py         # Run all ablation comparisons
|
|-- scripts/
|   |-- run_mcts.py                     # Entry point: generate MCTS data
|   |-- run_grpo.py                     # Entry point: GRPO training
|   |-- run_sft.py                      # Entry point: SFT baseline training
|   |-- run_eval.py                     # Entry point: evaluate a checkpoint
|   |-- run_loop.py                     # Entry point: full self-improvement loop
|   |-- run_ablations.py               # Entry point: all ablation studies
|   |-- start_vllm_server.sh           # Shell script to launch vLLM server
|
|-- tests/
|   |-- __init__.py
|   |-- test_mcts_node.py
|   |-- test_uct_selection.py
|   |-- test_sandbox_executor.py
|   |-- test_rewards.py
|   |-- test_answer_extraction.py
|   |-- test_data_loading.py
|   |-- test_prompt_builder.py
|   |-- integration/
|       |-- test_mcts_end_to_end.py
|       |-- test_grpo_training.py
|
|-- data/                               # gitignored; created at runtime
|   |-- raw/
|   |-- processed/
|   |-- mcts_traces/
|   |-- training/
|
|-- outputs/                            # gitignored; created at runtime
|   |-- checkpoints/
|   |-- eval_results/
|   |-- logs/
```

---

## 2. Environment Setup

### 2.1 Dependencies (pyproject.toml)

**Core ML stack:**
- `torch>=2.4.0` (CUDA 12.1)
- `transformers>=4.53.0`
- `trl>=1.0.0` (with `trl[vllm]` extra)
- `peft>=0.14.0`
- `bitsandbytes>=0.45.0`
- `accelerate>=1.4.0`
- `vllm>=0.8.5`

**Data and evaluation:**
- `datasets>=3.0.0`
- `math-verify>=0.8.0` (HuggingFace symbolic equivalence checker)
- `sympy>=1.13`

**Infrastructure:**
- `wandb>=0.19.0`
- `pyyaml>=6.0`
- `rich>=13.0` (progress bars)
- `pytest>=8.0`

**Code sandbox:**
- `RestrictedPython>=7.0` (optional, for AST-level restriction)

### 2.2 GPU Memory Budget (A100 80GB, 1.5B model)

| Setup | VRAM Usage |
|---|---|
| vLLM inference only (MCTS phase) | ~3 GB model + KV cache |
| GRPO QLoRA + vLLM colocate | ~1.5GB (4-bit model) + ~24GB (vLLM at 0.3) + ~10GB (optimizer/gradients) |
| Evaluation (vLLM greedy) | ~3 GB model + KV cache |

Strategy: Generate MCTS rollouts offline (server mode, full GPU) then run GRPO training as a separate phase. This avoids memory contention.

### 2.3 vLLM Server Launch

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Math-1.5B \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype bfloat16 --seed 42 \
    --enable-prefix-caching
```

---

## 3. Data Pipeline

### 3.1 Datasets to Download (`src/data/download.py`)

| Dataset | HuggingFace ID | Train | Test | Answer Format |
|---|---|---|---|---|
| GSM8K | `openai/gsm8k` | 7,473 | 1,319 | `#### <number>` |
| MATH | `hendrycks/competition_math` | 7,500 | 5,000 | `\boxed{answer}` |
| MATH-500 | `HuggingFaceH4/MATH-500` | - | 500 | `answer` column |

### 3.2 Preprocessing (`src/data/preprocess.py`)

Convert all problems to unified format:
```python
{
    "problem_id": str,
    "problem": str,            # question text
    "solution": str,           # ground truth answer (extracted from \boxed{} or ####)
    "source": "gsm8k" | "math",
    "level": int,              # 0 for GSM8K, 1-5 for MATH
    "subject": str,            # "arithmetic" for GSM8K, MATH subject for MATH
    "difficulty": float,       # initially from level, refined by MCTS later
}
```

**Prompt format** (conversational, for Qwen2.5-Math):
```python
[
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
    {"role": "user", "content": "<problem text>"}
]
```

### 3.3 Combined Training Set

Create 15K problem subset: all 7,473 GSM8K train + 7,500 MATH train (or subsample MATH to balance difficulty).

### 3.4 MCTS-to-Training Data Conversion (`src/data/mcts_dataset.py`)

**For GRPO**: Prompt-only dataset (GRPO generates its own completions). MCTS data informs the Q-value reward function.
```python
{"prompt": [system, user], "solution": str, "problem_id": str}
```

**For SFT baseline**: Prompt-completion pairs using best MCTS trajectory.
```python
{"prompt": [system, user], "completion": [{"role": "assistant", "content": trajectory_text}]}
```

---

## 4. MCTS Engine

### 4.1 MCTSNode (`src/mcts/node.py`)

```python
@dataclass
class MCTSNode:
    id: str
    step_text: str = ""           # Natural language reasoning step
    code_text: str = ""           # Python code for this step
    code_output: str = ""         # Execution output
    parent: Optional["MCTSNode"] = None
    children: list["MCTSNode"] = field(default_factory=list)
    depth: int = 0
    visit_count: int = 0          # N(s)
    total_value: float = 0.0      # q(s) - sum of backpropagated rewards
    is_terminal: bool = False
    final_answer: Optional[str] = None
    reward: Optional[float] = None  # +1 correct, -1 incorrect

    @property
    def q_value(self) -> float:
        """Q(s) = q(s) / N(s)"""
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0

    def get_trajectory(self) -> list["MCTSNode"]:
        """Path from root to this node."""

    def get_trajectory_text(self) -> str:
        """Concatenate all step_text from root to here."""
```

### 4.2 UCT Selection (`src/mcts/selection.py`)

```python
def uct_score(node: MCTSNode, c: float = 1.414) -> float:
    """UCT(s) = Q(s) + c * sqrt(ln(N_parent) / N(s))"""
    if node.visit_count == 0:
        return float('inf')
    return node.q_value + c * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)

def select_node(root: MCTSNode, c: float = 1.414) -> MCTSNode:
    """Descend tree by highest UCT until reaching a leaf or terminal node."""
```

### 4.3 Expansion (`src/mcts/expansion.py`) -- CRITICAL FILE

This is the most complex component. It connects the LLM to the MCTS tree.

```python
async def expand_node(
    node: MCTSNode,
    problem: str,
    vllm_client: VLLMClient,
    num_candidates: int = 4,     # k=4 candidates per expansion
    temperature: float = 0.7,
    max_tokens_per_step: int = 512,
) -> list[MCTSNode]:
    """
    1. Build prompt: system + user question + trajectory so far + step instruction
    2. Sample k completions via vLLM with temperature sampling
    3. Parse each completion into (step_text, code_text)
    4. Execute code_text in sandbox
    5. Filter: keep only candidates whose code executed successfully
    6. Check if any contain \boxed{} (terminal state)
    7. Create child MCTSNode for each valid candidate
    """
```

**Parsing strategy**: The model is prompted to produce "code-augmented CoT" in the format:
```
Step N: <natural language explanation>
```python
<code>
```
Result: <describe what code shows>
```

Use regex to extract the NL text and code blocks. Parse `\boxed{}` to detect terminal states.

### 4.4 Simulation/Rollout (`src/mcts/simulation.py`)

```python
async def simulate_to_terminal(
    node: MCTSNode,
    problem: str,
    ground_truth: str,
    vllm_client: VLLMClient,
    max_depth: int = 10,
) -> float:
    """
    From non-terminal leaf, greedily roll out (1 candidate/step, temp=0.8)
    until \boxed{} found or max_depth reached.
    Returns: +1.0 if correct, -1.0 if wrong or dead end.
    """
```

### 4.5 Backpropagation (`src/mcts/backpropagation.py`)

```python
def backpropagate(node: MCTSNode, reward: float) -> None:
    """Walk from node to root, incrementing visit_count and total_value."""
    current = node
    while current is not None:
        current.visit_count += 1
        current.total_value += reward
        current = current.parent
```

### 4.6 Main MCTS Loop (`src/mcts/tree.py`) -- CRITICAL FILE

```python
@dataclass
class MCTSConfig:
    num_rollouts: int = 8
    num_candidates: int = 4
    exploration_constant: float = 1.414
    max_depth: int = 10
    temperature: float = 0.7
    rollout_temperature: float = 0.8
    top_k_trajectories: int = 2

class MCTSTree:
    async def run(self, vllm_client: VLLMClient) -> list[MCTSTrajectory]:
        """
        For each of num_rollouts iterations:
          1. SELECT: descend from root using UCT to find leaf
          2. EXPAND: generate k candidates from leaf (code-filtered)
          3. SIMULATE: rollout each new child to terminal
          4. BACKPROPAGATE: update Q-values along path
        After all rollouts, extract top-k trajectories.
        """
```

### 4.7 Trajectory Extraction (`src/mcts/extract.py`)

```python
@dataclass
class MCTSTrajectory:
    problem: str
    steps: list[str]
    codes: list[str]
    final_answer: str
    is_correct: bool
    q_values: list[float]       # Q-value at each step
    visit_counts: list[int]     # Visit count at each step
    avg_q_value: float          # Average Q across steps
    trajectory_text: str

def extract_top_trajectories(root: MCTSNode, top_k: int = 2) -> list[MCTSTrajectory]:
    """
    Find all correct terminal nodes, trace back to root,
    compute per-step Q-values, rank by avg Q-value, return top-k.
    """
```

---

## 5. Code Execution Sandbox (`src/sandbox/executor.py`)

```python
FORBIDDEN_PATTERNS = [
    "import os", "import sys", "import subprocess",
    "import shutil", "__import__", "exec(", "eval(",
    "open(", "file(", "input(",
]

def execute_code_safely(code: str, timeout_seconds: int = 10) -> tuple[bool, str]:
    """
    1. Static validation (forbidden pattern check)
    2. Write code to temp file
    3. Execute via subprocess.run() with timeout + resource limits
    4. Capture stdout, truncate if >5000 chars
    Returns: (success, output_or_error)
    """
```

The subprocess approach provides isolation. On Linux (A100 server), use `resource.setrlimit(RLIMIT_AS, 512MB)` for memory limits. On macOS (dev), skip resource limits with platform check.

---

## 6. vLLM Integration (`src/inference/`)

### 6.1 Async Client (`src/inference/vllm_client.py`)

```python
class VLLMClient:
    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen/Qwen2.5-Math-1.5B", max_concurrent=64):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def generate(self, messages, temperature=0.7, max_tokens=512, n=1) -> list[str]:
        """Generate n completions for a single prompt."""

    async def generate_batch(self, prompts, temperature=0.7, max_tokens=512) -> list[str]:
        """Generate one completion per prompt, in parallel with semaphore."""
```

### 6.2 Prompt Builder (`src/inference/prompt_builder.py`)

Two modes:
- **TIR (Tool-Integrated Reasoning)**: For MCTS expansion — instructs model to produce code-augmented CoT
- **CoT**: For GRPO training prompts — standard step-by-step reasoning with `\boxed{}`

```python
SYSTEM_PROMPT_TIR = "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."
SYSTEM_PROMPT_COT = "Please reason step by step, and put your final answer within \\boxed{}."

def build_expansion_prompt(problem, trajectory_so_far, step_num) -> list[dict]:
    """Build prompt for MCTS expansion step."""

def build_grpo_prompt(problem) -> list[dict]:
    """Build prompt-only format for GRPO training."""
```

---

## 7. Reward Functions (`src/rewards/`)

### 7.1 Answer Extraction (`src/rewards/answer_extraction.py`)

```python
def extract_boxed_answer(text: str) -> str | None:
    """Extract content from \boxed{...}, handling nested braces. Takes the LAST \boxed occurrence."""

def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numeric answer after #### in GSM8K format."""
```

### 7.2 Accuracy Reward (`src/rewards/accuracy.py`) -- CRITICAL FILE

```python
def check_answer_equivalence(predicted: str, ground_truth: str) -> bool:
    """
    1. Try math_verify for symbolic comparison (\frac{1}{2} == 0.5)
    2. Fallback: normalize and compare strings
    """

def accuracy_reward_func(completions, solution, **kwargs) -> list[float]:
    """TRL-compatible: returns 1.0 for correct, 0.0 for incorrect."""
```

### 7.3 Format Reward (`src/rewards/format_reward.py`)

```python
def format_reward_func(completions, **kwargs) -> list[float]:
    """1.0 if has <think> tags + \boxed{}, 0.5 if only \boxed{}, 0.0 otherwise."""
```

### 7.4 Q-Value Process Reward (`src/rewards/qvalue_reward.py`) -- NOVEL CONTRIBUTION

```python
class QValueRewardFunction:
    """
    Uses MCTS Q-values as a process reward signal during GRPO training.
    For each problem, assigns a bonus reward proportional to the avg Q-value
    of the best MCTS trajectory found for that problem.
    """
    def __init__(self, mcts_data: dict, step_weight_by_visits: bool = True):
        self.mcts_data = mcts_data  # problem_id -> trajectory data

    def __call__(self, completions, problem_id=None, **kwargs) -> list[float]:
        """Returns Q-value-based reward for each completion."""
```

---

## 8. GRPO Training Pipeline (`src/training/grpo_runner.py`) -- CRITICAL FILE

```python
def run_grpo_training(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str,
    reward_funcs: list,
    reward_weights: list[float],
    round_num: int = 0,
):
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # GRPO config
    training_args = GRPOConfig(
        num_generations=8,              # Group size G
        max_completion_length=512,
        max_prompt_length=512,
        temperature=0.7,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_enable_sleep_mode=True,
        loss_type="dr_grpo",            # Length-bias corrected
        beta=0.0,                       # No KL penalty
        epsilon=0.2,                    # Clip range
        scale_rewards="group",
        reward_weights=reward_weights,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=1,
        bf16=True,
        report_to="wandb",
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model, args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(f"{output_dir}/round_{round_num}/final")
```

**Key design note**: GRPOTrainer expects a prompt-only dataset. It generates its own completions via vLLM, scores them with the reward functions, and trains. MCTS data feeds the Q-value reward function, not the completions directly.

### 8.1 SFT Baseline (`src/training/sft_runner.py`)

Same QLoRA config, but uses SFTTrainer with MCTS trajectories as supervised data (prompt-completion pairs). `learning_rate=2e-4`, `num_train_epochs=3`.

---

## 9. Self-Improvement Loop (`src/pipeline/self_improvement_loop.py`) -- CRITICAL FILE

```python
class SelfImprovementLoop:
    def __init__(self, base_model="Qwen/Qwen2.5-Math-1.5B", num_rounds=3):
        self.current_model = base_model

    async def run(self):
        for round_num in range(self.num_rounds):
            # Phase 1: Evaluate current model (pre-training baseline)
            pre_results = evaluate_model(self.current_model, gsm8k_test, math500)

            # Phase 2: MCTS data generation with current policy
            mcts_traces = await self.run_mcts_phase(train_problems, round_num)
            # -> Save traces to data/mcts_traces/round_N/

            # Phase 3: GRPO training with MCTS-informed rewards
            new_model = self.run_grpo_phase(train_problems, mcts_traces, round_num)
            # -> reward_funcs=[accuracy, format, qvalue], weights=[1.0, 0.1, 0.5]

            # Phase 4: Evaluate trained model (post-training)
            post_results = evaluate_model(new_model, gsm8k_test, math500)

            # Update model for next round
            self.current_model = new_model
```

### Adaptive Compute Allocation (Novel Contribution #4)

In `run_mcts_phase`, adjust rollouts per problem based on difficulty:
- Easy (first few rollouts all correct): 2-4 rollouts
- Medium: 8 rollouts (default)
- Hard (few/no correct rollouts): 16-64 rollouts

---

## 10. Evaluation Pipeline (`src/evaluation/evaluator.py`)

```python
def evaluate_model(model_path, gsm8k_test, math500, temperature=0.0, max_tokens=1024) -> dict:
    """
    Uses vLLM for fast batch greedy inference.
    Returns: {
        "gsm8k": {"accuracy": float, "correct": int, "total": int, "per_level": dict},
        "math500": {"accuracy": float, "correct": int, "total": int, "per_level": dict},
    }
    """
```

Evaluate after every round. Track per-difficulty-level accuracy for MATH-500 (levels 1-5).

---

## 11. Ablation Studies

| Experiment | MCTS Data? | Training | Q-Value Reward? | Tests |
|---|---|---|---|---|
| **MCTS+GRPO** (main) | Yes | GRPO | Yes | Full system |
| MCTS+GRPO (no Q) | Yes | GRPO | No | Does Q-value reward help? |
| MCTS+SFT | Yes | SFT | No | Does GRPO beat SFT on same data? |
| GRPO-only | No | GRPO | No | Does MCTS data improve GRPO? |
| Base model | No | None | No | Zero-shot baseline |

The **key comparison**: MCTS+GRPO vs. MCTS+SFT (same data, different training) and vs. GRPO without MCTS (same algorithm, different data source).

---

## 12. Logging and Experiment Tracking

**wandb integration at two levels:**

1. **GRPO Training** (automatic via TRL `report_to="wandb"`): loss, rewards, entropy, clip ratios, completion lengths, sampled completions
2. **MCTS and Pipeline** (custom): problems_solved, solve_rate, avg_q_value, avg_tree_depth per round; GSM8K/MATH-500 accuracy per round

**Checkpointing**: GRPO saves every 200 steps with `save_total_limit=3`. After each round, merge LoRA adapter into standalone model for next MCTS phase. MCTS traces saved as JSON per round.

---

## 13. Testing Strategy

### Unit Tests (no GPU needed, mock LLM)
- `test_mcts_node.py`: Node creation, q_value property, get_trajectory, tree linking
- `test_uct_selection.py`: UCT score with known values, unvisited=inf, select descends correctly
- `test_sandbox_executor.py`: Safe code runs, forbidden patterns rejected, timeout kills, memory limits
- `test_rewards.py`: accuracy_reward_func, format_reward_func with various inputs
- `test_answer_extraction.py`: `\boxed{}` with nested braces, `####` format, edge cases
- `test_prompt_builder.py`: Correct Qwen2.5-Math prompt structure
- `test_data_loading.py`: GSM8K/MATH preprocessing, unified format

### Integration Tests (small GPU or mock)
- `test_mcts_end_to_end.py`: Tiny MCTS (2 rollouts, 2 candidates) with mock LLM, verify tree structure
- `test_grpo_training.py`: 2 GRPO steps on 4 examples with tiny model, verify loss decreases

---

## 14. Implementation Schedule

### Week 1: Foundation (Days 1-5)
- **Day 1-2**: Project scaffolding (`pyproject.toml`, configs, `.gitignore`), data pipeline (`download.py`, `preprocess.py`), verify dataset download works
- **Day 2-3**: Sandbox executor + tests, answer extraction + tests (both `\boxed{}` and `####`), reward functions + tests
- **Day 3-4**: vLLM client (async OpenAI-compatible), prompt builder for Qwen2.5-Math
- **Day 4-5**: MCTSNode dataclass, UCT selection, backpropagation (pure algorithm, no LLM)

### Week 2: MCTS Engine (Days 6-10)
- **Day 6-7**: Expansion (LLM integration: prompt -> generate -> parse -> execute -> filter), simulation/rollout
- **Day 7-8**: MCTSTree main loop, trajectory extraction
- **Day 8-9**: `run_mcts.py` script, end-to-end test with vLLM server on 10 problems
- **Day 9-10**: Run MCTS on 100-problem subset, validate traces quality, tune hyperparams (temperature, num_candidates)

### Week 3: Training Pipeline (Days 11-14)
- **Day 11-12**: GRPO runner (QLoRA + vLLM colocate + multi-reward), test on 10 problems
- **Day 12-13**: SFT baseline runner, evaluation pipeline, MCTS-to-training-data conversion
- **Day 13-14**: Q-value reward function, self-improvement loop orchestrator, curriculum sorting

### Weeks 4-5: Experiments (Days 15-25)
- **Day 15-17**: Full Round 0 (MCTS on 15K problems ~25 hrs + GRPO training + eval)
- **Day 18-20**: Rounds 1-2 of self-improvement loop
- **Day 21-23**: All ablation experiments (MCTS+SFT, GRPO-only, no-Q-value)
- **Day 24-25**: Analysis, results compilation, comparison tables

---

## 15. Novel Contributions (Implementation Priority)

1. **MCTS Q-values as process rewards for GRPO** (high impact, moderate effort) -- `src/rewards/qvalue_reward.py`. TRL supports `reward_funcs=[accuracy, format, qvalue]` with configurable weights.

2. **Visit-count-weighted training** (high novelty, low effort) -- In `src/data/curriculum.py`. Weight problems by max visit count of "crux" nodes. Problems with high-visit-count steps = harder = more training signal.

3. **MCTS-based curriculum learning** (moderate impact, low effort) -- In `src/data/curriculum.py`. Sort training set easy-to-hard using MCTS solve rate. Train GRPO in curriculum batches.

4. **Adaptive compute allocation** (moderate impact, moderate effort) -- In `src/pipeline/self_improvement_loop.py`. 2-4 rollouts for easy, 16-64 for hard problems.

5. **Iterative MCTS->GRPO->MCTS self-play** (highest impact, highest effort) -- The full `self_improvement_loop.py`. 2-3 rounds on single GPU.

---

## 16. Risk Mitigations

| Risk | Mitigation |
|---|---|
| MCTS throughput too low | 1.5B model on vLLM: ~600 problems/hr. 15K problems in ~25 hrs. Feasible. |
| QLoRA + vLLM colocate OOM | 1.5B in 4-bit = 1.5GB. vLLM at 0.3 = 24GB. Leaves 55GB headroom. Reduce `num_generations` to 4 if needed. |
| Code execution unsafe | Subprocess sandbox with timeout + resource limits + static forbidden-pattern checks. Sufficient for research. |
| Q-value reward too noisy | Low weight (0.5 vs 1.0 accuracy). Ablation D tests this directly. |
| Model doesn't produce code-augmented CoT | Format reward incentivizes it. Qwen2.5-Math is designed for TIR mode. Fall back to pure CoT if needed. |

---

## 17. Verification Plan

After each phase, verify end-to-end:

1. **After data pipeline**: Check dataset sizes match expected (7473 GSM8K train, 7500 MATH train, 1319 GSM8K test, 500 MATH-500). Verify answer extraction on 50 random samples.
2. **After MCTS engine**: Run on 10 easy GSM8K problems. Expect >50% solve rate. Check Q-values are in [-1, 1] range. Check trajectories have 2-8 steps.
3. **After GRPO training**: Run 100-step training on 50 problems. Check loss decreases. Check reward/accuracy increases. Check wandb logs.
4. **After full Round 0**: Compare pre vs post eval on GSM8K test and MATH-500. Expect measurable improvement.
5. **After all rounds**: Compare MCTS+GRPO vs all ablations. The main hypothesis: MCTS+GRPO > MCTS+SFT > GRPO-only > base model.
