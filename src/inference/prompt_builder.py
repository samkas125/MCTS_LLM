"""Prompt formatting for Qwen2.5-Math models."""

SYSTEM_PROMPT_COT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

SYSTEM_PROMPT_TIR = (
    "Please integrate natural language reasoning with programs to solve the problem above, "
    "and put your final answer within \\boxed{}."
)

SYSTEM_PROMPT_MCTS = (
    "Solve math problems one step at a time. "
    "Each reply is ONE step only — do not jump to the final answer."
)

STEP_INSTRUCTION = (
    "Step {step_num}: write the next single reasoning step. "
    "Do NOT include \\boxed{{}} yet.\n"
)

FINAL_STEP_INSTRUCTION = (
    "Continue reasoning, or if you have enough to conclude, "
    "give the final answer as \\boxed{{<answer>}}.\n"
)


def build_expansion_prompt(
    problem: str,
    trajectory_so_far: str,
    step_num: int,
    allow_final_answer: bool = False,
) -> list[dict]:
    """Build prompt for MCTS expansion (generating a next reasoning step).

    Args:
        problem: The math problem text
        trajectory_so_far: Concatenated text of all steps so far
        step_num: Current step number (1-indexed)
        allow_final_answer: If True, permit \\boxed{} final answer in this step

    Returns:
        Conversational format messages list
    """
    user_content = problem + "\n\n"
    if trajectory_so_far:
        user_content += f"Progress so far:\n{trajectory_so_far}\n\n"

    if allow_final_answer:
        user_content += FINAL_STEP_INSTRUCTION
    else:
        user_content += STEP_INSTRUCTION.format(step_num=step_num)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_MCTS},
        {"role": "user", "content": user_content},
    ]


def build_grpo_prompt(problem: str) -> list[dict]:
    """Build prompt-only format for GRPO training.

    Args:
        problem: The math problem text

    Returns:
        Conversational format messages list
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COT},
        {"role": "user", "content": problem},
    ]


def build_eval_prompt(problem: str) -> list[dict]:
    """Build prompt for evaluation (greedy decoding).

    Same as GRPO prompt but used during eval with temperature=0.
    """
    return build_grpo_prompt(problem)
