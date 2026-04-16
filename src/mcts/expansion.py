"""MCTS node expansion: generate candidate reasoning steps via LLM.

This is the most critical component — it connects the LLM (via vLLM) to the
MCTS tree. Each expansion generates k candidate next steps, executes their
code in a sandbox, and filters for valid candidates.
"""

from __future__ import annotations

import re
from typing import Optional

from src.inference.prompt_builder import build_expansion_prompt
from src.inference.vllm_client import VLLMClient
from src.mcts.node import MCTSNode
from src.sandbox.executor import execute_code_safely


def parse_step_and_code(completion: str) -> tuple[str, str]:
    """Parse a completion into natural language step and Python code.

    Expected format from the model:
        Step N: <natural language explanation>
        ```python
        <code>
        ```
        Result: <code output reference>

    Returns:
        (step_text, code_text) — code_text may be empty if no code block found.
    """
    # Extract code blocks
    code_match = re.search(r"```python\s*\n(.*?)```", completion, re.DOTALL)
    code_text = code_match.group(1).strip() if code_match else ""

    # The step text is everything (the full completion serves as the reasoning step)
    step_text = completion.strip()

    return step_text, code_text


def check_terminal(completion: str) -> tuple[bool, Optional[str]]:
    """Check if a completion contains \\boxed{answer} indicating a final answer.

    Returns:
        (is_terminal, final_answer) — final_answer is None if not terminal.
    """
    matches = list(re.finditer(r"\\boxed\{", completion))
    if not matches:
        return False, None

    # Extract the last \boxed{} content (handles nested braces)
    last_match = matches[-1]
    start = last_match.end()
    depth = 1
    i = start
    while i < len(completion) and depth > 0:
        if completion[i] == "{":
            depth += 1
        elif completion[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        answer = completion[start : i - 1].strip()
        return True, answer

    return False, None


async def expand_node(
    node: MCTSNode,
    problem: str,
    vllm_client: VLLMClient,
    num_candidates: int = 4,
    temperature: float = 0.7,
    max_tokens_per_step: int = 512,
    sandbox_timeout: int = 10,
) -> list[MCTSNode]:
    """Generate k candidate next steps from the current node.

    Pipeline:
        1. Build prompt from problem + trajectory so far
        2. Sample k completions via vLLM with temperature sampling
        3. Parse each completion into (step_text, code_text)
        4. Execute code_text in sandbox
        5. Filter: keep only candidates whose code executed successfully
           (or candidates with no code — pure reasoning steps)
        6. Check if any contain \\boxed{} (terminal state)
        7. Create child MCTSNode for each valid candidate

    Args:
        node: Current node to expand from.
        problem: Original math problem text.
        vllm_client: vLLM client for generating completions.
        num_candidates: k — number of candidates to generate.
        temperature: Sampling temperature.
        max_tokens_per_step: Max tokens per generated step.
        sandbox_timeout: Timeout for code execution in seconds.

    Returns:
        List of new child nodes (may be < num_candidates if some fail).
    """
    trajectory_so_far = node.get_trajectory_text()
    step_num = node.depth + 1
    allow_final = node.depth >= 2

    prompt = build_expansion_prompt(
        problem, trajectory_so_far, step_num, allow_final_answer=allow_final
    )

    # Generate k candidates
    completions = await vllm_client.generate_n_for_prompt(
        messages=prompt,
        n=num_candidates,
        temperature=temperature,
        max_tokens=max_tokens_per_step,
    )

    children = []
    for completion in completions:
        if not completion or not completion.strip():
            continue

        step_text, code_text = parse_step_and_code(completion)

        # Execute code in sandbox (if code exists)
        code_output = ""
        if code_text:
            success, code_output = execute_code_safely(
                code_text, timeout_seconds=sandbox_timeout
            )
            if not success:
                continue  # Filter out failed code executions
        # If no code, the step is pure reasoning — still valid

        # Deduplicate: skip if this candidate is equivalent to one already accepted
        # in this expansion. Check (1) same execution output, (2) exact same text.
        is_duplicate = False
        for existing in children:
            if code_output and existing.code_output and code_output == existing.code_output:
                is_duplicate = True
                break
            if step_text.strip() == existing.step_text.strip():
                is_duplicate = True
                break
        if is_duplicate:
            continue

        is_terminal, final_answer = check_terminal(completion)

        child = MCTSNode(
            step_text=step_text,
            code_text=code_text,
            code_output=code_output,
            is_terminal=is_terminal,
            final_answer=final_answer,
        )
        node.add_child(child)
        children.append(child)

    return children
