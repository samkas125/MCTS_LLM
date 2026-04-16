"""MCTS simulation: rollout from a leaf node to a terminal state."""

from __future__ import annotations

from src.inference.vllm_client import VLLMClient
from src.mcts.expansion import expand_node
from src.mcts.node import MCTSNode
from src.rewards.accuracy import check_answer_equivalence


async def simulate_to_terminal(
    node: MCTSNode,
    problem: str,
    ground_truth: str,
    vllm_client: VLLMClient,
    max_depth: int = 10,
    temperature: float = 0.8,
) -> float:
    """From a non-terminal leaf, greedily roll out to a terminal state.

    Uses temperature sampling with 1 candidate per step (greedy rollout).
    Continues until \\boxed{} answer found or max_depth reached.

    Args:
        node: Starting (non-terminal) leaf node.
        problem: Original math problem text.
        ground_truth: Ground truth answer for correctness check.
        vllm_client: vLLM client for generation.
        max_depth: Maximum tree depth to explore.
        temperature: Sampling temperature for rollout (slightly higher for diversity).

    Returns:
        +1.0 if final answer matches ground_truth.
        -1.0 if final answer is wrong, max_depth exceeded, or dead end.
    """
    current = node

    for _ in range(max_depth - current.depth):
        if current.is_terminal:
            break

        # Generate single continuation (greedy rollout)
        children = await expand_node(
            current,
            problem,
            vllm_client,
            num_candidates=1,
            temperature=temperature,
        )

        if not children:
            return -1.0  # Dead end — no valid code produced

        current = children[0]

    # Evaluate the terminal state
    if current.is_terminal and current.final_answer is not None:
        is_correct = check_answer_equivalence(current.final_answer, ground_truth)
        reward = 1.0 if is_correct else -1.0
        current.reward = reward
        return reward

    return -1.0  # Max depth without reaching an answer
