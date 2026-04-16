"""MCTS tree orchestrator: the main search loop.

Ties together selection, expansion, simulation, and backpropagation
to perform Monte Carlo Tree Search over reasoning steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.inference.vllm_client import VLLMClient
from src.mcts.backpropagation import backpropagate
from src.mcts.expansion import expand_node
from src.mcts.extract import MCTSTrajectory, extract_top_trajectories, get_tree_stats
from src.mcts.node import MCTSNode
from src.mcts.selection import select_node
from src.mcts.simulation import simulate_to_terminal
from src.rewards.accuracy import check_answer_equivalence

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_rollouts: int = 8  # Number of MCTS iterations per problem
    num_candidates: int = 4  # k candidates per expansion
    exploration_constant: float = 1.414  # c in UCT formula
    max_depth: int = 10  # Max reasoning steps per trajectory
    temperature: float = 0.7  # Temperature for expansion
    rollout_temperature: float = 0.8  # Temperature for simulation
    top_k_trajectories: int = 2  # Top-k trajectories to extract
    sandbox_timeout: int = 10  # Code execution timeout in seconds
    max_tokens_per_step: int = 512  # Max tokens per generated step


class MCTSTree:
    """Monte Carlo Tree Search over reasoning steps.

    The root node contains the math problem. Each search iteration:
    1. SELECT: Descend from root using UCT to find a leaf.
    2. EXPAND: Generate k candidate next steps from the leaf.
    3. SIMULATE: Roll out each new child to a terminal state.
    4. BACKPROPAGATE: Update Q-values along the path.

    After all iterations, extract the top-k trajectories.
    """

    def __init__(
        self,
        problem: str,
        ground_truth: str,
        config: MCTSConfig | None = None,
    ):
        self.root = MCTSNode(step_text=problem)
        self.problem = problem
        self.ground_truth = ground_truth
        self.config = config or MCTSConfig()

    async def run(self, vllm_client: VLLMClient) -> list[MCTSTrajectory]:
        """Execute full MCTS search and return top trajectories.

        Args:
            vllm_client: vLLM client for LLM generation.

        Returns:
            List of top-k MCTSTrajectory objects, sorted by avg Q-value.
        """
        cfg = self.config

        for rollout_idx in range(cfg.num_rollouts):
            logger.debug(f"Rollout {rollout_idx + 1}/{cfg.num_rollouts}")

            # 1. Selection
            leaf = select_node(self.root, cfg.exploration_constant)

            if leaf.is_terminal:
                # Re-backpropagate existing terminal reward
                reward = leaf.reward if leaf.reward is not None else -1.0
                backpropagate(leaf, reward)
                continue

            # 2. Expansion
            children = await expand_node(
                leaf,
                self.problem,
                vllm_client,
                num_candidates=cfg.num_candidates,
                temperature=cfg.temperature,
                max_tokens_per_step=cfg.max_tokens_per_step,
                sandbox_timeout=cfg.sandbox_timeout,
            )

            if not children:
                # Dead end: no valid candidates produced
                backpropagate(leaf, -1.0)
                continue

            # 3. Simulation + 4. Backpropagation for each new child
            for child in children:
                if child.is_terminal:
                    # Evaluate terminal directly
                    is_correct = check_answer_equivalence(
                        child.final_answer, self.ground_truth
                    )
                    reward = 1.0 if is_correct else -1.0
                    child.reward = reward
                else:
                    # Roll out to terminal
                    reward = await simulate_to_terminal(
                        child,
                        self.problem,
                        self.ground_truth,
                        vllm_client,
                        max_depth=cfg.max_depth,
                        temperature=cfg.rollout_temperature,
                    )

                backpropagate(child, reward)

        # Extract best trajectories
        trajectories = extract_top_trajectories(
            self.root, top_k=cfg.top_k_trajectories
        )

        stats = get_tree_stats(self.root)
        logger.info(
            f"MCTS complete: {stats['total_nodes']} nodes, "
            f"{stats['correct_terminals']}/{stats['total_terminals']} correct, "
            f"Q={stats['root_q_value']:.3f}"
        )

        return trajectories

    def get_stats(self) -> dict:
        """Get tree statistics for logging."""
        return get_tree_stats(self.root)
