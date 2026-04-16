"""Q-value process reward from MCTS trajectories (novel contribution).

Uses MCTS Q-values as a step-level process reward signal during GRPO training.
This bridges process supervision (Lightman et al.) with RL training — DeepSeek-R1
uses only outcome rewards; rStar-Math uses process rewards for search but not RL.
"""

from __future__ import annotations


class QValueRewardFunction:
    """Reward function that uses MCTS Q-values as process rewards.

    For each GRPO-generated completion, looks up the corresponding problem's
    MCTS data and assigns a bonus reward proportional to the average Q-value
    of the best trajectory found for that problem.

    This provides a denser reward signal than outcome-only accuracy reward,
    encoding the MCTS tree's judgment about problem difficulty and solution quality.
    """

    def __init__(
        self,
        mcts_data: dict[str, dict],
        step_weight_by_visits: bool = True,
        max_reward: float = 1.0,
    ):
        """Initialize with MCTS trajectory data.

        Args:
            mcts_data: Mapping of problem_id -> best trajectory dict with keys:
                       'avg_q_value', 'visit_counts', 'q_values'
            step_weight_by_visits: If True, scale reward by normalized visit counts
                                  (novel contribution: crux-move weighting)
            max_reward: Maximum reward value (Q-values are normalized to [0, max_reward])
        """
        self.mcts_data = mcts_data
        self.step_weight_by_visits = step_weight_by_visits
        self.max_reward = max_reward

    def __call__(
        self,
        completions,
        problem_id: list[str] | None = None,
        **kwargs,
    ) -> list[float]:
        """Compute Q-value reward for each completion.

        The reward for a completion is the average Q-value of the best MCTS
        trajectory for the same problem, normalized to [0, max_reward].

        If step_weight_by_visits is True, problems whose MCTS trees required
        more search effort (high visit counts) get a boosted reward, focusing
        learning on the "crux moves" where the model struggled.

        Args:
            completions: list of completions (conversational or string format)
            problem_id: list of problem IDs to look up MCTS data

        Returns:
            list[float]: Q-value reward for each completion
        """
        rewards = []
        for i, completion in enumerate(completions):
            pid = problem_id[i] if problem_id else None

            if pid and pid in self.mcts_data:
                traj = self.mcts_data[pid]
                avg_q = traj.get("avg_q_value", 0.0)

                # Normalize Q-value from [-1, 1] to [0, max_reward]
                q_reward = max(0.0, (avg_q + 1.0) / 2.0) * self.max_reward

                # Optional: boost reward for problems with high-visit-count crux nodes
                if self.step_weight_by_visits:
                    visit_counts = traj.get("visit_counts", [1])
                    if visit_counts:
                        max_visits = max(visit_counts)
                        # Logarithmic scaling to avoid extreme weights
                        visit_boost = 1.0 + 0.2 * min(max_visits / 4.0, 3.0)
                        q_reward *= visit_boost

                rewards.append(min(q_reward, self.max_reward))
            else:
                rewards.append(0.0)

        return rewards


def build_qvalue_reward_from_traces(
    mcts_traces: dict[str, list[dict]],
    step_weight_by_visits: bool = True,
    max_reward: float = 1.0,
) -> QValueRewardFunction:
    """Build a QValueRewardFunction from loaded MCTS trace data.

    Takes the best trajectory (index 0) for each problem.
    """
    mcts_data = {}
    for pid, traj_list in mcts_traces.items():
        if traj_list:
            best = traj_list[0]
            mcts_data[pid] = {
                "avg_q_value": best.get("avg_q_value", 0.0)
                if isinstance(best, dict)
                else best.avg_q_value,
                "visit_counts": best.get("visit_counts", [1])
                if isinstance(best, dict)
                else best.visit_counts,
                "q_values": best.get("q_values", [])
                if isinstance(best, dict)
                else best.q_values,
            }

    return QValueRewardFunction(
        mcts_data=mcts_data,
        step_weight_by_visits=step_weight_by_visits,
        max_reward=max_reward,
    )
