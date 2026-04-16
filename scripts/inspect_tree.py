"""Run MCTS on a single problem and print the full tree structure."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from src.inference.vllm_client import VLLMClient
from src.mcts.node import MCTSNode
from src.mcts.tree import MCTSConfig, MCTSTree


def print_tree(node: MCTSNode, indent: int = 0, max_text: int = 80) -> None:
    if node.is_root:
        print(f"ROOT  N={node.visit_count}  Q={node.q_value:+.3f}  children={len(node.children)}")
        print(f"  {node.step_text[:max_text]!r}")
    else:
        if node.is_terminal:
            marker = "✓" if node.reward == 1.0 else "✗"
            kind = f"TERMINAL({marker})"
        elif node.is_leaf:
            kind = "LEAF"
        else:
            kind = "NODE"
        prefix = "  " * indent
        print(
            f"{prefix}├─ depth={node.depth}  N={node.visit_count}  "
            f"Q={node.q_value:+.3f}  [{kind}]"
        )
        text = node.step_text.replace("\n", " ")[:max_text]
        print(f"{prefix}│  {text!r}")
        if node.code_output:
            print(f"{prefix}│  output={node.code_output[:60]!r}")

    for child in node.children:
        print_tree(child, indent + 1, max_text)


def print_stats(root: MCTSNode) -> None:
    depths, leaves, terminals = [], [], []

    def walk(node):
        depths.append(node.depth)
        if node.is_terminal:
            terminals.append(node)
        if node.is_leaf:
            leaves.append(node)
        for c in node.children:
            walk(c)

    walk(root)
    correct = sum(1 for t in terminals if t.reward == 1.0)
    print(f"\n--- Stats ---")
    print(f"Total nodes : {len(depths)}")
    print(f"Max depth   : {max(depths)}")
    print(f"Leaf nodes  : {len(leaves)}")
    print(f"Terminals   : {len(terminals)}  ({correct} correct)")
    branching = [len(n.children) for n in [root] + leaves if n.children]
    if branching:
        print(f"Avg branching factor: {sum(branching)/len(branching):.2f}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="Problem index in dataset")
    parser.add_argument("--dataset", default="data/processed/train_combined.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=4)
    args = parser.parse_args()

    ds = load_dataset("json", data_files=args.dataset, split="train")
    problem = ds[args.index]
    print(f"Problem [{args.index}]: {problem['problem'][:120]}")
    print(f"Answer: {problem['solution']}\n")

    cfg = MCTSConfig(num_rollouts=args.rollouts, num_candidates=args.candidates)
    tree = MCTSTree(problem=problem["problem"], ground_truth=problem["solution"], config=cfg)
    client = VLLMClient(base_url=args.vllm_url, model=args.model)

    print("Running MCTS...\n")
    trajectories = await tree.run(client)

    print("=" * 70)
    print("TREE STRUCTURE")
    print("=" * 70)
    print_tree(tree.root)
    print_stats(tree.root)

    print(f"\n--- Top {len(trajectories)} trajectories ---")
    for i, traj in enumerate(trajectories):
        print(f"\n[{i}] steps={len(traj.steps)}  Q={traj.avg_q_value:.3f}  correct={traj.is_correct}")


if __name__ == "__main__":
    asyncio.run(main())
