"""MCTS node representing a single reasoning state in the search tree."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """A single node in the MCTS tree representing a reasoning state.

    The root node contains the math problem. Each child represents one
    intermediate reasoning step (natural language + optional Python code).
    The state at step i is the concatenation of the problem and all
    preceding steps: x + s1 + s2 + ... + s_{i-1}.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Content
    step_text: str = ""  # Natural language reasoning step
    code_text: str = ""  # Python code for this step
    code_output: str = ""  # Output from executing the code

    # Tree structure
    parent: Optional[MCTSNode] = field(default=None, repr=False)
    children: list[MCTSNode] = field(default_factory=list, repr=False)
    depth: int = 0

    # MCTS statistics
    visit_count: int = 0  # N(s)
    total_value: float = 0.0  # q(s) — sum of all backpropagated rewards

    # Terminal state
    is_terminal: bool = False
    final_answer: Optional[str] = None  # Extracted answer if terminal
    reward: Optional[float] = None  # +1 correct, -1 incorrect, None if not scored

    @property
    def q_value(self) -> float:
        """Q(s) = q(s) / N(s). Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        """True if this node has no children (unexpanded)."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """True if this is the root node (no parent)."""
        return self.parent is None

    def get_trajectory(self) -> list[MCTSNode]:
        """Return the path from root to this node (inclusive)."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))

    def get_trajectory_text(self) -> str:
        """Concatenate all step_text from root to this node."""
        parts = []
        for node in self.get_trajectory():
            if node.step_text and not node.is_root:
                parts.append(node.step_text)
                if node.code_output:
                    parts.append(f"Output: {node.code_output}")
        return "\n".join(parts)

    def add_child(self, child: MCTSNode) -> None:
        """Add a child node, setting parent and depth automatically."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

    def __repr__(self) -> str:
        status = "T" if self.is_terminal else "L" if self.is_leaf else "I"
        return (
            f"MCTSNode(id={self.id}, depth={self.depth}, "
            f"N={self.visit_count}, Q={self.q_value:.3f}, {status})"
        )
