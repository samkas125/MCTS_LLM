"""Tests for prompt builder."""

from src.inference.prompt_builder import (
    build_eval_prompt,
    build_expansion_prompt,
    build_grpo_prompt,
)


def test_grpo_prompt_structure():
    messages = build_grpo_prompt("What is 2+2?")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "2+2" in messages[1]["content"]
    assert "\\boxed{}" in messages[0]["content"]


def test_expansion_prompt_includes_trajectory():
    messages = build_expansion_prompt(
        problem="What is 2+2?",
        trajectory_so_far="Step 1: We need to add 2 and 2.",
        step_num=2,
    )
    assert len(messages) == 2
    assert "Step 1" in messages[1]["content"]
    assert "Step 2" in messages[1]["content"] or "step_num" not in messages[1]["content"]


def test_expansion_prompt_empty_trajectory():
    messages = build_expansion_prompt(
        problem="What is 2+2?",
        trajectory_so_far="",
        step_num=1,
    )
    assert "Progress so far" not in messages[1]["content"]


def test_expansion_tir_mode():
    messages = build_expansion_prompt("problem", "", 1, mode="tir")
    assert "programs" in messages[0]["content"]


def test_expansion_cot_mode():
    messages = build_expansion_prompt("problem", "", 1, mode="cot")
    assert "step by step" in messages[0]["content"]


def test_eval_prompt_same_as_grpo():
    eval_msgs = build_eval_prompt("test problem")
    grpo_msgs = build_grpo_prompt("test problem")
    assert eval_msgs == grpo_msgs


def test_expansion_late_step_includes_final_hint():
    messages = build_expansion_prompt("problem", "step1\nstep2", step_num=4)
    assert "final answer" in messages[1]["content"].lower()
