"""Tests for sandboxed code execution."""

from src.sandbox.executor import execute_code_safely, validate_code


def test_simple_code_executes():
    success, output = execute_code_safely("print(2 + 2)")
    assert success
    assert "4" in output


def test_math_code():
    code = "import math\nprint(math.sqrt(144))"
    success, output = execute_code_safely(code)
    assert success
    assert "12" in output


def test_sympy_code():
    code = "from sympy import symbols, solve\nx = symbols('x')\nprint(solve(x**2 - 4, x))"
    success, output = execute_code_safely(code)
    assert success
    assert "-2" in output and "2" in output


def test_forbidden_os_import():
    is_safe, error = validate_code("import os\nos.system('echo hack')")
    assert not is_safe
    assert "Forbidden" in error


def test_forbidden_subprocess():
    is_safe, error = validate_code("import subprocess\nsubprocess.run(['ls'])")
    assert not is_safe
    assert "Forbidden" in error


def test_forbidden_exec():
    is_safe, error = validate_code("exec('print(1)')")
    assert not is_safe
    assert "Forbidden" in error


def test_forbidden_open():
    is_safe, error = validate_code("f = open('/etc/passwd')")
    assert not is_safe
    assert "Forbidden" in error


def test_disallowed_import():
    is_safe, error = validate_code("import socket")
    assert not is_safe


def test_disallowed_import_not_in_forbidden():
    """Test that the import whitelist catches modules not in FORBIDDEN_PATTERNS."""
    is_safe, error = validate_code("import pickle")
    assert not is_safe
    assert "Disallowed" in error


def test_allowed_imports():
    code = "import math\nimport itertools\nprint(math.pi)"
    is_safe, _ = validate_code(code)
    assert is_safe


def test_timeout():
    code = "while True: pass"
    success, output = execute_code_safely(code, timeout_seconds=2)
    assert not success
    assert "timed out" in output.lower()


def test_empty_code():
    success, output = execute_code_safely("")
    assert not success
    assert "Empty" in output


def test_syntax_error():
    success, output = execute_code_safely("def f(:\n  pass")
    assert not success


def test_runtime_error():
    success, output = execute_code_safely("1/0")
    assert not success
    assert "ZeroDivision" in output


def test_output_truncation():
    code = "for i in range(10000): print(i)"
    success, output = execute_code_safely(code, max_output_chars=100)
    assert success
    assert len(output) <= 100
