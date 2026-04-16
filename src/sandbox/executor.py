"""Sandboxed Python code execution for MCTS reasoning steps."""

import os
import platform
import subprocess
import tempfile

FORBIDDEN_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import http",
    "import urllib",
    "import requests",
    "__import__",
    "exec(",
    "eval(",
    "open(",
    "file(",
    "input(",
    "breakpoint(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
]

ALLOWED_IMPORTS = {
    "math",
    "fractions",
    "decimal",
    "statistics",
    "itertools",
    "functools",
    "collections",
    "sympy",
    "numpy",
    "re",
    "string",
    "random",
    "cmath",
    "operator",
}


def validate_code(code: str) -> tuple[bool, str]:
    """Static validation before execution.

    Returns (is_safe, error_message).
    """
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in code:
            return False, f"Forbidden pattern: {pattern}"

    # Check imports are in allowed list
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            # Extract module name
            if stripped.startswith("from "):
                module = stripped.split()[1].split(".")[0]
            else:
                module = stripped.split()[1].split(".")[0].split(",")[0]

            if module not in ALLOWED_IMPORTS:
                return False, f"Disallowed import: {module}"

    return True, ""


def execute_code_safely(
    code: str,
    timeout_seconds: int = 10,
    max_output_chars: int = 5000,
    max_memory_mb: int = 512,
) -> tuple[bool, str]:
    """Execute Python code in a sandboxed subprocess with timeout.

    Strategy:
        1. Static validation (forbidden patterns, import whitelist)
        2. Write code to a temporary file
        3. Execute via subprocess.run() with timeout and resource limits
        4. Capture stdout, truncate if too long
        5. Clean up temp file

    Returns:
        (success, output) - success=True if code ran without error within timeout.
        output is stdout (or error message if failed).
    """
    if not code or not code.strip():
        return False, "Empty code"

    is_safe, error_msg = validate_code(code)
    if not is_safe:
        return False, error_msg

    # Add resource limits on Linux
    resource_header = ""
    if platform.system() == "Linux":
        memory_bytes = max_memory_mb * 1024 * 1024
        resource_header = (
            "import resource\n"
            f"resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))\n"
        )

    full_code = resource_header + code

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name

        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env={
                **os.environ,
                "PYTHONPATH": "",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        if result.returncode == 0:
            output = result.stdout[:max_output_chars]
            return True, output
        else:
            error = result.stderr[:max_output_chars]
            return False, error

    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except (OSError, UnboundLocalError):
            pass
