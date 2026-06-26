from pathlib import Path


def test_post_process_kinematic_solver_root_is_agent_facing_only():
    package_root = Path("post_process/kinematic_solver")
    root_entries = {path.name for path in package_root.iterdir() if path.name != "__pycache__"}

    assert root_entries <= {
        "__init__.py",
        "docs",
        "estimate_limit.py",
        "sdk",
        "tools",
        "utils",
    }
