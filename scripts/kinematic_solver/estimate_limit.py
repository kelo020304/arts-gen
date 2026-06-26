"""Compatibility entrypoint for the post_process kinematic solver."""

from __future__ import annotations

from post_process.kinematic_solver.estimate_limit import *  # noqa: F401,F403
from post_process.kinematic_solver.estimate_limit import main


if __name__ == "__main__":
    main()
