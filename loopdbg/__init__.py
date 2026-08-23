"""loopdbg — an autonomous debugging agent built as an exercise in loop engineering."""

from .loop import Outcome, run
from .brakes import default_brakes
from .tools import Workspace

__version__ = "0.1.0"
__all__ = ["run", "Outcome", "Workspace", "default_brakes"]
