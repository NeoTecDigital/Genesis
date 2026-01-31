"""Command router for Genesis CLI - imports from specialized modules."""

# Import core commands (test, discover, synthesize)
from src.cli.commands_core import cmd_test, cmd_discover, cmd_synthesize

# Import training commands (train, chat, eval)
from src.cli.commands_train import cmd_train, cmd_chat, cmd_eval

# Import helper utilities
from src.cli.commands_helpers import *

# Re-export all commands for CLI dispatcher
__all__ = ['cmd_test', 'cmd_discover', 'cmd_synthesize', 'cmd_train', 'cmd_chat', 'cmd_eval']
