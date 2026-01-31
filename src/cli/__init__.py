"""CLI command modules for Genesis."""
from .commands import (
    cmd_test, cmd_discover, cmd_synthesize,
    cmd_train, cmd_chat, cmd_eval
)

__all__ = [
    'cmd_test', 'cmd_discover', 'cmd_synthesize',
    'cmd_train', 'cmd_chat', 'cmd_eval'
]