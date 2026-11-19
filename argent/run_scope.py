"""Which branch of a run a piece of code is working for.

A room's rules (what to ignore) and a step's model parameters (temperature, top_p, ...) are installed
for the time the step works. They are kept on process-wide stacks, which is fine while one step runs at
a time — but two branches of a `parallel` fork run side by side, and one branch's `temperature` must
not reach the other's model call.

Each branch therefore works inside its own *scope*: a chain of ids, one per fork level. Something
installed in a scope is seen by that scope and by the scopes nested inside it, never by a sibling.
Code that has no scope at all (the main path, a helper thread that was not given the branch's context)
sees everything, which is what it saw before scopes existed — so the more restrictive reading wins
when in doubt.
"""

import contextvars
import itertools
from contextlib import contextmanager

_chain = contextvars.ContextVar("argent_run_scope", default=())
_ids = itertools.count(1)


def current() -> tuple:
    """The chain of the scope the calling code works in (empty on the main path)."""
    return _chain.get()


def visible(owner: tuple) -> bool:
    """Whether something installed under the scope `owner` applies to the calling code."""
    here = _chain.get()
    return not here or here[:len(owner)] == tuple(owner)


@contextmanager
def branch():
    """Work inside a new scope, nested in the current one."""
    token = _chain.set(_chain.get() + (next(_ids),))
    try:
        yield _chain.get()
    finally:
        _chain.reset(token)
