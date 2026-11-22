"""Reusable resources of computations: buffers, FFT plans, captured CUDA graphs.

Setting up such resources can cost more than the computation that uses them,
for example in the phase unwrapping on the GPU
(:func:`~libertem_holo.base.unwrap.unwrap_phase_irls`,
:func:`~libertem_holo.base.unwrap.unwrap_phase_admm`). A :class:`Workspace`
keeps them between calls.

It holds at most one entry per *slot* (one slot per kind of computation, such as
:code:`"unwrap"`). An entry is set up for a *key*, such as a shape and dtype; a
computation with another key replaces the slot's entry, and frees the memory
of the old one before it sets up the new one.

A workspace serves one computation at a time: a lock lets only one thread use
it at once. Functions that take a :code:`workspace` argument use the default
workspace (:func:`default_workspace`) if none is given, so concurrent calls from
several threads wait for each other. For calls that run in parallel, give each
thread its own :class:`Workspace`:

.. code-block:: python

    workspace = Workspace()  # one per thread
    for frame in frames:
        result = unwrap_phase_admm(frame, workspace=workspace)
    workspace.clear()

:func:`clear_default_workspace` frees the default workspace's memory, and
:meth:`Workspace.clear` that of another one.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from types import TracebackType

    # Only for annotations (`typing.Self` needs Python 3.11):
    from typing_extensions import Self

T = TypeVar("T")


class Workspace:
    """Reusable resources of computations, one entry per slot.

    Can be used as a context manager, which clears it on exit.
    """

    def __init__(self) -> None:
        """Create an empty workspace."""
        self._lock = threading.RLock()
        self._entries: dict[str, tuple[Hashable, object]] = {}

    @property
    def lock(self) -> threading.RLock:
        """The lock that a computation holds while it uses the workspace."""
        return self._lock

    def entry(self, slot: str, key: Hashable, build: Callable[[], T]) -> T:
        """Return the entry of `slot` for `key`, and set it up with `build` if needed.

        An entry of the slot for another key is dropped first, and cupy's memory
        pools are freed, so that the new entry can use the memory of the old one.
        """
        with self._lock:
            current = self._entries.get(slot)
            if current is not None:
                if current[0] == key:
                    return current[1]  # type: ignore[return-value]
                del self._entries[slot]
                del current
                _free_cupy_pools()
            value = build()
            self._entries[slot] = (key, value)
            return value

    def clear(self) -> None:
        """Drop all entries, and free cupy's memory pools."""
        with self._lock:
            self._entries.clear()
            _free_cupy_pools()

    def __enter__(self) -> Self:
        """Return the workspace."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clear the workspace."""
        self.clear()


_default: Workspace | None = None
_default_lock = threading.Lock()


def default_workspace() -> Workspace:
    """Return the default workspace, which functions use when given none."""
    global _default  # noqa: PLW0603
    with _default_lock:
        if _default is None:
            _default = Workspace()
        return _default


def clear_default_workspace() -> None:
    """Drop the entries of the default workspace, and free cupy's memory pools.

    Freeing the pools returns all memory that cupy's pools hold unused to the GPU
    and the host, not only that of the default workspace.
    """
    default_workspace().clear()


def _free_cupy_pools() -> None:
    """Return the unused memory of cupy's device and pinned memory pools."""
    try:
        import cupy  # noqa: PLC0415
    except ImportError:
        return
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()


__all__ = ["Workspace", "clear_default_workspace", "default_workspace"]
