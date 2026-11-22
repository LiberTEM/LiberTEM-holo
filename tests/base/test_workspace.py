"""Tests of `Workspace`, which keeps reusable resources, one entry per slot."""
import threading

from libertem_holo.base.workspace import (
    Workspace,
    clear_default_workspace,
    default_workspace,
)


class _Resource:
    built = 0

    def __init__(self, key: object) -> None:
        self.key = key
        _Resource.built += 1


def test_an_entry_is_built_once_per_key():
    workspace = Workspace()
    _Resource.built = 0
    first = workspace.entry("a", 1, lambda: _Resource(1))
    assert workspace.entry("a", 1, lambda: _Resource(1)) is first
    assert _Resource.built == 1


def test_another_key_replaces_the_entry_of_its_slot():
    workspace = Workspace()
    first = workspace.entry("a", 1, lambda: _Resource(1))
    second = workspace.entry("a", 2, lambda: _Resource(2))
    assert second is not first
    assert workspace.entry("a", 1, lambda: _Resource(1)) is not first
    # Other slots are independent:
    other = workspace.entry("b", 1, lambda: _Resource("b"))
    assert other.key == "b"
    assert workspace.entry("a", 1, lambda: _Resource(3)).key == 1


def test_clear_drops_all_entries():
    workspace = Workspace()
    first = workspace.entry("a", 1, lambda: _Resource(1))
    workspace.clear()
    assert workspace.entry("a", 1, lambda: _Resource(1)) is not first


def test_a_workspace_clears_itself_after_a_with_block():
    with Workspace() as workspace:
        first = workspace.entry("a", 1, lambda: _Resource(1))
    assert workspace.entry("a", 1, lambda: _Resource(1)) is not first


def test_the_default_workspace_is_one():
    assert default_workspace() is default_workspace()
    first = default_workspace().entry("test", 1, lambda: _Resource(1))
    clear_default_workspace()
    assert default_workspace().entry("test", 1, lambda: _Resource(1)) is not first
    clear_default_workspace()


def test_the_lock_lets_one_thread_use_a_workspace_at_a_time():
    workspace = Workspace()
    inside = []
    overlaps = []

    def use() -> None:
        with workspace.lock:
            inside.append(1)
            if len(inside) > 1:
                overlaps.append(1)
            # A computation may take the lock again (`entry` does):
            workspace.entry("a", 1, lambda: _Resource(1))
            inside.pop()

    threads = [threading.Thread(target=use) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlaps
