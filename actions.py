"""Dedicated worker thread for fire-and-forget OS actions (clicks, hotkeys).

Calling pyautogui directly inside HandGestureController.update() runs it on
the same thread that reads camera frames, submits detection requests, and
renders the preview window. The OS-level key/mouse simulation pyautogui does
can take a handful of milliseconds — easily enough to visibly stall frame
capture right when a click or app-switch fires, which is exactly the moment
you're watching the overlay most closely (it looks like a stutter or freeze
during the swipe).

Unlike CursorWorker (which only keeps the newest target — dropping stale
mouse positions is correct, since only the latest one matters), every action
submitted here must actually run: a click or a swipe is a discrete, one-shot
event, not continuous state, so ActionWorker queues and executes all of them
in order instead of coalescing to "just the latest."
"""

import queue
import threading
from typing import Callable


class ActionWorker:
    """Runs submitted zero-arg callables, in order, on a dedicated thread."""

    def __init__(self):
        self._queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="action-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, action: Callable[[], None]) -> None:
        """Queue an action to run ASAP on the worker thread. Never blocks the caller."""
        self._queue.put(action)

    def _run(self) -> None:
        while not self._stopped:
            try:
                action = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                action()
            except Exception:
                # A single misbehaving action should never kill the worker.
                pass

    def stop(self) -> None:
        self._stopped = True
        self._thread.join(timeout=1.0)