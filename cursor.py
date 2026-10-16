"""Dedicated cursor-movement worker thread."""

import threading
from typing import Optional, Tuple

try:
    import pyautogui
except ImportError:
    pyautogui = None


class CursorWorker:
    """Applies only the newest cursor target on a dedicated thread."""

    def __init__(self):
        self._condition = threading.Condition()
        self._target: Optional[Tuple[float, float]] = None
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run,
            name="cursor-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, x: float, y: float) -> None:
        with self._condition:
            self._target = (x, y)
            self._condition.notify()

    def _run(self) -> None:
        while not self._stopped:
            with self._condition:
                while self._target is None and not self._stopped:
                    self._condition.wait(timeout=0.05)

                if self._stopped:
                    return

                target = self._target
                self._target = None

            if target is None or pyautogui is None:
                continue

            try:
                pyautogui.moveTo(target[0], target[1], _pause=False)
            except TypeError:
                # Older pyautogui versions do not support _pause
                try:
                    pyautogui.moveTo(target[0], target[1])
                except Exception:
                    pass
            except Exception:
                pass

    def stop(self) -> None:
        self._stopped = True
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
