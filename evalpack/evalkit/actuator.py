"""Actuator interface — the seam that lets the executor / planner / judge be
written once and work across Layer 0 (kernel) and Layer 1 (browser).

An actuator mechanically pokes controls and reports what happened. It makes
*no* judgments (docs/EVAL_TESTSET_DESIGN.md §6.1): enumerate the ground-truth
controls, set values / click, and snapshot the resulting output.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schemas import WidgetSpec


class Actuator(ABC):
    """Drive a demo's controls deterministically and capture output state."""

    @abstractmethod
    def start(self) -> None:
        """Bring the demo to life (run the notebook / open the page)."""

    @abstractmethod
    def enumerate(self) -> list[WidgetSpec]:
        """Return the authoritative list of interactive controls."""

    @abstractmethod
    def set_values(self, mapping: dict[str, Any]) -> dict[str, bool]:
        """Set one or more controls by display name. Returns name -> success."""

    @abstractmethod
    def click(self, name: str, repeat: int = 1) -> bool:
        """Click a button-like control `repeat` times. Returns success."""

    @abstractmethod
    def snapshot(self, tag: str) -> tuple[str, list[str]]:
        """Capture current output state. Returns (content_hash, [png_paths])."""

    @abstractmethod
    def stop(self) -> None:
        """Tear down (shut the kernel / close the browser)."""

    def __enter__(self) -> "Actuator":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def get_actuator(lane: str, **kwargs: Any) -> Actuator:
    """Factory: pick an actuator by lane (see router.LANE)."""
    if lane == "kernel":
        from actuate_kernel import KernelActuator
        return KernelActuator(**kwargs)
    if lane == "browser":
        try:
            from actuate_browser import BrowserActuator  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise NotImplementedError(
                f"browser lane not available yet (needs probe + actuate_browser.py): {e}")
        return BrowserActuator(**kwargs)
    raise NotImplementedError(f"no actuator implemented for lane={lane!r}")
