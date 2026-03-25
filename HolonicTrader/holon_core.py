from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal, Any, Callable, Dict, List, Optional
import time
import uuid
import threading
import logging
from enum import Enum, auto
from collections import defaultdict

logger = logging.getLogger("HolonicCore")


# =============================================================================
# MESSAGE BUS — Inter-Holon pub/sub communication backbone
# =============================================================================

class MessageBus:
    """
    Centralized publish/subscribe message bus for inter-Holon communication.

    Standard QUANT-OPS topics:
        quant_ops.forensics  — Chronos publishes forensic reports
        quant_ops.security   — Aegis publishes vulnerability/anomaly reports
        quant_ops.fixes      — Helix publishes fix proposals and constraints
        quant_ops.strategy   — Atlas publishes capital allocation and rules
        quant_ops.cycle      — QuantOps orchestrator publishes cycle events
    """

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._history: Dict[str, List[Message]] = defaultdict(list)
        self._lock = threading.Lock()
        self._max_history = 100  # Per-topic retention

    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Register a callback for messages on a given topic."""
        with self._lock:
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)
                logger.debug(f"MessageBus: subscribed to '{topic}'")

    def unsubscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        """Remove a callback from a topic."""
        with self._lock:
            if callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)

    def publish(self, topic: str, message: Message) -> int:
        """
        Publish a message to all subscribers of a topic.
        Returns the number of subscribers notified.
        """
        with self._lock:
            subscribers = list(self._subscribers.get(topic, []))
            # Store in history
            history = self._history[topic]
            history.append(message)
            if len(history) > self._max_history:
                self._history[topic] = history[-self._max_history:]

        notified = 0
        for cb in subscribers:
            try:
                cb(message)
                notified += 1
            except Exception as e:
                logger.error(f"MessageBus: subscriber error on '{topic}': {e}")
        return notified

    def get_history(self, topic: str, limit: int = 10) -> List['Message']:
        """Retrieve recent messages from a topic's history."""
        with self._lock:
            return list(self._history.get(topic, []))[-limit:]

    def get_latest(self, topic: str) -> Optional['Message']:
        """Get the most recent message on a topic, or None."""
        with self._lock:
            history = self._history.get(topic, [])
            return history[-1] if history else None

    def clear_history(self, topic: Optional[str] = None) -> None:
        """Clear history for a specific topic or all topics."""
        with self._lock:
            if topic:
                self._history[topic] = []
            else:
                self._history.clear()

    @property
    def topics(self) -> List[str]:
        """List all topics that have subscribers or history."""
        with self._lock:
            return list(set(list(self._subscribers.keys()) + list(self._history.keys())))


# Global singleton message bus
_global_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get or create the global MessageBus singleton."""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus


class PositionState(Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    PENDING_EXIT = "PENDING_EXIT"
    CLOSED = "CLOSED"
    ZOMBIE = "ZOMBIE"

class OrderState(Enum):
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class Message:
    """
    Standard communication packet between Holons.
    """
    sender: str
    type: str
    payload: Any
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class Disposition:
    """
    Defines the autonomy and integration levels of a Holon.
    Autonomy: float
    Integration: float
    """
    autonomy: float
    integration: float

class Holon(ABC):
    """
    Abstract Base Class for all Holons in the system.
    """
    def __init__(self, name: str, disposition: Disposition, state: Literal['ACTIVE', 'PASSIVE', 'HIBERNATE'] = 'ACTIVE', message_bus: Optional[MessageBus] = None):
        self.name = name
        self.disposition = disposition
        self.state = state
        self.reputation = 1.0 # NEURAL INCENTIVE: Starting reputation
        self.message_bus = message_bus  # Optional pub/sub bus for inter-agent comms

    def update_reputation(self, reward: float):
        """
        Update the holon's reputation and adjust disposition autonomy accordingly.
        """
        self.reputation = max(0.1, self.reputation + reward)
        
        # Performance-based Autonomy scaling
        # Range: 0.1 (Integration) to 0.95 (Full Autonomy)
        new_autonomy = min(0.95, max(0.1, 0.8 * self.reputation))
        self.disposition.autonomy = new_autonomy
        self.disposition.integration = 1.0 - new_autonomy
        
        # print(f"[{self.name}] REPUTATION LOG: {self.reputation:.3f} (Autonomy: {self.disposition.autonomy:.2f})")


    def get_dashboard_state(self) -> dict:
        """
        Return a dict of data this holon exposes to the dashboard.
        Override in subclasses to provide agent-specific state.
        The keys become top-level fields in the hub_state payload.
        """
        return {}

    @abstractmethod
    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process an incoming message from another agent/system.
        """
        pass
