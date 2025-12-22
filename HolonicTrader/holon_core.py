from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Literal, Any
import time
import uuid

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
    def __init__(self, name: str, disposition: Disposition, state: Literal['ACTIVE', 'PASSIVE', 'HIBERNATE'] = 'ACTIVE'):
        self.name = name
        self.disposition = disposition
        self.state = state
        self.reputation = 1.0 # NEURAL INCENTIVE: Starting reputation

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


    @abstractmethod
    def receive_message(self, sender: Any, content: Any) -> None:
        """
        Process an incoming message from another agent/system.
        """
        pass
