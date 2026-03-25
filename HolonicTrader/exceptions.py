class HolonicTraderException(Exception):
    """Base exception class for HolonicTrader."""
    pass

class DeadMansSwitchTriggered(HolonicTraderException):
    """Raised when a critical systemic failure or irreconcilable position mismatch is detected, requiring an immediate hard halt."""
    pass
