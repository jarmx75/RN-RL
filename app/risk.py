import time
from typing import Dict, Any

class RiskEngine:
    def __init__(self, max_consecutive_losses: int = 3, operating_hours=(0, 24)):
        self.max_consecutive_losses = max_consecutive_losses
        self.operating_hours = operating_hours
        self.consecutive_losses = 0

    def can_trade(self, now: float, kpi_state: Dict[str, Any]) -> bool:
        hour = time.localtime(now).tm_hour
        if not (self.operating_hours[0] <= hour < self.operating_hours[1]):
            return False
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        return True

    def register_outcome(self, win: bool):
        if win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1