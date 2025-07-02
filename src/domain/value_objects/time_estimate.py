"""Time estimation value object using PERT method."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeEstimate:
    """
    PERT-based time estimation with three-point estimates.
    
    Uses optimistic, realistic, and pessimistic estimates to calculate
    expected duration and variance.
    """
    
    optimistic_hours: float
    realistic_hours: float
    pessimistic_hours: float
    
    def __post_init__(self):
        """Validate estimates follow optimistic <= realistic <= pessimistic."""
        if not (self.optimistic_hours <= self.realistic_hours <= self.pessimistic_hours):
            raise ValueError(
                "Time estimates must follow: optimistic <= realistic <= pessimistic"
            )
        if self.optimistic_hours < 0:
            raise ValueError("Time estimates must be non-negative")
    
    @property
    def expected_hours(self) -> float:
        """Calculate PERT expected duration."""
        return (
            self.optimistic_hours + 
            4 * self.realistic_hours + 
            self.pessimistic_hours
        ) / 6
    
    @property
    def variance(self) -> float:
        """Calculate estimate variance."""
        return ((self.pessimistic_hours - self.optimistic_hours) / 6) ** 2
    
    @property
    def standard_deviation(self) -> float:
        """Calculate standard deviation of estimate."""
        return self.variance ** 0.5
    
    @classmethod
    def from_hours(
        cls,
        optimistic: float,
        realistic: float,
        pessimistic: float
    ) -> 'TimeEstimate':
        """Create time estimate from hour values."""
        return cls(
            optimistic_hours=optimistic,
            realistic_hours=realistic,
            pessimistic_hours=pessimistic
        )