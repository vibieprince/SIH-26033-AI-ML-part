from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DemandEngineConfig:
    recency_half_life_hours: float = 48.0
    elasticity_alpha: float = -0.35
    multiplier_min_bound: float = 0.60
    multiplier_max_bound: float = 2.00
    
    urgency_weights: Dict[str, float] = field(default_factory=lambda: {
        "STANDARD": 1.0,
        "PRIORITY": 1.25,
        "CRITICAL": 1.50
    })
    
    buyer_weights: Dict[str, float] = field(default_factory=lambda: {
        "HOUSEHOLD": 1.0,
        "RETAILER": 1.10,
        "WHOLESALER": 1.25,
        "INSTITUTION": 1.30
    })

@dataclass
class OpportunityWeights:
    gap_weight: float = 0.40
    price_weight: float = 0.25
    weather_weight: float = 0.20
    distance_weight: float = 0.15
    notification_threshold: float = 75.0

config = DemandEngineConfig()
opp_config = OpportunityWeights()