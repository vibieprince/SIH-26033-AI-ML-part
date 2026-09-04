from typing import Dict, Tuple

class SupplyGapDetector:
    """Calculates supply deficit and rates alert severity level."""
    
    SEVERITY_THRESHOLDS = {
        'NONE': 0.00,
        'LOW': 0.15,
        'MEDIUM': 0.35,
        'HIGH': 0.60
    }

    @classmethod
    def analyze_gap(
        cls, 
        predicted_req_kg: float, 
        available_supply_kg: float
    ) -> Tuple[float, str, bool, str]:
        """Computes supply gap quantity, ratio severity, and recommendation messages."""
        gap_kg = max(0.0, predicted_req_kg - available_supply_kg)
        gap_ratio = gap_kg / predicted_req_kg if predicted_req_kg > 0 else 0.0
        
        if gap_ratio == 0.0:
            severity = "NONE"
            notify = False
            msg = "Supply is balanced with expected demand requirement."
        elif gap_ratio <= cls.SEVERITY_THRESHOLDS['LOW']:
            severity = "LOW"
            notify = False
            msg = "Minor deficit detected. Monitor incoming local platform orders."
        elif gap_ratio <= cls.SEVERITY_THRESHOLDS['MEDIUM']:
            severity = "MEDIUM"
            notify = True
            msg = f"Moderate shortage predicted. Notify nearby FPOs for additional {gap_kg:.1f} kg."
        elif gap_ratio <= cls.SEVERITY_THRESHOLDS['HIGH']:
            severity = "HIGH"
            notify = True
            msg = f"High demand pressure detected. High-priority alert triggered to regional farmers for {gap_kg:.1f} kg."
        else:
            severity = "CRITICAL"
            notify = True
            msg = f"CRITICAL supply deficit! Immediate procurement outreach required for {gap_kg:.1f} kg."
            
        return gap_kg, severity, notify, msg