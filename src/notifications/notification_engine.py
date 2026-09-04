import time
from dataclasses import dataclass
from typing import List, Dict, Any
from src.demand.demand_config import opp_config

@dataclass
class NotificationMessage:
    notification_id: str
    target_farmer_id: str
    crop: str
    opportunity_score: float
    message_text: str
    timestamp_epoch: float
    status: str

class NotificationEngine:
    def __init__(self):
        self._sent_notifications: List[NotificationMessage] = []

    def process_and_dispatch(
        self, opportunity_score: float, matched_farmers: List[Dict[str, Any]], context: Dict[str, Any]
    ) -> List[NotificationMessage]:
        
        if opportunity_score < opp_config.notification_threshold:
            return []

        dispatched = []
        for idx, farmer in enumerate(matched_farmers):
            msg_text = (
                f"ALERT: High Demand for {context.get('crop')} near {context.get('city')}! "
                f"Opportunity Score: {opportunity_score}/100. Your estimated match is {farmer.get('match_score')}%. "
                f"Contact hub to secure guaranteed price."
            )
            
            notification = NotificationMessage(
                notification_id=f"NOTIF-{int(time.time())}-{idx}",
                target_farmer_id=farmer.get("farmer_id", f"FARMER-{idx}"),
                crop=context.get("crop", "Crop"),
                opportunity_score=opportunity_score,
                message_text=msg_text,
                timestamp_epoch=time.time(),
                status="DISPATCHED"
            )
            
            self._sent_notifications.append(notification)
            dispatched.append(notification)

        return dispatched

    def get_history(self) -> List[NotificationMessage]:
        return self._sent_notifications

notification_engine = NotificationEngine()