from fastapi import APIRouter
from typing import List
from src.notifications.notification_engine import notification_engine, NotificationMessage

router = APIRouter(prefix="/notifications", tags=["Notifications"])

@router.get("/history", response_model=List[NotificationMessage])
async def get_notification_history():
    return notification_engine.get_history()