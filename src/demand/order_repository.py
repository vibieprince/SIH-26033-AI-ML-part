import time
from typing import List, Optional
from pydantic import BaseModel, Field

class Order(BaseModel):
    order_id: str
    crop: str
    quantity_kg: float
    timestamp_epoch: float = Field(default_factory=time.time)
    urgency: str = "STANDARD"  # STANDARD, PRIORITY, CRITICAL
    buyer_category: str = "RETAILER"  # HOUSEHOLD, RETAILER, WHOLESALER, INSTITUTION
    state: str
    district: str

class OrderRepository:
    def __init__(self):
        self._orders: List[Order] = []

    def add_order(self, order: Order) -> Order:
        self._orders.append(order)
        return order

    def get_active_orders(
        self, crop: str, state: str, district: Optional[str] = None, max_age_hours: float = 168.0
    ) -> List[Order]:
        cutoff = time.time() - (max_age_hours * 3600.0)
        return [
            ord for ord in self._orders
            if ord.crop.lower() == crop.lower()
            and ord.state.lower() == state.lower()
            and (district is None or ord.district.lower() == district.lower())
            and ord.timestamp_epoch >= cutoff
        ]

    def clear(self) -> None:
        self._orders.clear()

order_repo = OrderRepository()