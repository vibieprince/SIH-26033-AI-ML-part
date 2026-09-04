from fastapi import APIRouter, HTTPException
from typing import List, Optional
from src.demand.order_repository import order_repo, Order

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.post("/", response_model=Order)
async def submit_order(order: Order):
    if order.quantity_kg <= 0:
        raise HTTPException(status_code=400, detail="Order quantity must be greater than zero.")
    return order_repo.add_order(order)

@router.get("/", response_model=List[Order])
async def list_orders(crop: str, state: str, district: Optional[str] = None):
    return order_repo.get_active_orders(crop=crop, state=state, district=district)