# src/demand_store.py
"""
Thread-safe in-memory order store for Kisan Guard.

Normalizes all commodity/location lookups with .lower().strip() so
'Onion', 'onion', ' ONION ' are all treated as the same key.
Orders are stored per-session (server lifetime) and accumulate across
requests.  A classmethod clear() is provided for test teardown.
"""

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class DemandSignalStore:
    """
    Singleton in-memory order store.

    Thread-safe via a module-level RLock.  No DB dependency – designed
    for rapid prototyping and demo sessions where the API server runs
    continuously.
    """

    _lock: threading.RLock = threading.RLock()
    _orders: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _norm(value: str) -> str:
        """Normalize a string key for case-insensitive matching."""
        return value.lower().strip() if value else ""

    @classmethod
    def _commodity_match(cls, order: Dict[str, Any], commodity: str) -> bool:
        """True if the order's commodity equals the normalized search term."""
        if not commodity:
            return True
        norm_target = cls._norm(commodity)
        # Accept 'commodity' or legacy 'crop' key
        order_comm = cls._norm(order.get("commodity", "") or order.get("crop", ""))
        return order_comm == norm_target

    @classmethod
    def _location_match(cls, order: Dict[str, Any], location: str) -> bool:
        """True if the order location contains the normalized search term."""
        if not location:
            return True
        norm_target = cls._norm(location)
        # Accept 'location', 'district', or 'state' key
        for key in ("location", "district", "state"):
            if norm_target in cls._norm(order.get(key, "")):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @classmethod
    def add_order(
        cls,
        commodity: str,
        location: str,
        quantity_kg: float,
        buyer_type: str = "retailer",
        order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Append an incoming order to the in-memory store.

        Keys are stored in their original casing for display, but all
        lookups use the normalized versions via _norm().

        Returns the stored order dict.
        """
        order = {
            "order_id": order_id or f"ORD-{uuid.uuid4().hex[:8].upper()}",
            # Raw display values
            "commodity": commodity.strip(),
            "crop": commodity.strip(),          # legacy alias
            "location": location.strip(),
            "district": location.strip(),       # legacy alias
            "state": location.strip(),          # legacy alias
            "quantity_kg": float(quantity_kg),
            "buyer_type": buyer_type.strip(),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        with cls._lock:
            cls._orders.append(order)
        return order

    @classmethod
    def get_orders(
        cls,
        commodity: str = "",
        location: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Return all orders matching the (optional) commodity and location
        filters.  Empty string means "no filter" (returns all).
        """
        with cls._lock:
            snapshot = list(cls._orders)

        return [
            o for o in snapshot
            if cls._commodity_match(o, commodity) and cls._location_match(o, location)
        ]

    @classmethod
    def get_aggregated_demand_kg(cls, commodity: str, location: str) -> float:
        """
        Sum of quantity_kg for all orders matching the given commodity and
        location.  Uses normalized matching so 'Noida' == 'noida' == 'NOIDA'.

        Returns 0.0 if no matching orders exist.
        """
        matching = cls.get_orders(commodity=commodity, location=location)
        return round(sum(float(o.get("quantity_kg", 0.0)) for o in matching), 2)

    @classmethod
    def get_order_count(cls, commodity: str = "", location: str = "") -> int:
        """Returns the number of matching orders (useful for ratio calculations)."""
        return len(cls.get_orders(commodity=commodity, location=location))

    @classmethod
    def clear(cls) -> None:
        """Flush all stored orders.  Primarily used in tests."""
        with cls._lock:
            cls._orders.clear()

    @classmethod
    def total_orders(cls) -> int:
        """Total number of orders logged since server start."""
        with cls._lock:
            return len(cls._orders)