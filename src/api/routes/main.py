from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.forecast import router as forecast_router
from src.api.routes.orders import router as orders_router
from src.api.routes.notifications import router as notifications_router

app = FastAPI(
    title="Agri-Tech Demand & Opportunity Intelligence Engine",
    description="Production pipeline for ML supply prediction, live demand aggregation, gap calculation, and farmer matching.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(forecast_router)
app.include_router(orders_router)
app.include_router(notifications_router)

@app.get("/")
async def root_health_check():
    return {
        "status": "ONLINE",
        "system": "Agri-Tech Forecasting Intelligence API",
        "docs_url": "/docs"
    }