# main.py
from fastapi import FastAPI
from src.api.routes import router as api_router
from src.config import API_HOST, API_PORT, LOG_LEVEL
from src.utils.logger import setup_logging
import logging

# Setup logging before anything else
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Oracle to Snowflake Migration API",
    description="AI-powered service to migrate Oracle PL/SQL procedures and SQL Views to Snowflake.",
    version="1.0.0"
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Oracle to Snowflake Migration API starting up...")
    logger.info(f"API will listen on http://{API_HOST}:{API_PORT}/api/v1")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Oracle to Snowflake Migration API shutting down.")

# To run this file:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# or with the values from config/settings.py:
# uvicorn main:app --host $API_HOST --port $API_PORT --reload