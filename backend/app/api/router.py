"""Top-level API router. Placeholder routers gain endpoints in later phases (docs/plan.md)."""

from fastapi import APIRouter

from app.api.routes import (
    backtests,
    copy,
    discovery,
    portfolio,
    rankings,
    risk,
    strategies,
    system,
    wallets,
    ws,
)

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(wallets.router)
api_router.include_router(discovery.router)
api_router.include_router(rankings.router)
api_router.include_router(backtests.router)
api_router.include_router(strategies.router)
api_router.include_router(portfolio.router)
api_router.include_router(risk.router)
api_router.include_router(copy.router)
api_router.include_router(ws.router)
