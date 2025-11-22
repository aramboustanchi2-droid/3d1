"""FastAPI Dashboard for University Knowledge System
داشبورد API برای مشاهده وضعیت سیستم دانشگاهی.
"""
from __future__ import annotations
from fastapi import FastAPI, Query
from typing import Optional
from .university_monitor import (
    get_overview, get_agents, get_security_events,
    get_universities, get_documents
)

app = FastAPI(title="University Knowledge Dashboard", version="0.1.0")

@app.get("/stats")
def stats():
    return get_overview()

@app.get("/agents")
def agents():
    return get_agents()

@app.get("/security-events")
def security_events(limit: int = Query(100, ge=1, le=1000)):
    return get_security_events(limit)

@app.get("/universities")
def universities():
    return get_universities()

@app.get("/documents")
def documents(university_key: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    return get_documents(university_key, limit)

# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
