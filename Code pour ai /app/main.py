from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from .database import SessionLocal, engine
from . import models, schemas, crud
from .routers import users, health_data, ai_analysis, doctors, psoriasis, ai_concepts

app = FastAPI(
    title="Health AI API",
    description="API pour l'analyse de santé personnalisée avec IA",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Création des tables de la base de données
models.Base.metadata.create_all(bind=engine)

# Dépendance pour obtenir la session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Inclusion des routeurs
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(health_data.router, prefix="/api/health", tags=["health"])
app.include_router(ai_analysis.router, prefix="/api/ai-analysis", tags=["ai-analysis"])
app.include_router(doctors.router, prefix="/api/doctors", tags=["doctors"])
app.include_router(psoriasis.router, prefix="/api/psoriasis", tags=["psoriasis"])
app.include_router(ai_concepts.router, prefix="/api/ai", tags=["ai_concepts"])

@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur l'API Health AI",
        "version": "1.0.0",
        "documentation": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 