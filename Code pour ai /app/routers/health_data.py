from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime

from .. import crud, schemas, auth
from ..database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.HealthData)
def create_health_data(
    health_data: schemas.HealthDataCreate,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Vérifier que l'utilisateur crée des données pour lui-même
    if health_data.user_id != current_user.id and not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Vous ne pouvez pas créer des données de santé pour d'autres utilisateurs"
        )
    return crud.create_health_data(db=db, health_data=health_data)

@router.get("/me", response_model=List[schemas.HealthData])
def read_my_health_data(
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    return crud.get_user_health_data(db=db, user_id=current_user.id)

@router.get("/user/{user_id}", response_model=List[schemas.HealthData])
def read_user_health_data(
    user_id: int,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Seuls les médecins peuvent accéder aux données des autres utilisateurs"
        )
    return crud.get_user_health_data(db=db, user_id=user_id)

@router.get("/types", response_model=List[str])
def get_health_data_types():
    return [
        "sleep",
        "weight",
        "diet",
        "exercise",
        "medical_history",
        "medication",
        "wearable_data",
        "mood",
        "lab_results"
    ] 