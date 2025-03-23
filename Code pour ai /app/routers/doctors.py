from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from .. import crud, schemas, auth
from ..database import get_db

router = APIRouter()

@router.post("/review", response_model=schemas.DoctorReview)
def create_doctor_review(
    review: schemas.DoctorReviewCreate,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Seuls les médecins peuvent créer des avis"
        )
    
    # Vérifier que l'analyse existe
    analysis = crud.get_user_ai_analyses(db, review.analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analyse non trouvée"
        )
    
    return crud.create_doctor_review(db=db, review=review)

@router.get("/my-reviews", response_model=List[schemas.DoctorReview])
def read_my_reviews(
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Seuls les médecins peuvent accéder aux avis"
        )
    return crud.get_doctor_reviews(db=db, doctor_id=current_user.id)

@router.get("/analysis/{analysis_id}/reviews", response_model=List[schemas.DoctorReview])
def read_analysis_reviews(
    analysis_id: int,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Vérifier que l'utilisateur a accès à l'analyse
    analysis = crud.get_user_ai_analyses(db, analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Analyse non trouvée"
        )
    
    if analysis.user_id != current_user.id and not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Accès non autorisé"
        )
    
    return crud.get_analysis_reviews(db=db, analysis_id=analysis_id) 