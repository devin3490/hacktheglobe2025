from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import openai
from datetime import datetime

from .. import crud, schemas, auth
from ..database import get_db

router = APIRouter()

async def generate_health_profile(db: Session, user_id: int):
    # Récupérer toutes les données de santé de l'utilisateur
    health_data = crud.get_user_health_data(db, user_id=user_id)
    
    # Préparer les données pour l'analyse
    data_summary = {
        "sleep": [],
        "weight": [],
        "diet": [],
        "exercise": [],
        "mood": []
    }
    
    for data in health_data:
        if data.data_type in data_summary:
            data_summary[data.data_type].append(data.data)
    
    # Appeler l'API OpenAI pour l'analyse
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Vous êtes un expert en santé qui analyse les données de santé d'un utilisateur pour générer un profil de santé personnalisé."},
                {"role": "user", "content": f"Analyser les données suivantes et générer un profil de santé détaillé : {data_summary}"}
            ]
        )
        
        analysis_data = {
            "health_profile": response.choices[0].message.content,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Sauvegarder l'analyse
        crud.create_ai_analysis(
            db=db,
            analysis=schemas.AIAnalysisCreate(
                user_id=user_id,
                analysis_type="health_profile",
                analysis_data=analysis_data
            )
        )
    except Exception as e:
        print(f"Erreur lors de la génération du profil de santé : {str(e)}")

@router.post("/generate-profile", response_model=schemas.AIAnalysis)
async def create_health_profile(
    background_tasks: BackgroundTasks,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    # Lancer la génération du profil en arrière-plan
    background_tasks.add_task(generate_health_profile, db, current_user.id)
    
    return {
        "id": 0,  # L'ID sera généré par la base de données
        "user_id": current_user.id,
        "analysis_type": "health_profile",
        "analysis_data": {"status": "Génération en cours"},
        "timestamp": datetime.utcnow()
    }

@router.get("/me", response_model=List[schemas.AIAnalysis])
def read_my_analyses(
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    return crud.get_user_ai_analyses(db=db, user_id=current_user.id)

@router.get("/user/{user_id}", response_model=List[schemas.AIAnalysis])
def read_user_analyses(
    user_id: int,
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Seuls les médecins peuvent accéder aux analyses des autres utilisateurs"
        )
    return crud.get_user_ai_analyses(db=db, user_id=user_id) 