from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image
import io

from .. import crud, schemas, auth
from ..database import get_db
from ..ml.psoriasis_model import PsoriasisModel

router = APIRouter()
psoriasis_model = PsoriasisModel()

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_psoriasis(
    image: UploadFile = File(...),
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Analyse une image pour détecter le psoriasis avec une analyse détaillée"""
    try:
        # Lire l'image
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image = np.array(image)
        
        # Faire la prédiction avec analyse détaillée
        prediction = psoriasis_model.predict(image)
        
        # Sauvegarder les résultats dans la base de données
        health_data = {
            "type": "psoriasis_analysis",
            "image_name": image.filename,
            "prediction": prediction,
            "features": prediction["features"],
            "analysis": prediction["analysis"]
        }
        
        crud.create_health_data(
            db=db,
            health_data=schemas.HealthDataCreate(
                user_id=current_user.id,
                data_type="psoriasis",
                data=health_data
            )
        )
        
        return {
            "diagnostic": {
                "class": prediction["class"],
                "confidence": prediction["confidence"],
                "predictions": prediction["predictions"]
            },
            "caracteristiques_visuelles": {
                "texture": prediction["features"]["texture"]["description"],
                "couleurs": prediction["features"]["color"]["description"],
                "bords": prediction["features"]["edges"]["description"],
                "symetrie": prediction["features"]["symmetry"]["description"]
            },
            "analyse_detaillee": prediction["analysis"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse de l'image : {str(e)}"
        )

@router.get("/history", response_model=List[schemas.HealthData])
def get_psoriasis_history(
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Récupère l'historique des analyses de psoriasis"""
    health_data = crud.get_user_health_data(db=db, user_id=current_user.id)
    return [data for data in health_data if data.data_type == "psoriasis"]

@router.post("/train")
async def train_model(
    current_user: schemas.User = Depends(auth.get_current_active_user),
    db: Session = Depends(get_db)
):
    """Entraîne le modèle sur le dataset PAD-UFES-20"""
    if not current_user.is_doctor:
        raise HTTPException(
            status_code=403,
            detail="Seuls les médecins peuvent entraîner le modèle"
        )
    
    try:
        # Chemins vers les données d'entraînement et de validation
        train_data_dir = "data/train"
        validation_data_dir = "data/validation"
        
        # Entraîner le modèle
        history = psoriasis_model.train(
            train_data_dir=train_data_dir,
            validation_data_dir=validation_data_dir,
            epochs=10
        )
        
        return {
            "message": "Modèle entraîné avec succès",
            "history": {
                "accuracy": history.history['accuracy'],
                "val_accuracy": history.history['val_accuracy'],
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraînement du modèle : {str(e)}"
        ) 