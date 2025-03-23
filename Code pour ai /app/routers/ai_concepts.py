from fastapi import APIRouter, HTTPException
from typing import Dict, List
from pydantic import BaseModel

from ..ml.ai_concepts import AIConcepts, AIConcept

router = APIRouter()
ai_concepts = AIConcepts()

class ConceptResponse(BaseModel):
    """Modèle de réponse pour un concept d'IA"""
    concept: AIConcept
    related_concepts: List[AIConcept]

@router.get("/concepts", response_model=Dict[str, AIConcept])
async def get_all_concepts():
    """Récupère tous les concepts d'IA disponibles"""
    return ai_concepts.get_all_concepts()

@router.get("/concepts/{concept_id}", response_model=ConceptResponse)
async def get_concept(concept_id: str):
    """Récupère un concept d'IA spécifique avec ses concepts liés"""
    try:
        concept = ai_concepts.get_concept(concept_id)
        related = ai_concepts.get_related_concepts(concept_id)
        return {
            "concept": concept,
            "related_concepts": related
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/concepts/{concept_id}/explanation")
async def get_concept_explanation(concept_id: str):
    """Récupère une explication détaillée d'un concept d'IA"""
    try:
        return ai_concepts.generate_explanation(concept_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) 