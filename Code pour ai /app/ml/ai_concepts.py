from typing import Dict, List
from pydantic import BaseModel

class AIConcept(BaseModel):
    """Modèle pour représenter un concept d'IA"""
    name: str
    description: str
    examples: List[str] = []
    related_concepts: List[str] = []

class AIConcepts:
    """Classe pour gérer les concepts d'IA et leurs explications"""
    
    def __init__(self):
        self.concepts = {
            "neural_network": AIConcept(
                name="Réseau de neurones",
                description="Un ensemble d'algorithmes conçus pour reconnaître des motifs en interprétant des données sensorielles",
                examples=[
                    "Détection de motifs dans les images médicales",
                    "Reconnaissance de la parole",
                    "Analyse de texte"
                ],
                related_concepts=["deep_learning", "supervised_learning"]
            ),
            "deep_learning": AIConcept(
                name="Apprentissage profond",
                description="Utilise des réseaux de neurones avec de nombreuses couches pour analyser divers facteurs dans de grandes quantités de données",
                examples=[
                    "Détection du psoriasis dans les images de peau",
                    "Analyse des radiographies",
                    "Prédiction des maladies"
                ],
                related_concepts=["neural_network", "supervised_learning"]
            ),
            "supervised_learning": AIConcept(
                name="Apprentissage supervisé",
                description="Type d'apprentissage automatique où les modèles sont entraînés en utilisant des données étiquetées déjà accompagnées de la bonne réponse",
                examples=[
                    "Classification des maladies de peau",
                    "Diagnostic médical assisté",
                    "Prédiction des résultats de traitement"
                ],
                related_concepts=["neural_network", "deep_learning"]
            ),
            "unsupervised_learning": AIConcept(
                name="Apprentissage non supervisé",
                description="Traite des données qui ne sont pas étiquetées, le système essaie donc d'apprendre les motifs et la structure à partir des données elles-mêmes",
                examples=[
                    "Regroupement des types de maladies",
                    "Découverte de nouveaux sous-types de maladies",
                    "Analyse des patterns de santé"
                ],
                related_concepts=["neural_network", "deep_learning"]
            ),
            "classification": AIConcept(
                name="Classification",
                description="Processus d'apprentissage automatique dans lequel un système apprend à placer les données dans des catégories prédéfinies",
                examples=[
                    "Classification des types de psoriasis",
                    "Catégorisation des stades de maladie",
                    "Identification des facteurs de risque"
                ],
                related_concepts=["supervised_learning", "neural_network"]
            ),
            "training": AIConcept(
                name="Entraînement",
                description="Données utilisées pour entraîner les modèles d'apprentissage automatique : la qualité et la quantité des données d'entraînement peuvent influencer les performances du modèle",
                examples=[
                    "Dataset d'images de peau étiquetées",
                    "Données cliniques avec diagnostics confirmés",
                    "Historique des patients avec résultats"
                ],
                related_concepts=["validation", "test"]
            ),
            "validation": AIConcept(
                name="Validation",
                description="Utilisation d'une partie séparée du dataset pour affiner les performances du modèle sur de nouvelles données non vues",
                examples=[
                    "Validation croisée des diagnostics",
                    "Ajustement des hyperparamètres",
                    "Évaluation de la robustesse du modèle"
                ],
                related_concepts=["training", "test"]
            ),
            "test": AIConcept(
                name="Test",
                description="Utilisation d'une autre partie du dataset pour évaluer les performances finales du modèle dans un scénario réel",
                examples=[
                    "Test sur de nouveaux patients",
                    "Évaluation en conditions réelles",
                    "Validation externe du modèle"
                ],
                related_concepts=["training", "validation"]
            )
        }
    
    def get_concept(self, concept_id: str) -> AIConcept:
        """Récupère un concept d'IA par son ID"""
        if concept_id not in self.concepts:
            raise ValueError(f"Concept {concept_id} non trouvé")
        return self.concepts[concept_id]
    
    def get_all_concepts(self) -> Dict[str, AIConcept]:
        """Récupère tous les concepts d'IA"""
        return self.concepts
    
    def get_related_concepts(self, concept_id: str) -> List[AIConcept]:
        """Récupère les concepts liés à un concept donné"""
        concept = self.get_concept(concept_id)
        return [self.get_concept(related_id) for related_id in concept.related_concepts]
    
    def generate_explanation(self, concept_id: str) -> str:
        """Génère une explication détaillée d'un concept"""
        concept = self.get_concept(concept_id)
        related = self.get_related_concepts(concept_id)
        
        explanation = f"""
        Concept : {concept.name}
        
        Description :
        {concept.description}
        
        Exemples d'utilisation :
        {chr(10).join(f"- {example}" for example in concept.examples)}
        
        Concepts liés :
        {chr(10).join(f"- {c.name}" for c in related)}
        """
        
        return explanation 