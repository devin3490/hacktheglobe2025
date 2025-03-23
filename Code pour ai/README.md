# Application d'Analyse de Santé avec IA

Une application FastAPI pour l'analyse de données de santé avec des fonctionnalités d'intelligence artificielle.

## Fonctionnalités

### 1. Analyse de Données de Santé
- Stockage et récupération des données de santé
- Visualisation des données
- Analyse statistique des tendances

### 2. Détection du Psoriasis avec IA
- Classification hiérarchique du psoriasis
- Analyse détaillée des caractéristiques visuelles
- Recommandations personnalisées

#### Caractéristiques du Modèle
- Architectures avancées (EfficientNetV2, ResNet50, Custom)
- Classification hiérarchique (type et sévérité)
- Analyse multi-modalité (texture, couleur, bords, symétrie)
- Apprentissage hybride (supervisé et non supervisé)
- Augmentation de données en temps réel
- Métriques avancées (accuracy, AUC, precision, recall, F1-score)

#### Types de Psoriasis Détectés
- Plaque
- Guttate
- Inverse
- Pustulaire
- Érythrodermique

#### Niveaux de Sévérité
- Léger
- Modéré
- Sévère

### 3. Concepts d'IA
- Documentation des concepts d'IA
- Exemples pratiques
- Relations entre les concepts

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/votre-username/health-analysis-ai.git
cd health-analysis-ai
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement :
```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

## Utilisation

1. Lancer l'application :
```bash
uvicorn app.main:app --reload
```

2. Accéder à la documentation API :
```
http://localhost:8000/docs
```

### Endpoints Principaux

#### Données de Santé
- `POST /api/health/upload` : Télécharger des données de santé
- `GET /api/health/{user_id}` : Récupérer les données d'un utilisateur
- `GET /api/health/analysis/{user_id}` : Obtenir l'analyse des données

#### Détection du Psoriasis
- `POST /api/psoriasis/analyze` : Analyser une image pour le psoriasis
- `GET /api/psoriasis/history/{user_id}` : Historique des analyses
- `POST /api/psoriasis/train` : Entraîner le modèle (médecins uniquement)

#### Concepts d'IA
- `GET /api/ai/concepts` : Liste des concepts d'IA
- `GET /api/ai/concepts/{concept_id}` : Détails d'un concept
- `GET /api/ai/concepts/{concept_id}/explanation` : Explication détaillée

### Exemple d'Utilisation

1. Analyser une image pour le psoriasis :
```bash
curl -X POST http://localhost:8000/api/psoriasis/analyze \
  -H "Authorization: Bearer votre_token" \
  -F "image=@chemin_vers_votre_image.jpg"
```

2. Entraîner le modèle (médecins) :
```bash
curl -X POST http://localhost:8000/api/psoriasis/train \
  -H "Authorization: Bearer votre_token_medecin"
```

3. Consulter les concepts d'IA :
```bash
curl http://localhost:8000/api/ai/concepts
```

## Structure du Projet

```
app/
├── main.py              # Point d'entrée de l'application
├── routers/            # Routes de l'API
│   ├── health.py       # Routes pour les données de santé
│   ├── psoriasis.py    # Routes pour la détection du psoriasis
│   └── ai_concepts.py  # Routes pour les concepts d'IA
├── ml/                 # Modèles d'IA
│   ├── psoriasis_model.py  # Modèle de détection du psoriasis
│   └── ai_concepts.py      # Documentation des concepts d'IA
└── models/             # Modèles de données
    └── health.py       # Modèles pour les données de santé
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails. 