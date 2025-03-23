import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

# Ajout du chemin racine au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from app.ml.psoriasis_model import PsoriasisModel
except ImportError as e:
    print(f"Erreur d'importation : {e}")
    print(f"Chemin actuel : {current_dir}")
    print(f"Dossier parent : {parent_dir}")
    print(f"Contenu du dossier parent : {os.listdir(parent_dir)}")
    sys.exit(1)

def get_user_input():
    """Récupère les inputs de l'utilisateur"""
    print("\n=== Interface de Diagnostic du Psoriasis ===")
    print("\nOptions disponibles :")
    print("1. Analyser une image")
    print("2. Quitter")
    
    while True:
        try:
            choix = int(input("\nVotre choix (1-2) : "))
            if choix in [1, 2]:
                break
            print("Veuillez entrer 1 ou 2")
        except ValueError:
            print("Veuillez entrer un nombre valide")
    
    if choix == 2:
        return None
    
    print("\n=== Analyse d'Image ===")
    print("\nVeuillez fournir les informations suivantes :")
    
    # Chemin de l'image
    while True:
        image_path = input("\nChemin de l'image (ou 'q' pour quitter) : ")
        if image_path.lower() == 'q':
            return None
        if os.path.exists(image_path):
            break
        print("Ce fichier n'existe pas. Veuillez réessayer.")
    
    # Informations supplémentaires
    print("\nInformations supplémentaires (optionnel) :")
    age = input("Âge du patient (appuyez sur Entrée pour ignorer) : ")
    sexe = input("Sexe du patient (M/F, appuyez sur Entrée pour ignorer) : ")
    historique = input("Historique familial de psoriasis (Oui/Non, appuyez sur Entrée pour ignorer) : ")
    
    return {
        'image_path': image_path,
        'age': age if age else None,
        'sexe': sexe if sexe else None,
        'historique': historique if historique else None
    }

def test_model():
    # Initialisation du modèle
    model = PsoriasisModel()
    
    while True:
        # Récupération des inputs utilisateur
        user_input = get_user_input()
        if user_input is None:
            print("\nAu revoir !")
            break
            
        # Chargement de l'image
        image = cv2.imread(user_input['image_path'])
        if image is None:
            print(f"Erreur : Impossible de charger l'image {user_input['image_path']}")
            continue
        
        # Création de timestamps simulés pour le test temporel
        timestamps = np.array([
            datetime.now() + timedelta(days=i) 
            for i in range(5)
        ])
        
        print(f"\nAnalyse de l'image : {os.path.basename(user_input['image_path'])}")
        
        # Test de prédiction avec explication
        try:
            result = model.predict_with_explanation(image)
            print("\nRésultats de la prédiction :")
            print(f"Présence de psoriasis : {result['main']['prediction']}")
            print(f"Confiance : {result['main']['confidence']:.2f}")
            
            # Affichage de l'explication
            print("\nExplication du raisonnement :")
            for step in result['explanation']['decision_path']['decision_steps']:
                print(f"\nÉtape : {step['step']}")
                print(f"Décision : {step['decision']}")
                print(f"Niveau de confiance : {result['explanation']['confidence_analysis'][step['step']]['confidence_level']}")
                print("\nRaisonnement :")
                print(step['reasoning'])
            
            if result['main']['prediction'] == 'psoriasis':
                print(f"\nType : {result['type']['prediction']}")
                print(f"Sévérité : {result['severity']['prediction']}")
                
                # Analyse des probabilités alternatives
                print("\nProbabilités alternatives :")
                for task in ['type', 'severity']:
                    print(f"\n{task.capitalize()} :")
                    alt_probs = result['explanation']['confidence_analysis'][task]['alternative_probabilities']
                    for cls, prob in alt_probs.items():
                        print(f"- {cls}: {prob:.2f}")
                
                # Affichage des probabilités détaillées pour les médecins
                print("\nProbabilités détaillées pour les médecins :")
                print("\nTypes de psoriasis :")
                for type_name, prob in result['probabilities']['type'].items():
                    print(f"- {type_name}: {prob:.2%}")
                
                print("\nNiveaux de sévérité :")
                for severity_name, prob in result['probabilities']['severity'].items():
                    print(f"- {severity_name}: {prob:.2%}")
                
                # Analyse des caractéristiques cliniques
                print("\nCaractéristiques cliniques observées :")
                for feature in result['explanation']['decision_path']['decision_steps'][1]['key_features']:
                    print(f"- {feature['feature']}: {feature['contribution']}")
                
                # Affichage des informations patient si disponibles
                if any([user_input['age'], user_input['sexe'], user_input['historique']]):
                    print("\nInformations patient :")
                    if user_input['age']:
                        print(f"Âge : {user_input['age']} ans")
                    if user_input['sexe']:
                        print(f"Sexe : {user_input['sexe']}")
                    if user_input['historique']:
                        print(f"Historique familial : {user_input['historique']}")
            
        except Exception as e:
            print(f"Erreur lors de la prédiction : {str(e)}")
        
        # Test de l'analyse avancée
        try:
            advanced_result = model.predict_with_advanced_analysis(image, timestamps)
            
            # Affichage des résultats de l'analyse avancée
            print("\nAnalyse avancée :")
            
            # Analyse temporelle
            if advanced_result['advanced_analysis']['time_analysis']:
                print("\nStatistiques temporelles :")
                time_stats = advanced_result['advanced_analysis']['time_analysis']['time_statistics']
                print(f"Moyenne : {np.mean(time_stats['mean']):.2f}")
                print(f"Écart-type : {np.mean(time_stats['std']):.2f}")
            
            # Analyse de Fourier
            fourier = advanced_result['advanced_analysis']['fourier_transform']
            print("\nAnalyse fréquentielle :")
            print(f"Énergie basses fréquences : {fourier['frequency_bands']['low_frequency_energy']:.2f}")
            print(f"Énergie moyennes fréquences : {fourier['frequency_bands']['medium_frequency_energy']:.2f}")
            print(f"Énergie hautes fréquences : {fourier['frequency_bands']['high_frequency_energy']:.2f}")
            
            # Visualisation des résultats
            visualize_results(image, advanced_result)
            
        except Exception as e:
            print(f"Erreur lors de l'analyse avancée : {str(e)}")
        
        # Demander si l'utilisateur veut continuer
        while True:
            continuer = input("\nVoulez-vous analyser une autre image ? (Oui/Non) : ").lower()
            if continuer in ['oui', 'non']:
                break
            print("Veuillez répondre par 'Oui' ou 'Non'")
        
        if continuer == 'non':
            print("\nAu revoir !")
            break

def visualize_results(image, results):
    """Visualise les résultats de l'analyse"""
    plt.figure(figsize=(15, 10))
    
    # Image originale
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Image Originale")
    
    # Spectre de Fourier
    fourier = results['advanced_analysis']['fourier_transform']
    plt.subplot(2, 2, 2)
    plt.imshow(np.log(np.abs(fourier['magnitude_spectrum'])), cmap='gray')
    plt.title("Spectre de Fourier")
    
    # Attention weights
    attention = results['advanced_analysis']['attention_mechanism']
    plt.subplot(2, 2, 3)
    plt.imshow(attention['attention_weights'], cmap='hot')
    plt.title("Carte d'Attention")
    
    # Probabilités bayésiennes
    bayesian = results['advanced_analysis']['bayesian_probabilities']
    plt.subplot(2, 2, 4)
    plt.bar(range(len(bayesian['main']['prior_probabilities'])), 
            bayesian['main']['prior_probabilities'])
    plt.title("Probabilités Bayésiennes")
    
    plt.tight_layout()
    plt.show()

def verify_model_setup():
    """Vérifie que tout est correctement configuré"""
    print("\n=== Vérification de la Configuration ===")
    
    # Vérification des dépendances
    try:
        import tensorflow as tf
        print("✅ TensorFlow installé")
    except ImportError:
        print("❌ TensorFlow non installé. Installez-le avec : pip install tensorflow")
        return False
    
    try:
        import cv2
        print("✅ OpenCV installé")
    except ImportError:
        print("❌ OpenCV non installé. Installez-le avec : pip install opencv-python")
        return False
    
    # Vérification du dossier test_images
    if not os.path.exists("test_images"):
        print("❌ Dossier test_images non trouvé. Création...")
        os.makedirs("test_images")
    print("✅ Dossier test_images présent")
    
    # Vérification des images de test
    test_images = [f for f in os.listdir("test_images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not test_images:
        print("❌ Aucune image de test trouvée dans le dossier test_images")
        print("Veuillez ajouter des images de test dans ce dossier")
        return False
    print(f"✅ {len(test_images)} images de test trouvées")
    
    # Vérification des poids du modèle
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "app", "ml", "weights", "model_weights.h5")
    if not os.path.exists(weights_path):
        print("⚠️ Poids du modèle non trouvés")
        print("Le modèle tentera de les télécharger automatiquement lors de l'initialisation")
    else:
        print("✅ Poids du modèle présents")
    
    return True

if __name__ == "__main__":
    if verify_model_setup():
        print("\nConfiguration correcte. Démarrage du programme...")
        test_model()
    else:
        print("\nVeuillez corriger les problèmes ci-dessus avant de continuer.") 