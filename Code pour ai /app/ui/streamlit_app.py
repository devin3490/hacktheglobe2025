import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Ajout du chemin racine au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from app.ml.psoriasis_model import PsoriasisModel

def main():
    st.set_page_config(
        page_title="Détecteur de Psoriasis",
        page_icon="🔍",
        layout="wide"
    )
    
    # En-tête
    st.title("🔍 Détecteur de Psoriasis")
    st.markdown("""
    ### Analyse Intelligente des Images de Psoriasis
    Cette application utilise l'intelligence artificielle pour analyser les images de peau et détecter le psoriasis.
    """)
    
    # Initialisation du modèle
    @st.cache_resource
    def load_model():
        return PsoriasisModel()
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {str(e)}")
        return
    
    # Création de deux colonnes
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Télécharger une Image")
        uploaded_file = st.file_uploader("Choisissez une image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Lecture de l'image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Affichage de l'image
            st.image(image, channels="BGR", caption="Image téléchargée")
            
            # Informations patient
            with st.expander("Informations Patient (Optionnel)"):
                age = st.number_input("Âge", min_value=0, max_value=120, value=30)
                sex = st.selectbox("Sexe", ["Non spécifié", "Homme", "Femme", "Autre"])
                family_history = st.checkbox("Antécédents familiaux de psoriasis")
            
            # Bouton d'analyse
            if st.button("Analyser l'Image", type="primary"):
                with st.spinner("Analyse en cours..."):
                    try:
                        # Prédiction
                        prediction, explanation = model.predict_with_explanation(image)
                        
                        # Affichage des résultats
                        st.success("Analyse terminée !")
                        
                        # Création de trois colonnes pour les résultats
                        res_col1, res_col2, res_col3 = st.columns(3)
                        
                        with res_col1:
                            st.metric("Présence", prediction['main_prediction'])
                            st.metric("Confiance", f"{prediction['confidence']:.1f}%")
                        
                        with res_col2:
                            st.metric("Type", prediction['type_prediction'])
                            st.metric("Confiance Type", f"{prediction['type_confidence']:.1f}%")
                        
                        with res_col3:
                            st.metric("Sévérité", prediction['severity_prediction'])
                            st.metric("Confiance Sévérité", f"{prediction['severity_confidence']:.1f}%")
                        
                        # Affichage des explications
                        st.subheader("Explications")
                        st.markdown(explanation['main_decision'])
                        st.markdown(explanation['type_decision'])
                        st.markdown(explanation['severity_decision'])
                        
                        # Visualisations
                        st.subheader("Visualisations")
                        
                        # Création de deux colonnes pour les visualisations
                        vis_col1, vis_col2 = st.columns(2)
                        
                        with vis_col1:
                            # Graphique des probabilités
                            fig, ax = plt.subplots(figsize=(8, 4))
                            types = list(prediction['type_probabilities'].keys())
                            probs = list(prediction['type_probabilities'].values())
                            ax.bar(types, probs)
                            ax.set_title("Probabilités par Type")
                            ax.set_xticklabels(types, rotation=45)
                            st.pyplot(fig)
                        
                        with vis_col2:
                            # Graphique de sévérité
                            fig, ax = plt.subplots(figsize=(8, 4))
                            severities = list(prediction['severity_probabilities'].keys())
                            probs = list(prediction['severity_probabilities'].values())
                            ax.bar(severities, probs)
                            ax.set_title("Probabilités par Sévérité")
                            ax.set_xticklabels(severities, rotation=45)
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Erreur lors de l'analyse : {str(e)}")
    
    with col2:
        st.subheader("ℹ️ À propos")
        st.markdown("""
        Cette application utilise un modèle de deep learning pour :
        
        - Détecter la présence de psoriasis
        - Identifier le type de psoriasis
        - Évaluer la sévérité
        - Fournir des explications détaillées
        
        ### Comment utiliser l'application :
        
        1. Téléchargez une image de la zone affectée
        2. Remplissez les informations patient (optionnel)
        3. Cliquez sur "Analyser l'Image"
        4. Consultez les résultats et visualisations
        
        ### Précautions :
        
        - Assurez-vous que l'image est bien éclairée
        - La zone affectée doit être clairement visible
        - L'image doit être de bonne qualité
        """)
        
        # Ajout d'un footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Développé avec ❤️ pour la santé</p>
            <p>Version 1.0.0</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 