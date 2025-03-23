import os
import requests
import zipfile
import shutil
from pathlib import Path
import pandas as pd

def download_dataset():
    """Télécharge le dataset PAD-UFES-20 depuis Mendeley"""
    # URL du dataset
    url = "https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/d8a7f6d9-7db3-4c3b-9f55-75fad4c12702/file_downloaded"
    
    # Créer le dossier data s'il n'existe pas
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Télécharger le fichier
    print("Téléchargement du dataset...")
    response = requests.get(url, stream=True)
    zip_path = data_dir / "pad_ufes_20.zip"
    
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    # Extraire le fichier zip
    print("Extraction du dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Supprimer le fichier zip
    zip_path.unlink()
    
    # Organiser les données pour l'entraînement
    organize_data()

def organize_data():
    """Organise les données pour l'entraînement"""
    data_dir = Path("data")
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    
    # Créer les dossiers d'entraînement et de validation
    train_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)
    
    # Créer les sous-dossiers pour chaque classe
    for class_name in ["psoriasis", "normal"]:
        (train_dir / class_name).mkdir(exist_ok=True)
        (validation_dir / class_name).mkdir(exist_ok=True)
    
    # Lire le fichier CSV des métadonnées
    metadata = pd.read_csv(data_dir / "metadata.csv")
    
    # Répartir les images entre train et validation (80/20)
    for _, row in metadata.iterrows():
        image_path = data_dir / "images" / row["img_id"]
        if not image_path.exists():
            continue
        
        # Déterminer si l'image va dans train ou validation
        if row["split"] == "train":
            dest_dir = train_dir
        else:
            dest_dir = validation_dir
        
        # Déterminer la classe
        if row["diagnostic"] == "PSORIASIS":
            class_name = "psoriasis"
        else:
            class_name = "normal"
        
        # Copier l'image
        shutil.copy2(image_path, dest_dir / class_name / image_path.name)
    
    # Supprimer le dossier images original
    shutil.rmtree(data_dir / "images")

if __name__ == "__main__":
    download_dataset() 