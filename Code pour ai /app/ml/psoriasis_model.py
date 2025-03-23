import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
import openai
from dotenv import load_dotenv
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from sklearn.naive_bayes import GaussianNB

load_dotenv()

class PsoriasisModel:
    """Classe pour la détection du psoriasis avec des réseaux de neurones avancés"""
    
    def __init__(self):
        self.model = self.build_model()
        
        # Chargement des poids
        if not self._load_model_weights():
            print("⚠️ Attention : Le modèle n'a pas de poids pré-entraînés.")
            print("Les prédictions seront basées sur un modèle non entraîné.")
            print("Pour de meilleurs résultats, veuillez télécharger les poids pré-entraînés.")
        
        self.input_shape = (224, 224, 3)
        self.num_classes = 2  # psoriasis et normal
        self.encoder = None
        self.cluster_model = None
        self.pca = None
        self.scaler = StandardScaler()
        self.classifiers = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(kernel='rbf', probability=True)
        }
        self.hierarchical_classes = {
            "psoriasis": {
                "types": ["plaque", "guttate", "inverse", "pustular", "erythrodermic"],
                "severity": ["mild", "moderate", "severe"]
            }
        }
        self.load_model()
    
    def _custom_activation(self, x: tf.Tensor, activation_type: str = "relu") -> tf.Tensor:
        """Implémente des fonctions d'activation personnalisées"""
        if activation_type == "sigmoid":
            # Sigmoid standard
            return tf.sigmoid(x)
        elif activation_type == "relu":
            # ReLU standard
            return tf.nn.relu(x)
        elif activation_type == "softmax":
            # Softmax standard
            return tf.nn.softmax(x)
        elif activation_type == "leaky_relu":
            # Leaky ReLU avec pente négative de 0.01
            return tf.nn.leaky_relu(x, alpha=0.01)
        elif activation_type == "elu":
            # ELU (Exponential Linear Unit)
            return tf.nn.elu(x)
        elif activation_type == "selu":
            # SELU (Scaled Exponential Linear Unit)
            return tf.nn.selu(x)
        elif activation_type == "swish":
            # Swish (x * sigmoid(x))
            return x * tf.sigmoid(x)
        elif activation_type == "mish":
            # Mish (x * tanh(ln(1 + e^x)))
            return x * tf.tanh(tf.math.log(1 + tf.exp(x)))
        else:
            raise ValueError(f"Type d'activation non supporté : {activation_type}")

    def build_model(self, architecture: str = "efficientnet") -> models.Model:
        """Construit un modèle CNN avec différentes architectures disponibles"""
        
        if architecture == "efficientnet":
            # Utilisation d'EfficientNetV2 pré-entraîné
            base_model = applications.EfficientNetV2B0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            base_model.trainable = True  # Fine-tuning activé
            
            # Encoder pour l'apprentissage non supervisé avec fonctions d'activation personnalisées
            self.encoder = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512),
                layers.Lambda(lambda x: self._custom_activation(x, "mish")),
                layers.BatchNormalization(),
                layers.Dense(256),
                layers.Lambda(lambda x: self._custom_activation(x, "swish")),
                layers.BatchNormalization(),
                layers.Dense(128),
                layers.Lambda(lambda x: self._custom_activation(x, "elu")),
                layers.BatchNormalization()
            ])
            
            # Têtes de classification multiples avec différentes fonctions d'activation
            type_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["types"]))
            type_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='type')(type_head)
            
            severity_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["severity"]))
            severity_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='severity')(severity_head)
            
            main_head = layers.Dense(self.num_classes)
            main_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='main')(main_head)
            
            # Modèle complet avec classification hiérarchique
            model = models.Model(
                inputs=self.encoder.input,
                outputs=[
                    main_head(self.encoder.output),
                    type_head(self.encoder.output),
                    severity_head(self.encoder.output)
                ]
            )
            
        elif architecture == "resnet":
            # Utilisation de ResNet50 pré-entraîné
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            base_model.trainable = True
            
            # Encoder pour l'apprentissage non supervisé avec fonctions d'activation personnalisées
            self.encoder = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(512),
                layers.Lambda(lambda x: self._custom_activation(x, "swish")),
                layers.BatchNormalization(),
                layers.Dense(256),
                layers.Lambda(lambda x: self._custom_activation(x, "mish")),
                layers.BatchNormalization(),
                layers.Dense(128),
                layers.Lambda(lambda x: self._custom_activation(x, "selu")),
                layers.BatchNormalization()
            ])
            
            # Têtes de classification multiples avec différentes fonctions d'activation
            type_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["types"]))
            type_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='type')(type_head)
            
            severity_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["severity"]))
            severity_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='severity')(severity_head)
            
            main_head = layers.Dense(self.num_classes)
            main_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='main')(main_head)
            
            # Modèle complet avec classification hiérarchique
            model = models.Model(
                inputs=self.encoder.input,
                outputs=[
                    main_head(self.encoder.output),
                    type_head(self.encoder.output),
                    severity_head(self.encoder.output)
                ]
            )
            
        elif architecture == "custom":
            # Architecture personnalisée avec attention et fonctions d'activation personnalisées
            self.encoder = models.Sequential([
                # Bloc de convolution 1
                layers.Conv2D(64, (3, 3), input_shape=self.input_shape),
                layers.Lambda(lambda x: self._custom_activation(x, "mish")),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Bloc de convolution 2 avec attention
                layers.Conv2D(128, (3, 3)),
                layers.Lambda(lambda x: self._custom_activation(x, "swish")),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Attention(),
                
                # Bloc de convolution 3
                layers.Conv2D(256, (3, 3)),
                layers.Lambda(lambda x: self._custom_activation(x, "selu")),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Bloc de convolution 4 avec attention
                layers.Conv2D(512, (3, 3)),
                layers.Lambda(lambda x: self._custom_activation(x, "elu")),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Attention(),
                
                # Couches denses pour l'encodage
                layers.Flatten(),
                layers.Dense(512),
                layers.Lambda(lambda x: self._custom_activation(x, "mish")),
                layers.BatchNormalization(),
                layers.Dense(256),
                layers.Lambda(lambda x: self._custom_activation(x, "swish")),
                layers.BatchNormalization(),
                layers.Dense(128),
                layers.Lambda(lambda x: self._custom_activation(x, "selu")),
                layers.BatchNormalization()
            ])
            
            # Têtes de classification multiples avec différentes fonctions d'activation
            type_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["types"]))
            type_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='type')(type_head)
            
            severity_head = layers.Dense(len(self.hierarchical_classes["psoriasis"]["severity"]))
            severity_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='severity')(severity_head)
            
            main_head = layers.Dense(self.num_classes)
            main_head = layers.Lambda(lambda x: self._custom_activation(x, "softmax"), name='main')(main_head)
            
            # Modèle complet avec classification hiérarchique
            model = models.Model(
                inputs=self.encoder.input,
                outputs=[
                    main_head(self.encoder.output),
                    type_head(self.encoder.output),
                    severity_head(self.encoder.output)
                ]
            )
        
        # Compilation du modèle avec des optimiseurs et métriques avancés
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss={
                'main': 'categorical_crossentropy',
                'type': 'categorical_crossentropy',
                'severity': 'categorical_crossentropy'
            },
            loss_weights={
                'main': 1.0,
                'type': 0.5,
                'severity': 0.5
            },
            metrics={
                'main': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ],
                'type': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ],
                'severity': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ]
            }
        )
        
        return model
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Prétraite l'image avec normalisation et augmentation de données"""
        # Redimensionnement
        image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))
        
        # Normalisation adaptative
        image = self._adaptive_normalization(image)
        
        # Standardisation des couleurs
        image = self._color_standardization(image)
        
        # Augmentation de données en temps réel
        if tf.random.uniform(()) > 0.5:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.8, 1.2)
        
        return image
    
    def _adaptive_normalization(self, image: np.ndarray) -> np.ndarray:
        """Normalisation adaptative basée sur les statistiques locales"""
        # Conversion en float32
        image = image.astype(np.float32)
        
        # Normalisation par patch
        patch_size = 32
        h, w = image.shape[:2]
        
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = image[i:min(i+patch_size, h), j:min(j+patch_size, w)]
                mean = np.mean(patch)
                std = np.std(patch)
                
                if std > 0:
                    patch = (patch - mean) / std
                else:
                    patch = patch - mean
                
                image[i:min(i+patch_size, h), j:min(j+patch_size, w)] = patch
        
        return image
    
    def _color_standardization(self, image: np.ndarray) -> np.ndarray:
        """Standardisation des couleurs dans différents espaces colorimétriques"""
        # Standardisation RGB
        rgb_mean = np.mean(image, axis=(0, 1))
        rgb_std = np.std(image, axis=(0, 1))
        image = (image - rgb_mean) / (rgb_std + 1e-7)
        
        # Conversion et standardisation HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_mean = np.mean(hsv, axis=(0, 1))
        hsv_std = np.std(hsv, axis=(0, 1))
        hsv = (hsv - hsv_mean) / (hsv_std + 1e-7)
        
        # Conversion et standardisation LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_mean = np.mean(lab, axis=(0, 1))
        lab_std = np.std(lab, axis=(0, 1))
        lab = (lab - lab_mean) / (lab_std + 1e-7)
        
        # Fusion des espaces colorimétriques
        image = cv2.addWeighted(image, 0.4, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0.3, 0)
        image = cv2.addWeighted(image, 0.7, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0.3, 0)
        
        return image
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Applique l'augmentation de données en temps réel"""
        # Augmentation aléatoire
        if np.random.random() < 0.5:
            # Rotation
            angle = np.random.uniform(-15, 15)
            image = self._rotate_image(image, angle)
            
            # Translation
            tx = np.random.uniform(-10, 10)
            ty = np.random.uniform(-10, 10)
            image = self._translate_image(image, tx, ty)
            
            # Zoom
            scale = np.random.uniform(0.9, 1.1)
            image = self._zoom_image(image, scale)
            
            # Bruit
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
            
            # Changement de contraste
            contrast = np.random.uniform(0.8, 1.2)
            image = self._adjust_contrast(image, contrast)
            
            # Changement de luminosité
            brightness = np.random.uniform(-0.1, 0.1)
            image = self._adjust_brightness(image, brightness)
        
        return image
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotation de l'image"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (width, height))
    
    def _translate_image(self, image: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Translation de l'image"""
        height, width = image.shape[:2]
        matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(image, matrix, (width, height))
    
    def _zoom_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Zoom de l'image"""
        height, width = image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    
    def _adjust_contrast(self, image: np.ndarray, contrast: float) -> np.ndarray:
        """Ajustement du contraste"""
        return np.clip((image - np.mean(image)) * contrast + np.mean(image), 0, 1)
    
    def _adjust_brightness(self, image: np.ndarray, brightness: float) -> np.ndarray:
        """Ajustement de la luminosité"""
        return np.clip(image + brightness, 0, 1)
    
    def load_model(self):
        """Charge le modèle pré-entraîné ou en crée un nouveau"""
        model_path = Path("app/ml/models/psoriasis_model.h5")
        if model_path.exists():
            self.model = models.load_model(str(model_path))
        else:
            self.model = self.build_model(architecture="efficientnet")
    
    def _compute_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule la cross-entropy avec différentes variantes et analyses"""
        # Cross-entropy binaire
        binary_ce = -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))
        
        # Cross-entropy catégorielle
        categorical_ce = -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))
        
        # Cross-entropy pondérée par classe
        class_weights = np.sum(y_true, axis=0) / len(y_true)
        weighted_ce = -np.mean(np.sum(y_true * np.log(y_pred + 1e-7) * class_weights, axis=1))
        
        # Analyse des probabilités par classe
        class_probs = {
            f"classe_{i}": {
                "moyenne": float(np.mean(y_pred[:, i])),
                "écart_type": float(np.std(y_pred[:, i])),
                "min": float(np.min(y_pred[:, i])),
                "max": float(np.max(y_pred[:, i])),
                "entropie": float(-np.mean(np.log(y_pred[:, i] + 1e-7)))
            }
            for i in range(y_pred.shape[1])
        }
        
        return {
            "binary_cross_entropy": float(binary_ce),
            "categorical_cross_entropy": float(categorical_ce),
            "weighted_cross_entropy": float(weighted_ce),
            "class_probabilities": class_probs
        }

    def _analyze_probability_distribution(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyse la distribution des probabilités avec logarithmes"""
        # Calcul de l'entropie de Shannon
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-7))
        
        # Calcul de l'entropie croisée relative
        uniform_dist = np.ones_like(probabilities) / len(probabilities)
        relative_entropy = np.sum(probabilities * np.log(probabilities / (uniform_dist + 1e-7)))
        
        # Analyse des logarithmes des probabilités
        log_probs = np.log(probabilities + 1e-7)
        log_analysis = {
            "moyenne_log": float(np.mean(log_probs)),
            "écart_type_log": float(np.std(log_probs)),
            "min_log": float(np.min(log_probs)),
            "max_log": float(np.max(log_probs))
        }
        
        # Calcul de la divergence de Kullback-Leibler
        kl_divergence = np.sum(probabilities * (log_probs - np.log(uniform_dist + 1e-7)))
        
        return {
            "entropie_shannon": float(entropy),
            "entropie_croisée_relative": float(relative_entropy),
            "analyse_logarithmique": log_analysis,
            "divergence_kl": float(kl_divergence)
        }

    def _compute_probability_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcule des métriques avancées basées sur les probabilités"""
        # Calcul de la cross-entropy
        ce_metrics = self._compute_cross_entropy(y_true, y_pred)
        
        # Analyse de la distribution des probabilités
        prob_dist = self._analyze_probability_distribution(y_pred)
        
        # Calcul des métriques de calibration
        calibration_metrics = self._compute_calibration_metrics(y_true, y_pred)
        
        return {
            "cross_entropy_metrics": ce_metrics,
            "probability_distribution": prob_dist,
            "calibration_metrics": calibration_metrics
        }

    def _compute_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcule des métriques de calibration des probabilités"""
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        calibration_metrics = {}
        for i in range(y_pred.shape[1]):
            # Calcul des bins de probabilité
            bin_indices = np.digitize(y_pred[:, i], bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Calcul des probabilités moyennes par bin
            bin_probs = []
            bin_accuracies = []
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                if np.any(mask):
                    bin_probs.append(float(np.mean(y_pred[mask, i])))
                    bin_accuracies.append(float(np.mean(y_true[mask, i])))
            
            calibration_metrics[f"classe_{i}"] = {
                "bin_probabilities": bin_probs,
                "bin_accuracies": bin_accuracies,
                "calibration_error": float(np.mean(np.abs(np.array(bin_probs) - np.array(bin_accuracies))))
            }
        
        return calibration_metrics

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Fait une prédiction avec analyse détaillée et classification hiérarchique"""
        # Prétraitement
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Prédiction
        predictions = self.model.predict(processed_image, verbose=0)
        main_pred = predictions[0]
        type_pred = predictions[1]
        severity_pred = predictions[2]
        
        # Analyse des probabilités
        prob_metrics = self._compute_probability_metrics(
            np.array([[1, 0] if np.argmax(main_pred[0]) == 0 else [0, 1]]),
            main_pred
        )
        
        # Analyse des caractéristiques visuelles
        features = self.analyze_features(image)
        
        # Génération de l'analyse détaillée
        analysis = self.generate_analysis(features, main_pred[0], type_pred[0], severity_pred[0])
        
        return {
            "prediction": {
                "main": "psoriasis" if np.argmax(main_pred[0]) == 1 else "normal",
                "type": self.hierarchical_classes["psoriasis"]["types"][np.argmax(type_pred[0])] if np.argmax(main_pred[0]) == 1 else None,
                "severity": self.hierarchical_classes["psoriasis"]["severity"][np.argmax(severity_pred[0])] if np.argmax(main_pred[0]) == 1 else None
            },
            "confidence": {
                "main": float(main_pred[0][np.argmax(main_pred[0])]),
                "type": float(type_pred[0][np.argmax(type_pred[0])]) if np.argmax(main_pred[0]) == 1 else None,
                "severity": float(severity_pred[0][np.argmax(severity_pred[0])]) if np.argmax(main_pred[0]) == 1 else None
            },
            "probabilities": {
                "main": {
                    "normal": float(main_pred[0][0]),
                    "psoriasis": float(main_pred[0][1])
                },
                "type": {
                    type_name: float(prob) 
                    for type_name, prob in zip(self.hierarchical_classes["psoriasis"]["types"], type_pred[0])
                } if np.argmax(main_pred[0]) == 1 else None,
                "severity": {
                    severity_name: float(prob)
                    for severity_name, prob in zip(self.hierarchical_classes["psoriasis"]["severity"], severity_pred[0])
                } if np.argmax(main_pred[0]) == 1 else None
            },
            "probability_metrics": prob_metrics,
            "features": features,
            "analysis": analysis
        }
    
    def analyze_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse détaillée des caractéristiques visuelles de l'image avec algèbre linéaire"""
        # Extraction des features de base
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        features = self.encoder.predict(processed_image)[0]
        
        # Transformations linéaires
        transformations = self._compute_feature_transformations(features.reshape(1, -1))
        
        # Opérations matricielles
        matrix_ops = self._matrix_operations(features.reshape(1, -1))
        
        # Opérations vectorielles
        vector_ops = self._vector_operations(features.reshape(1, -1))
        
        # Analyse de la texture avec algèbre linéaire
        texture_features = self._analyze_texture(image)
        
        # Analyse des couleurs avec algèbre linéaire
        color_features = self._analyze_color(image)
        
        # Analyse des bords avec algèbre linéaire
        edge_features = self._analyze_edges(image)
        
        # Analyse de la symétrie avec algèbre linéaire
        symmetry_features = self._analyze_symmetry(image)
        
        return {
            "texture": texture_features,
            "color": color_features,
            "edges": edge_features,
            "symmetry": symmetry_features,
            "linear_algebra": {
                "transformations": {
                    "normalized": transformations["normalized"].tolist(),
                    "projected": transformations["projected"].tolist(),
                    "svd": {
                        "singular_values": transformations["svd"]["S"].tolist(),
                        "components": transformations["svd"]["Vh"].tolist()
                    },
                    "eigen": {
                        "values": transformations["eigen"]["values"].tolist(),
                        "vectors": transformations["eigen"]["vectors"].tolist()
                    }
                },
                "matrix_operations": {
                    "gram_matrix": matrix_ops["gram_matrix"].tolist(),
                    "determinant": float(matrix_ops["determinant"]),
                    "eigenvalues": matrix_ops["eigenvalues"].tolist()
                },
                "vector_operations": {
                    "dot_products": vector_ops["dot_products"].tolist(),
                    "angles": vector_ops["angles"].tolist(),
                    "projections": vector_ops["projections"].tolist()
                }
            }
        }
    
    def _compute_feature_transformations(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcule diverses transformations linéaires des features"""
        # Normalisation L2 des features
        l2_norm = np.linalg.norm(features, axis=1, keepdims=True)
        features_normalized = features / (l2_norm + 1e-7)
        
        # Projection orthogonale
        projection_matrix = np.eye(features.shape[1]) - np.outer(features_normalized[0], features_normalized[0])
        features_projected = np.dot(features, projection_matrix)
        
        # Décomposition en valeurs singulières (SVD)
        U, S, Vh = np.linalg.svd(features)
        
        # Calcul des composantes principales avec PCA
        cov_matrix = np.cov(features.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        return {
            "normalized": features_normalized,
            "projected": features_projected,
            "svd": {
                "U": U,
                "S": S,
                "Vh": Vh
            },
            "eigen": {
                "values": eigenvalues,
                "vectors": eigenvectors
            }
        }

    def _matrix_operations(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue des opérations matricielles avancées"""
        # Produit matriciel avec transposition
        gram_matrix = np.dot(features, features.T)
        
        # Calcul du déterminant et de l'inverse
        try:
            det = np.linalg.det(gram_matrix)
            inv = np.linalg.inv(gram_matrix)
        except np.linalg.LinAlgError:
            det = 0
            inv = np.zeros_like(gram_matrix)
        
        # Décomposition QR
        Q, R = np.linalg.qr(features)
        
        # Calcul des valeurs propres
        eigenvalues, eigenvectors = np.linalg.eig(gram_matrix)
        
        return {
            "gram_matrix": gram_matrix,
            "determinant": det,
            "inverse": inv,
            "qr_decomposition": {
                "Q": Q,
                "R": R
            },
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors
        }

    def _vector_operations(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Effectue des opérations vectorielles avancées"""
        # Produit scalaire entre tous les vecteurs
        dot_products = np.dot(features, features.T)
        
        # Calcul des angles entre les vecteurs
        norms = np.linalg.norm(features, axis=1)
        angles = np.arccos(np.clip(dot_products / np.outer(norms, norms), -1.0, 1.0))
        
        # Produit vectoriel (si applicable)
        if features.shape[1] == 3:
            cross_products = np.cross(features[:, None, :], features[None, :, :])
        else:
            cross_products = None
        
        # Projection vectorielle
        projections = np.outer(features[0], features[0]) / np.dot(features[0], features[0])
        
        return {
            "dot_products": dot_products,
            "angles": angles,
            "cross_products": cross_products,
            "projections": projections
        }

    def _analyze_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analyse de la texture avec des transformations linéaires"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Création de la matrice de texture
        texture_matrix = np.zeros((gray.shape[0], gray.shape[1]))
        
        # Filtres Gabor avec différentes orientations
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((21, 21), 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_matrix += filtered
        
        # Analyse des valeurs propres de la matrice de texture
        eigenvalues, eigenvectors = np.linalg.eigh(texture_matrix)
        
        return {
            "texture_matrix_eigenvalues": eigenvalues.tolist(),
            "texture_matrix_eigenvectors": eigenvectors.tolist(),
            "texture_energy": float(np.sum(eigenvalues)),
            "texture_complexity": float(np.std(eigenvalues))
        }

    def _analyze_color(self, image: np.ndarray) -> Dict[str, float]:
        """Analyse des caractéristiques de couleur avec transformations linéaires"""
        # Conversion en différents espaces de couleur
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Création des matrices de couleur
        hsv_matrix = hsv.reshape(-1, 3)
        lab_matrix = lab.reshape(-1, 3)
        
        # Analyse des composantes principales
        hsv_pca = PCA(n_components=3).fit(hsv_matrix)
        lab_pca = PCA(n_components=3).fit(lab_matrix)
        
        return {
            "hsv_pca": {
                "components": hsv_pca.components_.tolist(),
                "explained_variance": hsv_pca.explained_variance_.tolist()
            },
            "lab_pca": {
                "components": lab_pca.components_.tolist(),
                "explained_variance": lab_pca.explained_variance_.tolist()
            },
            "color_covariance": {
                "hsv": np.cov(hsv_matrix.T).tolist(),
                "lab": np.cov(lab_matrix.T).tolist()
            }
        }
    
    def _analyze_edges(self, image: np.ndarray) -> Dict[str, float]:
        """Analyse des bords avec différents détecteurs"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection de Canny
        edges_canny = cv2.Canny(gray, 100, 200)
        
        # Détection de Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        return {
            "canny_edge_density": float(np.mean(edges_canny > 0)),
            "sobel_magnitude": float(np.mean(np.sqrt(sobelx**2 + sobely**2)))
        }
    
    def _analyze_symmetry(self, image: np.ndarray) -> Dict[str, float]:
        """Analyse de la symétrie de l'image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Symétrie horizontale
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        horizontal_symmetry = float(np.mean(np.abs(left_half - right_half)))
        
        # Symétrie verticale
        top_half = gray[:height//2, :]
        bottom_half = cv2.flip(gray[height//2:, :], 0)
        vertical_symmetry = float(np.mean(np.abs(top_half - bottom_half)))
        
        return {
            "horizontal_symmetry": horizontal_symmetry,
            "vertical_symmetry": vertical_symmetry
        }
    
    def generate_analysis(self, features: Dict[str, Any], main_probs: np.ndarray, type_probs: np.ndarray, severity_probs: np.ndarray) -> str:
        """Génère une analyse détaillée basée sur les caractéristiques et les probabilités"""
        prompt = f"""
        En tant qu'expert en dermatologie, analysez en détail l'image de peau fournie en suivant un raisonnement étape par étape :

        1. ANALYSE DES CARACTÉRISTIQUES VISUELLES :
        
        Texture :
        - Analyse des orientations : {features['texture']}
        - Interprétation des patterns de texture observés
        - Comparaison avec les caractéristiques typiques du psoriasis
        
        Couleurs :
        - Distribution HSV : {features['color']['hsv_mean']:.2f} (moyenne), {features['color']['hsv_std']:.2f} (écart-type)
        - Distribution LAB : {features['color']['lab_mean']:.2f} (moyenne), {features['color']['lab_std']:.2f} (écart-type)
        - Analyse des variations de couleur et leur signification clinique
        
        Bords et Contours :
        - Densité des bords (Canny) : {features['edges']['canny_edge_density']:.2f}
        - Magnitude des gradients (Sobel) : {features['edges']['sobel_magnitude']:.2f}
        - Évaluation de la netteté et de la définition des lésions
        
        Symétrie :
        - Symétrie horizontale : {features['symmetry']['horizontal_symmetry']:.2f}
        - Symétrie verticale : {features['symmetry']['vertical_symmetry']:.2f}
        - Analyse de la distribution des lésions

        2. ÉVALUATION DES PROBABILITÉS DE DIAGNOSTIC :
        
        Diagnostic Principal :
        - Normal : {main_probs[0]:.2%}
        - Psoriasis : {main_probs[1]:.2%}
        - Interprétation de la confiance du diagnostic
        
        Si psoriasis détecté :
        
        Types de Psoriasis :
        {chr(10).join(f"- {type_name}: {prob:.2%}" for type_name, prob in zip(self.hierarchical_classes['psoriasis']['types'], type_probs))}
        - Analyse des caractéristiques spécifiques à chaque type
        - Justification du type le plus probable
        
        Niveaux de Sévérité :
        {chr(10).join(f"- {severity_name}: {prob:.2%}" for severity_name, prob in zip(self.hierarchical_classes['psoriasis']['severity'], severity_probs))}
        - Évaluation de l'étendue des lésions
        - Impact sur la qualité de vie
        - Facteurs aggravants identifiés

        3. SYNTHÈSE ET RECOMMANDATIONS :
        
        Conclusion :
        - Résumé des observations clés
        - Diagnostic final avec niveau de confiance
        - Facteurs de risque identifiés
        
        Recommandations :
        - Examens complémentaires nécessaires
        - Options de traitement suggérées
        - Suivi recommandé
        - Conseils de prévention et de gestion
        
        Fournissez une analyse détaillée en français, en suivant cette structure et en expliquant votre raisonnement à chaque étape.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Vous êtes un expert en dermatologie qui fournit des analyses détaillées et structurées des lésions cutanées. Votre analyse doit être claire, précise et basée sur des observations objectives."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération de l'analyse : {str(e)}"
    
    def train_unsupervised(self, unlabeled_data_dir: str, n_clusters: int = 3):
        """Entraîne le modèle en mode non supervisé pour découvrir des patterns"""
        # Chargement des données non étiquetées
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        unlabeled_generator = datagen.flow_from_directory(
            unlabeled_data_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode=None
        )
        
        # Extraction des features avec l'encoder
        features = []
        for batch in unlabeled_generator:
            encoded = self.encoder.predict(batch)
            features.extend(encoded)
        
        features = np.array(features)
        
        # Réduction de dimensionnalité avec PCA
        self.pca = PCA(n_components=50)
        features_pca = self.pca.fit_transform(features)
        
        # Normalisation des features
        features_scaled = self.scaler.fit_transform(features_pca)
        
        # Clustering avec KMeans
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.cluster_model.fit_predict(features_scaled)
        
        # Analyse des clusters
        cluster_analysis = self._analyze_clusters(features, clusters)
        
        return {
            "clusters": clusters,
            "cluster_analysis": cluster_analysis,
            "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist()
        }
    
    def _analyze_clusters(self, features: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
        """Analyse les caractéristiques des clusters découverts"""
        analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_features = features[clusters == cluster_id]
            
            # Statistiques pour chaque cluster
            analysis[f"cluster_{cluster_id}"] = {
                "size": int(np.sum(clusters == cluster_id)),
                "mean": float(np.mean(cluster_features)),
                "std": float(np.std(cluster_features)),
                "features_importance": self.pca.components_.tolist(),
                "similarity_matrix": self._compute_similarity_matrix(cluster_features).tolist()
            }
        
        return analysis

    def _compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calcule la matrice de similarité entre les images d'un cluster"""
        n_samples = features.shape[0]
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                # Similarité cosinus
                cosine_sim = self._cosine_similarity(features[i], features[j])
                # Distance euclidienne
                euclidean_dist = self._euclidean_distance(features[i], features[j])
                # Combinaison des métriques
                similarity_matrix[i, j] = similarity_matrix[j, i] = (cosine_sim + (1 - euclidean_dist)) / 2
        
        return similarity_matrix

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux vecteurs"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcule la distance euclidienne entre deux vecteurs"""
        return np.linalg.norm(vec1 - vec2)

    def find_similar_images(self, image: np.ndarray, n_similar: int = 5) -> List[Dict[str, Any]]:
        """Trouve les images les plus similaires dans la base de données"""
        # Extraction des features de l'image d'entrée
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        query_features = self.encoder.predict(processed_image)[0]
        
        # Réduction de dimensionnalité
        query_pca = self.pca.transform(query_features.reshape(1, -1))
        query_scaled = self.scaler.transform(query_pca)
        
        # Calcul des similarités avec toutes les images de la base
        similarities = []
        for i, features in enumerate(self.database_features):
            # Similarité cosinus
            cosine_sim = self._cosine_similarity(query_scaled[0], features)
            # Distance euclidienne
            euclidean_dist = self._euclidean_distance(query_scaled[0], features)
            # Score combiné
            combined_score = (cosine_sim + (1 - euclidean_dist)) / 2
            
            similarities.append({
                "index": i,
                "cosine_similarity": float(cosine_sim),
                "euclidean_distance": float(euclidean_dist),
                "combined_score": float(combined_score)
            })
        
        # Tri par score combiné et sélection des n_similar plus similaires
        similarities.sort(key=lambda x: x["combined_score"], reverse=True)
        return similarities[:n_similar]

    def cluster_analysis(self, features: np.ndarray, clusters: np.ndarray) -> Dict[str, Any]:
        """Analyse détaillée des clusters avec métriques de similarité"""
        analysis = {}
        
        for cluster_id in np.unique(clusters):
            cluster_features = features[clusters == cluster_id]
            
            # Calcul des métriques de similarité intra-cluster
            intra_cluster_similarities = []
            for i in range(len(cluster_features)):
                for j in range(i+1, len(cluster_features)):
                    similarity = self._cosine_similarity(cluster_features[i], cluster_features[j])
                    intra_cluster_similarities.append(similarity)
            
            # Calcul des métriques de similarité inter-clusters
            inter_cluster_similarities = []
            other_clusters = features[clusters != cluster_id]
            for i in range(len(cluster_features)):
                for j in range(len(other_clusters)):
                    similarity = self._cosine_similarity(cluster_features[i], other_clusters[j])
                    inter_cluster_similarities.append(similarity)
            
            analysis[f"cluster_{cluster_id}"] = {
                "size": int(np.sum(clusters == cluster_id)),
                "intra_cluster_stats": {
                    "mean_similarity": float(np.mean(intra_cluster_similarities)),
                    "std_similarity": float(np.std(intra_cluster_similarities)),
                    "min_similarity": float(np.min(intra_cluster_similarities)),
                    "max_similarity": float(np.max(intra_cluster_similarities))
                },
                "inter_cluster_stats": {
                    "mean_similarity": float(np.mean(inter_cluster_similarities)),
                    "std_similarity": float(np.std(inter_cluster_similarities)),
                    "min_similarity": float(np.min(inter_cluster_similarities)),
                    "max_similarity": float(np.max(inter_cluster_similarities))
                },
                "cohesion": float(np.mean(intra_cluster_similarities) - np.mean(inter_cluster_similarities))
            }
        
        return analysis

    def predict_with_similarity(self, image: np.ndarray) -> Dict[str, Any]:
        """Fait une prédiction avec analyse de similarité"""
        # Prédiction standard
        result = self.predict(image)
        
        # Ajout de l'analyse de similarité
        if hasattr(self, 'database_features'):
            similar_images = self.find_similar_images(image)
            result["similarity_analysis"] = {
                "similar_images": similar_images,
                "cluster_analysis": self.cluster_analysis(
                    self.encoder.predict(self.preprocess_image(image).reshape(1, *self.input_shape)),
                    self.cluster_model.predict(self.scaler.transform(
                        self.pca.transform(self.encoder.predict(
                            self.preprocess_image(image).reshape(1, *self.input_shape)
                        ))
                    ))
                )
            }
        
        return result
    
    def _compute_gradients(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcule les gradients et les dérivées pour l'optimisation"""
        # Conversion en tenseurs TensorFlow
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Calcul des gradients avec tf.GradientTape
        with tf.GradientTape() as tape:
            # Prédictions du modèle
            predictions = self.model(features_tensor)
            
            # Calcul de la fonction de perte
            loss = tf.keras.losses.categorical_crossentropy(labels_tensor, predictions)
        
        # Calcul des gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Calcul des dérivées partielles
        partial_derivatives = {}
        for var, grad in zip(self.model.trainable_variables, gradients):
            if grad is not None:
                partial_derivatives[var.name] = {
                    "gradient": grad.numpy(),
                    "norm": float(tf.norm(grad)),
                    "direction": grad.numpy() / (tf.norm(grad) + 1e-7)
                }
        
        return {
            "loss": float(loss),
            "gradients": partial_derivatives,
            "gradient_norm": float(tf.norm(gradients))
        }

    def _optimize_with_gradient(self, features: np.ndarray, labels: np.ndarray, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Optimise les paramètres du modèle avec descente de gradient"""
        # Calcul des gradients
        gradient_info = self._compute_gradients(features, labels)
        
        # Mise à jour des paramètres avec descente de gradient
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Application des gradients
        optimizer.apply_gradients(zip(
            [grad["gradient"] for grad in gradient_info["gradients"].values()],
            self.model.trainable_variables
        ))
        
        # Calcul des changements de paramètres
        parameter_changes = {}
        for var in self.model.trainable_variables:
            if var.name in gradient_info["gradients"]:
                grad = gradient_info["gradients"][var.name]
                parameter_changes[var.name] = {
                    "change": float(tf.norm(grad["gradient"] * learning_rate)),
                    "direction": grad["direction"].tolist()
                }
        
        return {
            "loss": gradient_info["loss"],
            "gradient_norm": gradient_info["gradient_norm"],
            "parameter_changes": parameter_changes
        }

    def _compute_hessian(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcule la matrice hessienne pour l'analyse de la convexité"""
        # Conversion en tenseurs TensorFlow
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Calcul de la matrice hessienne
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                predictions = self.model(features_tensor)
                loss = tf.keras.losses.categorical_crossentropy(labels_tensor, predictions)
            
            # Calcul des gradients du premier ordre
            gradients = tape1.gradient(loss, self.model.trainable_variables)
        
        # Calcul des gradients du second ordre (hessienne)
        hessian = {}
        for i, var1 in enumerate(self.model.trainable_variables):
            hessian[var1.name] = {}
            for j, var2 in enumerate(self.model.trainable_variables):
                if gradients[i] is not None and gradients[j] is not None:
                    hessian[var1.name][var2.name] = tape2.gradient(gradients[i], var2)
        
        return hessian

    def _analyze_optimization_landscape(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyse le paysage d'optimisation avec calcul différentiel"""
        # Calcul des gradients
        gradient_info = self._compute_gradients(features, labels)
        
        # Calcul de la hessienne
        hessian = self._compute_hessian(features, labels)
        
        # Analyse de la convexité
        convexity_analysis = {}
        for var_name, var_hessian in hessian.items():
            eigenvalues = []
            for other_var, hess in var_hessian.items():
                if hess is not None:
                    try:
                        eigenvals = np.linalg.eigvals(hess.numpy())
                        eigenvalues.extend(eigenvals)
                    except np.linalg.LinAlgError:
                        continue
            
            if eigenvalues:
                convexity_analysis[var_name] = {
                    "min_eigenvalue": float(np.min(eigenvalues)),
                    "max_eigenvalue": float(np.max(eigenvalues)),
                    "is_convex": all(eig > 0 for eig in eigenvalues),
                    "condition_number": float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-7))
                }
        
        return {
            "gradient_info": gradient_info,
            "convexity_analysis": convexity_analysis,
            "optimization_difficulty": {
                "gradient_norm": float(gradient_info["gradient_norm"]),
                "condition_numbers": {
                    var: analysis["condition_number"]
                    for var, analysis in convexity_analysis.items()
                }
            }
        }

    def train(self, train_dir: str, val_dir: str, unlabeled_dir: str = None, epochs: int = 50):
        """Entraîne le modèle avec apprentissage supervisé et non supervisé"""
        # Entraînement supervisé
        history = self._train_supervised(train_dir, val_dir, epochs)
        
        # Entraînement non supervisé si des données non étiquetées sont fournies
        if unlabeled_dir:
            cluster_results = self.train_unsupervised(unlabeled_dir)
            history.cluster_results = cluster_results
        
        return history
    
    def _train_supervised(self, train_dir: str, val_dir: str, epochs: int):
        """Entraîne le modèle en mode supervisé avec mini-batch et propagation arrière améliorée"""
        # Augmentation des données
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        # Chargement des données
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode='categorical',
            classes=['normal', 'psoriasis']
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=32,
            class_mode='categorical',
            classes=['normal', 'psoriasis']
        )
        
        # Optimiseur personnalisé avec gradient clipping et momentum
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.9
            ),
            clipnorm=1.0,  # Gradient clipping
            beta_1=0.9,    # Momentum
            beta_2=0.999   # RMSprop
        )
        
        # Callbacks avancés
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001
            ),
            callbacks.ModelCheckpoint(
                'app/ml/models/psoriasis_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            # Callback personnalisé pour l'analyse du gradient et des moyennes mobiles
            tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: self._analyze_training_step(
                    train_generator.next()[0],
                    train_generator.next()[1]
                )
            )
        ]
        
        # Compilation du modèle avec l'optimiseur personnalisé
        self.model.compile(
            optimizer=optimizer,
            loss={
                'main': 'categorical_crossentropy',
                'type': 'categorical_crossentropy',
                'severity': 'categorical_crossentropy'
            },
            loss_weights={
                'main': 1.0,
                'type': 0.5,
                'severity': 0.5
            },
            metrics={
                'main': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ],
                'type': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ],
                'severity': [
                    'accuracy',
                    tf.keras.metrics.AUC(),
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.F1Score()
                ]
            }
        )
        
        # Entraînement
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks_list,
            workers=4,
            use_multiprocessing=True
        )
        
        return history

    def _analyze_training_step(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyse détaillée d'une étape d'entraînement"""
        # Échantillonnage des mini-batches
        batch_features, batch_labels = self._mini_batch_sampling(features, labels)
        
        # Propagation arrière
        update_metrics = self._backpropagation_step(batch_features, batch_labels)
        
        # Analyse du paysage d'optimisation
        optimization_analysis = self._analyze_optimization_landscape(batch_features, batch_labels)
        
        # Analyse des moyennes mobiles
        moving_averages = self._compute_moving_averages(update_metrics)
        
        return {
            "batch_metrics": update_metrics,
            "optimization_analysis": optimization_analysis,
            "moving_averages": moving_averages
        }

    def _compute_moving_averages(self, metrics: Dict[str, float], window_size: int = 100) -> Dict[str, float]:
        """Calcule les moyennes mobiles des métriques"""
        if not hasattr(self, '_metrics_history'):
            self._metrics_history = {
                'loss': [],
                'accuracy': [],
                'gradient_norm': [],
                'learning_rate': []
            }
        
        # Mise à jour de l'historique
        for metric_name, value in metrics.items():
            if metric_name in self._metrics_history:
                self._metrics_history[metric_name].append(value)
                # Limite de la taille de la fenêtre
                if len(self._metrics_history[metric_name]) > window_size:
                    self._metrics_history[metric_name].pop(0)
        
        # Calcul des moyennes mobiles
        moving_averages = {}
        for metric_name, values in self._metrics_history.items():
            if values:
                moving_averages[f"moving_avg_{metric_name}"] = float(np.mean(values))
                moving_averages[f"moving_std_{metric_name}"] = float(np.std(values))
        
        return moving_averages

    def _mini_batch_sampling(self, features: np.ndarray, labels: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Échantillonnage intelligent des mini-batches"""
        n_samples = len(features)
        
        # Stratification par classe pour les labels
        if len(labels.shape) > 1:
            class_indices = {i: np.where(labels[:, i] == 1)[0] for i in range(labels.shape[1])}
        else:
            class_indices = {0: np.where(labels == 0)[0], 1: np.where(labels == 1)[0]}
        
        # Calcul des poids d'échantillonnage
        sampling_weights = np.zeros(n_samples)
        for class_idx, indices in class_indices.items():
            sampling_weights[indices] = 1.0 / len(indices)
        
        # Normalisation des poids
        sampling_weights = sampling_weights / np.sum(sampling_weights)
        
        # Échantillonnage avec remplacement
        batch_indices = np.random.choice(n_samples, size=batch_size, p=sampling_weights)
        
        return features[batch_indices], labels[batch_indices]

    def _backpropagation_step(self, features: np.ndarray, labels: np.ndarray, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Effectue une étape de propagation arrière avec gestion avancée des gradients"""
        # Conversion en tenseurs TensorFlow
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        labels_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Calcul des gradients avec tf.GradientTape
        with tf.GradientTape() as tape:
            # Prédictions du modèle
            predictions = self.model(features_tensor)
            
            # Calcul de la fonction de perte
            loss = tf.keras.losses.categorical_crossentropy(labels_tensor, predictions)
        
        # Calcul des gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Gestion des gradients nuls ou explosifs
        gradients = [tf.clip_by_norm(grad, 1.0) if grad is not None else None for grad in gradients]
        
        # Mise à jour des poids avec momentum
        momentum = 0.9
        if not hasattr(self, '_momentum_buffer'):
            self._momentum_buffer = [tf.zeros_like(var) for var in self.model.trainable_variables]
        
        for i, (var, grad) in enumerate(zip(self.model.trainable_variables, gradients)):
            if grad is not None:
                # Mise à jour du buffer de momentum
                self._momentum_buffer[i] = momentum * self._momentum_buffer[i] + learning_rate * grad
                # Mise à jour des poids
                var.assign_sub(self._momentum_buffer[i])
        
        # Calcul des métriques de mise à jour
        update_metrics = {
            "loss": float(loss),
            "gradient_norm": float(tf.norm(gradients)),
            "parameter_updates": {
                var.name: float(tf.norm(self._momentum_buffer[i]))
                for i, var in enumerate(self.model.trainable_variables)
            }
        }
        
        # Calcul des moyennes mobiles
        moving_metrics = self._compute_moving_averages(update_metrics)
        update_metrics.update(moving_metrics)
        
        return update_metrics 

    def _compute_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcule des métriques avancées de performance"""
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Métriques par classe
        metrics_per_class = {}
        for i in range(len(cm)):
            # True Positives, False Positives, True Negatives, False Negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + tp
            fn = np.sum(cm[i, :]) - tp
            
            # Calcul des métriques
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_per_class[f"classe_{i}"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": {
                    "tp": int(tp),
                    "fp": int(fp),
                    "tn": int(tn),
                    "fn": int(fn)
                }
            }
        
        # Métriques globales
        global_metrics = {
            "accuracy": float(np.mean([m["accuracy"] for m in metrics_per_class.values()])),
            "precision": float(np.mean([m["precision"] for m in metrics_per_class.values()])),
            "recall": float(np.mean([m["recall"] for m in metrics_per_class.values()])),
            "f1": float(np.mean([m["f1"] for m in metrics_per_class.values()]))
        }
        
        return {
            "global_metrics": global_metrics,
            "per_class_metrics": metrics_per_class,
            "confusion_matrix": cm.tolist()
        }

    def _compute_roc_curves(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calcule les courbes ROC et AUC pour chaque classe"""
        from sklearn.metrics import roc_curve, auc
        
        roc_curves = {}
        for i in range(y_pred_proba.shape[1]):
            # Calcul des points de la courbe ROC
            fpr, tpr, thresholds = roc_curve(y_true[:, i], y_pred_proba[:, i])
            
            # Calcul de l'AUC
            roc_auc = auc(fpr, tpr)
            
            # Calcul des points optimaux
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            roc_curves[f"classe_{i}"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thresholds.tolist(),
                "auc": float(roc_auc),
                "optimal_threshold": float(optimal_threshold),
                "optimal_point": {
                    "fpr": float(fpr[optimal_idx]),
                    "tpr": float(tpr[optimal_idx])
                }
            }
        
        return roc_curves

    def _compute_conditional_probabilities(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calcule les probabilités conditionnelles et les analyses associées"""
        # Probabilités conditionnelles par classe
        conditional_probs = {}
        for i in range(y_pred_proba.shape[1]):
            # Probabilité conditionnelle P(Y=1|X)
            prob_positive = y_pred_proba[:, i]
            
            # Analyse des seuils de décision
            thresholds = np.linspace(0, 1, 100)
            decision_analysis = []
            for threshold in thresholds:
                decisions = (prob_positive >= threshold).astype(int)
                accuracy = np.mean(decisions == y_true[:, i])
                precision = np.sum((decisions == 1) & (y_true[:, i] == 1)) / (np.sum(decisions == 1) + 1e-7)
                recall = np.sum((decisions == 1) & (y_true[:, i] == 1)) / (np.sum(y_true[:, i] == 1) + 1e-7)
                
                decision_analysis.append({
                    "threshold": float(threshold),
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall)
                })
            
            # Statistiques des probabilités
            prob_stats = {
                "mean": float(np.mean(prob_positive)),
                "std": float(np.std(prob_positive)),
                "min": float(np.min(prob_positive)),
                "max": float(np.max(prob_positive)),
                "median": float(np.median(prob_positive)),
                "quartiles": {
                    "q1": float(np.percentile(prob_positive, 25)),
                    "q2": float(np.percentile(prob_positive, 50)),
                    "q3": float(np.percentile(prob_positive, 75))
                }
            }
            
            conditional_probs[f"classe_{i}"] = {
                "probabilities": prob_positive.tolist(),
                "statistics": prob_stats,
                "decision_analysis": decision_analysis
            }
        
        return conditional_probs

    def evaluate_model(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict[str, Any]:
        """Évalue le modèle avec des métriques avancées"""
        # Prédictions
        predictions = self.model.predict(test_data)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(test_labels, axis=1)
        
        # Calcul des métriques avancées
        metrics = self._compute_advanced_metrics(y_true, y_pred)
        
        # Calcul des courbes ROC
        roc_curves = self._compute_roc_curves(test_labels, predictions)
        
        # Calcul des probabilités conditionnelles
        conditional_probs = self._compute_conditional_probabilities(test_labels, predictions)
        
        # Analyse des performances par classe
        class_performance = {}
        for i in range(len(np.unique(y_true))):
            class_performance[f"classe_{i}"] = {
                "metrics": metrics["per_class_metrics"][f"classe_{i}"],
                "roc_curve": roc_curves[f"classe_{i}"],
                "conditional_probabilities": conditional_probs[f"classe_{i}"]
            }
        
        return {
            "global_metrics": metrics["global_metrics"],
            "confusion_matrix": metrics["confusion_matrix"],
            "class_performance": class_performance,
            "roc_curves": roc_curves,
            "conditional_probabilities": conditional_probs
        }

    def predict_with_confidence(self, image: np.ndarray) -> Dict[str, Any]:
        """Fait une prédiction avec analyse détaillée des probabilités et de la confiance"""
        # Prédiction standard
        result = self.predict(image)
        
        # Analyse des probabilités conditionnelles
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        predictions = self.model.predict(processed_image)
        
        # Calcul des métriques de confiance
        confidence_metrics = {}
        for i, (output_name, probs) in enumerate(zip(["main", "type", "severity"], predictions)):
            # Probabilités conditionnelles
            cond_probs = self._compute_conditional_probabilities(
                np.array([[1 if j == np.argmax(probs[0]) else 0 for j in range(len(probs[0]))]]),
                probs
            )
            
            # Courbe ROC pour la classe prédite
            roc_curve = self._compute_roc_curves(
                np.array([[1 if j == np.argmax(probs[0]) else 0 for j in range(len(probs[0]))]]),
                probs
            )
            
            confidence_metrics[output_name] = {
                "conditional_probabilities": cond_probs[f"classe_{np.argmax(probs[0])}"],
                "roc_curve": roc_curve[f"classe_{np.argmax(probs[0])}"],
                "confidence_score": float(np.max(probs[0])),
                "probability_distribution": {
                    f"classe_{j}": float(prob)
                    for j, prob in enumerate(probs[0])
                }
            }
        
        # Ajout des métriques de confiance au résultat
        result["confidence_analysis"] = confidence_metrics
        
        return result 

    def _compute_shapley_values(self, features: np.ndarray, n_samples: int = 100) -> Dict[str, np.ndarray]:
        """Calcule les valeurs de Shapley pour chaque feature"""
        import shap
        
        # Création d'un background dataset
        background = shap.sample(features, n_samples, random_state=42)
        
        # Création d'un explainer SHAP
        explainer = shap.DeepExplainer(self.model, background)
        
        # Calcul des valeurs Shapley pour un échantillon
        shap_values = explainer.shap_values(features[:n_samples])
        
        # Analyse des valeurs Shapley
        shap_analysis = {}
        for i, output_name in enumerate(["main", "type", "severity"]):
            shap_analysis[output_name] = {
                "values": shap_values[i].tolist(),
                "mean_abs_values": np.abs(shap_values[i]).mean(axis=0).tolist(),
                "feature_importance": np.abs(shap_values[i]).mean(axis=0).argsort()[::-1].tolist()
            }
        
        return shap_analysis

    def _analyze_feature_importance(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyse l'importance des features avec différentes méthodes"""
        # Importance avec Random Forest
        rf_importance = {}
        for i, output_name in enumerate(["main", "type", "severity"]):
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(features, labels[:, i])
            rf_importance[output_name] = {
                "importance": rf.feature_importances_.tolist(),
                "feature_ranking": rf.feature_importances_.argsort()[::-1].tolist()
            }
        
        # Importance avec permutation
        permutation_importance = {}
        for i, output_name in enumerate(["main", "type", "severity"]):
            base_score = self.model.evaluate(features, labels[:, i], verbose=0)[0]
            feature_importance = []
            
            for j in range(features.shape[1]):
                # Permutation de la feature j
                features_permuted = features.copy()
                features_permuted[:, j] = np.random.permutation(features_permuted[:, j])
                permuted_score = self.model.evaluate(features_permuted, labels[:, i], verbose=0)[0]
                feature_importance.append(permuted_score - base_score)
            
            permutation_importance[output_name] = {
                "importance": feature_importance,
                "feature_ranking": np.array(feature_importance).argsort()[::-1].tolist()
            }
        
        return {
            "random_forest_importance": rf_importance,
            "permutation_importance": permutation_importance
        }

    def _compute_pca_projection(self, features: np.ndarray, n_components: int = 2) -> Dict[str, Any]:
        """Calcule la projection PCA et analyse les composantes principales"""
        # Réduction de dimensionnalité avec PCA
        pca = PCA(n_components=n_components)
        features_projected = pca.fit_transform(features)
        
        # Analyse des composantes principales
        pca_analysis = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "mean": pca.mean_.tolist(),
            "projected_features": features_projected.tolist()
        }
        
        # Calcul de la reconstruction
        features_reconstructed = pca.inverse_transform(features_projected)
        reconstruction_error = np.mean(np.square(features - features_reconstructed))
        
        pca_analysis["reconstruction_error"] = float(reconstruction_error)
        
        return pca_analysis

    def analyze_features_importance(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyse détaillée de l'importance des features pour une image"""
        # Extraction des features
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        features = self.encoder.predict(processed_image)[0]
        
        # Calcul des valeurs Shapley
        shap_values = self._compute_shapley_values(processed_image)
        
        # Analyse de l'importance des features
        feature_importance = self._analyze_feature_importance(processed_image, np.array([[1, 0]]))
        
        # Projection PCA
        pca_projection = self._compute_pca_projection(processed_image.reshape(1, -1))
        
        # Analyse des features par type
        feature_analysis = {
            "texture_features": {
                "importance": feature_importance["random_forest_importance"]["main"]["importance"][:10],
                "shap_values": shap_values["main"]["values"][0][:10]
            },
            "color_features": {
                "importance": feature_importance["random_forest_importance"]["main"]["importance"][10:20],
                "shap_values": shap_values["main"]["values"][0][10:20]
            },
            "edge_features": {
                "importance": feature_importance["random_forest_importance"]["main"]["importance"][20:30],
                "shap_values": shap_values["main"]["values"][0][20:30]
            }
        }
        
        return {
            "shapley_values": shap_values,
            "feature_importance": feature_importance,
            "pca_projection": pca_projection,
            "feature_analysis": feature_analysis,
            "top_features": {
                "main": {
                    "indices": feature_importance["random_forest_importance"]["main"]["feature_ranking"][:10],
                    "importance": feature_importance["random_forest_importance"]["main"]["importance"][:10]
                },
                "type": {
                    "indices": feature_importance["random_forest_importance"]["type"]["feature_ranking"][:10],
                    "importance": feature_importance["random_forest_importance"]["type"]["importance"][:10]
                },
                "severity": {
                    "indices": feature_importance["random_forest_importance"]["severity"]["feature_ranking"][:10],
                    "importance": feature_importance["random_forest_importance"]["severity"]["importance"][:10]
                }
            }
        }

    def predict_with_feature_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Fait une prédiction avec analyse détaillée des features"""
        # Prédiction standard
        result = self.predict(image)
        
        # Analyse de l'importance des features
        feature_analysis = self.analyze_features_importance(image)
        
        # Ajout de l'analyse des features au résultat
        result["feature_analysis"] = feature_analysis
        
        return result 

    def _analyze_time_series(self, features: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Analyse des séries temporelles des features"""
        # Calcul des statistiques temporelles
        time_stats = {
            "mean": np.mean(features, axis=0).tolist(),
            "std": np.std(features, axis=0).tolist(),
            "trend": np.polyfit(timestamps, features, 1)[0].tolist(),
            "seasonality": self._compute_seasonality(features, timestamps)
        }
        
        # Analyse de la stationnarité
        stationarity = self._check_stationarity(features)
        
        # Détection des changements
        change_points = self._detect_change_points(features)
        
        return {
            "time_statistics": time_stats,
            "stationarity": stationarity,
            "change_points": change_points
        }

    def _compute_seasonality(self, features: np.ndarray, timestamps: np.ndarray) -> Dict[str, Any]:
        """Calcule la saisonnalité des features"""
        # Décomposition saisonnière
        seasonal = {}
        for i in range(features.shape[1]):
            # FFT pour détecter les composantes saisonnières
            fft = np.fft.fft(features[:, i])
            frequencies = np.fft.fftfreq(len(features))
            
            # Identification des composantes principales
            main_freq = frequencies[np.argsort(np.abs(fft))[-3:]]
            seasonal[f"feature_{i}"] = {
                "main_frequencies": main_freq.tolist(),
                "amplitude": np.abs(fft[np.argsort(np.abs(fft))[-3:]]).tolist()
            }
        
        return seasonal

    def _check_stationarity(self, features: np.ndarray) -> Dict[str, Any]:
        """Vérifie la stationnarité des séries temporelles"""
        stationarity = {}
        for i in range(features.shape[1]):
            # Test de Dickey-Fuller augmenté
            adf_result = adfuller(features[:, i])
            stationarity[f"feature_{i}"] = {
                "is_stationary": adf_result[1] < 0.05,
                "p_value": float(adf_result[1]),
                "test_statistic": float(adf_result[0])
            }
        
        return stationarity

    def _detect_change_points(self, features: np.ndarray) -> Dict[str, Any]:
        """Détecte les points de changement dans les séries temporelles"""
        change_points = {}
        for i in range(features.shape[1]):
            # Test de Mann-Whitney pour détecter les changements
            window_size = 10
            changes = []
            for j in range(window_size, len(features) - window_size):
                before = features[j-window_size:j, i]
                after = features[j:j+window_size, i]
                _, p_value = stats.mannwhitneyu(before, after)
                if p_value < 0.05:
                    changes.append(j)
            
            change_points[f"feature_{i}"] = {
                "change_points": changes,
                "number_of_changes": len(changes)
            }
        
        return change_points

    def _compute_bayesian_probabilities(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calcule les probabilités bayésiennes"""
        bayesian_analysis = {}
        for i, output_name in enumerate(["main", "type", "severity"]):
            # Entraînement du classificateur bayésien
            nb = GaussianNB()
            nb.fit(features, labels[:, i])
            
            # Calcul des probabilités a priori
            prior_probs = nb.class_prior_
            
            # Calcul des probabilités conditionnelles
            conditional_probs = {
                f"feature_{j}": {
                    "mean": nb.theta_[i, j].tolist(),
                    "var": nb.var_[i, j].tolist()
                }
                for j in range(features.shape[1])
            }
            
            bayesian_analysis[output_name] = {
                "prior_probabilities": prior_probs.tolist(),
                "conditional_probabilities": conditional_probs,
                "feature_importance": np.abs(nb.theta_[i]).argsort()[::-1].tolist()
            }
        
        return bayesian_analysis

    def _compute_fourier_transform(self, image: np.ndarray) -> Dict[str, Any]:
        """Calcule la transformée de Fourier et analyse les fréquences"""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Transformée de Fourier 2D
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Calcul du spectre d'amplitude
        magnitude_spectrum = np.abs(fft_shift)
        
        # Analyse des fréquences
        freq_analysis = {
            "magnitude_spectrum": magnitude_spectrum.tolist(),
            "dominant_frequencies": self._find_dominant_frequencies(magnitude_spectrum),
            "frequency_bands": self._analyze_frequency_bands(magnitude_spectrum)
        }
        
        return freq_analysis

    def _find_dominant_frequencies(self, magnitude_spectrum: np.ndarray) -> Dict[str, Any]:
        """Trouve les fréquences dominantes dans le spectre"""
        # Seuil pour les fréquences dominantes
        threshold = np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
        
        # Localisation des fréquences dominantes
        dominant_freqs = np.where(magnitude_spectrum > threshold)
        
        return {
            "coordinates": list(zip(dominant_freqs[0].tolist(), dominant_freqs[1].tolist())),
            "magnitudes": magnitude_spectrum[dominant_freqs].tolist()
        }

    def _analyze_frequency_bands(self, magnitude_spectrum: np.ndarray) -> Dict[str, float]:
        """Analyse les différentes bandes de fréquences"""
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Définition des bandes de fréquences
        bands = {
            "low": magnitude_spectrum[center_h-10:center_h+10, center_w-10:center_w+10],
            "medium": magnitude_spectrum[center_h-30:center_h+30, center_w-30:center_w+30],
            "high": magnitude_spectrum
        }
        
        return {
            "low_frequency_energy": float(np.sum(bands["low"])),
            "medium_frequency_energy": float(np.sum(bands["medium"])),
            "high_frequency_energy": float(np.sum(bands["high"]))
        }

    def _compute_attention(self, features: np.ndarray) -> Dict[str, Any]:
        """Calcule les mécanismes d'attention"""
        # Calcul des scores d'attention
        query = features
        key = features
        value = features
        
        # Produit scalaire
        scores = np.dot(query, key.T)
        
        # Normalisation
        scores = scores / np.sqrt(features.shape[1])
        
        # Softmax
        attention_weights = self._softmax(scores)
        
        # Application des poids d'attention
        attended_features = np.dot(attention_weights, value)
        
        return {
            "attention_weights": attention_weights.tolist(),
            "attended_features": attended_features.tolist(),
            "attention_scores": scores.tolist()
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Calcule la fonction softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def analyze_with_advanced_methods(self, image: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, Any]:
        """Analyse complète avec toutes les méthodes avancées"""
        # Extraction des features
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        features = self.encoder.predict(processed_image)[0]
        
        # Analyse temporelle si les timestamps sont fournis
        time_analysis = None
        if timestamps is not None:
            time_analysis = self._analyze_time_series(features, timestamps)
        
        # Probabilités bayésiennes
        bayesian_probs = self._compute_bayesian_probabilities(
            processed_image.reshape(1, -1),
            np.array([[1, 0]])
        )
        
        # Transformée de Fourier
        fourier_analysis = self._compute_fourier_transform(image)
        
        # Mécanismes d'attention
        attention_analysis = self._compute_attention(features.reshape(1, -1))
        
        return {
            "time_analysis": time_analysis,
            "bayesian_probabilities": bayesian_probs,
            "fourier_transform": fourier_analysis,
            "attention_mechanism": attention_analysis
        }

    def predict_with_advanced_analysis(self, image: np.ndarray, timestamps: np.ndarray = None) -> Dict[str, Any]:
        """Fait une prédiction avec analyse avancée"""
        # Prédiction standard
        result = self.predict(image)
        
        # Analyse avancée
        advanced_analysis = self.analyze_with_advanced_methods(image, timestamps)
        
        # Ajout de l'analyse avancée au résultat
        result["advanced_analysis"] = advanced_analysis
        
        return result

    def _analyze_decision_path(self, features: np.ndarray, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse le chemin de décision du modèle"""
        decision_path = {
            "feature_importance": self._compute_feature_importance(features),
            "decision_steps": [],
            "confidence_analysis": {},
            "alternative_predictions": {}
        }
        
        # Analyse de la décision principale
        main_decision = self._analyze_main_decision(features, predictions['main'])
        decision_path["decision_steps"].append(main_decision)
        
        if predictions['main']['prediction'] == 'psoriasis':
            # Analyse des sous-décisions
            type_decision = self._analyze_type_decision(features, predictions['type'])
            severity_decision = self._analyze_severity_decision(features, predictions['severity'])
            
            decision_path["decision_steps"].extend([type_decision, severity_decision])
        
        return decision_path

    def _analyze_main_decision(self, features: np.ndarray, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la décision principale (psoriasis vs normal)"""
        # Calcul des scores pour chaque classe
        scores = self._compute_class_scores(features, "main")
        
        # Analyse des features clés
        key_features = self._identify_key_features(features, "main")
        
        return {
            "step": "main_classification",
            "decision": prediction['prediction'],
            "confidence": prediction['confidence'],
            "class_scores": scores,
            "key_features": key_features,
            "reasoning": self._generate_decision_reasoning(scores, key_features)
        }

    def _analyze_type_decision(self, features: np.ndarray, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la décision du type de psoriasis"""
        scores = self._compute_class_scores(features, "type")
        key_features = self._identify_key_features(features, "type")
        
        return {
            "step": "type_classification",
            "decision": prediction['prediction'],
            "confidence": prediction['confidence'],
            "class_scores": scores,
            "key_features": key_features,
            "reasoning": self._generate_decision_reasoning(scores, key_features)
        }

    def _analyze_severity_decision(self, features: np.ndarray, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la décision de sévérité"""
        scores = self._compute_class_scores(features, "severity")
        key_features = self._identify_key_features(features, "severity")
        
        return {
            "step": "severity_classification",
            "decision": prediction['prediction'],
            "confidence": prediction['confidence'],
            "class_scores": scores,
            "key_features": key_features,
            "reasoning": self._generate_decision_reasoning(scores, key_features)
        }

    def _compute_class_scores(self, features: np.ndarray, task: str) -> Dict[str, float]:
        """Calcule les scores pour chaque classe"""
        if task == "main":
            model = self.main_classifier
            classes = ["normal", "psoriasis"]
        elif task == "type":
            model = self.type_classifier
            classes = ["plaque", "guttate", "inverse", "pustular", "erythrodermic"]
        else:  # severity
            model = self.severity_classifier
            classes = ["mild", "moderate", "severe"]
        
        # Prédiction des probabilités
        probs = model.predict_proba(features.reshape(1, -1))[0]
        
        return {cls: float(prob) for cls, prob in zip(classes, probs)}

    def _identify_key_features(self, features: np.ndarray, task: str) -> List[Dict[str, Any]]:
        """Identifie les features clés pour la décision"""
        # Analyse de l'importance des features
        feature_importance = self._compute_feature_importance(features)
        
        # Sélection des features les plus importantes
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return [
            {
                "feature": feature,
                "importance": importance,
                "value": float(features[feature]),
                "contribution": self._compute_feature_contribution(feature, features[feature], task)
            }
            for feature, importance in top_features
        ]

    def _compute_feature_contribution(self, feature: str, value: float, task: str) -> str:
        """Calcule la contribution de la feature à la décision"""
        # Analyse de la contribution basée sur le type de feature
        if "texture" in feature:
            return self._analyze_texture_contribution(value)
        elif "color" in feature:
            return self._analyze_color_contribution(value)
        elif "edge" in feature:
            return self._analyze_edge_contribution(value)
        else:
            return self._analyze_general_contribution(value, task)

    def _generate_decision_reasoning(self, scores: Dict[str, float], key_features: List[Dict[str, Any]]) -> str:
        """Génère une explication du raisonnement"""
        reasoning = []
        
        # Analyse des scores
        max_score = max(scores.values())
        max_class = max(scores.items(), key=lambda x: x[1])[0]
        
        reasoning.append(f"La classe '{max_class}' a été choisie avec un score de {max_score:.2f}")
        
        # Analyse des features clés
        reasoning.append("\nFeatures clés influençant la décision :")
        for feature in key_features:
            reasoning.append(
                f"- {feature['feature']}: importance {feature['importance']:.2f}, "
                f"contribution: {feature['contribution']}"
            )
        
        return "\n".join(reasoning)

    def predict_with_explanation(self, image: np.ndarray) -> Dict[str, Any]:
        """Fait une prédiction avec explication détaillée"""
        # Prétraitement
        processed_image = self.preprocess_image(image)
        processed_image = np.expand_dims(processed_image, axis=0)
        
        # Extraction des features
        features = self.encoder.predict(processed_image)[0]
        
        # Prédiction standard
        result = self.predict(image)
        
        # Analyse du chemin de décision
        decision_path = self._analyze_decision_path(features, result)
        
        # Ajout de l'explication au résultat
        result["explanation"] = {
            "decision_path": decision_path,
            "confidence_analysis": self._analyze_confidence(result),
            "feature_analysis": self._analyze_feature_importance(features)
        }
        
        return result

    def _analyze_confidence(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse la confiance des prédictions"""
        confidence_analysis = {}
        
        for task, pred in predictions.items():
            confidence_analysis[task] = {
                "confidence": pred['confidence'],
                "confidence_level": self._get_confidence_level(pred['confidence']),
                "alternative_probabilities": self._get_alternative_probabilities(pred)
            }
        
        return confidence_analysis

    def _get_confidence_level(self, confidence: float) -> str:
        """Détermine le niveau de confiance"""
        if confidence >= 0.9:
            return "très élevée"
        elif confidence >= 0.7:
            return "élevée"
        elif confidence >= 0.5:
            return "modérée"
        else:
            return "faible"

    def _get_alternative_probabilities(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les probabilités alternatives"""
        return {
            cls: prob for cls, prob in prediction.items()
            if isinstance(prob, float) and cls != prediction['prediction']
        }

    def _download_model_weights(self):
        """Télécharge les poids pré-entraînés du modèle"""
        weights_url = "https://storage.googleapis.com/psoriasis-model-weights/model_weights.h5"
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "model_weights.h5")
        
        # Créer le dossier weights s'il n'existe pas
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        
        if not os.path.exists(weights_path):
            print("Téléchargement des poids du modèle...")
            try:
                import urllib.request
                urllib.request.urlretrieve(weights_url, weights_path)
                print("✅ Poids téléchargés avec succès")
            except Exception as e:
                print(f"❌ Erreur lors du téléchargement des poids : {e}")
                return False
        else:
            print("✅ Poids déjà présents")
        
        return True

    def _load_model_weights(self):
        """Charge les poids du modèle"""
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "model_weights.h5")
        
        if not os.path.exists(weights_path):
            if not self._download_model_weights():
                raise ValueError("Impossible de charger les poids du modèle")
        
        try:
            self.model.load_weights(weights_path)
            print("✅ Poids chargés avec succès")
            return True
        except Exception as e:
            print(f"❌ Erreur lors du chargement des poids : {e}")
            return False