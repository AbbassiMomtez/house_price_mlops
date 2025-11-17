"""
Evaluation Module
Évalue les performances du modèle entraîné
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_model(data_dir="model", model_path="models/model.pkl"):
    """
    Évalue le modèle entraîné sur les données de test
    
    Args:
        data_dir (str): Répertoire contenant les données traitées
        model_path (str): Chemin vers le fichier du modèle sauvegardé
        
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation
    """
    print("\n" + "="*60)
    print("📈 EVALUATION PIPELINE")
    print("="*60 + "\n")
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Modèle {model_path} non trouvé!")
        return None
    
    # --- ÉTAPE 1 : Chargement du modèle ---
    print("🔍 Étape 1: Chargement du modèle")
    model = load_model(model_path)
    
    if model is None:
        return None
    
    # --- ÉTAPE 2 : Chargement des données de test ---
    print("📂 Étape 2: Chargement des données de test")
    X_train, X_test, y_train, y_test = load_test_data(data_dir)
    
    if X_test is None:
        print(f"❌ Impossible de charger les données depuis {data_dir}")
        return None
    
    # --- ÉTAPE 3 : Évaluation du modèle ---
    print("📊 Étape 3: Évaluation du modèle\n")
    metrics = evaluate_on_splits(model, X_train, X_test, y_train, y_test)
    
    # --- ÉTAPE 4 : Affichage des résultats ---
    print("="*60)
    print("📋 RÉSUMÉ DES PERFORMANCES")
    print("="*60)
    print_metrics_summary(metrics)
    print("="*60)
    print("✅ EVALUATION TERMINÉE")
    print("="*60 + "\n")
    
    return metrics


def load_model(model_path):
    """
    Charge le modèle sauvegardé
    
    Args:
        model_path (str): Chemin vers le fichier du modèle
        
    Returns:
        LinearRegression: Modèle chargé ou None en cas d'erreur
    """
    try:
        model = joblib.load(model_path)
        print(f"✔ Modèle chargé: {model_path}\n")
        return model
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}\n")
        return None


def load_test_data(data_dir):
    """
    Charge les données de test et d'entraînement
    
    Args:
        data_dir (str): Répertoire contenant les données
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) ou (None, None, None, None) en cas d'erreur
    """
    try:
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]
        
        print(f"✔ X_train chargé: {X_train.shape}")
        print(f"✔ X_test chargé: {X_test.shape}")
        print(f"✔ y_train chargé: {y_train.shape}")
        print(f"✔ y_test chargé: {y_test.shape}\n")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError as e:
        print(f"❌ Erreur lors du chargement des données: {e}\n")
        return None, None, None, None


def evaluate_on_splits(model, X_train, X_test, y_train, y_test):
    """
    Évalue le modèle sur les ensembles d'entraînement et de test
    
    Args:
        model: Modèle entraîné
        X_train, X_test (pd.DataFrame): Features train/test
        y_train, y_test (pd.Series): Target train/test
        
    Returns:
        dict: Dictionnaire contenant toutes les métriques
    """
    metrics = {}
    
    # Prédictions sur train
    y_train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    metrics["train"] = {
        "RMSE": train_rmse,
        "MAE": train_mae,
        "R²": train_r2
    }
    
    print("--- TRAIN SET ---")
    print(f"RMSE : {train_rmse:.2f}")
    print(f"MAE  : {train_mae:.2f}")
    print(f"R²   : {train_r2:.4f}\n")
    
    # Prédictions sur test
    y_test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    metrics["test"] = {
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R²": test_r2
    }
    
    print("--- TEST SET ---")
    print(f"RMSE : {test_rmse:.2f}")
    print(f"MAE  : {test_mae:.2f}")
    print(f"R²   : {test_r2:.4f}\n")
    
    # Calcul du overfitting/underfitting
    rmse_diff = test_rmse - train_rmse
    r2_diff = train_r2 - test_r2
    
    metrics["analysis"] = {
        "RMSE_difference": rmse_diff,
        "R2_difference": r2_diff,
        "overfitting_status": "Possible overfitting" if r2_diff > 0.05 else "Normal"
    }
    
    return metrics


def print_metrics_summary(metrics):
    """
    Affiche un résumé formaté des métriques
    
    Args:
        metrics (dict): Dictionnaire contenant les métriques
    """
    if metrics is None:
        print("❌ Aucune métrique disponible")
        return
    
    print("\nMétriques d'entraînement:")
    for key, value in metrics["train"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMétriques de test:")
    for key, value in metrics["test"].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAnalyse:")
    print(f"  Différence RMSE (test - train): {metrics['analysis']['RMSE_difference']:.2f}")
    print(f"  Différence R² (train - test): {metrics['analysis']['R2_difference']:.4f}")
    print(f"  Status: {metrics['analysis']['overfitting_status']}")


if __name__ == "__main__":
    evaluate_model()
