"""
Training Module
Entraîne le modèle de régression linéaire
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib


def train_model(data_dir="data/processed", output_dir="models"):
    """
    Charge les données traitées, entraîne un modèle LinearRegression
    et le sauvegarde.
    
    Args:
        data_dir (str): Répertoire contenant les données traitées
        output_dir (str): Répertoire de sortie pour le modèle
        
    Returns:
        dict: Dictionnaire contenant le modèle et les chemins de sortie
    """
    print("\n" + "="*60)
    print("🔧 TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Vérifier que le répertoire de données existe
    if not os.path.exists(data_dir):
        print(f"❌ Erreur: Répertoire {data_dir} non trouvé!")
        return None
    
    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- ÉTAPE 1 : Chargement des données traitées ---
    print("📂 Étape 1: Chargement des données traitées")
    X, y, y_binned = load_processed_data(data_dir)
    
    if X is None:
        print(f"❌ Impossible de charger les données depuis {data_dir}")
        return None
    
    # --- ÉTAPE 2 : Split train/test ---
    print("📊 Étape 2: Train/Test split avec stratification")
    X_train, X_test, y_train, y_test = split_train_test(X, y, y_binned)
    
    # --- ÉTAPE 3 : Entraînement du modèle ---
    print("🚀 Étape 3: Entraînement du modèle")
    model = train_linear_regression(X_train, y_train)
    
    # --- ÉTAPE 4 : Sauvegarde du modèle et des splits ---
    print("💾 Étape 4: Sauvegarde du modèle et des données de split")
    output_paths = save_model_and_splits(
        model, X_train, X_test, y_train, y_test, output_dir
    )
    
    print("="*60)
    print("✅ TRAINING TERMINÉ")
    print("="*60 + "\n")
    
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "output_paths": output_paths
    }


def load_processed_data(data_dir):
    """
    Charge les fichiers de données traitées
    
    Args:
        data_dir (str): Répertoire contenant les données traitées
        
    Returns:
        tuple: (X, y, y_binned) ou (None, None, None) en cas d'erreur
    """
    try:
        X_path = os.path.join(data_dir, "X.csv")
        y_path = os.path.join(data_dir, "y.csv")
        y_binned_path = os.path.join(data_dir, "y_binned.csv")
        
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0]
        y_binned = pd.read_csv(y_binned_path).iloc[:, 0]
        
        print(f"✔ X chargé: {X.shape}")
        print(f"✔ y chargé: {y.shape}")
        print(f"✔ y_binned chargé: {y_binned.shape}\n")
        
        return X, y, y_binned
    
    except FileNotFoundError as e:
        print(f"❌ Erreur lors du chargement: {e}\n")
        return None, None, None


def split_train_test(X, y, y_binned, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles train/test avec stratification
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        y_binned (pd.Series): Target stratifiée
        test_size (float): Proportion du test set
        random_state (int): Graine aléatoire
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_binned
    )
    
    print(f"✔ Train set: {X_train.shape[0]} lignes ({(1-test_size)*100:.0f}%)")
    print(f"✔ Test set: {X_test.shape[0]} lignes ({test_size*100:.0f}%)\n")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Entraîne un modèle de régression linéaire
    
    Args:
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Target d'entraînement
        
    Returns:
        LinearRegression: Modèle entraîné
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"✔ Modèle entraîné avec succès")
    print(f"✔ Nombre de features: {len(model.coef_)}\n")
    
    return model


def save_model_and_splits(model, X_train, X_test, y_train, y_test, output_dir):
    """
    Sauvegarde le modèle entraîné et les splits de données
    
    Args:
        model (LinearRegression): Modèle entraîné
        X_train, X_test (pd.DataFrame): Features train/test
        y_train, y_test (pd.Series): Target train/test
        output_dir (str): Répertoire de destination
        
    Returns:
        dict: Dictionnaire avec les chemins des fichiers sauvegardés
    """
    model_path = os.path.join(output_dir, "model.pkl")
    X_train_path = os.path.join(output_dir, "X_train.csv")
    X_test_path = os.path.join(output_dir, "X_test.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")
    
    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"✔ Modèle sauvegardé: {model_path}")
    
    # Sauvegarde des splits pour l'évaluation
    X_train.to_csv(X_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_train.to_csv(y_train_path, index=False, header=["SalePrice"])
    y_test.to_csv(y_test_path, index=False, header=["SalePrice"])
    
    print(f"✔ X_train sauvegardé: {X_train_path}")
    print(f"✔ X_test sauvegardé: {X_test_path}")
    print(f"✔ y_train sauvegardé: {y_train_path}")
    print(f"✔ y_test sauvegardé: {y_test_path}\n")
    
    return {
        "model": model_path,
        "X_train": X_train_path,
        "X_test": X_test_path,
        "y_train": y_train_path,
        "y_test": y_test_path
    }


if __name__ == "__main__":
    train_model()
