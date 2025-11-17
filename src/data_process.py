"""
Data Processing Module
Prépare et traite les données pour l'entraînement du modèle
"""

import pandas as pd
import os
from pathlib import Path


def prepare_data(input_path="data/train.csv", output_dir="data/processed"):
    """
    Prépare les données : nettoyage, sélection des features numériques,
    séparation features/target et stratification.
    
    Args:
        input_path (str): Chemin vers le fichier CSV source
        output_dir (str): Répertoire de sortie pour les fichiers traités
        
    Returns:
        dict: Dictionnaire contenant les chemins vers les fichiers sauvegardés
    """
    print("\n" + "="*60)
    print("📊 DATA PROCESSING PIPELINE")
    print("="*60 + "\n")
    
    # Vérifier que le fichier d'entrée existe
    if not os.path.exists(input_path):
        print(f"❌ Erreur: Fichier {input_path} non trouvé!")
        return None
    
    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # --- ÉTAPE 1 : Chargement des données ---
    print("📂 Étape 1: Chargement des données")
    df = load_data(input_path)
    
    # --- ÉTAPE 2 : Sélection des features numériques ---
    print("📊 Étape 2: Sélection des features numériques")
    df_num = select_numeric_features(df)
    
    # --- ÉTAPE 3 : Séparation features/target ---
    print("📊 Étape 3: Séparation features/target et stratification")
    X, y, y_binned = split_features_target(df_num)
    
    # --- ÉTAPE 4 : Sauvegarde des données traités ---
    print("💾 Étape 4: Sauvegarde des données traités")
    output_paths = save_processed_data(X, y, y_binned, output_dir)
    
    print("="*60)
    print("✅ DATA PROCESSING TERMINÉ")
    print("="*60 + "\n")
    
    return output_paths


def load_data(csv_path):
    """
    Charge le fichier de données d'entraînement
    
    Args:
        csv_path (str): Chemin vers le fichier train.csv
        
    Returns:
        pd.DataFrame: Données chargées
    """
    df = pd.read_csv(csv_path)
    print(f"✔ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df


def select_numeric_features(df):
    """
    Sélectionne uniquement les colonnes numériques et supprime les valeurs manquantes
    
    Args:
        df (pd.DataFrame): DataFrame brutes
        
    Returns:
        pd.DataFrame: DataFrame avec colonnes numériques uniquement
    """
    df_num = df.select_dtypes(include=["int64", "float64"])
    
    rows_before = df_num.shape[0]
    df_num = df_num.dropna()
    rows_after = df_num.shape[0]
    
    dropped_rows = rows_before - rows_after
    print(f"✔ {dropped_rows} lignes avec valeurs manquantes supprimées")
    print(f"✔ Dimensions finales: {df_num.shape[0]} lignes, {df_num.shape[1]} colonnes\n")
    
    return df_num


def split_features_target(df_num):
    """
    Sépare les features (X) de la cible (y) et applique une stratification
    
    Args:
        df_num (pd.DataFrame): DataFrame avec colonnes numériques
        
    Returns:
        tuple: (X, y, y_binned) où y_binned est utilisé pour la stratification
    """
    # Définir la target
    y = df_num["SalePrice"]
    
    # Supprimer la target des features
    X = df_num.drop(columns=["SalePrice"])
    
    # Stratification par binning (10 bins)
    y_binned = pd.qcut(y, q=10, duplicates="drop")
    
    print(f"✔ X shape: {X.shape}")
    print(f"✔ y shape: {y.shape}")
    print(f"✔ Stratification: {y_binned.nunique()} bins créés\n")
    
    return X, y, y_binned


def save_processed_data(X, y, y_binned, output_dir):
    """
    Sauvegarde les données traitées en fichiers CSV
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        y_binned (pd.Series): Target stratifiée (pour référence)
        output_dir (str): Répertoire de destination
        
    Returns:
        dict: Dictionnaire avec les chemins des fichiers sauvegardés
    """
    X_path = os.path.join(output_dir, "X.csv")
    y_path = os.path.join(output_dir, "y.csv")
    y_binned_path = os.path.join(output_dir, "y_binned.csv")
    
    X.to_csv(X_path, index=False)
    y.to_csv(y_path, index=False, header=["SalePrice"])
    y_binned.to_csv(y_binned_path, index=False, header=["SalePrice_Binned"])
    
    print(f"✔ X sauvegardé: {X_path}")
    print(f"✔ y sauvegardé: {y_path}")
    print(f"✔ y_binned sauvegardé: {y_binned_path}\n")
    
    return {
        "X": X_path,
        "y": y_path,
        "y_binned": y_binned_path
    }


if __name__ == "__main__":
    prepare_data()
