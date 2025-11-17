"""
Main Pipeline Orchestrator
Exécute la pipeline complète : data_process.py -> train.py -> eval.py
"""

import os
import sys
from pathlib import Path

# Importer les modules de pipeline
from data_process import prepare_data
from train import train_model
from eval import evaluate_model


def run_full_pipeline(
    input_csv=r"house_price_mlops\data\train.csv",
    data_dir=r"house_price_mlops\data\processed",
    model_dir=r"house_price_mlops\model"
):
    """
    Exécute la pipeline complète d'apprentissage automatique
    
    Args:
        input_csv (str): Chemin vers le fichier CSV source
        data_dir (str): Répertoire pour les données traitées
        model_dir (str): Répertoire pour le modèle
    """
    print("\n" + "🚀" * 30)
    print("FULL ML PIPELINE - HOUSE PRICE PREDICTION")
    print("🚀" * 30 + "\n")
    
    # Vérifier que le fichier source existe
    if not os.path.exists(input_csv):
        print(f"❌ Erreur: Fichier {input_csv} non trouvé!")
        print("Veuillez placer le fichier train.csv dans le répertoire data/")
        return False
    
    # --- PHASE 1 : DATA PROCESSING ---
    print("\n" + "="*60)
    print("PHASE 1: DATA PROCESSING")
    print("="*60)
    try:
        data_paths = prepare_data(input_path=input_csv, output_dir=data_dir)
        if data_paths is None:
            print("❌ Erreur lors du traitement des données")
            return False
        print("✅ Phase 1 complétée")
    except Exception as e:
        print(f"❌ Erreur Phase 1: {e}")
        return False
    
    # --- PHASE 2 : TRAINING ---
    print("\n" + "="*60)
    print("PHASE 2: TRAINING")
    print("="*60)
    try:
        training_results = train_model(data_dir=data_dir, output_dir=model_dir)
        if training_results is None:
            print("❌ Erreur lors de l'entraînement du modèle")
            return False
        print("✅ Phase 2 complétée")
    except Exception as e:
        print(f"❌ Erreur Phase 2: {e}")
        return False
    
    # --- PHASE 3 : EVALUATION ---
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION")
    print("="*60)
    try:
        model_path = os.path.join(model_dir, "model.pkl")
        metrics = evaluate_model(data_dir=data_dir, model_path=model_path)
        if metrics is None:
            print("❌ Erreur lors de l'évaluation du modèle")
            return False
        print("✅ Phase 3 complétée")
    except Exception as e:
        print(f"❌ Erreur Phase 3: {e}")
        return False
    
    # --- RÉSUMÉ FINAL ---
    print("\n" + "🎉" * 30)
    print("✅ PIPELINE COMPLÈTE AVEC SUCCÈS!")
    print("🎉" * 30)
    print("\nRésumé:")
    print(f"  📁 Données traitées: {data_dir}/")
    print(f"  🤖 Modèle entraîné: {model_dir}/model.pkl")
    print(f"  📊 Performances:")
    if metrics:
        print(f"     - Train R²: {metrics['train']['R²']:.4f}")
        print(f"     - Test R²:  {metrics['test']['R²']:.4f}")
        print(f"     - Status:   {metrics['analysis']['overfitting_status']}")
    
    return True


def run_individual_pipeline(pipeline_name):
    """
    Exécute une pipeline individuelle
    
    Args:
        pipeline_name (str): 'process', 'train' ou 'eval'
    """
    if pipeline_name == "process":
        print("\n🔄 Exécution: DATA PROCESSING")
        prepare_data()
    
    elif pipeline_name == "train":
        print("\n🔄 Exécution: TRAINING")
        train_model()
    
    elif pipeline_name == "eval":
        print("\n🔄 Exécution: EVALUATION")
        evaluate_model()
    
    else:
        print(f"❌ Pipeline inconnue: {pipeline_name}")
        print("Options disponibles: process, train, eval, full")


if __name__ == "__main__":
    
    # Vérifier les arguments de ligne de commande
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "full":
            run_full_pipeline()
        elif command in ["process", "train", "eval"]:
            run_individual_pipeline(command)
        else:
            print("❌ Commande inconnue")
            print("\nUsage:")
            print("  python main.py full      # Exécuter la pipeline complète")
            print("  python main.py process   # Seulement traiter les données")
            print("  python main.py train     # Seulement entraîner le modèle")
            print("  python main.py eval      # Seulement évaluer le modèle")
    else:
        # Exécuter la pipeline complète par défaut
        run_full_pipeline()
