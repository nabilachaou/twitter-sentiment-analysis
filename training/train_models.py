import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Classe principale pour l'analyse de sentiment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.vectorizer = None
        self.best_model = None
        self.best_model_name = None
        self.models = {}
        self.results = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Configuration par défaut du système"""
        return {
            'test_size': 0.2,
            'val_size': 0.2,
            'random_state': 42,
            'max_features': 10000,
            'cv_folds': 3,  # Réduit de 5 à 3 pour accélérer
            'models_dir': 'models',
            'results_dir': 'results',
            'plots_dir': 'plots',
            'include_random_forest': False,  # Désactivé par défaut
            'sample_data': None,  # Pour tester sur un échantillon
            'use_incremental_learning': False  # Pour très gros datasets
        }
    
    def create_directories(self):
        """Créer les répertoires nécessaires"""
        for dir_name in ['models', 'results', 'plots']:
            os.makedirs(dir_name, exist_ok=True)
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Charger et prétraiter les données"""
        logger.info("📥 Chargement du dataset...")
        
        try:
            # Charger les données
            df = pd.read_csv(file_path, encoding='latin-1', header=None)
            df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
            
            logger.info(f"Dataset chargé: {len(df)} échantillons")
            
            # Option: échantillonner pour tests rapides
            if self.config.get('sample_data') and self.config['sample_data'] < len(df):
                df = df.sample(n=self.config['sample_data'], random_state=self.config['random_state'])
                logger.info(f"Échantillon utilisé: {len(df)} échantillons")
            
            # Mapper les sentiments
            sentiment_mapping = {0: "negative", 2: "neutral", 4: "positive"}
            df['sentiment'] = df['sentiment'].map(sentiment_mapping)
            
            # Vérifier la distribution des classes
            logger.info("Distribution des sentiments:")
            logger.info(df['sentiment'].value_counts())
            
            # Nettoyer les textes (supposons que clean_text existe)
            logger.info("🧹 Nettoyage des tweets...")
            try:
                from utils.preprocessing import clean_text
                df['clean_text'] = df['text'].apply(clean_text)
            except ImportError:
                logger.warning("Module preprocessing non trouvé, utilisation du texte brut")
                df['clean_text'] = df['text'].str.lower().str.strip()
            
            # Supprimer les valeurs manquantes
            df = df.dropna(subset=['clean_text', 'sentiment'])
            
            logger.info(f"Dataset après nettoyage: {len(df)} échantillons")
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """Diviser les données en ensembles d'entraînement, validation et test"""
        logger.info("🔀 Division des données...")
        
        X = df['clean_text']
        y = df['sentiment']
        
        # Division train/test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'], 
            stratify=y, 
            random_state=self.config['random_state']
        )
        
        # Division train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=self.config['val_size'], 
            stratify=y_train_val, 
            random_state=self.config['random_state']
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def vectorize_text(self, X_train: pd.Series, X_val: pd.Series, X_test: pd.Series):
        """Vectoriser les textes avec TF-IDF"""
        logger.info("🔤 Vectorisation TF-IDF...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=(1, 2),  # Unigrammes et bigrammes
            min_df=2,  # Ignorer les termes qui apparaissent moins de 2 fois
            max_df=0.8,  # Ignorer les termes qui apparaissent dans plus de 80% des documents
            stop_words='english'
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        X_test_vec = self.vectorizer.transform(X_test)
        
        logger.info(f"Forme des données vectorisées: {X_train_vec.shape}")
        
        return X_train_vec, X_val_vec, X_test_vec
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialiser les modèles avec hyperparamètres optimisés"""
        models = {
            "LogisticRegression": {
                'model': LogisticRegression(max_iter=1000, n_jobs=-1),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']  # saga plus rapide pour grands datasets
                }
            },
            "MultinomialNB": {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 1.0, 2.0]  # Réduit le nombre de paramètres
                }
            },
            "LinearSVC": {
                'model': LinearSVC(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10]  # Supprimé 'loss' pour accélérer
                }
            },
            "SGDClassifier": {  # Remplace RandomForest - beaucoup plus rapide
                'model': SGDClassifier(random_state=42, n_jobs=-1, max_iter=1000),
                'params': {
                    'alpha': [0.0001, 0.001, 0.01],
                    'loss': ['hinge', 'log_loss']
                }
            }
        }
        
        # Option: inclure RandomForest seulement si explicitement demandé
        if self.config.get('include_random_forest', False):
            models["RandomForest"] = {
                'model': RandomForestClassifier(
                    random_state=42, 
                    n_jobs=-1,
                    n_estimators=50,  # Réduit de 100-200 à 50
                    max_depth=10,     # Limite la profondeur
                    min_samples_split=10,  # Augmente pour accélérer
                    min_samples_leaf=5,    # Augmente pour accélérer
                    max_features='sqrt'    # Réduit le nombre de features
                ),
                'params': {
                    'n_estimators': [50, 100],  # Réduit les options
                    'max_depth': [10, 15]       # Limite les options
                }
            }
        
        return models
    
    def train_incremental_models(self, X_train_vec, y_train, X_val_vec, y_val, batch_size=10000):
        """Entraînement incrémental pour très gros datasets"""
        from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
        from sklearn.naive_bayes import MultinomialNB
        
        logger.info("🔄 Entraînement incrémental pour gros dataset...")
        
        # Modèles supportant l'apprentissage incrémental
        incremental_models = {
            "SGDClassifier": SGDClassifier(random_state=42, max_iter=1000),
            "PassiveAggressive": PassiveAggressiveClassifier(random_state=42, max_iter=1000),
            "MultinomialNB_Incremental": MultinomialNB()
        }
        
        best_score = 0
        n_samples = X_train_vec.shape[0]
        
        for name, model in incremental_models.items():
            logger.info(f"\n--- Entraînement incrémental de {name} ---")
            start_time = datetime.now()
            
            try:
                # Entraînement par batches
                for i in range(0, n_samples, batch_size):
                    end_idx = min(i + batch_size, n_samples)
                    X_batch = X_train_vec[i:end_idx]
                    y_batch = y_train.iloc[i:end_idx] if hasattr(y_train, 'iloc') else y_train[i:end_idx]
                    
                    if i == 0:
                        # Premier batch : fit
                        model.fit(X_batch, y_batch)
                    else:
                        # Batches suivants : partial_fit
                        if hasattr(model, 'partial_fit'):
                            model.partial_fit(X_batch, y_batch)
                        else:
                            # Pour MultinomialNB qui n'a pas partial_fit standard
                            continue
                    
                    if i % (batch_size * 5) == 0:  # Log tous les 5 batches
                        logger.info(f"Traité {min(end_idx, n_samples)}/{n_samples} échantillons")
                
                # Évaluation
                y_val_pred = model.predict(X_val_vec)
                accuracy = accuracy_score(y_val, y_val_pred)
                f1 = f1_score(y_val, y_val_pred, average='weighted')
                training_time = (datetime.now() - start_time).total_seconds()
                
                self.results[name] = {
                    'model': model,
                    'best_params': {},  # Pas d'optimisation d'hyperparamètres en incrémental
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'training_time': training_time,
                    'predictions': y_val_pred
                }
                
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
                logger.info(f"Temps d'entraînement: {training_time:.2f}s")
                
                if f1 > best_score:
                    best_score = f1
                    self.best_model = model
                    self.best_model_name = name
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement incrémental de {name}: {e}")
                continue
        
        logger.info(f"\n✅ Meilleur modèle incrémental: {self.best_model_name} (F1-Score: {best_score:.4f})")

    def train_and_evaluate_models(self, X_train_vec, X_val_vec, y_train, y_val):
        """Entraîner et évaluer tous les modèles"""
        
        # Vérifier si on doit utiliser l'apprentissage incrémental
        if self.config.get('use_incremental_learning', False) or X_train_vec.shape[0] > 500000:
            logger.info("Dataset volumineux détecté, utilisation de l'apprentissage incrémental...")
            self.train_incremental_models(X_train_vec, y_train, X_val_vec, y_val)
            return
        
        logger.info("🚀 Entraînement des modèles...")
        
        models = self.initialize_models()
        best_score = 0
        
        for name, model_config in models.items():
            logger.info(f"\n--- Entraînement de {name} ---")
            start_time = datetime.now()
            
            try:
                # Recherche par grille pour les hyperparamètres
                grid_search = GridSearchCV(
                    model_config['model'],
                    model_config['params'],
                    cv=self.config['cv_folds'],  # Utilise la config
                    scoring='f1_weighted',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train_vec, y_train)
                best_model = grid_search.best_estimator_
                
                # Prédictions sur validation
                y_val_pred = best_model.predict(X_val_vec)
                
                # Métriques
                accuracy = accuracy_score(y_val, y_val_pred)
                f1 = f1_score(y_val, y_val_pred, average='weighted')
                
                # Temps d'entraînement
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Stocker les résultats
                self.results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'training_time': training_time,
                    'predictions': y_val_pred
                }
                
                logger.info(f"Meilleurs paramètres: {grid_search.best_params_}")
                logger.info(f"Accuracy: {accuracy:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")
                logger.info(f"Temps d'entraînement: {training_time:.2f}s")
                
                # Sélectionner le meilleur modèle
                if f1 > best_score:
                    best_score = f1
                    self.best_model = best_model
                    self.best_model_name = name
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de {name}: {e}")
                continue
        
        logger.info(f"\n✅ Meilleur modèle: {self.best_model_name} (F1-Score: {best_score:.4f})")
    
    def evaluate_final_model(self, X_test_vec, y_test):
        """Évaluer le meilleur modèle sur l'ensemble de test"""
        logger.info(f"\n🧪 Évaluation finale du modèle {self.best_model_name}...")
        
        y_test_pred = self.best_model.predict(X_test_vec)
        
        # Métriques finales
        accuracy = accuracy_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        logger.info(f"Accuracy finale: {accuracy:.4f}")
        logger.info(f"F1-Score final: {f1:.4f}")
        
        # Rapport détaillé
        logger.info("\n📊 Rapport de classification:")
        print(classification_report(y_test, y_test_pred))
        
        return y_test_pred, accuracy, f1
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Matrice de confusion"):
        """Créer et sauvegarder la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title(title)
        plt.ylabel('Vrai label')
        plt.xlabel('Prédiction')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{self.best_model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self):
        """Comparer les performances des modèles"""
        if not self.results:
            return
            
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        f1_scores = [self.results[model]['f1_score'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy
        ax1.bar(models, accuracies, color='skyblue')
        ax1.set_title('Accuracy par modèle')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # F1-Score
        ax2.bar(models, f1_scores, color='lightcoral')
        ax2.set_title('F1-Score par modèle')
        ax2.set_ylabel('F1-Score')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models_and_results(self):
        """Sauvegarder les modèles et résultats"""
        logger.info("💾 Sauvegarde des modèles et résultats...")
        
        # Sauvegarder le meilleur modèle et le vectoriseur
        joblib.dump(self.best_model, f'models/best_model_{self.best_model_name}.pkl')
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        
        # Sauvegarder les résultats
        results_summary = {
            'best_model': self.best_model_name,
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': {
                name: {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'best_params': result['best_params']
                }
                for name, result in self.results.items()
            }
        }
        
        import json
        with open('results/training_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info("✅ Sauvegarde terminée")
    
    def predict_sentiment(self, texts):
        """Prédire le sentiment de nouveaux textes"""
        if self.best_model is None or self.vectorizer is None:
            raise ValueError("Modèle non entraîné. Veuillez d'abord entraîner le modèle.")
        
        # Vectoriser les textes
        texts_vec = self.vectorizer.transform(texts)
        
        # Prédire
        predictions = self.best_model.predict(texts_vec)
        probabilities = self.best_model.predict_proba(texts_vec)
        
        return predictions, probabilities
    
    def run_full_pipeline(self, data_path: str):
        """Exécuter le pipeline complet d'analyse de sentiment"""
        try:
            # Créer les répertoires
            self.create_directories()
            
            # Charger et prétraiter les données
            df = self.load_and_preprocess_data(data_path)
            
            # Diviser les données
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df)
            
            # Vectoriser
            X_train_vec, X_val_vec, X_test_vec = self.vectorize_text(X_train, X_val, X_test)
            
            # Entraîner les modèles
            self.train_and_evaluate_models(X_train_vec, X_val_vec, y_train, y_val)
            
            # Évaluation finale
            y_test_pred, final_accuracy, final_f1 = self.evaluate_final_model(X_test_vec, y_test)
            
            # Visualisations
            self.plot_confusion_matrix(y_test, y_test_pred)
            self.plot_model_comparison()
            
            # Sauvegarder
            self.save_models_and_results()
            
            logger.info(f"\n🎉 Pipeline terminé avec succès!")
            logger.info(f"Meilleur modèle: {self.best_model_name}")
            logger.info(f"Performance finale: Accuracy={final_accuracy:.4f}, F1={final_f1:.4f}")
            
            return self.best_model, self.vectorizer
            
        except Exception as e:
            logger.error(f"Erreur dans le pipeline: {e}")
            raise

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration personnalisée (optionnelle)
    config = {
        'test_size': 0.2,
        'val_size': 0.2,
        'random_state': 42,
        'max_features': 15000,
        'cv_folds': 5
    }
    
    # Initialiser l'analyseur
    analyzer = SentimentAnalyzer(config)
    
    # Exécuter le pipeline complet
    try:
        model, vectorizer = analyzer.run_full_pipeline("data/training.1600000.processed.noemoticon.csv")
        
        # Test de prédiction sur de nouveaux textes
        test_texts = [
            "I love this product! It's amazing!",
            "This is terrible, I hate it",
            "It's okay, nothing special"
        ]
        
        predictions, probabilities = analyzer.predict_sentiment(test_texts)
        
        print("\n🔮 Prédictions sur de nouveaux textes:")
        for text, pred, prob in zip(test_texts, predictions, probabilities):
            print(f"Texte: '{text}'")
            print(f"Sentiment: {pred}")
            print(f"Probabilités: {dict(zip(['negative', 'neutral', 'positive'], prob))}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Erreur: {e}")