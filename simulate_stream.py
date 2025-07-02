import time
import pandas as pd
import csv
import logging
import argparse
from pathlib import Path
from typing import Optional

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TweetStreamer:
    """Simulateur de streaming de tweets depuis un dataset CSV."""
    
    def __init__(self, dataset_path: str, stream_file: str, delay: float = 2.0):
        self.dataset_path = Path(dataset_path)
        self.stream_file = Path(stream_file)
        self.delay = delay
        self.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        
    def validate_files(self) -> bool:
        """Valide l'existence des fichiers nécessaires."""
        if not self.dataset_path.exists():
            logger.error(f"Le fichier dataset n'existe pas: {self.dataset_path}")
            return False
        return True
    
    def load_dataset(self) -> Optional[pd.DataFrame]:
        """Charge le dataset avec gestion d'erreurs."""
        try:
            df = pd.read_csv(
                self.dataset_path, 
                encoding='latin-1', 
                header=None,
                names=self.columns
            )
            logger.info(f"Dataset chargé: {len(df)} tweets")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement du dataset: {e}")
            return None
    
    def initialize_stream_file(self) -> bool:
        """Initialise le fichier de streaming avec les en-têtes."""
        try:
            # Créer le répertoire parent si nécessaire
            self.stream_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.stream_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
            
            logger.info(f"Fichier de streaming initialisé: {self.stream_file}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du fichier: {e}")
            return False
    
    def write_tweet_batch(self, tweets: pd.DataFrame) -> bool:
        """Écrit un lot de tweets dans le fichier de streaming."""
        try:
            with open(self.stream_file, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                for _, row in tweets.iterrows():
                    writer.writerow([
                        row['sentiment'],
                        row['id'],
                        row['date'],
                        row['query'],
                        row['user'],
                        row['text']
                    ])
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'écriture: {e}")
            return False
    
    def stream_tweets(self, batch_size: int = 1, max_tweets: Optional[int] = None) -> None:
        """Simule le streaming de tweets."""
        if not self.validate_files():
            return
        
        df = self.load_dataset()
        if df is None:
            return
        
        if not self.initialize_stream_file():
            return
        
        # Limiter le nombre de tweets si spécifié
        if max_tweets:
            df = df.head(max_tweets)
        
        logger.info(f"Démarrage du streaming de {len(df)} tweets...")
        logger.info(f"Taille des lots: {batch_size}, Délai: {self.delay}s")
        
        try:
            total_tweets = 0
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                if self.write_tweet_batch(batch):
                    total_tweets += len(batch)
                    logger.info(f"Lot {i//batch_size + 1}: {len(batch)} tweets ajoutés "
                              f"(Total: {total_tweets}/{len(df)})")
                else:
                    logger.error(f"Échec de l'écriture du lot {i//batch_size + 1}")
                    break
                
                # Pause entre les lots
                if i + batch_size < len(df):  # Pas de pause après le dernier lot
                    time.sleep(self.delay)
            
            logger.info(f"Streaming terminé! {total_tweets} tweets traités.")
            
        except KeyboardInterrupt:
            logger.info(f"Streaming interrompu par l'utilisateur. {total_tweets} tweets traités.")
        except Exception as e:
            logger.error(f"Erreur pendant le streaming: {e}")

def main():
    """Fonction principale avec arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description="Simulateur de streaming de tweets")
    parser.add_argument(
        '--dataset', 
        default="data/training.1600000.processed.noemoticon.csv",
        help="Chemin vers le dataset de tweets"
    )
    parser.add_argument(
        '--output', 
        default="data/stream_tweets.csv",
        help="Fichier de sortie pour le streaming"
    )
    parser.add_argument(
        '--delay', 
        type=float, 
        default=2.0,
        help="Délai entre les lots en secondes"
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=1,
        help="Nombre de tweets par lot"
    )
    parser.add_argument(
        '--max-tweets', 
        type=int,
        help="Nombre maximum de tweets à traiter"
    )
    
    args = parser.parse_args()
    
    # Créer et lancer le streamer
    streamer = TweetStreamer(
        dataset_path=args.dataset,
        stream_file=args.output,
        delay=args.delay
    )
    
    streamer.stream_tweets(
        batch_size=args.batch_size,
        max_tweets=args.max_tweets
    )

if __name__ == "__main__":
    main()