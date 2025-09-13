# %% [markdown]
"""
# Notebook SOTA : Système de Recommandation Hybride My Content

Ce notebook implémente un système de recommandation hybride combinant:
1. **Content-Based Filtering** : Utilise les embeddings d'articles + similarité cosinus
2. **Collaborative Filtering** : Utilise la librairie Surprise avec ratings implicites
3. **Système Hybride** : Combine intelligemment les deux approches

**Objectif** : Recommander 5 articles pertinents pour chaque utilisateur
**Architecture** : Compatible GCP Cloud Functions + Next.js frontend
"""

# %% [python]
# /// script
# dependencies = [
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "scikit-surprise>=1.1.3",
#     "matplotlib>=3.7.0",
#     "seaborn>=0.12.0",
#     "pickle-mixin>=1.0.2",
#     "jupytext>=1.15.0"
# ]
# ///

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Surprise library imports (obligatoire selon consignes)
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

print("📚 Imports réussis - Démarrage du notebook SOTA My Content")

# %% [markdown]
"""
## 1. Configuration et Chargement des Données
"""

# %% [python]
def setup_data_paths() -> dict:
    """Configure les chemins vers les fichiers de données."""
    base_path = Path("../data/news-portal-user-interactions-by-globocom")
    
    paths = {
        'articles_metadata': base_path / "articles_metadata.csv",
        'articles_embeddings': base_path / "articles_embeddings.pickle", 
        'clicks_dir': base_path / "clicks"
    }
    
    # Vérification existence
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"❌ Fichier manquant: {path}")
    
    print(f"✅ Tous les fichiers de données trouvés")
    return paths

def load_articles_metadata(path: Path) -> pd.DataFrame:
    """Charge les métadonnées des articles."""
    df = pd.read_csv(path)
    print(f"📄 Articles metadata: {df.shape[0]} articles, {df.shape[1]} colonnes")
    return df

def load_articles_embeddings(path: Path) -> np.ndarray:
    """Charge les embeddings précalculés des articles."""
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"🧮 Embeddings: forme {embeddings.shape}")
    return embeddings

# Configuration des chemins
data_paths = setup_data_paths()
articles_df = load_articles_metadata(data_paths['articles_metadata'])
embeddings_matrix = load_articles_embeddings(data_paths['articles_embeddings'])

# %% [python]
def load_clicks_data(clicks_dir: Path, sample_files: int = 5) -> pd.DataFrame:
    """Charge les données de clics (échantillon pour développement)."""
    click_files = sorted(list(clicks_dir.glob("clicks_hour_*.csv")))
    
    if not click_files:
        raise FileNotFoundError("❌ Aucun fichier de clics trouvé")
    
    # Limiter pour développement
    selected_files = click_files[:sample_files]
    
    dfs = []
    for file_path in selected_files:
        df_chunk = pd.read_csv(file_path)
        dfs.append(df_chunk)
        print(f"📊 Chargé {file_path.name}: {df_chunk.shape[0]} interactions")
    
    clicks_df = pd.concat(dfs, ignore_index=True)
    print(f"🎯 Total interactions chargées: {clicks_df.shape[0]}")
    return clicks_df

def explore_basic_stats(clicks_df: pd.DataFrame, articles_df: pd.DataFrame) -> dict:
    """Calcule les statistiques de base du dataset."""
    stats = {
        'nb_users': clicks_df['user_id'].nunique(),
        'nb_articles': clicks_df['click_article_id'].nunique(),
        'nb_sessions': clicks_df['session_id'].nunique(),
        'nb_interactions': len(clicks_df),
        'articles_metadata_count': len(articles_df)
    }
    
    print("\n📈 STATISTIQUES DU DATASET:")
    for key, value in stats.items():
        print(f"  {key}: {value:,}")
    
    return stats

# Chargement des données de clics
clicks_df = load_clicks_data(data_paths['clicks_dir'])
dataset_stats = explore_basic_stats(clicks_df, articles_df)

# %% [markdown]
"""
## 2. Content-Based Filtering Implementation

Utilise les embeddings précalculés pour calculer la similarité entre articles.
Stratégies implémentées:
- Dernier article cliqué par l'utilisateur
- Moyenne pondérée des articles cliqués
- Gestion des nouveaux utilisateurs
"""

# %% [python]
def create_user_article_matrix(clicks_df: pd.DataFrame) -> pd.DataFrame:
    """Crée une matrice utilisateur-article avec comptage des clics."""
    user_articles = clicks_df.groupby(['user_id', 'click_article_id']).size().reset_index()
    user_articles.columns = ['user_id', 'article_id', 'click_count']
    
    print(f"👥 Matrice créée: {user_articles['user_id'].nunique()} users, " +
          f"{user_articles['article_id'].nunique()} articles")
    return user_articles

def get_user_last_article(clicks_df: pd.DataFrame, user_id: int) -> int:
    """Retourne le dernier article cliqué par un utilisateur."""
    user_clicks = clicks_df[clicks_df['user_id'] == user_id]
    
    if user_clicks.empty:
        return None
        
    last_click = user_clicks.loc[user_clicks['click_timestamp'].idxmax()]
    return last_click['click_article_id']

def compute_cosine_similarities(embeddings: np.ndarray, article_idx: int) -> np.ndarray:
    """Calcule la similarité cosinus entre un article et tous les autres."""
    if article_idx >= len(embeddings):
        raise ValueError(f"Index article {article_idx} hors limites")
    
    article_embedding = embeddings[article_idx].reshape(1, -1)
    similarities = cosine_similarity(article_embedding, embeddings)[0]
    return similarities

def content_based_recommendations(user_id: int, embeddings: np.ndarray, 
                                clicks_df: pd.DataFrame, n_recommendations: int = 5) -> list:
    """Génère des recommandations content-based pour un utilisateur."""
    # Stratégie: utiliser le dernier article cliqué
    last_article_id = get_user_last_article(clicks_df, user_id)
    
    if last_article_id is None:
        return []  # Utilisateur inconnu
    
    # Calculer similarités
    similarities = compute_cosine_similarities(embeddings, last_article_id)
    
    # Obtenir les articles déjà vus par l'utilisateur
    user_articles = set(clicks_df[clicks_df['user_id'] == user_id]['click_article_id'])
    
    # Trier par similarité décroissante, exclure articles déjà vus
    article_scores = [(i, score) for i, score in enumerate(similarities) 
                     if i not in user_articles and score > 0]
    article_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Retourner top N recommandations
    recommendations = [article_id for article_id, _ in article_scores[:n_recommendations]]
    return recommendations

# Test du système content-based
user_articles_matrix = create_user_article_matrix(clicks_df)
test_user_id = clicks_df['user_id'].iloc[0]
cb_recommendations = content_based_recommendations(test_user_id, embeddings_matrix, clicks_df)
print(f"🎯 Content-Based pour user {test_user_id}: {cb_recommendations}")

# %% [markdown]
"""
## 3. Collaborative Filtering avec Surprise Library

Implémentation obligatoire selon les consignes du projet.
Utilise des ratings implicites basés sur le nombre de clics.
"""

# %% [python]
def create_implicit_ratings(clicks_df: pd.DataFrame) -> pd.DataFrame:
    """Crée des ratings implicites basés sur les clics."""
    # Compter les clics par utilisateur-article
    ratings = clicks_df.groupby(['user_id', 'click_article_id']).agg({
        'session_size': 'mean',  # Taille moyenne des sessions
        'click_timestamp': 'count'  # Nombre de clics
    }).reset_index()
    
    ratings.columns = ['user_id', 'article_id', 'avg_session_size', 'click_count']
    
    # Créer rating implicite : combinaison clics + session size
    ratings['implicit_rating'] = (
        ratings['click_count'] * 2 +  # Poids des clics répétés
        ratings['avg_session_size'] * 0.1  # Poids de l'engagement session
    )
    
    # Normaliser entre 1 et 5 pour Surprise
    max_rating = ratings['implicit_rating'].max()
    ratings['rating'] = 1 + (ratings['implicit_rating'] / max_rating) * 4
    
    print(f"⭐ Ratings créés: {len(ratings)} interactions, " +
          f"rating moyen: {ratings['rating'].mean():.2f}")
    
    return ratings[['user_id', 'article_id', 'rating']]

def train_surprise_model(ratings_df: pd.DataFrame, algorithm='SVD') -> tuple:
    """Entraîne un modèle Surprise sur les ratings implicites."""
    # Préparer les données pour Surprise
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df, reader)
    
    # Split train/test
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Sélection de l'algorithme
    if algorithm == 'SVD':
        model = SVD(random_state=42, n_factors=50, n_epochs=20)
    elif algorithm == 'NMF':
        model = NMF(random_state=42, n_factors=50, n_epochs=20)
    else:
        raise ValueError(f"Algorithme {algorithm} non supporté")
    
    # Entraînement
    model.fit(trainset)
    
    # Évaluation
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    
    print(f"🤖 Modèle {algorithm} entraîné - RMSE: {rmse:.3f}")
    return model, trainset

def collaborative_recommendations(user_id: int, model, trainset, 
                                n_recommendations: int = 5) -> list:
    """Génère des recommandations collaborative filtering."""
    try:
        # Vérifier si l'utilisateur existe dans le trainset
        inner_user_id = trainset.to_inner_uid(user_id)
    except ValueError:
        # Utilisateur inconnu
        return []
    
    # Obtenir tous les articles du trainset (inner IDs)
    all_inner_items = set(range(trainset.n_items))
    
    # Articles déjà vus par l'utilisateur (inner IDs)
    user_items = set([item for (item, _) in trainset.ur[inner_user_id]])
    
    # Articles non vus (inner IDs)
    unseen_inner_items = all_inner_items - user_items
    
    # Prédire les ratings pour les articles non vus
    predictions = []
    for inner_item_id in unseen_inner_items:
        # Convertir inner ID vers raw ID pour la prédiction
        raw_item_id = trainset.to_raw_iid(inner_item_id)
        pred = model.predict(user_id, raw_item_id)
        predictions.append((raw_item_id, pred.est))
    
    # Trier par rating prédit décroissant
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Retourner top N recommandations (raw IDs)
    recommendations = [article_id for article_id, _ in predictions[:n_recommendations]]
    return recommendations

# Entraînement du modèle collaboratif
ratings_df = create_implicit_ratings(clicks_df)
surprise_model, trainset = train_surprise_model(ratings_df, algorithm='SVD')

# Test collaborative filtering
cf_recommendations = collaborative_recommendations(test_user_id, surprise_model, trainset)
print(f"🤝 Collaborative pour user {test_user_id}: {cf_recommendations}")

# %% [markdown]
"""
## 4. Système Hybride

Combine les approches Content-Based et Collaborative Filtering.
Différentes stratégies de combinaison selon les performances individuelles.
"""

# %% [python]
def hybrid_recommendations(user_id: int, embeddings: np.ndarray, clicks_df: pd.DataFrame,
                          surprise_model, trainset, n_recommendations: int = 5,
                          weight_cb: float = 0.6, weight_cf: float = 0.4) -> list:
    """
    Génère des recommandations hybrides combinant CB et CF.
    
    Args:
        weight_cb: Poids du Content-Based (0.0 à 1.0)
        weight_cf: Poids du Collaborative Filtering (0.0 à 1.0)
    """
    # Obtenir recommandations des deux systèmes
    cb_recs = content_based_recommendations(user_id, embeddings, clicks_df, n_recommendations*2)
    cf_recs = collaborative_recommendations(user_id, surprise_model, trainset, n_recommendations*2)
    
    # Scoring hybride
    hybrid_scores = {}
    
    # Scorer les recommandations Content-Based
    for i, article_id in enumerate(cb_recs):
        score = weight_cb * (1.0 - i / len(cb_recs))  # Score décroissant
        hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
    
    # Scorer les recommandations Collaborative
    for i, article_id in enumerate(cf_recs):
        score = weight_cf * (1.0 - i / len(cf_recs))  # Score décroissant  
        hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
    
    # Trier par score hybride décroissant
    sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Retourner top N
    final_recommendations = [article_id for article_id, _ in sorted_recommendations[:n_recommendations]]
    return final_recommendations

def evaluate_recommendation_coverage(recommendations: list, total_articles: int) -> dict:
    """Évalue la couverture et diversité des recommandations."""
    metrics = {
        'nb_recommendations': len(recommendations),
        'coverage_rate': len(set(recommendations)) / total_articles if total_articles > 0 else 0,
        'diversity_rate': len(set(recommendations)) / len(recommendations) if recommendations else 0
    }
    return metrics

# Test du système hybride
hybrid_recs = hybrid_recommendations(test_user_id, embeddings_matrix, clicks_df, 
                                   surprise_model, trainset)
print(f"🔗 Hybride pour user {test_user_id}: {hybrid_recs}")

# Évaluation
total_articles = clicks_df['click_article_id'].nunique()
coverage_metrics = evaluate_recommendation_coverage(hybrid_recs, total_articles)
print(f"📊 Métriques couverture: {coverage_metrics}")

# %% [markdown]
"""
## 5. Sauvegarde des Modèles et Configuration

Préparation pour déploiement GCP Cloud Functions.
"""

# %% [python]
def save_models_for_deployment(surprise_model, embeddings: np.ndarray, 
                              articles_df: pd.DataFrame, output_dir: str = "../scripts/models") -> dict:
    """Sauvegarde les modèles entraînés pour déploiement."""
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = {}
    
    # Sauvegarder le modèle Surprise
    with open(f"{output_dir}/surprise_model.pkl", 'wb') as f:
        pickle.dump(surprise_model, f)
    saved_files['surprise_model'] = f"{output_dir}/surprise_model.pkl"
    
    # Sauvegarder les embeddings (possibilité PCA si trop volumineux)
    if embeddings.shape[1] > 1000:  # Si > 1000 dimensions, appliquer PCA
        pca = PCA(n_components=min(500, embeddings.shape[1]))
        embeddings_reduced = pca.fit_transform(embeddings)
        
        with open(f"{output_dir}/embeddings_pca.pkl", 'wb') as f:
            pickle.dump({'embeddings': embeddings_reduced, 'pca': pca}, f)
        saved_files['embeddings'] = f"{output_dir}/embeddings_pca.pkl"
        
        print(f"🗜️ Embeddings réduits: {embeddings.shape} → {embeddings_reduced.shape}")
    else:
        with open(f"{output_dir}/embeddings.pkl", 'wb') as f:
            pickle.dump(embeddings, f)
        saved_files['embeddings'] = f"{output_dir}/embeddings.pkl"
    
    # Sauvegarder métadonnées articles essentielles
    articles_meta = articles_df[['article_id', 'category_id', 'words_count']].copy()
    articles_meta.to_pickle(f"{output_dir}/articles_metadata.pkl")
    saved_files['articles_metadata'] = f"{output_dir}/articles_metadata.pkl"
    
    # Configuration système
    config = {
        'model_version': '1.0.0',
        'n_recommendations': 5,
        'hybrid_weights': {'content_based': 0.6, 'collaborative': 0.4},
        'embeddings_shape': embeddings.shape,
        'total_articles': len(articles_df),
        'training_stats': dataset_stats
    }
    
    with open(f"{output_dir}/config.pkl", 'wb') as f:
        pickle.dump(config, f)
    saved_files['config'] = f"{output_dir}/config.pkl"
    
    print("💾 Modèles sauvegardés avec succès:")
    for name, path in saved_files.items():
        print(f"  {name}: {path}")
    
    return saved_files

# Sauvegarde pour déploiement
saved_model_paths = save_models_for_deployment(surprise_model, embeddings_matrix, articles_df)

# %% [markdown]
"""
## 6. Validation et Tests Finaux

Tests de validation du système complet avant mise en production.
"""

# %% [python]
def test_recommendation_system(sample_users: list, n_tests: int = 5) -> pd.DataFrame:
    """Teste le système sur un échantillon d'utilisateurs."""
    results = []
    
    for user_id in sample_users[:n_tests]:
        try:
            # Test des 3 approches
            cb_recs = content_based_recommendations(user_id, embeddings_matrix, clicks_df)
            cf_recs = collaborative_recommendations(user_id, surprise_model, trainset)  
            hybrid_recs = hybrid_recommendations(user_id, embeddings_matrix, clicks_df,
                                               surprise_model, trainset)
            
            results.append({
                'user_id': user_id,
                'content_based': len(cb_recs),
                'collaborative': len(cf_recs),
                'hybrid': len(hybrid_recs),
                'cb_recs': cb_recs[:3],  # 3 premières recommandations
                'cf_recs': cf_recs[:3],
                'hybrid_recs': hybrid_recs[:3]
            })
            
        except Exception as e:
            print(f"⚠️ Erreur pour user {user_id}: {e}")
    
    return pd.DataFrame(results)

# Tests de validation
sample_user_ids = clicks_df['user_id'].unique()[:10].tolist()
test_results = test_recommendation_system(sample_user_ids)

print("\n🧪 RÉSULTATS TESTS SYSTÈME:")
print(test_results[['user_id', 'content_based', 'collaborative', 'hybrid']])

print(f"\n✅ NOTEBOOK TERMINÉ AVEC SUCCÈS")
print(f"📈 Statistiques finales:")
print(f"   - {len(clicks_df)} interactions traitées")
print(f"   - {clicks_df['user_id'].nunique()} utilisateurs")  
print(f"   - {clicks_df['click_article_id'].nunique()} articles")
print(f"   - Modèles prêts pour déploiement GCP")

# %% [markdown]
"""
## 🎯 Prochaines Étapes

1. **API Development**: Créer l'API FastAPI dans `scripts/main_api.py`
2. **Tests API**: Valider les endpoints de recommandation  
3. **Déploiement GCP**: Cloud Functions + Cloud Storage
4. **Frontend Next.js**: Interface utilisateur avec Framer Motion
5. **Monitoring**: Suivi des performances et métriques

**Commande pour lancer l'API:**
```bash
uv run uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```
"""