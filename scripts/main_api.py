# /// script
# dependencies = [
#     "fastapi>=0.104.0",
#     "uvicorn[standard]>=0.24.0",
#     "pandas>=2.0.0",
#     "numpy>=1.24.0",
#     "scikit-learn>=1.3.0",
#     "scikit-surprise>=1.1.3",
#     "pydantic>=2.5.0",
#     "python-multipart>=0.0.6"
# ]
# requires-python = ">=3.9"
# ///

"""
API Principale - Système de Recommandation My Content

API FastAPI pour servir les recommandations d'articles hybrides.
Compatible avec la commande: uv run uvicorn main_api:app --reload --host 127.0.0.1 --port 8000

Endpoints:
- GET /recommend/{user_id}: Recommande 5 articles pour un utilisateur
- GET /health: Vérification du statut de l'API
- GET /stats: Statistiques du système
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modèles pydantic pour les réponses API
class RecommendationResponse(BaseModel):
    user_id: int = Field(..., description="ID de l'utilisateur")
    recommendations: List[int] = Field(..., description="Liste des 5 articles recommandés")
    method: str = Field(..., description="Méthode utilisée (hybrid/content-based/collaborative)")
    confidence: float = Field(..., description="Score de confiance (0.0 à 1.0)")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Statut de l'API")
    models_loaded: bool = Field(..., description="Modèles chargés avec succès")
    version: str = Field(..., description="Version de l'API")

class StatsResponse(BaseModel):
    total_users: int
    total_articles: int
    model_version: str
    api_version: str

class ArticleMetadata(BaseModel):
    article_id: int
    category_id: Optional[int] = None
    words_count: Optional[int] = None
    title: str
    category: str
    description: str

class ArticlesMetadataResponse(BaseModel):
    articles: List[ArticleMetadata]

# Instance FastAPI
app = FastAPI(
    title="My Content Recommendation API",
    description="API de recommandation hybride d'articles",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour les modèles
models_data = {}
is_loaded = False

def load_models() -> bool:
    """Charge tous les modèles nécessaires."""
    global models_data, is_loaded
    
    try:
        models_dir = Path("models")
        
        # Vérifier existence du dossier models
        if not models_dir.exists():
            logger.error(f"❌ Dossier models introuvable: {models_dir}")
            return False
        
        # Charger modèle Surprise
        with open(models_dir / "surprise_model.pkl", 'rb') as f:
            models_data['surprise_model'] = pickle.load(f)
        
        # Charger embeddings
        embeddings_file = models_dir / "embeddings.pkl"
        pca_file = models_dir / "embeddings_pca.pkl"
        
        if pca_file.exists():
            with open(pca_file, 'rb') as f:
                emb_data = pickle.load(f)
                models_data['embeddings'] = emb_data['embeddings']
                models_data['pca'] = emb_data.get('pca')
        elif embeddings_file.exists():
            with open(embeddings_file, 'rb') as f:
                models_data['embeddings'] = pickle.load(f)
        else:
            raise FileNotFoundError("Fichier embeddings introuvable")
        
        # Charger métadonnées articles
        with open(models_dir / "articles_metadata.pkl", 'rb') as f:
            articles_df = pickle.load(f)
            models_data['articles_metadata'] = articles_df
            # Créer un dictionnaire pour accès rapide
            models_data['articles_dict'] = articles_df.set_index('article_id').to_dict('index')
        
        # Charger configuration
        with open(models_dir / "config.pkl", 'rb') as f:
            models_data['config'] = pickle.load(f)
        
        logger.info("✅ Tous les modèles chargés avec succès")
        is_loaded = True
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles: {e}")
        return False

def content_based_recommendations(user_id: int, n_recs: int = 5) -> List[int]:
    """Génère des recommandations content-based utilisant les embeddings."""
    if not is_loaded:
        return []
    
    try:
        embeddings = models_data['embeddings']
        
        # Simuler le dernier article pour l'utilisateur (à améliorer avec vraies données)
        # En production, il faudrait une base de données des interactions utilisateur
        np.random.seed(user_id)
        last_article_idx = np.random.randint(0, len(embeddings))
        
        # Calculer similarité cosinus
        article_embedding = embeddings[last_article_idx].reshape(1, -1)
        similarities = cosine_similarity(article_embedding, embeddings)[0]
        
        # Obtenir top articles similaires (exclure l'article lui-même)
        similar_indices = np.argsort(similarities)[::-1][1:n_recs+1]
        
        return similar_indices.tolist()
        
    except Exception as e:
        logger.error(f"❌ Erreur content-based pour user {user_id}: {e}")
        return []

def collaborative_recommendations(user_id: int, n_recs: int = 5) -> List[int]:
    """Génère des recommandations collaborative filtering avec Surprise."""
    if not is_loaded:
        return []
    
    try:
        surprise_model = models_data['surprise_model']
        config = models_data['config']
        total_articles = config['total_articles']
        
        # Prédire des ratings pour tous les articles pour cet utilisateur
        predictions = []
        for article_id in range(min(1000, total_articles)):  # Limiter pour performance
            try:
                pred = surprise_model.predict(user_id, article_id)
                predictions.append((article_id, pred.est))
            except:
                continue
        
        # Trier par rating prédit décroissant
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner top N articles
        recommendations = [article_id for article_id, _ in predictions[:n_recs]]
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Erreur collaborative pour user {user_id}: {e}")
        return []

def get_hybrid_recommendations(user_id: int, n_recs: int = 5) -> Dict[str, Any]:
    """Génère des recommandations hybrides combinant CB et CF."""
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    
    try:
        config = models_data['config']
        hybrid_weights = config['hybrid_weights']
        weight_cb = hybrid_weights['content_based']
        weight_cf = hybrid_weights['collaborative']
        
        # Obtenir recommandations des deux systèmes
        cb_recs = content_based_recommendations(user_id, n_recs * 2)
        cf_recs = collaborative_recommendations(user_id, n_recs * 2)
        
        # Scoring hybride
        hybrid_scores = {}
        
        # Scorer les recommandations Content-Based
        for i, article_id in enumerate(cb_recs):
            score = weight_cb * (1.0 - i / len(cb_recs)) if cb_recs else 0
            hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
        
        # Scorer les recommandations Collaborative
        for i, article_id in enumerate(cf_recs):
            score = weight_cf * (1.0 - i / len(cf_recs)) if cf_recs else 0
            hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
        
        # Si pas de recommandations hybrides, fallback sur recommandations aléatoires
        if not hybrid_scores:
            logger.warning(f"Aucune recommandation hybride pour user {user_id}, fallback aléatoire")
            np.random.seed(user_id)
            total_articles = config['total_articles']
            recommendations = np.random.choice(range(total_articles), 
                                             size=min(n_recs, total_articles), 
                                             replace=False).tolist()
            method = 'random_fallback'
            confidence = 0.3
        else:
            # Trier par score hybride décroissant
            sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [article_id for article_id, _ in sorted_recs[:n_recs]]
            method = 'hybrid'
            confidence = min(0.95, 0.6 + len(hybrid_scores) * 0.05)  # Confiance basée sur nombre de recs
        
        return {
            'recommendations': recommendations,
            'method': method,
            'confidence': confidence,
            'metadata': {
                'cb_count': len(cb_recs),
                'cf_count': len(cf_recs),
                'hybrid_candidates': len(hybrid_scores),
                'weights': {'content_based': weight_cb, 'collaborative': weight_cf},
                'algorithm_version': config['model_version']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur génération recommandations hybrides pour user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne de recommandation")

@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage de l'API."""
    logger.info("🚀 Démarrage API My Content")
    success = load_models()
    
    if not success:
        logger.error("❌ Impossible de charger les modèles")
    else:
        logger.info("✅ API prête à servir les recommandations")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de vérification de santé de l'API."""
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        models_loaded=is_loaded,
        version="1.0.0"
    )

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def recommend_articles(
    user_id: int,
    n_recommendations: int = Query(5, ge=1, le=10, description="Nombre de recommandations (1-10)")
):
    """
    Recommande des articles pour un utilisateur donné.
    
    - **user_id**: ID de l'utilisateur pour lequel générer des recommandations
    - **n_recommendations**: Nombre d'articles à recommander (défaut: 5)
    """
    # Validation manuelle du user_id
    if user_id < 0:
        raise HTTPException(
            status_code=422,
            detail="user_id doit être un entier positif ou zéro"
        )
    
    if not is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Service indisponible: modèles non chargés"
        )
    
    try:
        # Générer recommandations
        result = get_hybrid_recommendations(user_id, n_recommendations)
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=result['recommendations'],
            method=result['method'],
            confidence=result['confidence']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur inattendue pour user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Retourne les statistiques du système."""
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    
    config = models_data['config']
    training_stats = config.get('training_stats', {})
    
    return StatsResponse(
        total_users=training_stats.get('nb_users', 0),
        total_articles=training_stats.get('nb_articles', 0),
        model_version=config.get('model_version', '1.0.0'),
        api_version="1.0.0"
    )

@app.get("/articles/metadata", response_model=ArticlesMetadataResponse)
async def get_articles_metadata(article_ids: str = Query(..., description="IDs d'articles séparés par des virgules")):
    """Retourne les métadonnées pour une liste d'articles."""
    if not is_loaded:
        raise HTTPException(status_code=503, detail="Modèles non chargés")
    
    try:
        # Parser les IDs d'articles
        ids = [int(x.strip()) for x in article_ids.split(',')]
        
        articles_dict = models_data['articles_dict']
        articles = []
        
        # Catégories simulées pour la démo
        categories_map = {
            1: "Actualités", 2: "Sport", 3: "Technologie", 4: "Culture", 5: "Économie",
            6: "Politique", 7: "Science", 8: "Santé", 9: "Lifestyle", 10: "Voyage"
        }
        
        for article_id in ids:
            if article_id in articles_dict:
                article_data = articles_dict[article_id]
                category_id = article_data.get('category_id', 1)
                category = categories_map.get(category_id, "Actualités")
                
                # Générer titre et description basés sur l'ID pour la démo
                titles = [
                    f"Les dernières innovations en {category.lower()}",
                    f"Analyse approfondie : {category} en 2024",
                    f"Découverte majeure dans le domaine {category.lower()}",
                    f"Tendances actuelles en {category.lower()}",
                    f"Expertise {category.lower()} : ce qu'il faut savoir"
                ]
                
                descriptions = [
                    f"Une exploration détaillée des développements récents en {category.lower()} avec des insights d'experts.",
                    f"Analyse complète des enjeux et opportunités du secteur {category.lower()}.",
                    f"Découvrez les dernières tendances et innovations qui façonnent {category.lower()}.",
                    f"Guide complet sur les évolutions actuelles en {category.lower()}.",
                    f"Perspective d'expert sur les défis et solutions en {category.lower()}."
                ]
                
                # Utiliser l'ID pour choisir titre/description de façon reproductible
                np.random.seed(article_id)
                title = np.random.choice(titles)
                description = np.random.choice(descriptions)
                
                articles.append(ArticleMetadata(
                    article_id=article_id,
                    category_id=category_id,
                    words_count=article_data.get('words_count', np.random.randint(200, 1000)),
                    title=title,
                    category=category,
                    description=description
                ))
            else:
                # Article non trouvé, créer des données par défaut
                articles.append(ArticleMetadata(
                    article_id=article_id,
                    category_id=1,
                    words_count=500,
                    title=f"Article {article_id}",
                    category="Actualités",
                    description="Contenu personnalisé sélectionné par notre système de recommandation."
                ))
        
        return ArticlesMetadataResponse(articles=articles)
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail="Format d'IDs invalide")
    except Exception as e:
        logger.error(f"❌ Erreur récupération métadonnées: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne")

@app.get("/")
async def root():
    """Endpoint racine avec informations de base."""
    return {
        "message": "My Content Recommendation API",
        "version": "1.0.0",
        "status": "running" if is_loaded else "loading",
        "endpoints": {
            "recommendations": "/recommend/{user_id}",
            "health": "/health", 
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Gestionnaire d'erreur 404 personnalisé."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint non trouvé", "available_endpoints": ["/", "/recommend/{user_id}", "/health", "/stats", "/docs"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Gestionnaire d'erreur 500 personnalisé."""
    logger.error(f"Erreur interne: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Erreur interne du serveur", "support": "Vérifiez les logs"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Configuration pour développement
    uvicorn.run(
        "main_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )