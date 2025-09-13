"""
Main Cloud Run Function - Système de Recommandation My Content
Adapté pour la nouvelle architecture GCP Cloud Run avec functions_framework

URL: https://my-content-serverless-83529683395.europe-west1.run.app
"""

import functions_framework
from flask import jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any
from google.cloud import storage
import io
from sklearn.metrics.pairwise import cosine_similarity

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales pour cache des modèles
_models_cache = {}
_is_loaded = False

# Configuration GCP
BUCKET_NAME = 'air-paradis-models'
MODELS_PATH = 'my-content-embeddings/'

def load_models_from_gcs():
    """Charge les modèles depuis Google Cloud Storage."""
    global _models_cache, _is_loaded
    
    if _is_loaded:
        return True
    
    try:
        logger.info(f"🔄 Chargement modèles depuis gs://{BUCKET_NAME}/{MODELS_PATH}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        models_to_load = [
            'surprise_model.pkl',
            'embeddings.pkl', 
            'articles_metadata.pkl',
            'config.pkl'
        ]
        
        for model_file in models_to_load:
            blob_path = f"{MODELS_PATH}{model_file}"
            blob = bucket.blob(blob_path)
            
            if not blob.exists():
                raise FileNotFoundError(f"Model {blob_path} not found in bucket")
            
            # Télécharger et désérialiser
            model_bytes = blob.download_as_bytes()
            model_data = pickle.load(io.BytesIO(model_bytes))
            
            # Stocker dans cache
            model_key = model_file.replace('.pkl', '')
            _models_cache[model_key] = model_data
            logger.info(f"✅ {model_file} chargé ({len(model_bytes)} bytes)")
        
        # Créer dictionnaire d'accès rapide pour articles
        if 'articles_metadata' in _models_cache:
            articles_df = _models_cache['articles_metadata']
            _models_cache['articles_dict'] = articles_df.set_index('article_id').to_dict('index')
        
        _is_loaded = True
        logger.info("🎉 Tous les modèles chargés avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur chargement modèles: {e}")
        return False

def content_based_recs(user_id: int, n_recs: int = 5) -> List[int]:
    """Recommandations content-based."""
    try:
        embeddings = _models_cache['embeddings']
        
        # Simuler dernier article (reproductible)
        np.random.seed(user_id)
        last_article_idx = np.random.randint(0, len(embeddings))
        
        # Similarité cosinus
        article_embedding = embeddings[last_article_idx].reshape(1, -1)
        similarities = cosine_similarity(article_embedding, embeddings)[0]
        
        # Top articles similaires
        similar_indices = np.argsort(similarities)[::-1][1:n_recs+1]
        return similar_indices.tolist()
        
    except Exception as e:
        logger.error(f"❌ Content-based error: {e}")
        return []

def collaborative_recs(user_id: int, n_recs: int = 5) -> List[int]:
    """Recommandations collaborative filtering."""
    try:
        surprise_model = _models_cache['surprise_model']
        config = _models_cache['config']
        total_articles = config['total_articles']
        
        # Prédire ratings (optimisé pour Cloud Run)
        predictions = []
        for article_id in range(min(300, total_articles)):
            try:
                pred = surprise_model.predict(user_id, article_id)
                predictions.append((article_id, pred.est))
            except:
                continue
        
        # Trier par rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [article_id for article_id, _ in predictions[:n_recs]]
        
    except Exception as e:
        logger.error(f"❌ Collaborative error: {e}")
        return []

def get_hybrid_recs(user_id: int, n_recs: int = 5) -> Dict[str, Any]:
    """Recommandations hybrides."""
    if not _is_loaded:
        return {
            'error': 'Models not loaded',
            'recommendations': [],
            'method': 'error',
            'confidence': 0.0
        }
    
    try:
        config = _models_cache['config']
        hybrid_weights = config['hybrid_weights']
        weight_cb = hybrid_weights['content_based']
        weight_cf = hybrid_weights['collaborative']
        
        # Obtenir recommandations
        cb_recs = content_based_recs(user_id, n_recs * 2)
        cf_recs = collaborative_recs(user_id, n_recs * 2)
        
        # Scoring hybride
        hybrid_scores = {}
        
        # Scorer CB
        for i, article_id in enumerate(cb_recs):
            score = weight_cb * (1.0 - i / len(cb_recs)) if cb_recs else 0
            hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
        
        # Scorer CF
        for i, article_id in enumerate(cf_recs):
            score = weight_cf * (1.0 - i / len(cf_recs)) if cf_recs else 0
            hybrid_scores[article_id] = hybrid_scores.get(article_id, 0) + score
        
        # Fallback si pas de résultats
        if not hybrid_scores:
            logger.warning(f"No hybrid recs for user {user_id}, random fallback")
            np.random.seed(user_id)
            total_articles = config['total_articles']
            recommendations = np.random.choice(
                range(total_articles), 
                size=min(n_recs, total_articles), 
                replace=False
            ).tolist()
            method = 'random_fallback'
            confidence = 0.3
        else:
            # Trier par score
            sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            recommendations = [article_id for article_id, _ in sorted_recs[:n_recs]]
            method = 'hybrid'
            confidence = min(0.95, 0.6 + len(hybrid_scores) * 0.05)
        
        return {
            'user_id': user_id,
            'recommendations': recommendations,
            'method': method,
            'confidence': confidence,
            'metadata': {
                'cb_count': len(cb_recs),
                'cf_count': len(cf_recs),
                'hybrid_candidates': len(hybrid_scores),
                'weights': {'content_based': weight_cb, 'collaborative': weight_cf},
                'algorithm_version': config['model_version'],
                'source': 'gcp_cloud_run'
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Hybrid error for user {user_id}: {e}")
        return {
            'error': str(e),
            'recommendations': [],
            'method': 'error',
            'confidence': 0.0
        }

def get_articles_metadata_gcp(article_ids: List[int]) -> Dict[str, Any]:
    """Récupère métadonnées articles."""
    if not _is_loaded:
        return {'articles': []}
    
    try:
        articles_dict = _models_cache['articles_dict']
        articles = []
        
        # Catégories
        categories_map = {
            1: "Actualités", 2: "Sport", 3: "Technologie", 4: "Culture", 5: "Économie",
            6: "Politique", 7: "Science", 8: "Santé", 9: "Lifestyle", 10: "Voyage"
        }
        
        for article_id in article_ids:
            if article_id in articles_dict:
                article_data = articles_dict[article_id]
                category_id = article_data.get('category_id', 1)
                category = categories_map.get(category_id, "Actualités")
                
                # Générer titre/description reproductible
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
                
                np.random.seed(article_id)
                title = np.random.choice(titles)
                description = np.random.choice(descriptions)
                
                articles.append({
                    'article_id': article_id,
                    'category_id': category_id,
                    'words_count': article_data.get('words_count', np.random.randint(200, 1000)),
                    'title': title,
                    'category': category,
                    'description': description
                })
            else:
                articles.append({
                    'article_id': article_id,
                    'category_id': 1,
                    'words_count': 500,
                    'title': f"Article {article_id}",
                    'category': "Actualités",
                    'description': "Contenu personnalisé sélectionné par notre système."
                })
        
        return {'articles': articles}
        
    except Exception as e:
        logger.error(f"❌ Metadata error: {e}")
        return {'articles': []}

@functions_framework.http
def my_content_http(request):
    """Point d'entrée HTTP Cloud Run Function."""
    
    # CORS Headers - Support pour Next.js
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, Cache-Control, Authorization, Accept',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS, PUT, DELETE',
        'Access-Control-Max-Age': '86400',  # Cache preflight 24h
        'Vary': 'Origin'
    }
    
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    try:
        # Chargement des modèles au premier appel
        if not _is_loaded:
            if not load_models_from_gcs():
                return jsonify({
                    'error': 'Failed to load models',
                    'status': 503
                }), 503, headers
        
        # Router les endpoints basé sur le path
        path = request.path
        method = request.method
        
        if method == 'GET':
            if path == '/health':
                config = _models_cache.get('config', {})
                return jsonify({
                    'status': 'healthy',
                    'models_loaded': _is_loaded,
                    'version': '1.0.0-cloud-run',
                    'source': 'gcp_cloud_run',
                    'model_version': config.get('model_version', '1.0.0')
                }), 200, headers
            
            elif path == '/stats':
                config = _models_cache.get('config', {})
                training_stats = config.get('training_stats', {})
                return jsonify({
                    'total_users': training_stats.get('nb_users', 0),
                    'total_articles': training_stats.get('nb_articles', 0),
                    'model_version': config.get('model_version', '1.0.0'),
                    'api_version': '1.0.0-cloud-run',
                    'source': 'gcp_cloud_run'
                }), 200, headers
            
            elif path.startswith('/recommend/'):
                # Extraire user_id du path
                try:
                    user_id = int(path.split('/')[-1])
                    n_recommendations = int(request.args.get('n_recommendations', 5))
                    
                    result = get_hybrid_recs(user_id, n_recommendations)
                    
                    if 'error' in result:
                        return jsonify(result), 500, headers
                    
                    return jsonify(result), 200, headers
                except ValueError:
                    return jsonify({'error': 'Invalid user ID'}), 422, headers
            
            elif path == '/articles/metadata':
                article_ids_param = request.args.get('article_ids', '')
                try:
                    article_ids = [int(x.strip()) for x in article_ids_param.split(',')]
                    result = get_articles_metadata_gcp(article_ids)
                    return jsonify(result), 200, headers
                except ValueError:
                    return jsonify({'error': 'Invalid article IDs format'}), 422, headers
            
            elif path == '/' or path == '':
                return jsonify({
                    'message': 'My Content Recommendation API - GCP Cloud Run',
                    'version': '1.0.0-cloud-run',
                    'endpoints': ['/health', '/stats', '/recommend/{user_id}', '/articles/metadata'],
                    'models_loaded': _is_loaded,
                    'source': 'gcp_cloud_run',
                    'url': 'https://my-content-serverless-83529683395.europe-west1.run.app',
                    'port': os.environ.get('PORT', '8080'),
                    'debug': {
                        'path': path,
                        'method': method,
                        'environment': 'cloud_run'
                    }
                }), 200, headers
        
        return jsonify({'error': 'Endpoint not found'}), 404, headers
        
    except Exception as e:
        logger.error(f"❌ Function error: {e}")
        return jsonify({
            'error': str(e),
            'status': 500,
            'source': 'gcp_cloud_run'
        }), 500, headers