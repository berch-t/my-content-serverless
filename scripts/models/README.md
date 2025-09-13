# Modèles ML - My Content Recommendation System

Les modèles ML sont stockés dans Google Cloud Storage pour la production :
- Bucket: `gs://air-paradis-models`
- Path: `my-content-embeddings/`

## Modèles disponibles

- `surprise_model.pkl` - Modèle Collaborative Filtering (SVD)
- `embeddings.pkl` - Embeddings des articles (364 MB)
- `articles_metadata.pkl` - Métadonnées des articles
- `config.pkl` - Configuration du système

## Développement local

Pour développer localement, les modèles doivent être présents dans ce répertoire.
Ils ne sont pas versionnés git à cause de leur taille.

## Production

En production, les modèles sont automatiquement chargés depuis Google Cloud Storage
par la Cloud Function `my-content-serverless`.