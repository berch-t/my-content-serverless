# My Content - Système de Recommandation Hybride

🚀 **Application de recommandation de contenu ultra-moderne** utilisant Next.js 14+, intelligence artificielle hybride (Content-Based + Collaborative Filtering), et architecture cloud native avec Google Cloud Platform.

![My Content App Interface](.app/public/UI.png)

## ✨ Aperçu

My Content est une application de recommandation personnalisée qui combine deux approches d'IA pour offrir des suggestions d'articles précises et pertinentes :
- **Content-Based Filtering** : Analyse de la similarité des contenus
- **Collaborative Filtering** : Apprentissage des préférences utilisateurs
- **Système Hybride** : Fusion intelligente des deux approches (60% Content-Based + 40% Collaborative)

## 🛠️ Stack Technologique

### Frontend
- **Next.js 14+** avec App Router
- **TypeScript** pour la sécurité des types
- **Tailwind CSS** avec design system moderne
- **Framer Motion** pour les animations avancées
- **Lucide React** pour les icônes
- **Responsive Design** mobile-first

### Backend & Intelligence Artificielle
- **FastAPI** (développement local)
- **Google Cloud Run** (production serverless)
- **scikit-learn** pour les algorithmes ML
- **scikit-surprise** pour le collaborative filtering
- **Pandas & NumPy** pour le traitement des données
- **Google Cloud Storage** pour la persistance des modèles

### Architecture Cloud
- **Google Cloud Platform** (GCP)
- **Cloud Run** pour l'API serverless
- **Cloud Storage** pour les modèles ML
- **GitHub Actions** pour CI/CD automatique

## 🚀 Démarrage Rapide

### Prérequis
- Node.js 18+
- Python 3.11+
- Compte Google Cloud Platform (pour la production)

### Installation

```bash
# Cloner le projet
git clone https://github.com/berch-t/my-content-serverless
cd app

# Installer les dépendances
npm install

# Configurer l'environnement
cp .env.local.example .env.local
```

### Configuration `.env.local`

```bash
# Mode API par défaut ('local', 'cloud', ou 'auto')
NEXT_PUBLIC_API_MODE=auto

# URL de l'API locale (FastAPI)
NEXT_PUBLIC_LOCAL_API_URL=http://127.0.0.1:8000

# URL de la Cloud Function GCP 
NEXT_PUBLIC_CLOUD_FUNCTION_URL=votre_url_cloud_function

# Configuration Next.js
NEXT_PUBLIC_APP_ENV=development # (dev ou production)
NEXT_PUBLIC_APP_VERSION=1.0.0
```

### Lancement

```bash
# Démarrer en mode développement
npm run dev

# Build pour production
npm run build

# Lancer en production
npm run start

# Vérifications qualité
npm run lint
npm run type-check
```

L'application sera disponible sur **http://localhost:3000**

## 🏗️ Architecture de l'Application

```
app/
├── app/                          # Next.js App Router
│   ├── globals.css              # Styles globaux avec variables CSS
│   ├── layout.tsx               # Layout principal avec providers
│   └── page.tsx                 # Page d'accueil avec logique principale
├── components/                   # Composants React réutilisables
│   ├── ui/                      # Design system de base
│   │   ├── card.tsx            # Composant Card avec variants
│   │   ├── badge.tsx           # Badges de statut
│   │   ├── button.tsx          # Boutons avec animations
│   │   ├── input.tsx           # Inputs avec validation
│   │   ├── ClientOnly.tsx      # Wrapper client-side
│   │   └── LightRays.tsx       # Effet lumineux animé
│   ├── ApiModeSelector.tsx      # Sélecteur Local/Cloud avec health checks
│   ├── UserIdInput.tsx          # Saisie utilisateur avec validation
│   ├── RecommendationCard.tsx   # Carte article avec animations
│   └── RecommendationsList.tsx  # Liste des recommandations
├── lib/                         # Services et utilitaires
│   ├── utils.ts                 # Fonctions utilitaires
│   └── api-manager.ts           # Client API avec fallback intelligent
└── public/                      # Assets statiques
    ├── UI.png                   # Screenshot de l'application
    └── logo.png                 # Logo de l'application
```

## 🎯 Fonctionnalités Principales

### 🤖 Intelligence Artificielle Hybride
- **Algorithme Content-Based** : Similarité cosinus sur embeddings d'articles
- **Algorithme Collaborative** : SVD Matrix Factorization avec Surprise
- **Fusion Hybride** : Scoring pondéré optimisé pour la précision
- **Confidence Scoring** : Score de confiance pour chaque recommandation

### 🔄 Architecture Multi-API
- **Mode Auto** : Fallback intelligent Local → Cloud
- **Mode Local** : API FastAPI pour le développement rapide
- **Mode Cloud** : Google Cloud Run pour la production
- **Health Monitoring** : Surveillance temps réel des APIs
- **Retry Logic** : Gestion automatique des erreurs réseau

### 🎨 Interface Utilisateur 
- **Design System** cohérent avec Tailwind CSS
- **Animations Framer Motion** : Transitions fluides et micro-interactions
- **Responsive Design** : Adaptation mobile/tablet/desktop
- **Mode Sombre** : Support natif avec système variables CSS
- **Accessibilité** : Composants accessibles par défaut

### ⚡ Performance & Optimisation
- **Server-Side Rendering** : Next.js App Router pour SEO optimal
- **Code Splitting** : Chargement optimisé des composants
- **Lazy Loading** : Images et composants chargés à la demande
- **Caching Strategy** : Cache intelligent des requêtes API

## 📊 Utilisation

### 1. Démarrage du Backend (Développement)

```bash
# Depuis le répertoire racine
cd ../scripts

# Activer l'environnement Python
source .content_reco/bin/activate  # Linux/Mac
# ou
.content_reco\\Scripts\\activate   # Windows

# Démarrer l'API locale
uv run uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

### 2. Utilisation de l'Application

1. **Ouvrir l'application** : `http://localhost:3000`
2. **Choisir le mode API** : Local, Cloud, ou Auto
3. **Entrer un ID utilisateur** : Exemple : 123, 456, 789, 1001...
4. **Découvrir les recommandations** : 5 articles personnalisés avec scores de confiance
5. **Explorer les détails** : Cliquer sur les cartes pour plus d'informations

### 3. Types d'Utilisateurs de Test

- **Utilisateurs actifs** (ID 1-500) : Profils riches avec historique
- **Utilisateurs moyens** (ID 500-1500) : Quelques interactions
- **Nouveaux utilisateurs** (ID 1500+) : Recommandations par fallback

## 🔧 Développement

### Structure des Composants

Tous les composants suivent les conventions React modernes :

```typescript
// Exemple : RecommendationCard.tsx
interface RecommendationCardProps {
  article: Article
  rank: number
  confidence: number
  onArticleClick?: (articleId: number) => void
}

export function RecommendationCard({ 
  article, 
  rank, 
  confidence,
  onArticleClick 
}: RecommendationCardProps) {
  // Logique avec hooks TypeScript
  // Animations Framer Motion
  // Styling Tailwind CSS
}
```

### API Client

Le gestionnaire d'API offre une interface unifiée :

```typescript
// lib/api-manager.ts
const apiManager = new ApiManager()

// Appel avec fallback automatique
const recommendations = await apiManager.getRecommendations(userId, 5)
const metadata = await apiManager.getArticlesMetadata(articleIds)
const health = await apiManager.getHealth()
```

### Système de Design

Variables CSS personnalisées dans `globals.css` :

```css
:root {
  --primary: 262 83% 58%;
  --secondary: 215 28% 17%;
  --accent: 142 76% 36%;
  --background: 224 71% 4%;
  --foreground: 213 31% 91%;
}
```

## 🚀 Déploiement

### Production avec Vercel (Recommandé)

```bash
# Installer Vercel CLI
npm i -g vercel

# Déployer
vercel --prod

# Configuration automatique :
# - Next.js optimizations
# - Environment variables
# - Domain custom
```

### Variables d'Environnement Production

```env
NEXT_PUBLIC_API_MODE=cloud
NEXT_PUBLIC_CLOUD_FUNCTION_URL=https://your-cloud-run-url
NEXT_PUBLIC_APP_ENV=production
```

### Google Cloud Platform

Le backend est déployé automatiquement via GitHub Actions :

```yaml
# .github/workflows/deploy-gcp.yml
- Déploiement Cloud Run automatique
- Upload des modèles ML vers Cloud Storage  
- Tests d'intégration end-to-end
- Rollback automatique en cas d'erreur
```

## 📈 Monitoring & Analytics

### Health Checks Intégrés

- **API Locale** : Vérification toutes les 30s
- **Cloud Function** : Monitoring de latence et disponibilité
- **Modèles ML** : Vérification du chargement des modèles

### Métriques de Performance

- **Temps de réponse API** : Affiché en temps réel
- **Taux de succès** : Pourcentage de requêtes réussies
- **Confidence des recommandations** : Score de qualité ML

## 🤝 Contribution

### Standards de Code

- **TypeScript strict** : Pas d'`any`, types explicites
- **ESLint + Prettier** : Formatage automatique
- **Conventional Commits** : Messages de commit structurés
- **Tests unitaires** : Jest + React Testing Library

### Workflow de Développement

1. Fork du repository
2. Création d'une branche feature
3. Développement avec tests
4. Pull Request avec description détaillée
5. Review et merge automatique

## 📄 License

Ce projet est développé dans le cadre du **Projet 10 OpenClassrooms** - Parcours Data Scientist / AI Engineer.

---

**🎯 My Content** - Recommandations intelligentes powered by AI hybride et architecture cloud native.

Made with ❤️ using Next.js, TypeScript, Tailwind CSS, Framer Motion & Google Cloud Platform.