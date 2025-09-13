export interface RecommendationResponse {
  user_id: number
  recommendations: number[]
  method: string
  confidence: number
}

export interface HealthResponse {
  status: string
  models_loaded: boolean
  version: string
}

export interface StatsResponse {
  total_users: number
  total_articles: number
  model_version: string
  api_version: string
}

export interface ArticleMetadata {
  article_id: number
  category_id?: number
  words_count?: number
  title: string
  category: string
  description: string
}

export interface ArticlesMetadataResponse {
  articles: ArticleMetadata[]
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  async getRecommendations(userId: number, nRecommendations: number = 5): Promise<RecommendationResponse> {
    const response = await fetch(`${this.baseUrl}/recommend/${userId}?n_recommendations=${nRecommendations}`)
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  }

  async getHealth(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`)
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  }

  async getStats(): Promise<StatsResponse> {
    const response = await fetch(`${this.baseUrl}/stats`)
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  }

  async getArticlesMetadata(articleIds: number[]): Promise<ArticlesMetadataResponse> {
    const idsParam = articleIds.join(',')
    const response = await fetch(`${this.baseUrl}/articles/metadata?article_ids=${idsParam}`)
    
    if (!response.ok) {
      throw new Error(`Erreur API: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  }
}

export const apiClient = new ApiClient()