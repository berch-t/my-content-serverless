/**
 * Gestionnaire intelligent d'API - My Content
 * Permet de basculer entre API locale et Cloud Function avec fallback automatique
 */

export type ApiMode = 'local' | 'cloud' | 'auto'

export interface RecommendationResponse {
  user_id: number
  recommendations: number[]
  method: string
  confidence: number
  metadata?: {
    source?: string
    [key: string]: any
  }
}

export interface HealthResponse {
  status: string
  models_loaded: boolean
  version: string
  source?: string
}

export interface StatsResponse {
  total_users: number
  total_articles: number
  model_version: string
  api_version: string
  source?: string
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

interface ApiConfig {
  baseUrl: string
  timeout: number
  retries: number
  name: string
}

interface ApiCallOptions {
  timeout?: number
  retries?: number
  preferredMode?: ApiMode
}

class ApiManager {
  private currentMode: ApiMode = 'auto'
  private lastSuccessfulMode: 'local' | 'cloud' | null = null
  
  private configs: Record<'local' | 'cloud', ApiConfig> = {
    local: {
      baseUrl: process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://127.0.0.1:8000',
      timeout: 5000,
      retries: 2,
      name: 'API Locale'
    },
    cloud: {
      baseUrl: process.env.NEXT_PUBLIC_CLOUD_FUNCTION_URL || '',
      timeout: 45000, // Cloud Functions peut avoir cold start
      retries: 1,
      name: 'Cloud Function'
    }
  }

  constructor() {
    // Charger mode depuis localStorage si disponible
    if (typeof window !== 'undefined') {
      const savedMode = localStorage.getItem('api-mode') as ApiMode
      if (savedMode && ['local', 'cloud', 'auto'].includes(savedMode)) {
        this.currentMode = savedMode
      }
    }
  }

  /**
   * Définit le mode d'API à utiliser
   */
  setMode(mode: ApiMode) {
    this.currentMode = mode
    if (typeof window !== 'undefined') {
      localStorage.setItem('api-mode', mode)
    }
    console.log(`🔄 Mode API changé: ${mode}`)
  }

  /**
   * Récupère le mode actuel
   */
  getMode(): ApiMode {
    return this.currentMode
  }

  /**
   * Effectue un appel API intelligent avec fallback
   */
  private async makeApiCall<T>(
    endpoint: string,
    options: ApiCallOptions = {}
  ): Promise<{ data: T; source: 'local' | 'cloud' }> {
    const modes = this.getModesToTry(options.preferredMode)
    let lastError: Error | null = null

    for (const mode of modes) {
      if (!this.configs[mode].baseUrl) {
        console.warn(`⚠️  ${this.configs[mode].name} URL non configurée`)
        continue
      }

      try {
        console.log(`🔍 Tentative ${this.configs[mode].name}: ${endpoint}`)
        const data = await this.callSingleApi<T>(mode, endpoint, options)
        
        // Succès - mémoriser ce mode
        this.lastSuccessfulMode = mode
        console.log(`✅ Succès ${this.configs[mode].name}: ${endpoint}`)
        
        return { data, source: mode }
      } catch (error) {
        const err = error instanceof Error ? error : new Error('Erreur inconnue')
        
        // Messages d'erreur plus explicites
        if (err.name === 'AbortError') {
          console.warn(`⏱️ Timeout ${this.configs[mode].name}: ${this.configs[mode].timeout}ms dépassé`)
          lastError = new Error(`Timeout ${this.configs[mode].timeout}ms dépassé pour ${this.configs[mode].name}`)
        } else if (err.message.includes('Failed to fetch')) {
          console.warn(`🌐 Connexion échouée ${this.configs[mode].name}: Serveur inaccessible`)
          lastError = new Error(`Impossible de contacter ${this.configs[mode].name}`)
        } else {
          console.warn(`❌ Échec ${this.configs[mode].name}: ${err.message}`)
          lastError = err
        }
      }
    }

    // Aucun mode n'a fonctionné
    throw new Error(`Tous les modes d'API ont échoué. Dernière erreur: ${lastError?.message || 'Inconnue'}`)
  }

  /**
   * Détermine quels modes essayer dans quel ordre
   */
  private getModesToTry(preferredMode?: ApiMode): ('local' | 'cloud')[] {
    const effectiveMode = preferredMode || this.currentMode

    switch (effectiveMode) {
      case 'local':
        return ['local']
      case 'cloud':
        return ['cloud']
      case 'auto':
      default:
        // Mode auto : essaie le dernier succès d'abord, puis l'autre
        if (this.lastSuccessfulMode) {
          return this.lastSuccessfulMode === 'local' ? ['local', 'cloud'] : ['cloud', 'local']
        }
        // Priorité par défaut : local puis cloud (local plus rapide pour dev)
        return ['local', 'cloud']
    }
  }

  /**
   * Effectue un appel vers un mode spécifique
   */
  private async callSingleApi<T>(
    mode: 'local' | 'cloud',
    endpoint: string,
    options: ApiCallOptions = {}
  ): Promise<T> {
    const config = this.configs[mode]
    const timeout = options.timeout || config.timeout
    const maxRetries = options.retries ?? config.retries

    let lastError: Error | null = null

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeout)

        const url = `${config.baseUrl}${endpoint}`
        const startTime = Date.now()

        const response = await fetch(url, {
          signal: controller.signal,
          method: 'GET',
          mode: 'cors',
          credentials: 'omit',
          headers: {
            'Accept': 'application/json',
            'Cache-Control': 'no-cache',
          },
        })

        clearTimeout(timeoutId)
        const duration = Date.now() - startTime

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data = await response.json()
        
        // Ajouter métadonnées de source
        if (data && typeof data === 'object') {
          data.metadata = { ...data.metadata, source: mode, responseTime: duration }
        }

        return data
      } catch (error) {
        const err = error instanceof Error ? error : new Error('Erreur inconnue')
        
        // Messages d'erreur spécifiques
        if (err.name === 'AbortError') {
          lastError = new Error(`Timeout ${config.timeout}ms dépassé`)
        } else if (err.message.includes('Failed to fetch')) {
          lastError = new Error(`Serveur inaccessible`)
        } else {
          lastError = err
        }
        
        if (attempt < maxRetries) {
          console.log(`🔄 Retry ${attempt + 1}/${maxRetries} pour ${config.name}`)
          await this.sleep(1000 * (attempt + 1)) // Backoff exponentiel
        }
      }
    }

    throw lastError || new Error('Erreur inconnue')
  }

  /**
   * Utilitaire pour attendre
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }

  /**
   * MÉTHODES PUBLIQUES - Interface unifiée
   */

  async getRecommendations(
    userId: number, 
    nRecommendations: number = 5,
    options?: ApiCallOptions
  ): Promise<{ data: RecommendationResponse; source: 'local' | 'cloud' }> {
    const endpoint = `/recommend/${userId}?n_recommendations=${nRecommendations}`
    return this.makeApiCall<RecommendationResponse>(endpoint, options)
  }

  async getHealth(
    options?: ApiCallOptions
  ): Promise<{ data: HealthResponse; source: 'local' | 'cloud' }> {
    return this.makeApiCall<HealthResponse>('/health', options)
  }

  async getStats(
    options?: ApiCallOptions
  ): Promise<{ data: StatsResponse; source: 'local' | 'cloud' }> {
    return this.makeApiCall<StatsResponse>('/stats', options)
  }

  async getArticlesMetadata(
    articleIds: number[],
    options?: ApiCallOptions
  ): Promise<{ data: ArticlesMetadataResponse; source: 'local' | 'cloud' }> {
    const idsParam = articleIds.join(',')
    const endpoint = `/articles/metadata?article_ids=${idsParam}`
    return this.makeApiCall<ArticlesMetadataResponse>(endpoint, options)
  }

  /**
   * Test la disponibilité des APIs
   */
  async checkApiAvailability(): Promise<{
    local: { available: boolean; responseTime: number | null; error?: string }
    cloud: { available: boolean; responseTime: number | null; error?: string }
  }> {
    const results = {
      local: { available: false, responseTime: null as number | null, error: undefined as string | undefined },
      cloud: { available: false, responseTime: null as number | null, error: undefined as string | undefined }
    }

    // Test en parallèle
    const tests = Object.entries(this.configs).map(async ([mode, config]) => {
      if (!config.baseUrl) {
        return { mode: mode as 'local' | 'cloud', available: false, responseTime: null, error: 'URL non configurée' }
      }

      try {
        const startTime = Date.now()
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 5000)

        const response = await fetch(`${config.baseUrl}/health`, {
          signal: controller.signal,
          headers: { 'Cache-Control': 'no-cache' }
        })

        clearTimeout(timeoutId)
        const responseTime = Date.now() - startTime

        if (response.ok) {
          const data = await response.json()
          return {
            mode: mode as 'local' | 'cloud',
            available: data.status === 'healthy' || data.models_loaded === true,
            responseTime,
          }
        } else {
          return {
            mode: mode as 'local' | 'cloud',
            available: false,
            responseTime,
            error: `HTTP ${response.status}`
          }
        }
      } catch (error) {
        return {
          mode: mode as 'local' | 'cloud',
          available: false,
          responseTime: null,
          error: error instanceof Error ? error.message : 'Erreur de connexion'
        }
      }
    })

    const testResults = await Promise.all(tests)
    testResults.forEach(result => {
      results[result.mode] = {
        available: result.available,
        responseTime: result.responseTime,
        error: result.error
      }
    })

    return results
  }

  /**
   * Informations sur la configuration actuelle
   */
  getConfigInfo() {
    return {
      currentMode: this.currentMode,
      lastSuccessfulMode: this.lastSuccessfulMode,
      configs: this.configs,
      isLocalConfigured: !!this.configs.local.baseUrl,
      isCloudConfigured: !!this.configs.cloud.baseUrl,
    }
  }
}

// Instance singleton
export const apiManager = new ApiManager()

// Export de l'ancienne interface pour compatibilité
export const apiClient = {
  async getRecommendations(userId: number, nRecommendations: number = 5) {
    const { data } = await apiManager.getRecommendations(userId, nRecommendations)
    return data
  },
  
  async getHealth() {
    const { data } = await apiManager.getHealth()
    return data
  },
  
  async getStats() {
    const { data } = await apiManager.getStats()
    return data
  },
  
  async getArticlesMetadata(articleIds: number[]) {
    const { data } = await apiManager.getArticlesMetadata(articleIds)
    return data
  }
}