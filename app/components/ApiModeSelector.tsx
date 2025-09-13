"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Home, Cloud, Wifi, WifiOff, Clock, Zap } from "lucide-react"

type ApiMode = 'local' | 'cloud'

interface ApiStatus {
  available: boolean
  responseTime: number | null
  error?: string
}

interface ApiModeSelectorProps {
  currentMode: ApiMode
  onModeChange: (mode: ApiMode) => void
}

export function ApiModeSelector({ currentMode, onModeChange }: ApiModeSelectorProps) {
  const [localStatus, setLocalStatus] = useState<ApiStatus>({ available: false, responseTime: null })
  const [cloudStatus, setCloudStatus] = useState<ApiStatus>({ available: false, responseTime: null })
  const [isChecking, setIsChecking] = useState(false)

  const checkApiHealth = async (mode: ApiMode): Promise<ApiStatus> => {
    const baseUrl = mode === 'local' 
      ? process.env.NEXT_PUBLIC_LOCAL_API_URL || 'http://127.0.0.1:8000'
      : process.env.NEXT_PUBLIC_CLOUD_FUNCTION_URL || 'https://my-content-serverless-83529683395.europe-west1.run.app'

    if (!baseUrl) {
      return { available: false, responseTime: null, error: 'URL non configurée' }
    }

    try {
      const startTime = Date.now()
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), mode === 'local' ? 5000 : 20000)

      const response = await fetch(`${baseUrl}/health`, {
        signal: controller.signal,
        method: 'GET',
        mode: 'cors',
        credentials: 'omit',
        headers: { 
          'Accept': 'application/json',
          'Cache-Control': 'no-cache' 
        }
      })

      clearTimeout(timeoutId)
      const responseTime = Date.now() - startTime

      if (response.ok) {
        const data = await response.json()
        return {
          available: data.status === 'healthy' || data.models_loaded === true,
          responseTime,
        }
      } else {
        return { available: false, responseTime, error: `HTTP ${response.status}` }
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Erreur de connexion')
      
      let errorMessage = 'Erreur de connexion'
      if (err.name === 'AbortError') {
        errorMessage = `Timeout ${mode === 'local' ? '5s' : '20s'} dépassé`
      } else if (err.message.includes('Failed to fetch')) {
        errorMessage = 'Serveur inaccessible'
      } else if (err.message) {
        errorMessage = err.message
      }
      
      return {
        available: false,
        responseTime: null,
        error: errorMessage
      }
    }
  }

  const checkAllApis = async () => {
    setIsChecking(true)
    try {
      const [local, cloud] = await Promise.all([
        checkApiHealth('local'),
        checkApiHealth('cloud')
      ])
      setLocalStatus(local)
      setCloudStatus(cloud)
    } finally {
      setIsChecking(false)
    }
  }

  // Vérification initiale et périodique
  useEffect(() => {
    checkAllApis()
    const interval = setInterval(checkAllApis, 30000) // Toutes les 30s
    return () => clearInterval(interval)
  }, [])

  const modes = [
    {
      id: 'local' as ApiMode,
      name: 'Local',
      icon: Home,
      description: 'API FastAPI locale',
      color: 'emerald',
      status: localStatus,
      benefits: ['🚀 Ultra rapide', '💰 Gratuit', '🔧 Debug facile']
    },
    {
      id: 'cloud' as ApiMode,
      name: 'Cloud',
      icon: Cloud,
      description: 'Google Cloud Function',
      color: 'blue',
      status: cloudStatus,
      benefits: ['☁️ Scalable', '🌍 Disponible partout', '🛡️ Sécurisé']
    }
  ]

  const formatResponseTime = (time: number | null) => {
    if (time === null) return '?'
    if (time < 100) return `${time}ms`
    if (time < 1000) return `${Math.round(time)}ms`
    return `${(time / 1000).toFixed(1)}s`
  }

  const getStatusColor = (status: ApiStatus) => {
    if (status.available) return 'text-green-400 bg-green-900/30 border-green-400/50'
    if (status.error?.includes('URL non configurée')) return 'text-gray-300 bg-gray-800/30 border-gray-400/50'
    return 'text-red-400 bg-red-900/30 border-red-400/50'
  }

  const getStatusIcon = (status: ApiStatus) => {
    return status.available ? Wifi : WifiOff
  }

  return (
    <Card className="w-full max-w-4xl mx-auto border-0 glass-effect bg-white/10 dark:bg-white/5 backdrop-blur-lg shadow-xl">
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-foreground">Mode d'API</h3>
            <p className="text-sm text-muted-foreground opacity-80">
              Choisissez entre l'API locale ou la Cloud Function
            </p>
          </div>
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={checkAllApis}
            disabled={isChecking}
            className="p-2 rounded-lg bg-white/10 hover:bg-white/20 dark:bg-white/10 dark:hover:bg-white/20 transition-colors disabled:opacity-50 text-foreground"
          >
            <motion.div
              animate={isChecking ? { rotate: 360 } : { rotate: 0 }}
              transition={{ duration: 1, repeat: isChecking ? Infinity : 0, ease: "linear" }}
            >
              <Zap className="h-4 w-4" />
            </motion.div>
          </motion.button>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          {modes.map((mode, index) => {
            const isActive = currentMode === mode.id
            const StatusIcon = getStatusIcon(mode.status)

            return (
              <motion.div
                key={mode.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => onModeChange(mode.id)}
                className={`
                  relative p-4 rounded-xl cursor-pointer transition-all duration-300
                  ${isActive 
                    ? `border-2 border-purple-400 bg-purple-500/20 shadow-lg` 
                    : 'border border-white/20 hover:border-white/30 bg-white/10 hover:bg-white/15'
                  }
                `}
              >
                {/* Indicateur actif */}
                {isActive && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="absolute -top-2 -right-2 w-6 h-6 bg-primary rounded-full flex items-center justify-center"
                  >
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ duration: 1, repeat: Infinity }}
                      className="w-2 h-2 bg-white rounded-full"
                    />
                  </motion.div>
                )}

                <div className="flex items-start gap-3 mb-3">
                  <motion.div
                    whileHover={{ rotate: 5 }}
                    className={`p-2 rounded-lg ${
                      isActive 
                        ? `bg-purple-500/30 text-purple-200` 
                        : 'bg-white/20 text-foreground'
                    }`}
                  >
                    <mode.icon className="h-5 w-5" />
                  </motion.div>
                  
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-medium text-foreground">{mode.name}</h4>
                      <StatusIcon className={`h-4 w-4 ${mode.status.available ? 'text-green-600' : 'text-red-500'}`} />
                    </div>
                    <p className="text-sm text-muted-foreground opacity-80">{mode.description}</p>
                  </div>
                </div>

                {/* Status */}
                <div className="flex items-center gap-2 mb-3">
                  <Badge
                    variant="outline"
                    className={`text-xs ${getStatusColor(mode.status)}`}
                  >
                    {mode.status.available ? 'Disponible' : mode.status.error || 'Indisponible'}
                  </Badge>
                  
                  {mode.status.responseTime !== null && (
                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {formatResponseTime(mode.status.responseTime)}
                    </div>
                  )}
                </div>

                {/* Benefits */}
                <div className="space-y-1">
                  {mode.benefits.map((benefit, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 + i * 0.05 }}
                      className="text-xs text-muted-foreground opacity-70"
                    >
                      {benefit}
                    </motion.div>
                  ))}
                </div>

                {/* Glow effect for active mode */}
                {isActive && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className={`absolute inset-0 rounded-xl bg-gradient-to-r from-purple-400/10 to-purple-600/10 pointer-events-none -z-1`}
                  />
                )}
              </motion.div>
            )
          })}
        </div>

        {/* Status global */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-4 p-3 rounded-lg bg-white/10 backdrop-blur-sm"
        >
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground opacity-80">Mode actuel :</span>
            <div className="flex items-center gap-2">
              <Badge variant={modes.find(m => m.id === currentMode)?.status.available ? "default" : "destructive"}>
                {modes.find(m => m.id === currentMode)?.name} 
                {modes.find(m => m.id === currentMode)?.status.available ? ' ✅' : ' ❌'}
              </Badge>
            </div>
          </div>
        </motion.div>
      </CardContent>
    </Card>
  )
}