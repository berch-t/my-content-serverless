"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { RecommendationCard } from "./RecommendationCard"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Brain, Users, Zap, TrendingUp } from "lucide-react"
import type { RecommendationResponse, ArticleMetadata } from "@/lib/api-manager"
import { apiManager } from "@/lib/api-manager"

interface RecommendationsListProps {
  data: RecommendationResponse
  onBack: () => void
}

export function RecommendationsList({ data, onBack }: RecommendationsListProps) {
  const [articlesMetadata, setArticlesMetadata] = useState<ArticleMetadata[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchArticlesMetadata = async () => {
      try {
        setLoading(true)
        const { data: response } = await apiManager.getArticlesMetadata(data.recommendations)
        setArticlesMetadata(response.articles)
      } catch (error) {
        console.error('Erreur chargement métadonnées:', error)
      } finally {
        setLoading(false)
      }
    }

    if (data.recommendations.length > 0) {
      fetchArticlesMetadata()
    }
  }, [data.recommendations])

  const getArticleMetadata = (articleId: number): ArticleMetadata | undefined => {
    return articlesMetadata.find(article => article.article_id === articleId)
  }
  const getMethodIcon = (method: string) => {
    switch (method.toLowerCase()) {
      case 'hybrid':
        return <Zap className="h-4 w-4" />
      case 'content-based':
        return <Brain className="h-4 w-4" />
      case 'collaborative':
        return <Users className="h-4 w-4" />
      default:
        return <TrendingUp className="h-4 w-4" />
    }
  }

  const getMethodDescription = (method: string) => {
    switch (method.toLowerCase()) {
      case 'hybrid':
        return "Combinaison optimale de l'analyse de contenu et des préférences communautaires"
      case 'content-based':
        return "Basé sur l'analyse du contenu des articles que vous avez consultés"
      case 'collaborative':
        return "Basé sur les préférences d'utilisateurs ayant des goûts similaires"
      default:
        return "Recommandations personnalisées"
    }
  }

  const getMethodColor = (method: string) => {
    switch (method.toLowerCase()) {
      case 'hybrid':
        return "bg-gradient-to-r from-purple-500 to-blue-500"
      case 'content-based':
        return "bg-gradient-to-r from-green-500 to-emerald-500"
      case 'collaborative':
        return "bg-gradient-to-r from-orange-500 to-red-500"
      default:
        return "bg-gradient-to-r from-gray-500 to-slate-500"
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-6xl mx-auto space-y-8"
    >
      {/* Header */}
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-8"
      >
        <h1 className="text-3xl font-bold gradient-text mb-2">
          Vos Recommandations
        </h1>
        <p className="text-muted-foreground">
          Utilisateur #{data.user_id} - {data.recommendations.length} articles sélectionnés
        </p>
      </motion.div>

      {/* Method Information */}
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card className="border-0 glass-effect bg-white/10 dark:bg-white/5 backdrop-blur-lg">
          <CardHeader>
            <div className="flex items-center gap-3">
              <motion.div
                whileHover={{ scale: 1.1, rotate: 5 }}
                className={`p-3 rounded-lg text-white ${getMethodColor(data.method)}`}
              >
                {getMethodIcon(data.method)}
              </motion.div>
              <div className="flex-1">
                <CardTitle className="flex items-center gap-2 text-foreground">
                  Méthode {data.method.charAt(0).toUpperCase() + data.method.slice(1)}
                  <Badge variant="outline" className="text-xs">
                    {Math.round(data.confidence * 100)}% confiance
                  </Badge>
                </CardTitle>
                <CardDescription className="text-sm">
                  {getMethodDescription(data.method)}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
        </Card>
      </motion.div>

      {/* Recommendations Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {loading ? (
          // Loading skeleton
          Array.from({ length: data.recommendations.length }).map((_, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="h-64 bg-gray-200 dark:bg-gray-700 rounded-xl animate-pulse"
            />
          ))
        ) : (
          data.recommendations.map((articleId, index) => {
            const metadata = getArticleMetadata(articleId)
            return (
              <RecommendationCard
                key={articleId}
                articleId={articleId}
                title={metadata?.title}
                description={metadata?.description}
                category={metadata?.category}
                readTime={metadata?.words_count ? Math.max(1, Math.ceil(metadata.words_count / 200)) : undefined}
                confidence={data.confidence}
                index={index}
                onClick={() => {
                  // Simuler ouverture d'article (dans une vraie app, rediriger vers l'article)
                  const url = `https://example.com/article/${articleId}`
                  window.open(url, '_blank')
                  console.log(`Ouverture article ${articleId}: ${metadata?.title}`)
                }}
              />
            )
          })
        )}
      </div>

      {/* Statistics */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="text-center pt-8"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 text-sm text-muted-foreground">
          <TrendingUp className="h-4 w-4" />
          Recommandations générées par IA en temps réel
        </div>
      </motion.div>
    </motion.div>
  )
}

// Badge component for method confidence
const BadgeComponent = ({ variant = "default", className = "", children, ...props }: {
  variant?: "default" | "secondary" | "destructive" | "outline"
  className?: string
  children: React.ReactNode
}) => {
  const variants = {
    default: "bg-primary text-primary-foreground hover:bg-primary/80",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/80",
    outline: "text-foreground border border-input bg-background hover:bg-accent hover:text-accent-foreground",
  }

  return (
    <div
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${variants[variant]} ${className}`}
      {...props}
    >
      {children}
    </div>
  )
}