"use client"

import { motion } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { BookOpen, Clock, TrendingUp, Eye } from "lucide-react"

interface RecommendationCardProps {
  articleId: number
  title?: string
  description?: string
  category?: string
  readTime?: number
  confidence: number
  index: number
  onClick?: () => void
}

export function RecommendationCard({
  articleId,
  title = `Article ${articleId}`,
  description = "Découvrez ce contenu personnalisé sélectionné spécialement pour vous par notre système de recommandation hybride.",
  category = "Actualités",
  readTime = Math.floor(Math.random() * 10) + 2,
  confidence,
  index,
  onClick
}: RecommendationCardProps) {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "text-green-600 bg-green-50"
    if (confidence >= 0.6) return "text-blue-600 bg-blue-50"
    return "text-orange-600 bg-orange-50"
  }

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return "Hautement recommandé"
    if (confidence >= 0.6) return "Recommandé"
    return "Potentiellement intéressant"
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      whileHover={{ scale: 1.02, y: -4 }}
      className="h-full"
    >
      <Card className="h-full cursor-pointer group hover:shadow-xl transition-all duration-300 border-0 glass-effect bg-white/10 dark:bg-white/5 backdrop-blur-lg">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <motion.div
                  whileHover={{ rotate: 15 }}
                  className="p-1.5 rounded-md bg-primary/10 text-primary"
                >
                  <BookOpen className="h-4 w-4" />
                </motion.div>
                <span className="text-xs font-medium text-muted-foreground px-2 py-1 rounded-full bg-secondary">
                  {category}
                </span>
              </div>
              <CardTitle className="text-lg leading-tight group-hover:text-primary transition-colors">
                {title}
              </CardTitle>
            </div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className={`text-xs font-medium px-2 py-1 rounded-full ${getConfidenceColor(confidence)}`}
            >
              {Math.round(confidence * 100)}%
            </motion.div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <CardDescription className="text-sm leading-relaxed">
            {description}
          </CardDescription>
          
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <Clock className="h-3.5 w-3.5" />
                <span>{readTime} min</span>
              </div>
              <div className="flex items-center gap-1">
                <TrendingUp className="h-3.5 w-3.5" />
                <span>{getConfidenceText(confidence)}</span>
              </div>
            </div>
            <div className="text-xs font-mono opacity-70">
              #{articleId}
            </div>
          </div>
          
          <motion.div whileHover={{ scale: 1.01 }} whileTap={{ scale: 0.99 }}>
            <Button 
              variant="outline" 
              size="sm" 
              className="w-full group-hover:bg-primary group-hover:text-primary-foreground transition-all duration-300"
              onClick={onClick}
            >
              <Eye className="h-4 w-4 mr-2" />
              Lire l'article
            </Button>
          </motion.div>
        </CardContent>
      </Card>
    </motion.div>
  )
}