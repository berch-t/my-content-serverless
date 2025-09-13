"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import dynamic from "next/dynamic"

// Import components dynamically to avoid SSR hydration issues
const UserIdInput = dynamic(() => import("@/components/UserIdInput").then(mod => ({ default: mod.UserIdInput })), { ssr: false })
const RecommendationsList = dynamic(() => import("@/components/RecommendationsList").then(mod => ({ default: mod.RecommendationsList })), { ssr: false })
const ApiModeSelector = dynamic(() => import("@/components/ApiModeSelector").then(mod => ({ default: mod.ApiModeSelector })), { ssr: false })
import { apiManager, type ApiMode, type RecommendationResponse } from "@/lib/api-manager"
import { Card, CardContent } from "@/components/ui/card"
import ClientOnly from "@/components/ui/ClientOnly"
import Image from "next/image"

// Import LightRays dynamically
const LightRays = dynamic(() => import("@/components/ui/LightRays"), { ssr: false })

// Import icons dynamically
const Brain = dynamic(() => import("lucide-react").then(mod => ({ default: mod.Brain })), { ssr: false })
const Sparkles = dynamic(() => import("lucide-react").then(mod => ({ default: mod.Sparkles })), { ssr: false })
const Zap = dynamic(() => import("lucide-react").then(mod => ({ default: mod.Zap })), { ssr: false })

export default function Home() {
  const [loading, setLoading] = useState(false)
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiMode, setApiMode] = useState<ApiMode>(apiManager.getMode())
  const [apiSource, setApiSource] = useState<'local' | 'cloud' | null>(null)

  const handleGetRecommendations = async (userId: number) => {
    setLoading(true)
    setError(null)
    setApiSource(null)

    try {
      const { data, source } = await apiManager.getRecommendations(userId, 5)
      setRecommendations(data)
      setApiSource(source)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Une erreur est survenue")
    } finally {
      setLoading(false)
    }
  }

  const handleModeChange = (mode: ApiMode) => {
    setApiMode(mode)
    apiManager.setMode(mode)
  }

  const handleBack = () => {
    setRecommendations(null)
    setError(null)
  }

  const floatingElements = [
    { icon: Brain, delay: 0, x: "10%", y: "20%" },
    { icon: Sparkles, delay: 0.5, x: "80%", y: "15%" },
    { icon: Zap, delay: 1, x: "15%", y: "70%" },
    { icon: Brain, delay: 1.5, x: "85%", y: "75%" },
  ]

  return (
    <main className="min-h-full relative" style={{minHeight: '90vh'}}>
      {/* Light Rays Background */}
      <div className="fixed inset-0 z-0">
        <LightRays
          raysOrigin="top-center"
          raysColor="#ac40ff"
          raysSpeed={1.2}
          lightSpread={0.8}
          rayLength={3.0}
          followMouse={true}
          mouseInfluence={0.08}
          noiseAmount={0.05}
          distortion={0.1}
          fadeDistance={2.0}
          saturation={1.2}
          className="opacity-70"
        />
      </div>
      
      {/* Animated Background Elements */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {floatingElements.map((element, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0, scale: 0 }}
            animate={{ 
              opacity: [0, 0.1, 0.05, 0.1],
              scale: [0, 1, 1.2, 1],
              rotate: [0, 180, 360]
            }}
            transition={{
              duration: 8,
              delay: element.delay,
              repeat: Infinity,
              repeatType: "reverse"
            }}
            className="absolute text-primary/20"
            style={{ left: element.x, top: element.y }}
          >
            <element.icon className="h-16 w-16" />
          </motion.div>
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-4 py-6">
        <div className="space-y-12">
            {/* Hero Section */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="text-center space-y-6 max-w-4xl mx-auto"
            >
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="flex justify-center"
              >
                <Image
                  src="/logo.png"
                  alt="My Content - AI Recommendation System"
                  width={200}
                  height={60}
                  priority
                  className="max-w-full h-auto"
                />
              </motion.div>
              
              {/* Horizontal separator line */}
              <motion.div
                initial={{ opacity: 0, scaleX: 0 }}
                animate={{ opacity: 1, scaleX: 1 }}
                transition={{ duration: 0.8, delay: 0.6 }}
                className="w-3/4 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent mx-auto"
              />
              
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto leading-relaxed"
              >
                Recommandation de contenus servi par un modèle de prédiction, rating et filtrage collaboratif
              </motion.p>
              
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.6, delay: 0.6 }}
                className="flex flex-wrap justify-center gap-4 pt-4"
              >
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm">
                  <Brain className="h-4 w-4" />
                  Content-Based
                </div>
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 text-blue-600 text-sm">
                  <Sparkles className="h-4 w-4" />
                  Collaborative
                </div>
                <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-purple-500/10 text-purple-600 text-sm">
                  <Zap className="h-4 w-4" />
                  Hybrid AI
                </div>
              </motion.div>
            </motion.div>

            {/* User Input Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <UserIdInput onSubmit={handleGetRecommendations} loading={loading} />
            </motion.div>

            {/* Error Display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="max-w-md mx-auto"
              >
                <Card className="border-destructive/50 bg-destructive/5">
                  <CardContent className="pt-6">
                    <p className="text-destructive text-center">{error}</p>
                  </CardContent>
                </Card>
              </motion.div>
            )}
            
            {/* Recommendations Display */}
            {recommendations && (
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, delay: 0.3 }}
                className="pt-12"
              >
                <RecommendationsList data={recommendations} onBack={handleBack} />
              </motion.div>
            )}
            
            {/* API Mode Selector - always visible at bottom */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
              className="pt-12"
            >
              <ApiModeSelector
                currentMode={apiMode}
                onModeChange={handleModeChange}
              />
            </motion.div>

            {/* Features Section */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 1 }}
              className="grid md:grid-cols-3 gap-8 pt-16"
            >
              {[
                {
                  icon: Brain,
                  title: "Analyse Intelligente",
                  description: "Notre IA analyse le contenu des articles pour comprendre vos préférences"
                },
                {
                  icon: Sparkles,
                  title: "Recommandations Personnalisées",
                  description: "Algorithme hybride combinant analyse de contenu et filtrage collaboratif"
                },
                {
                  icon: Zap,
                  title: "Temps Réel",
                  description: "Recommandations générées instantanément avec un score de confiance"
                }
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 1.2 + index * 0.2 }}
                  whileHover={{ scale: 1.05 }}
                  className="text-center space-y-4"
                >
                  <div className="mx-auto w-16 h-16 bg-gradient-to-br from-primary/20 to-blue-500/20 rounded-full flex items-center justify-center">
                    <feature.icon className="h-8 w-8 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {feature.description}
                  </p>
                </motion.div>
              ))}
            </motion.div>
          </div>
      </div>
    </main>
  )
}