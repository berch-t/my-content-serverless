"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { User, Sparkles, ArrowRight } from "lucide-react"

interface UserIdInputProps {
  onSubmit: (userId: number) => void
  loading?: boolean
}

export function UserIdInput({ onSubmit, loading = false }: UserIdInputProps) {
  const [userId, setUserId] = useState("")
  const [error, setError] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    
    const id = parseInt(userId)
    if (isNaN(id) || id < 0) {
      setError("Veuillez entrer un ID utilisateur valide (nombre positif)")
      return
    }
    
    onSubmit(id)
  }

  const suggestedUsers = [123, 456, 789, 1001, 2024]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="w-full max-w-md mx-auto"
    >
      <Card className="border-0 glass-effect bg-white/20 dark:bg-white/10 backdrop-blur-lg shadow-xl">
        <CardHeader className="text-center pb-6">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4"
          >
            <User className="h-8 w-8 text-primary" />
          </motion.div>
          <CardTitle className="text-2xl gradient-text">
            Découvrez vos recommandations
          </CardTitle>
          <CardDescription className="text-base">
            Entrez votre ID utilisateur pour obtenir 5 articles personnalisés
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="userId" className="text-sm font-medium text-foreground">
                ID Utilisateur
              </label>
              <div className="relative">
                <Input
                  id="userId"
                  type="number"
                  placeholder="Ex: 123"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  className="pr-12"
                  min="0"
                />
                <motion.div
                  animate={{ rotate: loading ? 360 : 0 }}
                  transition={{ duration: 1, repeat: loading ? Infinity : 0 }}
                  className="absolute right-3 top-1/2 -translate-y-1/2"
                >
                  <Sparkles className="h-4 w-4 text-primary" />
                </motion.div>
              </div>
              {error && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-sm text-destructive"
                >
                  {error}
                </motion.p>
              )}
            </div>
            
            <Button
              type="submit"
              className="w-full"
              size="lg"
              disabled={loading || !userId}
            >
              {loading ? (
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="mr-2"
                >
                  <Sparkles className="h-4 w-4" />
                </motion.div>
              ) : (
                <ArrowRight className="h-4 w-4 mr-2" />
              )}
              {loading ? "Génération..." : "Obtenir mes recommandations"}
            </Button>
          </form>
          
          <div className="pt-4 border-t">
            <p className="text-sm text-muted-foreground mb-3 text-center opacity-80">
              Ou essayez avec ces utilisateurs d'exemple :
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {suggestedUsers.map((id) => (
                <motion.button
                  key={id}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setUserId(id.toString())}
                  className="px-3 py-1 text-xs bg-white/20 hover:bg-white/30 dark:bg-white/10 dark:hover:bg-white/20 rounded-full transition-colors text-foreground"
                  disabled={loading}
                >
                  {id}
                </motion.button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}