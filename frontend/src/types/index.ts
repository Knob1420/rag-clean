export interface ChatSettings {
  mode: 'quick' | 'agent'
  top_k: number
  use_hyde: boolean
  use_rerank: boolean
  rerank_top_k: number
}

export interface Source {
  chunk_id: string
  doc_id: string
  doc_name: string
  score: number
  snippet: string
}

export interface ThinkingStep {
  iteration: number
  action: string
  step_content: string
  duration: number
  thought?: string
}

export interface Message {
  id: string
  session_id: string
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  created_at: string
  thinkingSteps?: ThinkingStep[]
}

export interface Session {
  id: string
  title: string
  created_at: string
  updated_at: string
  settings: ChatSettings
  messages?: Message[]
}

export interface ChatRequest {
  query: string
  mode: 'quick' | 'agent'
  top_k: number
  use_hyde: boolean
  use_rerank: boolean
  rerank_top_k: number
  session_id?: string
  file?: File | null
}

export interface ChatResponse {
  answer: string
  sources: Source[]
  time: Record<string, number>
  usage: TokenUsage
  chunks_count: number
}

export interface TokenUsage {
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
}

export type Theme = 'light' | 'dark'