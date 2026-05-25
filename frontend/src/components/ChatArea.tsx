import { Component, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { useAppStore } from '../stores/appStore'
import type { Message } from '../types'

// ErrorBoundary to catch rendering errors and prevent white screen
class ChatErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean; error: string }> {
  state = { hasError: false, error: '' }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex-1 flex items-center justify-center text-red-500 text-sm p-8">
          <div className="text-center">
            <p className="font-medium mb-2">渲染出错</p>
            <p className="text-xs text-gray-500 mb-3">{this.state.error}</p>
            <button
              className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded text-xs"
              onClick={() => this.setState({ hasError: false, error: '' })}
            >
              重试
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

export function ChatArea() {
  const { messages, isStreaming, settings } = useAppStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Check if any message has thinking steps (for the "Thinking..." fallback)
  const hasThinkingSteps = messages.some(m => m.thinkingSteps && m.thinkingSteps.length > 0)

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-white dark:bg-black">
    <ChatErrorBoundary>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-4">🚀</div>
              <h2 className="text-xl font-medium text-gray-700 dark:text-gray-300 mb-2">
                之江太空计算 RAG 助手
              </h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                询问关于卫星、空间计算产品的问题
              </p>
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, idx) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'assistant'}
              />
            ))}
            {isStreaming && !(settings.mode === 'agent' && hasThinkingSteps) && (
              <div className="flex items-center gap-2 text-gray-500 dark:text-gray-400 text-sm">
                <span className="animate-pulse">Thinking...</span>
              </div>
            )}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
    </ChatErrorBoundary>
    </div>
  )
}

function MessageBubble({ message, isStreaming }: { message: Message; isStreaming: boolean }) {
  const isUser = message.role === 'user'
  const [sourcesExpanded, setSourcesExpanded] = useState(false)

  // Per-message thinking steps collapse state
  const [thinkingCollapsed, setThinkingCollapsed] = useState(false)
  const thinkingSteps = message.thinkingSteps || []

  // Auto-collapse thinking steps once streaming finishes
  useEffect(() => {
    if (!isStreaming && thinkingSteps.length > 0) {
      setThinkingCollapsed(true)
    } else if (isStreaming && thinkingSteps.length > 0) {
      setThinkingCollapsed(false)
    }
  }, [isStreaming, thinkingSteps.length])

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`
          max-w-[85%] rounded-2xl px-5 py-4
          ${isUser
            ? 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
            : 'bg-white dark:bg-[#111] border border-gray-200 dark:border-gray-800 text-gray-900 dark:text-white'
          }
        `}
      >
        {/* Thinking steps for agent mode */}
        {!isUser && thinkingSteps.length > 0 && (
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
            <button
              onClick={() => setThinkingCollapsed(!thinkingCollapsed)}
              className="flex items-center gap-1 hover:text-gray-700 dark:hover:text-gray-300 mb-1"
            >
              <svg
                className={`w-3 h-3 transition-transform ${thinkingCollapsed ? '' : 'rotate-90'}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span>思考过程 ({thinkingSteps.length} 步){isStreaming ? ' 🔄' : ''}</span>
            </button>
            {!thinkingCollapsed && (
              <div className="pl-4 space-y-1 mt-2">
                {thinkingSteps.map((step, idx) => {
                  const isCurrent = isStreaming && idx === thinkingSteps.length - 1 && step.duration === 0
                  return (
                    <div key={step.iteration} className="flex items-start gap-2">
                      <span className="text-gray-400 flex-shrink-0">{step.iteration}.</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1">
                          <span className="font-medium text-blue-600 dark:text-blue-400">
                            {step.action}
                          </span>
                          {isCurrent && (
                            <span className="inline-block w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse" />
                          )}
                          {step.duration > 0 && (
                            <span className="text-gray-400">({step.duration}s)</span>
                          )}
                        </div>
                        {step.thought && (
                          <p className="text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2 italic">
                            {step.thought}
                          </p>
                        )}
                        {step.step_content && (
                          <p className="text-gray-400 dark:text-gray-500 mt-0.5">
                            → {step.step_content}
                          </p>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* Content - render as markdown for assistant */}
        {isUser ? (
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
        ) : (
          <div className="text-sm markdown-body">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {/* Sources - collapsible */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={() => setSourcesExpanded(!sourcesExpanded)}
              className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
            >
              <svg
                className={`w-3 h-3 transition-transform ${sourcesExpanded ? 'rotate-90' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span>参考文档 ({message.sources.length})</span>
            </button>

            {sourcesExpanded && (
              <div className="mt-2 space-y-1 pl-4">
                {message.sources.map((source, i) => (
                  <div key={i} className="text-xs text-gray-600 dark:text-gray-400 flex items-center gap-1">
                    <svg className="w-3 h-3 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <span className="truncate">{source.doc_name}</span>
                    <span className="text-gray-400">({(source.score ?? 0).toFixed(2)})</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}