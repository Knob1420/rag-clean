import { create } from 'zustand'
import type { Session, Message, ChatSettings, Theme, Source } from '../types'

const API_BASE = ''

interface AppState {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void

  sessions: Session[]
  currentSessionId: string | null
  loadSessions: () => Promise<void>
  createSession: () => Promise<Session>
  deleteSession: (id: string) => Promise<void>
  selectSession: (id: string) => Promise<void>
  updateSessionTitle: (id: string, title: string) => void

  messages: Message[]
  loadMessages: (sessionId: string) => Promise<void>
  addMessage: (message: Message) => void
  clearMessages: () => void

  settings: ChatSettings
  updateSettings: (settings: Partial<ChatSettings>) => void

  isStreaming: boolean
  sendMessage: (content: string, file: File | null) => Promise<void>

  uploadedFile: File | null
  setUploadedFile: (file: File | null) => void

  settingsPanelOpen: boolean
  setSettingsPanelOpen: (open: boolean) => void

  sidebarOpen: boolean
  setSidebarOpen: (open: boolean) => void

}

// ─── 消息持久化辅助 ───
function _saveMessages(sessionId: string, messages: Message[]) {
  try {
    localStorage.setItem(`rag_msgs_${sessionId}`, JSON.stringify(messages))
  } catch {}
}

function _loadMessages(sessionId: string): Message[] {
  try {
    const stored = localStorage.getItem(`rag_msgs_${sessionId}`)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

export const useAppStore = create<AppState>((set, get) => ({
  theme: 'light',
  setTheme: (theme) => set({ theme }),
  toggleTheme: () => set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' })),

  sessions: [],
  currentSessionId: null,

  loadSessions: async () => {
    // sessions 由前端本地管理，后端无此接口
    const stored = localStorage.getItem('rag_sessions')
    if (stored) {
      try {
        const sessions = JSON.parse(stored)
        set({ sessions })
      } catch {}
    }
  },

  createSession: async () => {
    // 本地创建 session
    const session = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      title: '新对话',
      created_at: new Date().toISOString(),
    }
    const sessions = [session, ...get().sessions]
    localStorage.setItem('rag_sessions', JSON.stringify(sessions))
    set((state) => ({
      sessions: [session, ...state.sessions],
      currentSessionId: session.id,
      messages: [],
    }))
    return session
  },

  deleteSession: async (id) => {
    // 本地删除 session + 对应消息
    const sessions = get().sessions.filter((s) => s.id !== id)
    localStorage.setItem('rag_sessions', JSON.stringify(sessions))
    localStorage.removeItem(`rag_msgs_${id}`)
    set((state) => ({
      sessions: state.sessions.filter((s) => s.id !== id),
      currentSessionId: state.currentSessionId === id ? null : state.currentSessionId,
      messages: state.currentSessionId === id ? [] : state.messages,
    }))
  },

  selectSession: async (id) => {
    const messages = _loadMessages(id)
    set({ currentSessionId: id, messages })
  },

  updateSessionTitle: (id, title) => {
    const sessions = get().sessions.map((s) => (s.id === id ? { ...s, title } : s))
    localStorage.setItem('rag_sessions', JSON.stringify(sessions))
    set({ sessions })
  },

  messages: [],

  loadMessages: async (_sessionId) => {
    // 消息由前端本地管理
    set({ messages: [] })
  },

  addMessage: (message) => {
    set((state) => {
      const messages = [...state.messages, message]
      // 自动持久化
      if (state.currentSessionId) {
        _saveMessages(state.currentSessionId, messages)
      }
      return { messages }
    })
  },

  clearMessages: () => set({ messages: [] }),

  settings: {
    mode: 'quick',
    top_k: 20,
    use_hyde: false,
    use_rerank: true,
    rerank_top_k: 10,
  },

  updateSettings: (newSettings) => set((state) => ({ settings: { ...state.settings, ...newSettings } })),

  isStreaming: false,

  sendMessage: async (content, _file) => {
    const { currentSessionId, settings, addMessage, updateSessionTitle } = get()

    let sessionId = currentSessionId
    if (!sessionId) {
      const session = await get().createSession()
      sessionId = session.id
    }

    const userMessage: Message = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      session_id: sessionId,
      role: 'user',
      content,
      sources: [],
      created_at: new Date().toISOString(),
    }
    addMessage(userMessage)

    const messages = get().messages
    if (messages.length === 1) {
      const title = content.slice(0, 30) + (content.length > 30 ? '...' : '')
      updateSessionTitle(sessionId, title)
    }

    set({ isStreaming: true })

    const payload = {
      query: content,
      mode: settings.mode,
      top_k: settings.top_k,
      use_hyde: settings.use_hyde,
      use_rerank: settings.use_rerank,
      rerank_top_k: settings.rerank_top_k,
    }

    const assistantMessage: Message = {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      session_id: sessionId,
      role: 'assistant',
      content: '',
      sources: [],
      created_at: new Date().toISOString(),
    }
    addMessage(assistantMessage)

    if (settings.mode === 'agent') {
      // Initialize thinkingSteps on the assistant message
      set((state) => ({
        messages: state.messages.map((m) =>
          m.id === assistantMessage.id ? { ...m, thinkingSteps: [] } : m
        ),
      }))
    }

    // ─── SSE 事件处理（带缓冲批处理）───
    let sources: Source[] = []

    // 缓冲区：避免每个 token 都触发昂贵的 state 更新
    let tokenBuffer = ''           // answer token 缓冲
    let thoughtBuffer = ''         // thought_token 缓冲
    let flushTimerId: ReturnType<typeof setTimeout> | null = null

    function flushBuffers() {
      flushTimerId = null
      const tokens = tokenBuffer
      const thoughts = thoughtBuffer
      tokenBuffer = ''
      thoughtBuffer = ''
      if (!tokens && !thoughts) return

      set((state) => ({
        messages: state.messages.map((m) => {
          if (m.id !== assistantMessage.id) return m
          const steps = m.thinkingSteps || []
          const currentIteration = steps.length > 0 ? steps[steps.length - 1].iteration : 0
          return {
            ...m,
            content: m.content + tokens,
            thinkingSteps: thoughts
              ? steps.map((s) =>
                  s.iteration === currentIteration
                    ? { ...s, thought: (s.thought || '') + thoughts }
                    : s
                )
              : steps,
          }
        }),
      }))
    }

    function scheduleFlush() {
      if (!flushTimerId) {
        flushTimerId = setTimeout(flushBuffers, 80)  // 每 80ms 批量刷新一次
      }
    }

    function handleEvent(eventType: string, data: any) {
      if (eventType === 'token') {
        const token = data.content || ''
        if (token) {
          tokenBuffer += token
          scheduleFlush()
        }
      } else if (eventType === 'sources') {
        flushBuffers()  // 先刷新挂起的 token
        sources = data.sources || []
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === assistantMessage.id ? { ...m, sources } : m
          ),
        }))
      } else if (eventType === 'step_start') {
        flushBuffers()  // 先刷新挂起的 token
        if (settings.mode === 'agent') {
          const action = data.action || 'think'
          const step = {
            iteration: data.iteration,
            action: action,
            step_content: data.step_content || '',
            thought: '',
            duration: 0,
          }
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === assistantMessage.id
                ? { ...m, thinkingSteps: [...(m.thinkingSteps || []), step] }
                : m
            ),
          }))
        }
      } else if (eventType === 'step_end') {
        flushBuffers()  // 先刷新挂起的 token
        if (settings.mode === 'agent') {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === assistantMessage.id
                ? {
                    ...m,
                    thinkingSteps: (m.thinkingSteps || []).map((s) =>
                      s.iteration === data.iteration
                        ? { ...s, duration: data.duration || 0, step_content: data.step_content || '', action: data.action || s.action }
                        : s
                    ),
                  }
                : m
            ),
          }))
        }
      } else if (eventType === 'thought') {
        flushBuffers()
        if (settings.mode === 'agent') {
          set((state) => {
            const msg = state.messages.find((m) => m.id === assistantMessage.id)
            const steps = msg?.thinkingSteps || []
            const existingStep = steps.find(s => s.iteration === data.iteration)
            let newSteps: typeof steps
            if (existingStep) {
              newSteps = steps.map((s) =>
                s.iteration === data.iteration
                  ? { ...s, thought: data.thought || '' }
                  : s
              )
            } else {
              newSteps = [...steps, {
                iteration: data.iteration,
                action: 'thinking',
                step_content: '',
                thought: data.thought || '',
                duration: 0,
              }]
            }
            return {
              messages: state.messages.map((m) =>
                m.id === assistantMessage.id ? { ...m, thinkingSteps: newSteps } : m
              ),
            }
          })
        }
      } else if (eventType === 'thought_token') {
        if (settings.mode === 'agent') {
          const content = data.content || ''
          if (content) {
            thoughtBuffer += content
            scheduleFlush()
          }
        }
      } else if (eventType === 'done') {
        flushBuffers()  // 确保所有挂起的 token 都已刷新
      }
      // heartbeat, tool_arg 等事件无需处理
    }

    // SSE 解析
    let eventBuffer = { event: '', data: '' }

    async function processStream() {
      // 10 分钟超时 — agent 查询可能很慢
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10 * 60 * 1000)

      try {
        console.log('[SSE] connecting to', `${API_BASE}/api/v1/chat/stream`)
        const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          signal: controller.signal,
        })

        console.log('[SSE] response status:', response.status, response.statusText)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)

        const reader = response.body?.getReader()
        if (!reader) throw new Error('No reader available')

        const decoder = new TextDecoder()

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const text = decoder.decode(value, { stream: true })
          const lines = text.split('\n')

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventBuffer.event = line.slice(7).trim()
            } else if (line.startsWith('data: ')) {
              eventBuffer.data = line.slice(6).trim()
            } else if (line.trim() === '' && eventBuffer.event && eventBuffer.data) {
              const eventType = eventBuffer.event
              let data
              try {
                data = JSON.parse(eventBuffer.data)
              } catch (e) {
                console.error('[SSE Parse Error]', eventBuffer.data.slice(0, 100))
                eventBuffer = { event: '', data: '' }
                continue
              }

              // 立即处理事件
              handleEvent(eventType, data)

              eventBuffer = { event: '', data: '' }
            }
          }
        }
      } catch (e: any) {
        console.error('[SSE] error:', e?.name, e?.message, e?.cause)
        console.error('[SSE] API_BASE was:', API_BASE)
        console.error('[SSE] URL was:', `${API_BASE}/api/v1/chat/stream`)

        // 区分超时 / 网络错误 / 其他错误，给出可读提示
        let userMsg: string
        if (e?.name === 'AbortError') {
          userMsg = '请求超时，Agent 推理时间过长，请稍后重试。'
        } else if (e instanceof TypeError && e.message === 'Failed to fetch') {
          userMsg = '网络连接中断，请检查网络后重试。'
        } else if (e?.message?.startsWith('HTTP')) {
          userMsg = `服务器错误（${e.message}），请稍后重试。`
        } else {
          userMsg = `连接异常：${e?.message || e}，请重试。`
        }

        const errorMsg: Message = {
          id: Date.now().toString(36) + Math.random().toString(36).slice(2),
          session_id: sessionId!,
          role: 'assistant',
          content: userMsg,
          sources: [],
          created_at: new Date().toISOString(),
        }
        set((state) => ({ messages: [...state.messages, errorMsg] }))
      } finally {
        clearTimeout(timeoutId)
        flushBuffers()  // 确保最后一批 token 不丢失
        set({ isStreaming: false, uploadedFile: null })
        // 流式结束后持久化消息
        const { currentSessionId, messages } = get()
        if (currentSessionId) {
          _saveMessages(currentSessionId, messages)
        }
      }
    }

    await processStream()
  },

  uploadedFile: null,
  setUploadedFile: (file) => set({ uploadedFile: file }),

  settingsPanelOpen: false,
  setSettingsPanelOpen: (open) => set({ settingsPanelOpen: open }),

  sidebarOpen: false,
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

}))