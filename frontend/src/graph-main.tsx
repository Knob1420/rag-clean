import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { GraphView } from './components/GraphView'
import './index.css'

// Graph page entry point
function GraphPage() {
  return (
    <div className="h-screen flex flex-col bg-white dark:bg-black text-gray-900 dark:text-white">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <button
            onClick={() => window.location.href = '/'}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-lg font-semibold">知识图谱</h1>
        </div>
        <div className="text-sm text-gray-500">
          {new Date().toLocaleDateString('zh-CN')}
        </div>
      </header>

      {/* Graph container */}
      <div className="flex-1 overflow-hidden">
        <GraphView />
      </div>
    </div>
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <GraphPage />
  </StrictMode>,
)