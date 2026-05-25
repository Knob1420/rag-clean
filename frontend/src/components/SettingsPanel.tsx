import { useAppStore } from '../stores/appStore'

export function SettingsPanel() {
  const { settingsPanelOpen, setSettingsPanelOpen, settings, updateSettings } = useAppStore()

  return (
    <>
      {/* Overlay */}
      {settingsPanelOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-40"
          onClick={() => setSettingsPanelOpen(false)}
        />
      )}

      {/* Panel */}
      <div
        className={`
          fixed top-0 right-0 bottom-0 w-80 z-50
          bg-white dark:bg-[#111]
          border-l border-gray-200 dark:border-gray-800
          transform transition-transform duration-200 ease-in-out
          ${settingsPanelOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-800">
          <h2 className="font-semibold text-gray-900 dark:text-white">设置</h2>
          <button
            onClick={() => setSettingsPanelOpen(false)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
          >
            <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Settings */}
        <div className="p-4 space-y-6 overflow-y-auto h-full">
          {/* Top K */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">召回数量</label>
              <span className="text-sm text-blue-600 dark:text-blue-400 font-medium">{settings.top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              value={settings.top_k}
              onChange={(e) => updateSettings({ top_k: Number(e.target.value) })}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>1</span>
              <span>50</span>
            </div>
          </div>

          {/* HyDE */}
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">启用 HyDE</label>
            <button
              onClick={() => updateSettings({ use_hyde: !settings.use_hyde })}
              className={`relative w-11 h-6 rounded-full transition-colors ${
                settings.use_hyde ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  settings.use_hyde ? 'translate-x-5' : ''
                }`}
              />
            </button>
          </div>

          {/* Rerank */}
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">启用 Rerank</label>
            <button
              onClick={() => updateSettings({ use_rerank: !settings.use_rerank })}
              className={`relative w-11 h-6 rounded-full transition-colors ${
                settings.use_rerank ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
              }`}
            >
              <span
                className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                  settings.use_rerank ? 'translate-x-5' : ''
                }`}
              />
            </button>
          </div>

          {/* Rerank Top K */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Rerank 保留数量</label>
              <span className="text-sm text-blue-600 dark:text-blue-400 font-medium">{settings.rerank_top_k}</span>
            </div>
            <input
              type="range"
              min="1"
              max="50"
              value={settings.rerank_top_k}
              onChange={(e) => updateSettings({ rerank_top_k: Number(e.target.value) })}
              className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>1</span>
              <span>50</span>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}