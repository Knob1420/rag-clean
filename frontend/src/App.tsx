import { useEffect } from 'react'
import { useAppStore } from './stores/appStore'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { ChatArea } from './components/ChatArea'
import { InputBox } from './components/InputBox'
import { SettingsPanel } from './components/SettingsPanel'
import './index.css'

function App() {
  const { theme } = useAppStore()

  // Apply theme class to html element
  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  return (
    <div className="h-screen flex flex-col bg-white dark:bg-black text-gray-900 dark:text-white">
      <Header />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar />
        <main className="flex-1 flex flex-col overflow-hidden">
          <ChatArea />
          <InputBox />
        </main>
        <SettingsPanel />
      </div>
    </div>
  )
}

export default App