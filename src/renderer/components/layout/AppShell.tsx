import { useState } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'

export default function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="flex min-h-screen bg-black">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="flex-1 flex flex-col min-h-screen">
        <button
          onClick={() => setSidebarOpen(true)}
          className="fixed top-4 left-4 z-10 p-2 text-white text-xl leading-none hover:opacity-70 transition-opacity"
          aria-label="Open menu"
        >
          ☰
        </button>
        <Outlet />
      </div>
    </div>
  )
}
