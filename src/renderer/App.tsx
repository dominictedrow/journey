import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { useAppStore } from './store/appStore'
import AppShell from './components/layout/AppShell'
import OnboardingPage from './pages/OnboardingPage'
import HomePage from './pages/HomePage'
import LibraryPage from './pages/LibraryPage'
import ScanningPage from './pages/ScanningPage'
import FormatDescriptionPage from './pages/FormatDescriptionPage'
import ProfilePage from './pages/ProfilePage'

export default function App() {
  const { initialized, isOnboarded, loadProfile } = useAppStore()

  useEffect(() => {
    loadProfile()
  }, [])

  if (!initialized) {
    return <div className="min-h-screen bg-black" />
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <Routes>
        {!isOnboarded ? (
          <>
            <Route path="/onboarding" element={<OnboardingPage />} />
            <Route path="*" element={<Navigate to="/onboarding" replace />} />
          </>
        ) : (
          <>
            <Route path="/onboarding" element={<Navigate to="/home" replace />} />
            <Route element={<AppShell />}>
              <Route index element={<Navigate to="/home" replace />} />
              <Route path="/home" element={<HomePage />} />
              <Route path="/library" element={<LibraryPage />} />
              <Route path="/scan" element={<ScanningPage />} />
              <Route path="/format/:id" element={<FormatDescriptionPage />} />
              <Route path="/profile" element={<ProfilePage />} />
            </Route>
          </>
        )}
      </Routes>
    </div>
  )
}
