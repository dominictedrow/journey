import { create } from 'zustand'
import { apiFetch } from '../api/client'

interface Profile {
  id: number
  username: string
  pfpUrl: string | null
}

interface Account {
  id: string
  username: string
  followersCount: number | null
}

interface AppState {
  initialized: boolean
  isOnboarded: boolean
  profile: Profile | null
  account: Account | null
  loadProfile: () => Promise<void>
}

export const useAppStore = create<AppState>((set) => ({
  initialized: false,
  isOnboarded: false,
  profile: null,
  account: null,

  loadProfile: async () => {
    try {
      const data = await apiFetch<{ profile: Profile; account: Account | null } | null>('/api/profile')
      set({
        initialized: true,
        profile: data?.profile ?? null,
        account: data?.account ?? null,
        isOnboarded: !!(data?.profile && data?.account),
      })
    } catch {
      set({ initialized: true })
    }
  },
}))
