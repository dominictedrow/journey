import { create } from 'zustand';
import { apiFetch } from '../api/client';
export const useAppStore = create((set) => ({
    initialized: false,
    isOnboarded: false,
    profile: null,
    account: null,
    loadProfile: async () => {
        const data = await apiFetch('/api/profile');
        set({
            initialized: true,
            profile: data?.profile ?? null,
            account: data?.account ?? null,
            isOnboarded: !!(data?.profile && data?.account),
        });
    },
}));
