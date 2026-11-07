import { create } from 'zustand';
export const usePipelineStore = create((set) => ({
    jobId: null,
    progress: 0,
    status: 'idle',
    setJob: (jobId) => set({ jobId, status: 'running', progress: 0 }),
    setProgress: (progress) => set({ progress }),
    setStatus: (status) => set({ status }),
    reset: () => set({ jobId: null, progress: 0, status: 'idle' }),
}));
