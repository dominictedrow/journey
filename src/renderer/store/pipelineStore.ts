import { create } from 'zustand'

interface PipelineState {
  jobId: string | null
  progress: number
  status: 'idle' | 'running' | 'done' | 'error'
  setJob: (jobId: string) => void
  setProgress: (progress: number) => void
  setStatus: (status: PipelineState['status']) => void
  reset: () => void
}

export const usePipelineStore = create<PipelineState>((set) => ({
  jobId: null,
  progress: 0,
  status: 'idle',
  setJob: (jobId) => set({ jobId, status: 'running', progress: 0 }),
  setProgress: (progress) => set({ progress }),
  setStatus: (status) => set({ status }),
  reset: () => set({ jobId: null, progress: 0, status: 'idle' }),
}))
