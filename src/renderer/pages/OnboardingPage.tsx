import { useState, useEffect } from 'react'
import { IG_GRADIENT } from '../constants/styles'
import InstagramButton from '../components/shared/InstagramButton'
import TextInput from '../components/shared/TextInput'
import { apiFetch } from '../api/client'
import { useAppStore } from '../store/appStore'

type Step = 'loading' | 'name' | 'igToken' | 'grokKey' | 'connecting' | 'done' | 'error'

interface EnvConfigured {
  igToken: boolean
  grokKey: boolean
}

export default function OnboardingPage() {
  const [name, setName] = useState('')
  const [igToken, setIgToken] = useState('')
  const [grokKey, setGrokKey] = useState('')
  const [step, setStep] = useState<Step>('loading')
  const [errorMsg, setErrorMsg] = useState('')
  const [errorReturnStep, setErrorReturnStep] = useState<'name' | 'igToken' | 'grokKey'>('name')
  const [envConfigured, setEnvConfigured] = useState<EnvConfigured>({ igToken: false, grokKey: false })
  const { loadProfile } = useAppStore()

  useEffect(() => {
    apiFetch<{ envConfigured: EnvConfigured }>('/api/settings')
      .then((s) => setEnvConfigured(s.envConfigured))
      .catch(() => {})
      .finally(() => setStep('name'))
  }, [])

  function nextAfterName() {
    if (!envConfigured.igToken) {
      setStep('igToken')
    } else if (!envConfigured.grokKey) {
      setStep('grokKey')
    } else {
      void createProfile()
    }
  }

  function nextAfterIgToken() {
    if (!envConfigured.grokKey) {
      setStep('grokKey')
    } else {
      void createProfile()
    }
  }

  async function createProfile() {
    setStep('connecting')
    setErrorMsg('')
    try {
      await apiFetch('/api/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: name.trim(),
          igToken: igToken.trim() || undefined,
          grokKey: grokKey.trim() || undefined,
        }),
      })
      setStep('done')
      await loadProfile()
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Something went wrong')
      setErrorReturnStep(envConfigured.igToken && envConfigured.grokKey ? 'name' : envConfigured.igToken ? 'grokKey' : 'igToken')
      setStep('error')
    }
  }

  return (
    <div className="min-h-screen bg-black flex items-center justify-center p-8">
      <div className="w-full max-w-sm flex flex-col gap-6">
        <div>
          <h1
            className="text-3xl font-bold bg-clip-text text-transparent"
            style={{ backgroundImage: IG_GRADIENT }}
          >
            Instana
          </h1>
          <p className="text-neutral-400 mt-1 text-sm">Instagram analytics for creators</p>
        </div>

        {step === 'loading' && (
          <div className="h-1 rounded-full overflow-hidden bg-neutral-800">
            <div className="h-full rounded-full animate-pulse w-1/2" style={{ background: IG_GRADIENT }} />
          </div>
        )}

        {step === 'name' && (
          <>
            <div className="flex flex-col gap-2">
              <label className="text-white text-sm">Your name</label>
              <TextInput
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Jon"
                onKeyDown={(e) => e.key === 'Enter' && name.trim() && nextAfterName()}
                autoFocus
              />
            </div>
            <InstagramButton onClick={nextAfterName} disabled={!name.trim()}>
              Continue
            </InstagramButton>
          </>
        )}

        {step === 'igToken' && (
          <>
            <div className="flex flex-col gap-2">
              <label className="text-white text-sm">Instagram Access Token</label>
              <TextInput
                value={igToken}
                onChange={(e) => setIgToken(e.target.value)}
                placeholder="IGAAS..."
                onKeyDown={(e) => e.key === 'Enter' && igToken.trim() && nextAfterIgToken()}
                autoFocus
              />
              <p className="text-neutral-500 text-xs">
                Paste your long-lived Instagram Graph API access token
              </p>
            </div>
            <InstagramButton onClick={nextAfterIgToken} disabled={!igToken.trim()}>
              Continue
            </InstagramButton>
          </>
        )}

        {step === 'grokKey' && (
          <>
            <div className="flex flex-col gap-2">
              <label className="text-white text-sm">xAI API Key</label>
              <TextInput
                value={grokKey}
                onChange={(e) => setGrokKey(e.target.value)}
                placeholder="xai-..."
                onKeyDown={(e) => e.key === 'Enter' && grokKey.trim() && createProfile()}
                autoFocus
              />
              <p className="text-neutral-500 text-xs">
                Your xAI Grok API key for video classification
              </p>
            </div>
            <InstagramButton onClick={() => void createProfile()} disabled={!grokKey.trim()}>
              Create Profile
            </InstagramButton>
          </>
        )}

        {step === 'connecting' && (
          <div className="flex flex-col gap-3 text-center">
            <p className="text-white">Setting up your profile…</p>
            <div className="h-1 rounded-full overflow-hidden bg-neutral-800 mt-2">
              <div
                className="h-full rounded-full animate-pulse"
                style={{ background: IG_GRADIENT, width: '60%' }}
              />
            </div>
          </div>
        )}

        {step === 'done' && (
          <div className="text-center flex flex-col gap-2">
            <p className="text-green-400 text-lg">Connected!</p>
            <p className="text-neutral-400 text-sm">Taking you to your dashboard…</p>
          </div>
        )}

        {step === 'error' && (
          <div className="flex flex-col gap-4">
            <p className="text-red-400 text-sm">{errorMsg}</p>
            <button
              onClick={() => setStep(errorReturnStep)}
              className="text-neutral-400 text-sm underline hover:text-white"
            >
              Try again
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
