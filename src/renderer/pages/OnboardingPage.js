import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "react/jsx-runtime";
import { useState } from 'react';
import { IG_GRADIENT } from '../constants/styles';
import InstagramButton from '../components/shared/InstagramButton';
import TextInput from '../components/shared/TextInput';
import { apiFetch } from '../api/client';
import { useAppStore } from '../store/appStore';
export default function OnboardingPage() {
    const [name, setName] = useState('');
    const [igToken, setIgToken] = useState('');
    const [grokKey, setGrokKey] = useState('');
    const [step, setStep] = useState('name');
    const [errorMsg, setErrorMsg] = useState('');
    const [errorReturnStep, setErrorReturnStep] = useState('name');
    const { loadProfile } = useAppStore();
    async function createProfile() {
        setStep('connecting');
        setErrorMsg('');
        try {
            await apiFetch('/api/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: name.trim(),
                    igToken: igToken.trim(),
                    grokKey: grokKey.trim(),
                }),
            });
            setStep('done');
            await loadProfile();
        }
        catch (err) {
            setErrorMsg(err instanceof Error ? err.message : 'Something went wrong');
            setErrorReturnStep('grokKey');
            setStep('error');
        }
    }
    return (_jsx("div", { className: "min-h-screen bg-black flex items-center justify-center p-8", children: _jsxs("div", { className: "w-full max-w-sm flex flex-col gap-6", children: [_jsxs("div", { children: [_jsx("h1", { className: "text-3xl font-bold bg-clip-text text-transparent", style: { backgroundImage: IG_GRADIENT }, children: "Instana" }), _jsx("p", { className: "text-neutral-400 mt-1 text-sm", children: "Instagram analytics for creators" })] }), step === 'name' && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("label", { className: "text-white text-sm", children: "Your name" }), _jsx(TextInput, { value: name, onChange: (e) => setName(e.target.value), placeholder: "e.g. Jon", onKeyDown: (e) => e.key === 'Enter' && name.trim() && setStep('igToken'), autoFocus: true })] }), _jsx(InstagramButton, { onClick: () => setStep('igToken'), disabled: !name.trim(), children: "Continue" })] })), step === 'igToken' && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("label", { className: "text-white text-sm", children: "Instagram Access Token" }), _jsx(TextInput, { value: igToken, onChange: (e) => setIgToken(e.target.value), placeholder: "IGAAS...", onKeyDown: (e) => e.key === 'Enter' && igToken.trim() && setStep('grokKey'), autoFocus: true }), _jsx("p", { className: "text-neutral-500 text-xs", children: "Paste your long-lived Instagram Graph API access token" })] }), _jsx(InstagramButton, { onClick: () => setStep('grokKey'), disabled: !igToken.trim(), children: "Continue" })] })), step === 'grokKey' && (_jsxs(_Fragment, { children: [_jsxs("div", { className: "flex flex-col gap-2", children: [_jsx("label", { className: "text-white text-sm", children: "xAI API Key" }), _jsx(TextInput, { value: grokKey, onChange: (e) => setGrokKey(e.target.value), placeholder: "xai-...", onKeyDown: (e) => e.key === 'Enter' && grokKey.trim() && createProfile(), autoFocus: true }), _jsx("p", { className: "text-neutral-500 text-xs", children: "Your xAI Grok API key for video classification" })] }), _jsx(InstagramButton, { onClick: createProfile, disabled: !grokKey.trim(), children: "Create Profile" })] })), step === 'connecting' && (_jsxs("div", { className: "flex flex-col gap-3 text-center", children: [_jsx("p", { className: "text-white", children: "Setting up your profile\u2026" }), _jsx("div", { className: "h-1 rounded-full overflow-hidden bg-neutral-800 mt-2", children: _jsx("div", { className: "h-full rounded-full animate-pulse", style: { background: IG_GRADIENT, width: '60%' } }) })] })), step === 'done' && (_jsxs("div", { className: "text-center flex flex-col gap-2", children: [_jsx("p", { className: "text-green-400 text-lg", children: "Connected!" }), _jsx("p", { className: "text-neutral-400 text-sm", children: "Taking you to your dashboard\u2026" })] })), step === 'error' && (_jsxs("div", { className: "flex flex-col gap-4", children: [_jsx("p", { className: "text-red-400 text-sm", children: errorMsg }), _jsx("button", { onClick: () => setStep(errorReturnStep), className: "text-neutral-400 text-sm underline hover:text-white", children: "Try again" })] }))] }) }));
}
