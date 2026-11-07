import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAppStore } from './store/appStore';
import AppShell from './components/layout/AppShell';
import OnboardingPage from './pages/OnboardingPage';
import HomePage from './pages/HomePage';
import LibraryPage from './pages/LibraryPage';
import ScanningPage from './pages/ScanningPage';
import FormatDescriptionPage from './pages/FormatDescriptionPage';
import ProfilePage from './pages/ProfilePage';
export default function App() {
    const { initialized, isOnboarded, loadProfile } = useAppStore();
    useEffect(() => {
        loadProfile();
    }, []);
    if (!initialized) {
        return _jsx("div", { className: "min-h-screen bg-black" });
    }
    return (_jsx("div", { className: "min-h-screen bg-black text-white", children: _jsx(Routes, { children: !isOnboarded ? (_jsxs(_Fragment, { children: [_jsx(Route, { path: "/onboarding", element: _jsx(OnboardingPage, {}) }), _jsx(Route, { path: "*", element: _jsx(Navigate, { to: "/onboarding", replace: true }) })] })) : (_jsxs(_Fragment, { children: [_jsx(Route, { path: "/onboarding", element: _jsx(Navigate, { to: "/home", replace: true }) }), _jsxs(Route, { element: _jsx(AppShell, {}), children: [_jsx(Route, { index: true, element: _jsx(Navigate, { to: "/home", replace: true }) }), _jsx(Route, { path: "/home", element: _jsx(HomePage, {}) }), _jsx(Route, { path: "/library", element: _jsx(LibraryPage, {}) }), _jsx(Route, { path: "/scan", element: _jsx(ScanningPage, {}) }), _jsx(Route, { path: "/format/:id", element: _jsx(FormatDescriptionPage, {}) }), _jsx(Route, { path: "/profile", element: _jsx(ProfilePage, {}) })] })] })) }) }));
}
