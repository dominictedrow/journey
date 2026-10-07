import { useEffect, useState } from 'react';
import StatusDot from './components/StatusDot.jsx';
import Home from './tabs/Home.jsx';
import OnDemand from './tabs/OnDemand.jsx';
import Assignments from './tabs/Assignments.jsx';
import Gifts from './tabs/Gifts.jsx';
import Settings from './tabs/Settings.jsx';

export default function App() {
  const [skills, setSkills] = useState([]);
  const [results, setResults] = useState({});
  const [runningId, setRunningId] = useState(null);
  const [tab, setTab] = useState('home');

  useEffect(() => {
    window.cwork.skills.list().then(setSkills);
    window.cwork.skills.lastResults().then((envelopes) => {
      const map = {};
      for (const e of envelopes) map[e.skill] = e;
      setResults(map);
    });
  }, []);

  function statusFor(id) {
    if (runningId === id) return 'running';
    const result = results[id];
    if (!result) return 'stale';
    return result.ok ? 'fresh' : 'failed';
  }

  async function runSkill(id, input) {
    setRunningId(id);
    try {
      const envelope = await window.cwork.skills.run(id, input);
      setResults((prev) => ({ ...prev, [id]: envelope }));
      return envelope;
    } finally {
      setRunningId(null);
    }
  }

  const activeSkill = skills.find((s) => s.id === tab);

  return (
    <div className="app-shell">
      <div className="sidebar">
        <div className="titlebar" />
        <div className="sidebar-title">cwork</div>

        <button className={`sidebar-item ${tab === 'home' ? 'active' : ''}`} onClick={() => setTab('home')}>
          <span className="label">Home</span>
        </button>

        {skills.map((s) => (
          <button key={s.id} className={`sidebar-item ${tab === s.id ? 'active' : ''}`} onClick={() => setTab(s.id)}>
            <StatusDot status={statusFor(s.id)} />
            <span className="label">{s.label}</span>
          </button>
        ))}

        <button
          className={`sidebar-item ${tab === 'assignments' ? 'active' : ''}`}
          onClick={() => setTab('assignments')}
        >
          <span className="label">Assignments</span>
        </button>

        <button
          className={`sidebar-item ${tab === 'people' ? 'active' : ''}`}
          onClick={() => setTab('people')}
        >
          <span className="label">People</span>
        </button>

        <div style={{ flex: 1 }} />

        <button className={`sidebar-item ${tab === 'settings' ? 'active' : ''}`} onClick={() => setTab('settings')}>
          <span className="label">Settings</span>
        </button>
      </div>

      <div className="content">
        {tab === 'home' && <Home skills={skills} results={results} />}
        {tab === 'assignments' && <Assignments />}
        {tab === 'people' && <Gifts />}
        {tab === 'settings' && <Settings />}
        {activeSkill && (
          <OnDemand
            skill={activeSkill}
            result={results[activeSkill.id]}
            running={runningId === activeSkill.id}
            onRun={(input) => runSkill(activeSkill.id, input)}
          />
        )}
      </div>
    </div>
  );
}
