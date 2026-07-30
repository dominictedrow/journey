import { useEffect, useState } from 'react';
import SkillOutput from './SkillOutput.jsx';

export default function RunPanel({ skill, result, running, onRun }) {
  const [log, setLog] = useState([]);
  const [inputs, setInputs] = useState({});

  useEffect(() => {
    const unsubscribe = window.cwork.skills.onLog(({ id, line }) => {
      if (id === skill.id) setLog((prev) => [...prev, line]);
    });
    return unsubscribe;
  }, [skill.id]);

  async function handleRun() {
    setLog([]);
    await onRun(skill.inputs.length ? inputs : undefined);
  }

  return (
    <div className="run-panel">
      {skill.inputs.length > 0 && (
        <div className="form-row">
          {skill.inputs.map((field) => (
            <input
              key={field.name}
              type="text"
              placeholder={field.placeholder || field.label}
              value={inputs[field.name] || ''}
              onChange={(e) => setInputs((prev) => ({ ...prev, [field.name]: e.target.value }))}
            />
          ))}
        </div>
      )}
      <button className="btn" disabled={running} onClick={handleRun}>
        {running ? 'Running…' : 'Run'}
      </button>
      {log.length > 0 && <div className="log">{log.join('\n')}</div>}
      {result && <SkillOutput envelope={result} />}
    </div>
  );
}
