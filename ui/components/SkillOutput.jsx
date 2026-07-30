// The single universal renderer for the envelope contract (plan.md's "one
// idea that keeps this light"). Every skill's result — whatever it is —
// renders through here via four section types: stat, list, table, text.

function Section({ section }) {
  return (
    <div className="section">
      <h4>{section.title}</h4>
      {section.type === 'stat' && (
        <div className="stat-row">
          {section.stats.map((s, i) => (
            <div className="stat" key={i}>
              <span className="value">{s.value}</span>
              <span className="label">{s.label}</span>
            </div>
          ))}
        </div>
      )}
      {section.type === 'list' && (
        <ul className="data-list">
          {section.items.map((item, i) => (
            <li key={i}>{typeof item === 'string' ? item : item.label}</li>
          ))}
        </ul>
      )}
      {section.type === 'table' && (
        <table className="data-table">
          <thead>
            <tr>
              {section.columns.map((c, i) => (
                <th key={i}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {section.rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {section.type === 'text' && <p>{section.body}</p>}
    </div>
  );
}

export default function SkillOutput({ envelope }) {
  if (!envelope) return null;

  if (!envelope.ok) {
    return (
      <div className="section">
        <h4>Error</h4>
        <p style={{ color: 'var(--red)' }}>{envelope.error || 'Skill failed'}</p>
      </div>
    );
  }

  return (
    <div>
      <p>{envelope.summary}</p>
      {envelope.sections.map((section, i) => (
        <Section key={i} section={section} />
      ))}
    </div>
  );
}
