import Card from '../components/Card.jsx';
import Empty from '../components/Empty.jsx';
import SkillOutput from '../components/SkillOutput.jsx';

export default function Home({ skills, results }) {
  const scheduled = skills.filter((s) => s.kind === 'scheduled');

  return (
    <div>
      <div className="content-header">
        <h1>Home</h1>
        <p>Latest results from scheduled skills.</p>
      </div>
      {scheduled.length === 0 ? (
        <Empty>No scheduled skills yet — these land in Phase 2 (concepts, releases).</Empty>
      ) : (
        <div className="card-grid">
          {scheduled.map((s) => {
            const result = results[s.id];
            return (
              <Card
                key={s.id}
                title={s.label}
                meta={result ? new Date(result.ran_at).toLocaleString() : 'never run'}
              >
                {result ? <SkillOutput envelope={result} /> : <Empty>No results yet.</Empty>}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
