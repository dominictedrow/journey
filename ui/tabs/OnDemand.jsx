import RunPanel from '../components/RunPanel.jsx';
import SkillOutput from '../components/SkillOutput.jsx';
import Empty from '../components/Empty.jsx';

export default function OnDemand({ skill, result, running, onRun }) {
  if (!skill) return null;

  return (
    <div>
      <div className="content-header">
        <h1>{skill.label}</h1>
      </div>
      {skill.kind === 'on-demand' ? (
        <RunPanel skill={skill} result={result} running={running} onRun={onRun} />
      ) : result ? (
        <SkillOutput envelope={result} />
      ) : (
        <Empty>No results yet — this skill runs on its own schedule.</Empty>
      )}
    </div>
  );
}
