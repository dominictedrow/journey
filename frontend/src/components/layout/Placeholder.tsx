export function Placeholder({ title, phase }: { title: string; phase: number }) {
  return (
    <div>
      <h1 className="text-2xl font-semibold">{title}</h1>
      <p className="mt-4 text-slate-400">
        This dashboard is implemented in Phase {phase} — see <code>docs/plan.md</code>.
      </p>
    </div>
  );
}
