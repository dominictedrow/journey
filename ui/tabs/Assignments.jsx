import { useEffect, useState } from 'react';

const GROUP_ORDER = ['Overdue', 'Due Today', 'Due This Week', 'Later', 'Done'];

function groupAssignments(assignments) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const weekEnd = new Date(today);
  weekEnd.setDate(weekEnd.getDate() + 7);

  const groups = { Overdue: [], 'Due Today': [], 'Due This Week': [], Later: [], Done: [] };

  for (const a of assignments) {
    if (a.done) {
      groups.Done.push(a);
      continue;
    }
    const due = new Date(a.due_date + 'T00:00:00');
    if (due < today) groups.Overdue.push(a);
    else if (due.getTime() === today.getTime()) groups['Due Today'].push(a);
    else if (due <= weekEnd) groups['Due This Week'].push(a);
    else groups.Later.push(a);
  }

  return groups;
}

export default function Assignments() {
  const [classes, setClasses] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [newClassName, setNewClassName] = useState('');
  const [form, setForm] = useState({ title: '', classId: '', dueDate: '' });

  async function refresh() {
    const [c, a] = await Promise.all([window.cwork.classes.list(), window.cwork.assignments.list()]);
    setClasses(c);
    setAssignments(a);
  }

  useEffect(() => {
    refresh();
  }, []);

  async function addClass() {
    if (!newClassName.trim()) return;
    await window.cwork.classes.add(newClassName.trim());
    setNewClassName('');
    refresh();
  }

  async function renameClass(cls) {
    const name = window.prompt('Rename class', cls.name);
    if (name && name.trim() && name.trim() !== cls.name) {
      await window.cwork.classes.rename(cls.id, name.trim());
      refresh();
    }
  }

  async function deleteClass(id) {
    await window.cwork.classes.delete(id);
    refresh();
  }

  async function addAssignment(e) {
    e.preventDefault();
    if (!form.title.trim() || !form.classId || !form.dueDate) return;
    await window.cwork.assignments.add({
      title: form.title.trim(),
      classId: Number(form.classId),
      dueDate: form.dueDate,
    });
    setForm((f) => ({ title: '', classId: f.classId, dueDate: '' }));
    refresh();
  }

  async function toggleDone(a) {
    await window.cwork.assignments.setDone(a.id, !a.done);
    refresh();
  }

  async function deleteAssignment(id) {
    await window.cwork.assignments.delete(id);
    refresh();
  }

  const groups = groupAssignments(assignments);

  return (
    <div>
      <div className="content-header">
        <h1>Assignments</h1>
        <p>Manually tracked — no scraping, no grades.</p>
      </div>

      <div className="section">
        <h4>Classes</h4>
        <div>
          {classes.map((c) => (
            <span className="class-chip" key={c.id}>
              <span onClick={() => renameClass(c)} style={{ cursor: 'pointer' }}>
                {c.name}
              </span>
              <button onClick={() => deleteClass(c.id)}>×</button>
            </span>
          ))}
        </div>
        <div className="form-row">
          <input
            type="text"
            placeholder="New class name"
            value={newClassName}
            onChange={(e) => setNewClassName(e.target.value)}
          />
          <button className="btn secondary" onClick={addClass}>
            Add class
          </button>
        </div>
      </div>

      <div className="section">
        <h4>New assignment</h4>
        <form className="form-row" onSubmit={addAssignment}>
          <input
            type="text"
            placeholder="Title"
            value={form.title}
            onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))}
          />
          <select value={form.classId} onChange={(e) => setForm((f) => ({ ...f, classId: e.target.value }))}>
            <option value="">Class…</option>
            {classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          <input
            type="date"
            value={form.dueDate}
            onChange={(e) => setForm((f) => ({ ...f, dueDate: e.target.value }))}
          />
          <button className="btn" type="submit">
            Add
          </button>
        </form>
      </div>

      {GROUP_ORDER.map((label) =>
        groups[label].length > 0 ? (
          <div className="assignment-group" key={label}>
            <h3>{label}</h3>
            {groups[label].map((a) => (
              <div
                className={`assignment-row ${label === 'Overdue' ? 'overdue' : ''} ${a.done ? 'done' : ''}`}
                key={a.id}
              >
                <input type="checkbox" checked={!!a.done} onChange={() => toggleDone(a)} />
                <span className="title">{a.title}</span>
                <span className="class-name">{a.class_name}</span>
                <span className="due">{a.due_date}</span>
                <button className="btn danger" onClick={() => deleteAssignment(a.id)}>
                  Delete
                </button>
              </div>
            ))}
          </div>
        ) : null
      )}
    </div>
  );
}
