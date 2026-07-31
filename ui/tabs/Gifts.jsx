import { useEffect, useState } from 'react';

const GROUP_ORDER = ['This Week', 'This Month', 'Later'];
const MONTH_NAMES = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

function isValidMonthDay(month, day) {
  const probe = new Date(2000, month - 1, day);
  return probe.getMonth() === month - 1 && probe.getDate() === day;
}

function daysUntilNext(month, day) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  let next = new Date(today.getFullYear(), month - 1, day);
  if (next < today) next = new Date(today.getFullYear() + 1, month - 1, day);
  return Math.round((next - today) / (1000 * 60 * 60 * 24));
}

function groupOccasions(occasions) {
  const groups = { 'This Week': [], 'This Month': [], Later: [] };
  for (const o of occasions) {
    const days = daysUntilNext(o.month, o.day);
    const entry = { ...o, daysUntil: days };
    if (days <= 7) groups['This Week'].push(entry);
    else if (days <= 30) groups['This Month'].push(entry);
    else groups.Later.push(entry);
  }
  for (const key of GROUP_ORDER) groups[key].sort((a, b) => a.daysUntil - b.daysUntil);
  return groups;
}

export default function Gifts() {
  const [people, setPeople] = useState([]);
  const [occasions, setOccasions] = useState([]);
  const [newPersonName, setNewPersonName] = useState('');
  const [form, setForm] = useState({ label: '', personId: '', month: '', day: '' });

  async function refresh() {
    const [p, o] = await Promise.all([window.cwork.people.list(), window.cwork.occasions.list()]);
    setPeople(p);
    setOccasions(o);
  }

  useEffect(() => {
    refresh();
  }, []);

  async function addPerson() {
    if (!newPersonName.trim()) return;
    await window.cwork.people.add(newPersonName.trim());
    setNewPersonName('');
    refresh();
  }

  async function renamePerson(person) {
    const name = window.prompt('Rename person', person.name);
    if (name && name.trim() && name.trim() !== person.name) {
      await window.cwork.people.rename(person.id, name.trim());
      refresh();
    }
  }

  async function deletePerson(id) {
    await window.cwork.people.delete(id);
    refresh();
  }

  async function addOccasion(e) {
    e.preventDefault();
    const month = Number(form.month);
    const day = Number(form.day);
    if (!form.label.trim() || !form.personId || !month || !day) return;
    if (!isValidMonthDay(month, day)) return;
    await window.cwork.occasions.add({
      personId: Number(form.personId),
      label: form.label.trim(),
      month,
      day,
    });
    setForm((f) => ({ label: '', personId: f.personId, month: '', day: '' }));
    refresh();
  }

  async function deleteOccasion(id) {
    await window.cwork.occasions.delete(id);
    refresh();
  }

  const groups = groupOccasions(occasions);

  return (
    <div>
      <div className="content-header">
        <h1>People</h1>
        <p>Manually tracked people and occasions — the Gifts skill surfaces ideas as dates approach.</p>
      </div>

      <div className="section">
        <h4>People</h4>
        <div>
          {people.map((p) => (
            <span className="class-chip" key={p.id}>
              <span onClick={() => renamePerson(p)} style={{ cursor: 'pointer' }}>
                {p.name}
              </span>
              <button onClick={() => deletePerson(p.id)}>×</button>
            </span>
          ))}
        </div>
        <div className="form-row">
          <input
            type="text"
            placeholder="New person name"
            value={newPersonName}
            onChange={(e) => setNewPersonName(e.target.value)}
          />
          <button className="btn secondary" onClick={addPerson}>
            Add person
          </button>
        </div>
      </div>

      <div className="section">
        <h4>New occasion</h4>
        <form className="form-row" onSubmit={addOccasion}>
          <input
            type="text"
            placeholder="Label (e.g. Birthday)"
            value={form.label}
            onChange={(e) => setForm((f) => ({ ...f, label: e.target.value }))}
          />
          <select value={form.personId} onChange={(e) => setForm((f) => ({ ...f, personId: e.target.value }))}>
            <option value="">Person…</option>
            {people.map((p) => (
              <option key={p.id} value={p.id}>
                {p.name}
              </option>
            ))}
          </select>
          <input
            type="number"
            min="1"
            max="12"
            placeholder="Month"
            value={form.month}
            onChange={(e) => setForm((f) => ({ ...f, month: e.target.value }))}
          />
          <input
            type="number"
            min="1"
            max="31"
            placeholder="Day"
            value={form.day}
            onChange={(e) => setForm((f) => ({ ...f, day: e.target.value }))}
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
            {groups[label].map((o) => (
              <div className="assignment-row" key={o.id}>
                <span className="title">
                  {o.person_name} — {o.label}
                </span>
                <span className="class-name">
                  {MONTH_NAMES[o.month - 1]} {o.day}
                </span>
                <span className="due">{o.daysUntil === 0 ? 'today' : `in ${o.daysUntil}d`}</span>
                <button className="btn danger" onClick={() => deleteOccasion(o.id)}>
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
