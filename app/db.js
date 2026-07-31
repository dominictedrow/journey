// Direct node:sqlite access to the same DB file skills/_lib/store.py uses.
// Classes/Assignments CRUD and the Home tab's cached skill_runs read go
// straight through here — no Python round-trip for plain reads/writes.

const { DatabaseSync } = require('node:sqlite');
const path = require('node:path');
const os = require('node:os');
const fs = require('node:fs');

const SCHEMA = `
CREATE TABLE IF NOT EXISTS claude_cache (
    cache_key   TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skill_runs (
    skill_id    TEXT PRIMARY KEY,
    envelope    TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    ran_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS classes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    due_date    TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS people (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS occasions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    month       INTEGER NOT NULL,
    day         INTEGER NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
`;

function dbPath() {
  const dir = path.join(os.homedir(), 'Library', 'Application Support', 'cwork');
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  return path.join(dir, 'cwork.db');
}

let db;

function getDb() {
  if (!db) {
    db = new DatabaseSync(dbPath());
    db.exec('PRAGMA foreign_keys = ON');
    db.exec(SCHEMA);
    fs.chmodSync(dbPath(), 0o600);
  }
  return db;
}

// --- skill_runs (Home tab cache) ---

function recordSkillRun(skillId, envelope) {
  getDb()
    .prepare(
      `INSERT INTO skill_runs (skill_id, envelope, ok, ran_at)
       VALUES (?, ?, ?, datetime('now'))
       ON CONFLICT(skill_id) DO UPDATE SET
         envelope = excluded.envelope, ok = excluded.ok, ran_at = excluded.ran_at`
    )
    .run(skillId, JSON.stringify(envelope), envelope.ok ? 1 : 0);
}

function listSkillRuns() {
  return getDb().prepare('SELECT skill_id, envelope, ok, ran_at FROM skill_runs').all();
}

// --- classes ---

function listClasses() {
  return getDb().prepare('SELECT id, name, created_at FROM classes ORDER BY name').all();
}

function addClass(name) {
  const result = getDb().prepare('INSERT INTO classes (name) VALUES (?)').run(name);
  return { id: Number(result.lastInsertRowid), name };
}

function renameClass(id, name) {
  getDb().prepare('UPDATE classes SET name = ? WHERE id = ?').run(name, id);
}

function deleteClass(id) {
  getDb().prepare('DELETE FROM classes WHERE id = ?').run(id);
}

// --- assignments ---

function listAssignments() {
  return getDb()
    .prepare(
      `SELECT a.id, a.title, a.due_date, a.done, a.class_id, c.name AS class_name
       FROM assignments a JOIN classes c ON c.id = a.class_id
       ORDER BY a.due_date`
    )
    .all();
}

function addAssignment({ classId, title, dueDate }) {
  const result = getDb()
    .prepare('INSERT INTO assignments (class_id, title, due_date) VALUES (?, ?, ?)')
    .run(classId, title, dueDate);
  return { id: Number(result.lastInsertRowid) };
}

function setAssignmentDone(id, done) {
  getDb().prepare('UPDATE assignments SET done = ? WHERE id = ?').run(done ? 1 : 0, id);
}

function deleteAssignment(id) {
  getDb().prepare('DELETE FROM assignments WHERE id = ?').run(id);
}

// --- people ---

function listPeople() {
  return getDb().prepare('SELECT id, name, created_at FROM people ORDER BY name').all();
}

function addPerson(name) {
  const result = getDb().prepare('INSERT INTO people (name) VALUES (?)').run(name);
  return { id: Number(result.lastInsertRowid), name };
}

function renamePerson(id, name) {
  getDb().prepare('UPDATE people SET name = ? WHERE id = ?').run(name, id);
}

function deletePerson(id) {
  getDb().prepare('DELETE FROM people WHERE id = ?').run(id);
}

// --- occasions ---

function listOccasions() {
  return getDb()
    .prepare(
      `SELECT o.id, o.label, o.month, o.day, o.person_id, p.name AS person_name
       FROM occasions o JOIN people p ON p.id = o.person_id
       ORDER BY o.month, o.day`
    )
    .all();
}

function addOccasion({ personId, label, month, day }) {
  const result = getDb()
    .prepare('INSERT INTO occasions (person_id, label, month, day) VALUES (?, ?, ?, ?)')
    .run(personId, label, month, day);
  return { id: Number(result.lastInsertRowid) };
}

function deleteOccasion(id) {
  getDb().prepare('DELETE FROM occasions WHERE id = ?').run(id);
}

module.exports = {
  recordSkillRun,
  listSkillRuns,
  listClasses,
  addClass,
  renameClass,
  deleteClass,
  listAssignments,
  addAssignment,
  setAssignmentDone,
  deleteAssignment,
  listPeople,
  addPerson,
  renamePerson,
  deletePerson,
  listOccasions,
  addOccasion,
  deleteOccasion,
};
