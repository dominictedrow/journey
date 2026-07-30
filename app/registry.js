// THE skill registry. Adding a skill later means writing Python and adding
// one entry here — no other UI code changes. `id` doubles as the Python
// module name (invoked as `python -m skills.<id>`).

module.exports = [
  {
    id: 'hello',
    label: 'Hello',
    kind: 'on-demand', // 'scheduled' (lands on Home) | 'on-demand' (own tab, run button)
    inputs: [], // [{ name, label, placeholder }] — on-demand skills only
  },
];
