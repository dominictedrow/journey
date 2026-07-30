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
  {
    id: 'concepts',
    label: 'Concepts',
    kind: 'scheduled',
    schedule: { hour: 7, minute: 0 }, // daily 7:00 AM
    inputs: [],
  },
  {
    id: 'releases',
    label: 'Releases',
    kind: 'scheduled',
    schedule: { day: 1, hour: 9, minute: 0 }, // monthly, 1st at 9:00 AM
    inputs: [],
  },
  {
    id: 'projects',
    label: 'Projects',
    kind: 'on-demand',
    inputs: [{ name: 'concept', label: 'Concept', placeholder: 'e.g. transformer attention' }],
  },
  {
    id: 'places',
    label: 'Places',
    kind: 'on-demand',
    inputs: [{ name: 'query', label: 'Query', placeholder: 'e.g. calm study spot' }],
  },
  {
    id: 'meals',
    label: 'Meals',
    kind: 'on-demand',
    inputs: [{ name: 'craving', label: 'Craving (optional)', placeholder: 'e.g. chicken' }],
  },
];
