// THE skill registry. Adding a skill later means writing Python and adding
// one entry here — no other UI code changes. `id` doubles as the Python
// module name (invoked as `python -m skills.<id>`).

module.exports = [
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
  {
    id: 'weather',
    label: 'Weather',
    kind: 'scheduled',
    schedule: { hour: 6, minute: 30 }, // daily 6:30 AM, ahead of concepts at 7:00
    inputs: [],
  },
  {
    id: 'assignments_prep',
    label: 'Assignments Prep',
    kind: 'on-demand',
    inputs: [],
  },
  {
    id: 'wishlist',
    label: 'Wishlist',
    kind: 'on-demand',
    inputs: [{ name: 'item', label: 'Item', placeholder: 'e.g. PS5 Pro restock' }],
  },
  {
    id: 'trips',
    label: 'Trips',
    kind: 'on-demand',
    inputs: [
      { name: 'query', label: 'Query', placeholder: 'e.g. quiet beach town' },
      { name: 'hours', label: 'Hours away (optional)', placeholder: 'e.g. 3' },
    ],
  },
  {
    id: 'gifts',
    label: 'Gifts',
    kind: 'scheduled',
    schedule: { weekday: 0, hour: 8, minute: 0 }, // weekly, Sunday 8:00 AM
    inputs: [],
  },
  {
    id: 'reads',
    label: 'Reads',
    kind: 'scheduled',
    schedule: { day: 1, hour: 10, minute: 0 }, // monthly, 1st at 10:00 AM, after releases at 9:00
    inputs: [],
  },
];
