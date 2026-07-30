import { useEffect, useState } from 'react';

function toCsv(arr) {
  return (arr || []).join(', ');
}

function fromCsv(str) {
  return str
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

// Number('') is 0, not NaN — a blank field would otherwise silently save
// as a valid-looking 0 instead of failing validation. Blank -> NaN so the
// main-process range/finite checks actually reject it.
function numOrNaN(str) {
  const trimmed = String(str).trim();
  return trimmed === '' ? NaN : Number(trimmed);
}

export default function Settings() {
  const [loaded, setLoaded] = useState(false);
  const [home, setHome] = useState({ label: '', lat: '', lon: '' });
  const [homeStatus, setHomeStatus] = useState(null);

  const [meals, setMeals] = useState({
    protein_min_g: '',
    calorie_min: '',
    calorie_max: '',
    required_vitamins: '',
    avoid: '',
  });
  const [mealsStatus, setMealsStatus] = useState(null);

  useEffect(() => {
    window.cwork.settings.get().then((s) => {
      if (s.home) {
        setHome({ label: s.home.label, lat: String(s.home.lat), lon: String(s.home.lon) });
      }
      setMeals({
        protein_min_g: String(s.meals.protein_min_g),
        calorie_min: String(s.meals.calorie_min),
        calorie_max: String(s.meals.calorie_max),
        required_vitamins: toCsv(s.meals.required_vitamins),
        avoid: toCsv(s.meals.avoid),
      });
      setLoaded(true);
    });
  }, []);

  async function saveHome(e) {
    e.preventDefault();
    setHomeStatus(null);
    try {
      await window.cwork.settings.updateHome({
        label: home.label,
        lat: numOrNaN(home.lat),
        lon: numOrNaN(home.lon),
      });
      setHomeStatus({ ok: true, text: 'Saved — places will use this as "near me".' });
    } catch (err) {
      setHomeStatus({ ok: false, text: err.message });
    }
  }

  async function saveMeals(e) {
    e.preventDefault();
    setMealsStatus(null);
    try {
      await window.cwork.settings.updateMeals({
        protein_min_g: numOrNaN(meals.protein_min_g),
        calorie_min: numOrNaN(meals.calorie_min),
        calorie_max: numOrNaN(meals.calorie_max),
        required_vitamins: fromCsv(meals.required_vitamins),
        avoid: fromCsv(meals.avoid),
      });
      setMealsStatus({ ok: true, text: 'Saved — next meals run uses these constraints.' });
    } catch (err) {
      setMealsStatus({ ok: false, text: err.message });
    }
  }

  if (!loaded) {
    return (
      <div>
        <div className="content-header">
          <h1>Settings</h1>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="content-header">
        <h1>Settings</h1>
      </div>

      <div className="section">
        <h4>Home location</h4>
        <p className="hint">Used by places as the "near me" origin. Look up coordinates on openstreetmap.org.</p>
        <form className="form-grid" onSubmit={saveHome}>
          <label>
            Label
            <input
              type="text"
              placeholder="e.g. Bay Ho, San Diego, CA"
              value={home.label}
              onChange={(e) => setHome((h) => ({ ...h, label: e.target.value }))}
              required
            />
          </label>
          <label>
            Latitude
            <input
              type="number"
              step="any"
              placeholder="32.8239077"
              value={home.lat}
              onChange={(e) => setHome((h) => ({ ...h, lat: e.target.value }))}
              required
            />
          </label>
          <label>
            Longitude
            <input
              type="number"
              step="any"
              placeholder="-117.2087893"
              value={home.lon}
              onChange={(e) => setHome((h) => ({ ...h, lon: e.target.value }))}
              required
            />
          </label>
          <button className="btn" type="submit">
            Save
          </button>
        </form>
        {homeStatus && <p className={`settings-status ${homeStatus.ok ? 'ok' : 'error'}`}>{homeStatus.text}</p>}
      </div>

      <div className="section">
        <h4>Meal macros</h4>
        <p className="hint">Hard constraints enforced on every meals run — checked in the prompt and re-validated in Python.</p>
        <form className="form-grid" onSubmit={saveMeals}>
          <label>
            Protein floor (g)
            <input
              type="number"
              min="0"
              value={meals.protein_min_g}
              onChange={(e) => setMeals((m) => ({ ...m, protein_min_g: e.target.value }))}
              required
            />
          </label>
          <label>
            Calorie min
            <input
              type="number"
              min="0"
              value={meals.calorie_min}
              onChange={(e) => setMeals((m) => ({ ...m, calorie_min: e.target.value }))}
              required
            />
          </label>
          <label>
            Calorie max
            <input
              type="number"
              min="0"
              value={meals.calorie_max}
              onChange={(e) => setMeals((m) => ({ ...m, calorie_max: e.target.value }))}
              required
            />
          </label>
          <label>
            Required vitamins (comma-separated)
            <input
              type="text"
              placeholder="D, B12, Iron"
              value={meals.required_vitamins}
              onChange={(e) => setMeals((m) => ({ ...m, required_vitamins: e.target.value }))}
            />
          </label>
          <label>
            Avoid (comma-separated, optional)
            <input
              type="text"
              placeholder="peanuts, shellfish"
              value={meals.avoid}
              onChange={(e) => setMeals((m) => ({ ...m, avoid: e.target.value }))}
            />
          </label>
          <button className="btn" type="submit">
            Save
          </button>
        </form>
        {mealsStatus && <p className={`settings-status ${mealsStatus.ok ? 'ok' : 'error'}`}>{mealsStatus.text}</p>}
      </div>
    </div>
  );
}
