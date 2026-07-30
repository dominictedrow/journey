// Direct fs read/write for the shared config Settings edits:
//   ~/Library/Application Support/cwork/config.json — holds both "home"
//   (read by skills/places) and "meals" (read by skills/meals as an
//   override). Both live outside the repo and outside the packaged app's
//   Resources dir on purpose: skills/meals/config.json ships as the
//   default template only and is never written to — it's a tracked file
//   in dev and lives inside the code-signed .app bundle once packaged, so
//   writing there would dirty git in dev and invalidate the bundle's
//   signature (or get silently overwritten by a future reinstall) in prod.
// Same spirit as db.js: plain local CRUD, validated here, no shell/Python
// round-trip.

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');

const DEFAULT_MEALS = {
  protein_min_g: 40,
  calorie_min: 500,
  calorie_max: 900,
  required_vitamins: [],
  avoid: [],
};

function configPath() {
  const dir = path.join(os.homedir(), 'Library', 'Application Support', 'cwork');
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  return path.join(dir, 'config.json');
}

// The shipped default template — read-only, never written to. Used only to
// pre-fill Settings the first time, before Dom has ever saved an override.
function mealsDefaultsPath(resourcesRoot) {
  return path.join(resourcesRoot, 'skills', 'meals', 'config.json');
}

function readJson(file, fallback) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch {
    return fallback;
  }
}

function getSettings(resourcesRoot) {
  const config = readJson(configPath(), {});
  const meals = config.meals || readJson(mealsDefaultsPath(resourcesRoot), DEFAULT_MEALS);
  return { home: config.home || null, meals };
}

function updateHome(resourcesRoot, home) {
  const { lat, lon, label } = home || {};
  if (typeof lat !== 'number' || !Number.isFinite(lat) || lat < -90 || lat > 90) {
    throw new Error('lat must be a number between -90 and 90');
  }
  if (typeof lon !== 'number' || !Number.isFinite(lon) || lon < -180 || lon > 180) {
    throw new Error('lon must be a number between -180 and 180');
  }
  if (typeof label !== 'string' || !label.trim()) {
    throw new Error('label must be a non-empty string');
  }

  const file = configPath();
  const config = readJson(file, {});
  config.home = { lat, lon, label: label.trim() };
  fs.writeFileSync(file, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
  return config.home;
}

function updateMeals(meals) {
  const { protein_min_g, calorie_min, calorie_max, required_vitamins, avoid } = meals || {};
  if (typeof protein_min_g !== 'number' || !Number.isFinite(protein_min_g) || protein_min_g < 0) {
    throw new Error('protein_min_g must be a non-negative number');
  }
  if (typeof calorie_min !== 'number' || !Number.isFinite(calorie_min) || calorie_min < 0) {
    throw new Error('calorie_min must be a non-negative number');
  }
  if (typeof calorie_max !== 'number' || !Number.isFinite(calorie_max) || calorie_max < calorie_min) {
    throw new Error('calorie_max must be a number greater than or equal to calorie_min');
  }
  if (!Array.isArray(required_vitamins) || !required_vitamins.every((v) => typeof v === 'string' && v.trim())) {
    throw new Error('required_vitamins must be an array of non-empty strings');
  }
  if (!Array.isArray(avoid) || !avoid.every((v) => typeof v === 'string' && v.trim())) {
    throw new Error('avoid must be an array of non-empty strings');
  }

  const out = {
    protein_min_g,
    calorie_min,
    calorie_max,
    required_vitamins: required_vitamins.map((v) => v.trim()),
    avoid: avoid.map((v) => v.trim()),
  };
  const file = configPath();
  const config = readJson(file, {});
  config.meals = out;
  fs.writeFileSync(file, JSON.stringify(config, null, 2) + '\n', { mode: 0o600 });
  return out;
}

module.exports = { getSettings, updateHome, updateMeals };
