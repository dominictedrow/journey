"""Build the self-contained Kaggriculture submission from strategy_follower.py."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "strategy_follower.py"
DEFAULT_OUTPUT = ROOT / "main.py"
SCIENTIFIC_IMPORTS = {"numpy", "pandas", "scipy", "sklearn", "torch", "tensorflow"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _agent_prefix(source):
    """Keep source through agent(), dropping demos and all later statements."""
    tree = ast.parse(source)
    agents = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
              and node.name == "agent"]
    if len(agents) != 1:
        raise ValueError("editable source must define exactly one top-level agent()")
    agent = agents[0]
    return "\n".join(source.splitlines()[:agent.end_lineno]) + "\n"


def _replace_watcher_import(source, replacement):
    tree = ast.parse(source)
    target = next((node for node in tree.body if isinstance(node, ast.Try) and
                   any(isinstance(child, ast.Import) and
                       any(alias.name == "shared_features" for alias in child.names)
                       for child in node.body)), None)
    if target is None:
        raise ValueError("watcher import block was not found")
    lines = source.splitlines()
    lines[target.lineno - 1:target.end_lineno] = replacement.splitlines()
    return "\n".join(lines) + "\n"


class _PrefixNames(ast.NodeTransformer):
    def __init__(self, prefix, names):
        self.prefix = prefix
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = self.prefix + node.id
        return node

    def visit_FunctionDef(self, node):
        if node.name in self.names:
            node.name = self.prefix + node.name
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.name in self.names:
            node.name = self.prefix + node.name
        return self.generic_visit(node)


def _embedded_module(path, prefix, exports):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom) or node.module != "__future__"]
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    names.update(target.id for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                 for target in ((node.targets if isinstance(node, ast.Assign) else [node.target]))
                 if isinstance(target, ast.Name))
    tree = _PrefixNames(prefix, names).visit(tree)
    ast.fix_missing_locations(tree)
    missing = set(exports) - names
    if missing:
        raise ValueError(f"{path} is missing exports: {sorted(missing)}")
    body = ast.unparse(tree)
    attrs = ", ".join(f"{name!r}: {prefix}{name}" for name in exports)
    return body + f"\n{prefix}namespace = type({prefix!r}, (), {{{attrs}}})\n"


def _validate_watcher(model_path, report_path, features_path):
    if not model_path or not report_path:
        raise ValueError("--watcher-model and --watcher-report must be supplied together")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    accepted = {name for name, result in report.get("outputs", {}).items()
                if result.get("accepted") is True and result.get("status") == "accepted"}
    if any(report["outputs"][name].get("leave_one_family_out_gate_passed") is not True
           for name in accepted):
        raise ValueError("accepted watcher output did not pass leave-one-family-out gating")
    model = _load_module(model_path, "_selected_watcher_model")
    features = _load_module(features_path, "_selected_watcher_features")
    model_outputs = set(getattr(model, "MODELS", {}))
    if not model_outputs or model_outputs != accepted:
        raise ValueError("watcher module outputs do not exactly match report's accepted outputs")
    if (model.SCHEMA_VERSION != features.SCHEMA_VERSION or
            model.SCHEMA_HASH != features.SCHEMA_HASH or
            tuple(model.FEATURE_ORDER) != tuple(features.FEATURE_ORDER)):
        raise ValueError("watcher schema hash or feature order differs from shared_features")
    metadata = report.get("metadata", {})
    if (metadata.get("schema_version") != model.SCHEMA_VERSION or
            metadata.get("schema_hash") != model.SCHEMA_HASH):
        raise ValueError("watcher report schema does not match the selected model")


def _watcher_replacement(model_path=None, report_path=None, features_path=None):
    if model_path is None and report_path is None:
        return "_watcher_features = None\n_watcher_model = None"
    features_path = Path(features_path or ROOT / "shared_features.py")
    _validate_watcher(Path(model_path), Path(report_path), features_path)
    exports = ("SCHEMA_VERSION", "SCHEMA_HASH", "FEATURE_ORDER")
    features = _embedded_module(features_path, "_wf_", exports + ("extract_features",))
    model = _embedded_module(Path(model_path), "_wm_", exports + ("MODELS", "score"))
    return features + model + "\n_watcher_features = _wf_namespace\n_watcher_model = _wm_namespace"


def _validate_watcher_build_mode(source_text, watcher_model, watcher_report):
    """Reject builds where the configured backend and embedded payload disagree."""
    tree = ast.parse(source_text)
    backend = None
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and
                node.targets[0].id == "WATCHER_BACKEND"):
            backend = ast.literal_eval(node.value)
            break
    if backend not in {"off", "rules", "linear_svm"}:
        raise ValueError(f"unsupported WATCHER_BACKEND: {backend!r}")
    supplied = watcher_model is not None or watcher_report is not None
    if backend == "linear_svm" and not supplied:
        raise ValueError(
            "WATCHER_BACKEND='linear_svm' requires --watcher-model and --watcher-report")
    if supplied and backend != "linear_svm":
        raise ValueError(
            "watcher model inputs require WATCHER_BACKEND='linear_svm' in the source")


def _validate_source(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.split(".")[0] in SCIENTIFIC_IMPORTS for name in names):
                raise ValueError(f"scientific runtime dependency remains: {names}")
    callables = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if not callables or not isinstance(callables[-1], (ast.FunctionDef, ast.AsyncFunctionDef)) or callables[-1].name != "agent":
        raise ValueError("agent(obs) is not the last top-level callable")


def _valid_quantity(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_unit_action(action, label):
    if not isinstance(action, list) or not action or not isinstance(action[0], str):
        raise ValueError(f"{label} is not a nonempty action list: {action!r}")
    if any(isinstance(value, float) and not math.isfinite(value) for value in action[1:]):
        raise ValueError(f"{label} contains a non-finite argument: {action!r}")


def _validate_market_order(order):
    if not isinstance(order, list) or not order or not isinstance(order[0], str):
        raise ValueError(f"invalid market order: {order!r}")
    name = order[0]
    if name in {"HIRE", "BUY_LAND", "NOOP"}:
        valid = len(order) == 1
    elif name in {"BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT", "SELL"}:
        valid = len(order) == 3 and isinstance(order[1], str) and _valid_quantity(order[2])
    else:
        valid = False
    if not valid:
        raise ValueError(f"invalid market order grammar: {order!r}")


def _validate_action(action, obs):
    if not isinstance(action, dict) or set(action) != {"farmer", "hands", "market"}:
        raise ValueError(f"agent returned an invalid action contract: {action!r}")
    if not isinstance(action["hands"], list) or not isinstance(action["market"], list):
        raise ValueError(f"agent returned non-list action fields: {action!r}")
    _validate_unit_action(action["farmer"], "farmer action")
    for index, hand_action in enumerate(action["hands"]):
        _validate_unit_action(hand_action, f"hand action {index}")
    player = obs.get("player", 0)
    farms = obs.get("farms", [])
    if not isinstance(player, int) or not 0 <= player < len(farms):
        raise ValueError("fixture observation has no valid focal player")
    expected_hands = len(farms[player].get("hands", []))
    if len(action["hands"]) != expected_hands:
        raise ValueError(f"agent returned {len(action['hands'])} hand actions for {expected_hands} hands")
    if len(action["market"]) > 10:
        raise ValueError("agent exceeded the 10-order market cap")
    for order in action["market"]:
        _validate_market_order(order)


def _validate_action_contract(path):
    module = _load_module(path, "_built_submission_contract")
    fixtures = sorted((ROOT.parent / "kaggle_replays" / "replays").glob("episode-*-replay.json"))
    if not fixtures:
        raise ValueError("the checked-in replay fixture is required for the action contract check")
    checked = 0
    for fixture in fixtures[:4]:
        replay = json.loads(fixture.read_text(encoding="utf-8"))
        steps = replay.get("steps", [])
        sample_indexes = sorted({0, len(steps) // 4, len(steps) // 2,
                                 3 * len(steps) // 4, max(0, len(steps) - 1)})
        for index in sample_indexes:
            for state in steps[index]:
                if state.get("status") not in {"ACTIVE", "DONE"}:
                    continue
                obs = state["observation"]
                _validate_action(module.agent(obs), obs)
                checked += 1
    if checked < 10:
        raise ValueError(f"insufficient replay observations for action contract check: {checked}")


def build(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, watcher_model=None, watcher_report=None,
          features=ROOT / "shared_features.py"):
    source, output = Path(source), Path(output)
    source_text = source.read_text(encoding="utf-8")
    _validate_watcher_build_mode(source_text, watcher_model, watcher_report)
    rendered = _agent_prefix(source_text)
    rendered = _replace_watcher_import(rendered, _watcher_replacement(watcher_model, watcher_report, features))
    _validate_source(rendered)
    output.write_text(rendered, encoding="utf-8")
    try:
        _validate_action_contract(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return {"source": str(source), "source_sha256": _sha256(source), "output": str(output),
            "output_sha256": _sha256(output), "watcher_embedded": watcher_model is not None}


def _literal_constants(path):
    wanted = {
        "ANIMAL_MIX", "LIVESTOCK_HANDS", "MELON_HANDS", "MAX_MELON_TILES",
        "LAND_TARGET_QUADRANTS", "SELL_MIN_PRICE_FRAC", "ENABLE_DEADLINE_PLANNER",
        "ENABLE_ADAPTIVE_TARGETS", "ENABLE_TOWN_WATCHER", "ENABLE_OPPONENT_WATCHER",
        "WATCHER_BACKEND", "HIRING_RAMP_MODE", "HIRING_CEILING",
        "LIVESTOCK_SPECIALISTS", "CROP_TRANSITION_MODE", "SALE_POLICY_MODE",
        "LAND_EXPANSION_MODE",
    }
    constants = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                constants[name] = ast.literal_eval(node.value)
    return constants


def write_manifest(build_result, manifest_path, features_path=ROOT / "shared_features.py",
                   watcher_model=None, watcher_report=None, readiness_report=None):
    features = _load_module(Path(features_path), "_manifest_features")
    report = None
    accepted = []
    if watcher_report:
        report = json.loads(Path(watcher_report).read_text(encoding="utf-8"))
        accepted = sorted(name for name, result in report.get("outputs", {}).items()
                          if result.get("accepted") is True and result.get("status") == "accepted")
    readiness = None
    if readiness_report:
        readiness = json.loads(Path(readiness_report).read_text(encoding="utf-8"))
    manifest = {
        "build_timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"path": build_result["source"], "sha256": build_result["source_sha256"]},
        "submission": {"path": build_result["output"], "sha256": build_result["output_sha256"],
                       "watcher_embedded": build_result["watcher_embedded"]},
        "constants": _literal_constants(build_result["source"]),
        "watcher": {
            "backend": _literal_constants(build_result["source"]).get("WATCHER_BACKEND"),
            "accepted_output_ids": accepted,
            "schema_version": features.SCHEMA_VERSION,
            "schema_hash": features.SCHEMA_HASH,
            "feature_order": list(features.FEATURE_ORDER),
            "model_sha256": _sha256(watcher_model) if watcher_model else None,
            "acceptance_report_sha256": _sha256(watcher_report) if watcher_report else None,
            "readiness_report_sha256": _sha256(readiness_report) if readiness_report else None,
            "dataset_hashes": (report or readiness or {}).get("metadata", {}).get(
                "dataset_hashes", (readiness or {}).get("dataset_hashes", {})),
        },
        "verification": {"static": "pending", "behavioral": "pending"},
    }
    Path(manifest_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watcher-model", type=Path)
    parser.add_argument("--watcher-report", type=Path)
    parser.add_argument("--features", type=Path, default=ROOT / "shared_features.py")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--readiness-report", type=Path)
    args = parser.parse_args(argv)
    result = build(args.source, args.output, args.watcher_model, args.watcher_report, args.features)
    if args.manifest:
        write_manifest(result, args.manifest, args.features, args.watcher_model,
                       args.watcher_report, args.readiness_report)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
