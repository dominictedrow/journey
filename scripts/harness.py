"""Shared simulation harness for testing Kaggriculture strategies.

run_matches() is the one entry point every strategy agent (in strategies/*/)
should use to score itself against a baseline opponent.
"""

import os
import statistics
import math
from concurrent.futures import ProcessPoolExecutor

from kaggle_environments import make


def _final_money(env, player_idx):
    final_step = env.steps[-1]
    return final_step[player_idx].observation["farms"][player_idx]["money"]


def _run_episode(agent, opponent, agent_idx, episode_steps, seed=None):
    # kaggle_environments is an expensive import (pulls in OpenSpiel's huge
    # game list) -- ProcessPoolExecutor pays that cost once per worker
    # process (reused across all tasks assigned to it), not once per
    # episode, so this only pays off once n_episodes is meaningfully larger
    # than core count.
    agents = [agent, opponent] if agent_idx == 0 else [opponent, agent]
    config = {"episodeSteps": episode_steps}
    if seed is not None:
        config["seed"] = seed
    env = make("kaggriculture", configuration=config, debug=False)
    env.run(agents)
    return _final_money(env, agent_idx)


def _run_episode_instrumented(agent, opponent, agent_idx, episode_steps, seed=None):
    from evaluation_metrics import episode_metrics
    agents = [agent, opponent] if agent_idx == 0 else [opponent, agent]
    config = {"episodeSteps": episode_steps}
    if seed is not None:
        config["seed"] = seed
    env = make("kaggriculture", configuration=config, debug=False)
    env.run(agents)
    return episode_metrics(env, agent_idx)


def run_matches(agent, opponent, n_episodes, episode_steps=720, n_workers=None, seeds=None):
    """Run n_episodes of agent vs opponent, split evenly across both player
    orders, and return stats on the agent's own final money (not win/loss).

    `seeds`, if given, must have length n_episodes -- episode i uses
    seeds[i] for weed-spawn/shop-unlock RNG (kaggriculture's `seed` config
    key). Reusing the same seeds list across two configs being compared
    makes it a paired comparison (see compare_paired below): both sides
    face identical RNG per episode, collapsing the between-episode variance
    that swamps small unpaired samples (1st.md "Measurement problem").
    """
    if seeds is not None and len(seeds) != n_episodes:
        raise ValueError("seeds must have length n_episodes")
    # ponytail: ProcessPoolExecutor needs picklable callables. Plain
    # top-level functions and built-in strings ("random", "starter") work
    # fine. The phase-1 sweep engines' make_agent(...) factories return
    # closures, which don't pickle -- self-play against an old phase-1
    # config would need that config rebuilt inside the worker (factory +
    # kwargs passed separately), not attempted here since it wasn't asked
    # for.
    workers = min(n_workers or os.cpu_count(), n_episodes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_episode, agent, opponent, i % 2, episode_steps,
                        seeds[i] if seeds is not None else None)
            for i in range(n_episodes)
        ]
        moneys = [f.result() for f in futures]

    return _stats(moneys)


def _stats(moneys):
    return {
        "n": len(moneys),
        "mean": statistics.mean(moneys),
        "median": statistics.median(moneys),
        "min": min(moneys),
        "max": max(moneys),
        "stdev": statistics.stdev(moneys) if len(moneys) > 1 else 0.0,
    }


def compare_paired(agent_a, agent_b, opponent, n_episodes, episode_steps=720, n_workers=None, seed_base=0):
    """Paired A/B: run agent_a and agent_b against the same opponent using
    the same seed list (and the same player-order alternation), so episode
    i is an apples-to-apples pairing. Returns each side's own stats plus the
    paired difference (a - b) stats, which is the number that should drive
    a keep-or-revert call (1st.md wants n>=20 here).
    """
    seeds = list(range(seed_base, seed_base + n_episodes))
    workers = min(n_workers or os.cpu_count(), 2 * n_episodes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs_a = [pool.submit(_run_episode, agent_a, opponent, i % 2, episode_steps, seeds[i]) for i in range(n_episodes)]
        futs_b = [pool.submit(_run_episode, agent_b, opponent, i % 2, episode_steps, seeds[i]) for i in range(n_episodes)]
        moneys_a = [f.result() for f in futs_a]
        moneys_b = [f.result() for f in futs_b]
    diffs = [a - b for a, b in zip(moneys_a, moneys_b)]
    diff_stdev = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    margin = 1.96 * diff_stdev / math.sqrt(n_episodes) if n_episodes else 0.0
    return {
        "a": _stats(moneys_a),
        "b": _stats(moneys_b),
        "diff_mean": statistics.mean(diffs),
        "diff_stdev": diff_stdev,
        "diff_min": min(diffs),
        "diff_max": max(diffs),
        "diff_ci95": [statistics.mean(diffs) - margin, statistics.mean(diffs) + margin],
        "wins": sum(diff > 0 for diff in diffs),
        "ties": sum(diff == 0 for diff in diffs),
        "losses": sum(diff < 0 for diff in diffs),
        "n": n_episodes,
    }


def run_matches_instrumented(agent, opponent, n_episodes, episode_steps=720,
                             n_workers=None, seeds=None):
    """Run matches and return both raw and aggregated phase-3 telemetry."""
    from evaluation_metrics import summarize_episodes
    if seeds is not None and len(seeds) != n_episodes:
        raise ValueError("seeds must have length n_episodes")
    workers = min(n_workers or os.cpu_count(), n_episodes)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_episode_instrumented, agent, opponent, i % 2,
                        episode_steps, seeds[i] if seeds is not None else None)
            for i in range(n_episodes)
        ]
        episodes = [future.result() for future in futures]
    return {"summary": summarize_episodes(episodes), "episodes": episodes}


def run_causal_triplet(*args, **kwargs):
    """Lazy compatibility entry point for the development-only adapter."""
    from kaggle_replays.causal_sim import run_causal_triplet as _run
    return _run(*args, **kwargs)


def _demo():
    stats = run_matches("pass", "random", n_episodes=4, episode_steps=50)
    assert stats["n"] == 4
    assert stats["min"] <= stats["mean"] <= stats["max"]
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["mean"] >= 0

    seeded = run_matches("pass", "random", n_episodes=4, episode_steps=50, seeds=[1, 2, 3, 4])
    assert seeded["n"] == 4

    paired = compare_paired("pass", "starter", "random", n_episodes=4, episode_steps=50)
    assert paired["n"] == 4
    # same seed list on both sides -> "pass" (no-op) is deterministic, so its
    # own stdev collapses to 0 even though the opponent's RNG still varies
    assert paired["a"]["stdev"] == 0.0

    print("harness self-check OK:", stats)


if __name__ == "__main__":
    _demo()
