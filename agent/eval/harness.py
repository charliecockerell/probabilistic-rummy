"""
Head-to-head evaluation harness.

Sits above the policy + inference layers (it never reads either's internals).
Two *agents* — each a `BotPlayer` wrapping a policy and its own belief state —
play full games on a real `GameState`; the harness routes the visible signals
of each turn to the opposing agent's belief. It then aggregates a metric panel
over N games and sweeps hyperparameters.

Extensibility (the point of the registry)
------------------------------------------
A policy is anything implementing the `Policy` protocol below
(choose_draw / choose_discard / should_knock / knock_ev). New policies
(MCTS, RL, ...) plug in by adding one line to `POLICY_REGISTRY`; nothing else
in the harness changes, because it only ever talks to the protocol.

    from agent.eval.harness import run_match, sweep

    stats = run_match(("probabilistic", {"alpha": 0.1}), ("greedy", {}), n_games=100)
    df    = sweep({"alpha": [0.0, 0.1, 0.2]}, n_games=100)   # vs greedy
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np

from agent.cards import Card, find_best_melds
from agent.game import GameState
from agent.bot import BotPlayer, GreedyPolicy
from agent.eval.opponents import MeldSeekerPolicy, RationalOpponent
from agent.policy import ProbabilisticPolicy
from agent.policy.search import RolloutSearchPolicy


# ── Policy interface + registry ────────────────────────────────────────────────

class Policy(Protocol):
    """The contract every policy must satisfy. The harness only ever calls these."""
    def choose_draw(self, hand: List[Card], bs, discard_top: Optional[Card]) -> str: ...
    def choose_discard(self, hand: List[Card], bs, forbidden: Optional[Card] = None) -> Card: ...
    def should_knock(self, hand: List[Card], bs) -> bool: ...
    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]: ...


# Hyperparameters that configure the *belief* (BotPlayer), not the policy object.
BELIEF_PARAMS = {"mu", "nu", "lam"}


def _greedy_factory(**kw) -> Policy:
    return GreedyPolicy()


def _meld_seeker_factory(**kw) -> Policy:
    return MeldSeekerPolicy()


def _rational_factory(**kw) -> Policy:
    return RationalOpponent(w=kw.get("w", 2.0))


def _search_factory(**kw) -> Policy:
    return RolloutSearchPolicy(
        alpha=kw.get("alpha", 0.1),
        gamma=kw.get("gamma", 0.0),
        kappa=kw.get("kappa", 0.0),
        n_determinizations=kw.get("n_determinizations", 24),
        knock_samples=kw.get("knock_samples", 200),
        rollout_policy=kw.get("rollout_policy", "naive"),
        seed=kw.get("seed"),
    )


def _search_smart_factory(**kw) -> Policy:
    """Search with the probabilistic baseline as the playout policy."""
    kw.setdefault("rollout_policy", "probabilistic")
    return _search_factory(**kw)


def _prob_factory(**kw) -> Policy:
    return ProbabilisticPolicy(
        alpha=kw.get("alpha", 0.1),
        gamma=kw.get("gamma", 0.0),
        kappa=kw.get("kappa", 0.0),
        knock_samples=kw.get("knock_samples", 400),
        seed=kw.get("seed"),
    )


# name -> (policy factory, uses_belief). Register MCTS / RL here when built.
POLICY_REGISTRY: Dict[str, Tuple[Callable[..., Policy], bool]] = {
    "greedy": (_greedy_factory, False),
    "meld_seeker": (_meld_seeker_factory, False),
    "rational": (_rational_factory, False),
    "probabilistic": (_prob_factory, True),
    "search": (_search_factory, True),
    "search_smart": (_search_smart_factory, True),
}


def make_agent(kind: str, seat: int, seed: Optional[int] = None, **params) -> BotPlayer:
    """Build a seated agent: a BotPlayer wrapping a registered policy + its belief."""
    if kind not in POLICY_REGISTRY:
        raise ValueError(f"unknown policy '{kind}'. Registered: {list(POLICY_REGISTRY)}")
    factory, uses_belief = POLICY_REGISTRY[kind]

    belief_kw = {k: params[k] for k in BELIEF_PARAMS if k in params}
    policy_kw = {k: v for k, v in params.items() if k not in BELIEF_PARAMS}
    if uses_belief and seed is not None:
        policy_kw.setdefault("seed", seed)        # reproducible knock sampling

    policy = factory(**policy_kw)
    return BotPlayer(seat, kind, policy=policy, uses_belief=uses_belief, **belief_kw)


# ── A single game ──────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    winner: Optional[int]          # 0, 1, or None (draw / timeout)
    result_type: str               # 'gin' | 'undercut' | 'knock' | 'draw' | 'timeout'
    knocker: Optional[int]
    score: int                     # points awarded this hand (to the winner)
    deadwood: List[int]            # final deadwood value per seat [seat0, seat1]
    turns: int


def _final_deadwood(hand: List[Card]) -> int:
    _, dw = find_best_melds(hand)
    return sum(c.value for c in dw)


def play_game(agent0: BotPlayer, agent1: BotPlayer,
              seed: Optional[int] = None, max_turns: int = 200) -> GameResult:
    """Play one full game between two seated agents and return the outcome.

    `seed` controls the deal and any in-game reshuffle (the engine uses the
    `random` module); per-agent sampling RNGs are seeded when the agent is built.
    """
    if seed is not None:
        random.seed(seed)

    game = GameState()                       # fresh scores [0, 0]
    face_up = game.discard_pile[-1]
    agents = [agent0, agent1]
    agent0.start(game.hands[0], face_up)
    agent1.start(game.hands[1], face_up)

    turns = 0
    while not game.game_over and turns < max_turns:
        actor = agents[game.current_player]
        observer = agents[1 - game.current_player]
        top_before = game.discard_pile[-1] if game.discard_pile else None

        rec = actor.play_turn(game)
        turns += 1

        # Route the visible signals of this turn to the opposing belief.
        if observer.bs is not None and rec.drawn is not None:
            observer.saw_opponent_draw(rec.source, top_before)
            if not rec.knocked and rec.discard is not None:
                observer.saw_opponent_discard(rec.discard)

    return _result_from_game(game, turns)


def _result_from_game(game: GameState, turns: int) -> GameResult:
    ki = game.knock_info
    if ki is not None:
        knocker, opp = ki["knocker"], 1 - ki["knocker"]
        dw = [0, 0]
        dw[knocker] = ki["knocker_dw_value"]
        dw[opp] = ki["opponent_dw_value"]
        # Engine scoring is authoritative (knock-score sign fixed in game.py).
        return GameResult(game.winner, ki["result_type"], knocker, ki["score"], dw, turns)

    # No knock: declared draw (stock empty) or hit the turn cap.
    dw = [_final_deadwood(game.hands[0]), _final_deadwood(game.hands[1])]
    rtype = "draw" if game.game_over else "timeout"
    return GameResult(None, rtype, None, 0, dw, turns)


# ── A match: N games, aggregated from the subject's perspective ─────────────────

PolicySpec = Tuple[str, dict]      # (kind, params)


def _points_to(res: GameResult, seat: int) -> int:
    """Signed points this hand from `seat`'s perspective."""
    if res.winner is None:
        return 0
    return res.score if res.winner == seat else -res.score


def _play_indexed(subject: PolicySpec, opponent: PolicySpec,
                  seed: int, sub_seat: int) -> GameResult:
    """Build both agents for one game and play it. Pure function of its args
    (no shared state), so it is safe to run in a worker process."""
    sub_kind, sub_params = subject
    opp_kind, opp_params = opponent
    opp_seat = 1 - sub_seat

    sub_agent = make_agent(sub_kind, sub_seat, seed=seed * 7 + 1, **sub_params)
    opp_agent = make_agent(opp_kind, opp_seat, seed=seed * 7 + 2, **opp_params)
    a0, a1 = (sub_agent, opp_agent) if sub_seat == 0 else (opp_agent, sub_agent)
    return play_game(a0, a1, seed=seed)


def run_match(subject: PolicySpec, opponent: PolicySpec, n_games: int = 100,
              base_seed: int = 0, alternate_seats: bool = False,
              n_jobs: int = 1) -> dict:
    """Play `n_games` between `subject` and `opponent`; return a metric panel
    (all metrics reported from the subject's perspective — no single objective).

    Each game is an independent random deal (base_seed + i). With
    `alternate_seats=True` the subject swaps seats on odd games to cancel the
    first-move advantage; default False keeps the simple fixed-seat setup.

    `n_jobs` parallelises across games via joblib (-1 = all cores). Games are
    independent and each carries a fixed seed, so the aggregated panel is
    identical to the serial run regardless of `n_jobs` (only the per-process
    meld cache differs, which is lossless). Default 1 runs serially and needs no
    joblib — worth it for a slow subject (e.g. search) at large `n_games`.
    """
    seeds = [base_seed + i for i in range(n_games)]
    subject_seats = [1 if (alternate_seats and i % 2 == 1) else 0
                     for i in range(n_games)]

    if n_jobs in (1, None):
        results = [_play_indexed(subject, opponent, s, seat)
                   for s, seat in zip(seeds, subject_seats)]
    else:
        from joblib import Parallel, delayed   # lazy: only needed when parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(_play_indexed)(subject, opponent, s, seat)
            for s, seat in zip(seeds, subject_seats))

    return _aggregate(results, subject_seats, n_games)


def _aggregate(results: List[GameResult], subject_seats: List[int], n: int) -> dict:
    wins = losses = draws = timeouts = 0
    knocks = gins = undercut_against = undercut_by = 0
    dw_sum = margin_sum = turns_sum = 0

    for res, s in zip(results, subject_seats):
        dw_sum += res.deadwood[s]
        turns_sum += res.turns
        margin_sum += _points_to(res, s)

        if res.result_type == "draw":
            draws += 1
        elif res.result_type == "timeout":
            timeouts += 1

        if res.winner == s:
            wins += 1
        elif res.winner is not None:
            losses += 1

        if res.knocker == s:
            knocks += 1
            if res.result_type == "gin":
                gins += 1
            elif res.result_type == "undercut":
                undercut_against += 1          # subject knocked and got undercut
        elif res.knocker is not None and res.result_type == "undercut":
            undercut_by += 1                   # subject undercut the opponent

    return {
        "n_games": n,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
        "timeout_rate": timeouts / n,
        "avg_deadwood": dw_sum / n,
        "avg_score_margin": margin_sum / n,
        "knock_rate": knocks / n,
        "gin_rate": gins / n,
        "undercut_against_rate": undercut_against / n,   # subject knocked, lost to undercut
        "undercut_by_rate": undercut_by / n,             # subject undercut opponent
        "avg_turns": turns_sum / n,
    }


# ── Hyperparameter sweep ───────────────────────────────────────────────────────

def sweep(grid: Dict[str, list],
          subject_kind: str = "probabilistic",
          opponent: PolicySpec = ("greedy", {}),
          fixed: Optional[dict] = None,
          n_games: int = 100,
          base_seed: int = 0,
          alternate_seats: bool = False,
          n_jobs: int = 1) -> List[dict]:
    """Grid-search `subject_kind` against `opponent` over the cartesian product
    of `grid` (param -> list of values). `fixed` sets params held constant.

    Returns one record per configuration: {**params, **metric_panel}. Pass the
    list to `pandas.DataFrame(...)` in a notebook. Sweepable params include the
    policy knobs (alpha, gamma, kappa) and the belief knobs (mu, nu, lam).

    `n_jobs` parallelises across grid cells via joblib (-1 = all cores). The
    cells are independent, so this is a near-linear speedup; n_jobs=1 (default)
    runs serially and needs no joblib. Results are identical either way — each
    game's seed is fixed regardless of execution order — but note the meld cache
    is per-process, so parallel workers don't share cross-cell cache hits.
    """
    fixed = fixed or {}
    keys = list(grid)
    param_list = [{**fixed, **dict(zip(keys, combo))}
                  for combo in product(*(grid[k] for k in keys))]

    if n_jobs in (1, None):
        stats_list = [run_match((subject_kind, p), opponent, n_games=n_games,
                                base_seed=base_seed, alternate_seats=alternate_seats)
                      for p in param_list]
    else:
        from joblib import Parallel, delayed   # lazy: only needed when parallel
        stats_list = Parallel(n_jobs=n_jobs)(
            delayed(run_match)((subject_kind, p), opponent, n_games=n_games,
                               base_seed=base_seed, alternate_seats=alternate_seats)
            for p in param_list)

    return [{**p, **s} for p, s in zip(param_list, stats_list)]
