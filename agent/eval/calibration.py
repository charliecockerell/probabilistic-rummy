"""
Belief-calibration diagnostic.

Validates the *inference* directly, independent of win rate: across a game, how
much probability does the BeliefState put on the opponent's ACTUAL cards? A
working belief should pull this above the uniform prior as it accumulates the
opponent's draws and discards. The soft meld-inference assumes a meld-collecting
opponent, so the curve should climb faster and higher against `meld_seeker` than
against belief-illegible `greedy`.

Run (saves calibration.png in the repo root):
    python -m agent.eval.calibration [n_games]
"""

from __future__ import annotations
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from agent.game import GameState
from agent.eval.harness import make_agent


def _mean_p_true(bs, opp_hand: List) -> float:
    """Mean probability the belief assigns to the opponent's current cards."""
    return float(np.mean([bs.prob(c) for c in opp_hand])) if opp_hand else 0.0


def calibration_curve(opponent_kind: str, n_games: int = 40, base_seed: int = 0,
                      knock_samples: int = 40, max_turns: int = 200,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Probabilistic agent (seat 0) vs `opponent_kind` (seat 1). Returns
    (turns, mean P on opp's true cards, SE, prior) aggregated over games."""
    by_turn: Dict[int, List[float]] = defaultdict(list)
    priors: List[float] = []

    for i in range(n_games):
        seed = base_seed + i
        prob = make_agent("probabilistic", 0, seed=seed * 7 + 1,
                          alpha=0.1, kappa=0.0, knock_samples=knock_samples)
        opp = make_agent(opponent_kind, 1, seed=seed * 7 + 2)

        random.seed(seed)
        game = GameState()
        face_up = game.discard_pile[-1]
        prob.start(game.hands[0], face_up)
        opp.start(game.hands[1], face_up)
        agents = [prob, opp]
        priors.append(_mean_p_true(prob.bs, game.hands[1]))

        t = 0
        while not game.game_over and t < max_turns:
            actor = agents[game.current_player]
            observer = agents[1 - game.current_player]
            top_before = game.discard_pile[-1] if game.discard_pile else None

            rec = actor.play_turn(game)
            t += 1
            if observer.bs is not None and rec.drawn is not None:
                observer.saw_opponent_draw(rec.source, top_before)
                if not rec.knocked and rec.discard is not None:
                    observer.saw_opponent_discard(rec.discard)
            # How well does the prob agent's belief track the opponent's hand now?
            by_turn[t].append(_mean_p_true(prob.bs, game.hands[1]))

    # keep turns that enough games reached, so the tail isn't one lucky game
    keep = [t for t in sorted(by_turn) if len(by_turn[t]) >= max(3, n_games // 4)]
    mean = np.array([np.mean(by_turn[t]) for t in keep])
    se = np.array([np.std(by_turn[t]) / np.sqrt(len(by_turn[t])) for t in keep])
    return np.array(keep), mean, se, float(np.mean(priors))


def main(n_games: int = 40) -> None:
    import pathlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    prior = None
    for kind, color in [("greedy", "tab:blue"), ("meld_seeker", "tab:orange")]:
        turns, mean, se, prior = calibration_curve(kind, n_games=n_games)
        ax.plot(turns, mean, color=color, label=f"vs {kind}")
        ax.fill_between(turns, mean - se, mean + se, color=color, alpha=0.2)
    ax.axhline(prior, ls="--", color="grey", label=f"uniform prior ~{prior:.2f}")
    ax.set_xlabel("turn")
    ax.set_ylabel("mean P on opponent's actual cards")
    ax.set_title("Belief calibration: does P concentrate on the true hand?")
    ax.legend()
    out = pathlib.Path(__file__).resolve().parents[2] / "calibration.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
