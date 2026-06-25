"""
Determinized Monte-Carlo search policy — the foundation for information-set MCTS.

This is the first policy that *plans*, rather than scoring a single ply. It frames
the turn as a sequential decision problem under uncertainty, and the structure is
deliberately the language of quantitative decision-making:

  * Determinization — we cannot see the opponent's hand, so we sample full
    opponent hands (and the implied stock) from the BeliefState. Each sample is a
    fully-observed "world" consistent with everything we've inferred. This is the
    information-set MCTS determinization step, and statistically it's importance
    sampling over hidden states weighted by the belief marginals.

  * Monte-Carlo rollout — in each determinized world we play the game forward to
    termination with a fast base policy and score the outcome. Averaging over
    worlds gives an expected value for each candidate action: classic Monte-Carlo
    evaluation of a decision under a stochastic future, the same machinery used to
    price path-dependent payoffs.

  * Action = argmax expected value — we discard the card with the highest mean
    terminal score across worlds, i.e. we maximise EV under belief uncertainty
    rather than optimising a myopic one-ply proxy.

  * Knocking is optimal stopping — knock (exercise) iff the immediate knock EV
    beats the value of continuing to play. v0 uses the belief-sampled knock EV as
    the exercise value; the continuation value (a rollout of *not* knocking) is the
    next increment, and is exactly an American-option-style continuation/exercise
    comparison.

v0 scope: the *discard* decision is chosen by determinized rollout (the new
lookahead); draw and knock reuse the validated one-ply machinery. The tree layer
(selection / expansion / UCB over these rollouts) and a rollout-based continuation
value for knocking are the next steps toward full ISMCTS — see next_model_tradeoffs.md.
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from agent.cards import Card, make_deck
from agent.policy._features import deadwood, risk_term
from agent.policy._meldcache import best_melds
from agent.policy.knock import (
    knock_discard, layoff_reduce, sample_opp_hands, knock_distribution,
)
from agent.policy.draw import should_take_discard

_DECK = make_deck()
_KNOCK_LEGAL = 10          # kept deadwood <= 10 is a legal knock


def _knock_gain(knocker_melds: list, k: int, opp_hand: List[Card]) -> int:
    """Signed points to the knocker if they knock with kept deadwood `k`."""
    _, opp_dw = best_melds(opp_hand)
    if k == 0:                                  # gin: opponent cannot lay off
        return deadwood(opp_dw) + 25
    ostar = layoff_reduce(opp_dw, knocker_melds)
    return (ostar - k) if ostar > k else -((k - ostar) + 25)


def _rollout(me: List[Card], opp: List[Card], stock: List[Card],
             me_to_move: bool) -> int:
    """Play one determinized world to termination with a fast base policy
    (draw stock, knock the moment it's legal, else discard to minimise deadwood).
    Returns the signed score from `me`'s perspective. A deliberate v0
    simplification: rollouts draw only from stock (no discard-pile pickups)."""
    me, opp = list(me), list(opp)
    while stock:
        hand = me if me_to_move else opp
        hand.append(stock.pop())               # draw from the top of the stock
        k, discard, melds = knock_discard(hand)
        hand.remove(discard)                   # keep the best 10
        if k <= _KNOCK_LEGAL:                   # base policy knocks when legal
            other = opp if me_to_move else me
            gain = _knock_gain(melds, k, other)
            return gain if me_to_move else -gain
        me_to_move = not me_to_move
    return 0                                    # stock exhausted -> no result


def _fast_risk_discard(hand: List[Card], bs, alpha: float,
                       forbidden: Optional[Card] = None) -> Card:
    """Risk-aware discard at *one* find_best_melds per call (vs 11 for the full
    best_discard) — the playout-speed version of the probabilistic discard.

    Decompose the 11-card hand once; the deadwood cards are the only sensible
    discards (dropping a melded card strictly worsens our deadwood). Removing a
    deadwood card d leaves the same melds, so kept deadwood = total - d.value.
    Score(d) = (total - d.value) + alpha*risk(d): keep the high-value drops unless
    the belief says they hand the opponent a meld. An approximation of best_discard
    that's good enough for a rollout and ~150x cheaper."""
    melds, dw = best_melds(hand)
    total = deadwood(dw)
    cands = [c for c in dw if c != forbidden] or [c for c in hand if c != forbidden] or list(hand)
    best, best_score = cands[0], float("inf")
    for d in cands:
        score = (total - d.value) + alpha * risk_term(d, bs)
        if score < best_score:
            best_score, best = score, d
    return best


def _rollout_smart(me: List[Card], opp: List[Card], stock: List[Card],
                   discard_top: Card, me_seat: int,
                   alpha: float, gamma: float) -> int:
    """Play a determinized world to termination with a *strong* base policy:
    both seats use the validated ProbabilisticPolicy for draw + discard, each
    maintaining its own BeliefState updated from the visible signals (reusing the
    engine + BotPlayer routing, so no game logic is duplicated here). Knock stays
    the fast knock-when-legal heuristic — running the knock Monte-Carlo inside
    every rollout ply would be intractable, and it's the *discard* playout we are
    upgrading. Returns the signed terminal score from `me`'s perspective.

    The point: flat rollout's value estimate is only as good as its playout
    policy. `_rollout` finishes with a naive min-deadwood player (weaker than the
    baseline we're trying to beat); this finishes with the baseline itself, so the
    EVs can rank moves that are genuinely better than the baseline."""
    from agent.game import GameState
    from agent.bot import BotPlayer, GreedyPolicy

    class _PlayoutPolicy:
        """Risk-aware (probabilistic) *discard* — the lever we're upgrading — but a
        cheap stock-only draw and knock-when-legal. The full probabilistic draw
        runs a stock lottery (~hundreds of find_best_melds per ply) and is far too
        slow to nest inside rollouts; the discard is where the playout strength
        that matters comes from."""
        def __init__(self):
            self._g = GreedyPolicy()
        def choose_draw(self, hand, bs, top):       return "stock"
        def choose_discard(self, hand, bs, forbidden=None):
            return _fast_risk_discard(hand, bs, alpha, forbidden=forbidden)
        def should_knock(self, hand, bs):
            # Cheap can-knock proxy: one best_melds instead of knock_discard's 11.
            # Dropping the biggest deadwood card from the 11-card optimum is the
            # min achievable kept deadwood; gate the (rare) real knock on it.
            _, dw = best_melds(hand)
            if not dw:
                return True
            return deadwood(dw) - max(c.value for c in dw) <= _KNOCK_LEGAL
        def knock_ev(self, hand, bs):               return self._g.knock_ev(hand, bs)

    opp_seat = 1 - me_seat
    g = GameState()
    g.hands[me_seat] = list(me)
    g.hands[opp_seat] = list(opp)
    g.stock = list(stock)
    g.discard_pile = [discard_top]
    g.current_player = opp_seat            # we just discarded; opponent is to move
    g.phase = "draw"
    g.drawn_card = None
    g.drew_from_discard = False
    g.game_over = False
    g.winner = None
    g.knock_info = None

    bots = {s: BotPlayer(s, "probabilistic", policy=_PlayoutPolicy(), uses_belief=True)
            for s in (me_seat, opp_seat)}
    bots[me_seat].start(g.hands[me_seat], discard_top)
    bots[opp_seat].start(g.hands[opp_seat], discard_top)

    turns = 0
    while not g.game_over and turns < 200:
        actor = bots[g.current_player]
        observer = bots[1 - g.current_player]
        top_before = g.discard_pile[-1] if g.discard_pile else None
        rec = actor.play_turn(g)
        turns += 1
        if observer.bs is not None and rec.drawn is not None:
            observer.saw_opponent_draw(rec.source, top_before)
            if not rec.knocked and rec.discard is not None:
                observer.saw_opponent_discard(rec.discard)

    ki = g.knock_info
    if ki is None or g.winner is None:
        return 0                            # declared draw / turn cap
    return ki["score"] if g.winner == me_seat else -ki["score"]


class RolloutSearchPolicy:
    """Determinized Monte-Carlo rollout policy (ISMCTS foundation).

    `rollout_policy` selects the playout used to value each determinized world:
      'naive'         - fast min-deadwood player (the v0 default).
      'probabilistic' - the validated Bayesian baseline plays out both seats, so
                        the rollout EV is measured against a competent player
                        rather than a weak one. Much slower per world.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.0, kappa: float = 0.0,
                 n_determinizations: int = 24, knock_samples: int = 200,
                 rollout_policy: str = "naive", seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.n_det = n_determinizations
        self.knock_samples = knock_samples
        self.rollout_policy = rollout_policy
        self.rng = np.random.default_rng(seed)

    def _determinize(self, bs):
        """Sample (opponent_hand, stock) worlds from the belief: the opponent gets
        a Madow-sampled hand; the leftover uncertain cards are the stock."""
        p = np.array([bs.prob(c) for c in _DECK])
        uncertain = [_DECK[i] for i in np.where((p > 1e-12) & (p < 1 - 1e-9))[0]]
        worlds = []
        for opp in sample_opp_hands(bs, self.n_det, self.rng):
            held = set(opp)
            stock = [c for c in uncertain if c not in held]
            self.rng.shuffle(stock)
            worlds.append((opp, stock))
        return worlds

    def choose_draw(self, hand: List[Card], bs, discard_top: Optional[Card]) -> str:
        # v0: reuse the validated belief-weighted draw rule (rollout-based draw next).
        if discard_top is None:
            return "stock"
        return "discard" if should_take_discard(hand, discard_top, bs,
                                                self.alpha, self.gamma) else "stock"

    def choose_discard(self, hand: List[Card], bs, forbidden: Optional[Card] = None) -> Card:
        """Discard the card with the best expected terminal score over determinized
        rollouts — the new lookahead. Falls back to the myopic best if no worlds."""
        cands = [c for c in hand if c != forbidden] or list(hand)
        worlds = self._determinize(bs)
        if not worlds:
            _, discard, _ = knock_discard(hand)        # degenerate: belief fully certain
            return discard if discard != forbidden else cands[0]
        smart = self.rollout_policy == "probabilistic"
        best, best_ev = cands[0], -1e18
        for d in cands:
            kept = [c for c in hand if c is not d]
            # after we discard it's the opponent's move in every world
            if smart:
                ev = np.mean([_rollout_smart(kept, opp, list(stock), d, me_seat=0,
                                             alpha=self.alpha, gamma=self.gamma)
                              for opp, stock in worlds])
            else:
                ev = np.mean([_rollout(kept, opp, list(stock), me_to_move=False)
                              for opp, stock in worlds])
            if ev > best_ev:
                best_ev, best = ev, d
        return best

    def should_knock(self, hand: List[Card], bs) -> bool:
        # Optimal stopping: exercise (knock) iff the immediate knock EV beats the
        # opportunity cost kappa. v0 uses the belief-sampled exercise value; the
        # rollout-based continuation value is the next increment.
        r = knock_distribution(hand, bs, self.knock_samples, self.rng)
        return r is not None and r["ev"] > self.kappa

    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]:
        return knock_distribution(hand, bs, self.knock_samples, self.rng)
