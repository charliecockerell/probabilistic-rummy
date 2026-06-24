"""
Bot player for interactive (human-vs-bot) play.

This is application glue that sits *above* the policy and inference layers:
it maintains a BeliefState over the opponent's (human's) hand, feeds it the
visible events of the game, and asks a policy for each action. It never reads
the opponent's hand — only its own hand and the belief.

Two opponents are exposed via a common interface
(choose_draw / choose_discard / should_knock / knock_ev):

  GreedyPolicy        - the sim.py baseline: take the discard iff it lowers
                        deadwood, discard greedily, knock the moment legal.
                        Ignores the belief entirely.
  ProbabilisticPolicy - the Bayesian agent from agent.policy.

Belief event routing mirrors notebooks/bayesian_signals.ipynb: a stock draw
is deferred to the following discard (so the stock-draw step is counted once),
and declining the face-up top is recorded as a pass signal.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from agent.cards import Card, find_best_melds
from agent.inference import BeliefState
from agent.eval.sim import _should_draw_discard, _random_discard
from agent.policy import ProbabilisticPolicy, can_knock, knock_discard


@dataclass
class TurnRecord:
    """Structured outcome of one bot turn — lets a caller (e.g. the eval harness)
    route the visible signals to the *other* seat's belief state."""
    seat: int
    source: str                       # 'stock' | 'discard'
    drawn: Optional[Card]             # card drawn (None if the stock ran out)
    knocked: bool
    discard: Optional[Card]           # card discarded / knock-discard (None if game ended on draw)
    ended: bool                       # True if the game is over after this turn


class GreedyPolicy:
    """sim.py's greedy baseline, wrapped in the policy interface (belief ignored)."""

    def choose_draw(self, hand: List[Card], bs, discard_top: Optional[Card]) -> str:
        if discard_top is not None and _should_draw_discard(hand, discard_top):
            return "discard"
        return "stock"

    def choose_discard(self, hand: List[Card], bs, forbidden: Optional[Card] = None) -> Card:
        if forbidden is None:
            return _random_discard(hand)
        # Min-deadwood discard, excluding the card just taken from the discard pile.
        cands = [c for c in hand if c != forbidden] or list(hand)
        best, best_dw = None, float("inf")
        for card in cands:
            _, dw = find_best_melds([c for c in hand if c is not card])
            v = sum(c.value for c in dw)
            if v < best_dw:
                best_dw, best = v, card
        return best

    def should_knock(self, hand: List[Card], bs) -> bool:
        return can_knock(hand)            # knock as soon as it is legal

    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]:
        k, discard, _ = knock_discard(hand)
        return None if k > 10 else {"k": k, "discard": discard}


def make_policy(kind: str, **kw):
    if kind == "greedy":
        return GreedyPolicy(), False           # (policy, uses_belief)
    if kind == "probabilistic":
        return ProbabilisticPolicy(
            alpha=kw.get("alpha", 0.1),
            gamma=kw.get("gamma", 0.0),
            kappa=kw.get("kappa", 0.0),
            knock_samples=kw.get("knock_samples", 400),
            seed=kw.get("seed"),
        ), True
    raise ValueError(f"unknown opponent kind: {kind}")


class BotPlayer:
    """
    Drives one seat. `seat` is the bot's index (1 by convention; human is 0).

    Observation hooks are called by the server as the human acts; take_turn()
    performs a full bot turn on the GameState and returns a short description.
    """

    NAMES = {"greedy": "Greedy bot", "probabilistic": "Probabilistic bot"}

    def __init__(self, seat: int, kind: str,
                 mu: float = 0.4, nu: float = 2.0, lam: float = 0.4,
                 policy=None, uses_belief: Optional[bool] = None, **kw):
        self.seat = seat
        self.kind = kind
        self.name = self.NAMES.get(kind, "Bot")
        # A prebuilt policy may be injected (e.g. by the eval registry, so new
        # policies like MCTS/RL slot in without touching make_policy here).
        if policy is not None:
            self.policy = policy
            self.uses_belief = bool(uses_belief)
        else:
            self.policy, self.uses_belief = make_policy(kind, **kw)
        self.mu, self.nu, self.lam = mu, nu, lam
        self.bs: Optional[BeliefState] = None
        self._opp_drew_from_stock = False

    # -- lifecycle -----------------------------------------------------------

    def start(self, own_hand: List[Card], face_up: Card) -> None:
        self.bs = BeliefState(list(own_hand), face_up) if self.uses_belief else None
        self._opp_drew_from_stock = False

    # -- observing the opponent (human) --------------------------------------

    def saw_opponent_draw(self, source: str, top: Optional[Card]) -> None:
        if self.bs is None:
            return
        if source == "discard" and top is not None:
            self.bs.observe_opponent_draw_discard_bayesian(top, nu=self.nu)
            self._opp_drew_from_stock = False
        else:  # stock: they declined the face-up top, then drew unseen
            if top is not None:
                self.bs.observe_opponent_pass_discard(top, mu=self.mu)
            self._opp_drew_from_stock = True

    def saw_opponent_discard(self, card: Card) -> None:
        if self.bs is None:
            return
        if self._opp_drew_from_stock:
            self.bs.observe_stock_draw_then_discard_bayesian(card, lambda_=self.lam)
        else:
            self.bs.observe_opponent_discard_bayesian(card, lambda_=self.lam)
        self._opp_drew_from_stock = False

    # -- observing our own actions -------------------------------------------

    def _saw_own_draw(self, source: str, card: Card) -> None:
        if self.bs is None:
            return
        if source == "discard":
            self.bs.observe_own_draw_discard(card)
        else:
            self.bs.observe_own_draw_stock(card)

    def _saw_own_discard(self, card: Card) -> None:
        if self.bs is not None:
            self.bs.observe_own_discard(card)

    # -- acting --------------------------------------------------------------

    def play_turn(self, game) -> TurnRecord:
        """Play one full bot turn (draw, then knock or discard) on `game`,
        updating this bot's own belief. Returns a TurnRecord so a caller can
        feed the visible signals to the opposing seat."""
        seat = self.seat
        top = game.discard_pile[-1] if game.discard_pile else None
        hand10 = list(game.hands[seat])

        source = self.policy.choose_draw(hand10, self.bs, top)
        game.draw(source)
        if game.game_over:            # stock exhausted -> engine declared a draw
            return TurnRecord(seat, source, None, knocked=False, discard=None, ended=True)

        drawn = game.drawn_card
        self._saw_own_draw(source, drawn)

        hand11 = list(game.hands[seat])

        # Can't throw back a card just taken from the discard pile.
        forbidden = drawn if source == "discard" else None

        if self.policy.should_knock(hand11, self.bs):
            info = self.policy.knock_ev(hand11, self.bs)
            discard = info["discard"]
            # Only knock if the knock-discard is legal; otherwise play on this turn.
            if discard != forbidden:
                game.knock(str(discard))
                return TurnRecord(seat, source, drawn, knocked=True, discard=discard, ended=True)

        discard = self.policy.choose_discard(hand11, self.bs, forbidden=forbidden)
        self._saw_own_discard(discard)
        game.discard(str(discard))
        return TurnRecord(seat, source, drawn, knocked=False, discard=discard,
                          ended=game.game_over)

    def take_turn(self, game) -> str:
        """Play one turn and return a human-readable description (used by the UI)."""
        rec = self.play_turn(game)
        sym = {"discard": "the discard", "stock": "the stock"}.get(rec.source, rec.source)
        if rec.drawn is None:
            return f"{self.name} could not draw — the stock is empty."
        if rec.knocked:
            return f"{self.name} drew from {sym} and knocked, discarding {rec.discard}."
        return f"{self.name} drew from {sym} and discarded {rec.discard}."

    # -- diagnostics ---------------------------------------------------------

    def belief_sum(self) -> Optional[float]:
        return None if self.bs is None else self.bs.hand_size_belief
