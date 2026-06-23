"""
Action-selection policy over a fixed belief model.

The policy reads only the BeliefState public API (`bs.prob`, `bs.stock_size`)
and the agent's own hand. It never touches inference internals or game-engine
state, so it stays swappable (greedy vs MCTS vs RL) over the same beliefs.

Three decisions, matching a gin-rummy turn:
  - draw    : take the discard top or draw from stock          (10-card hand)
  - discard : which card to throw                              (11-card hand)
  - knock   : whether to end the hand, and the EV of doing so  (11-card hand)
"""

from __future__ import annotations
from typing import List, Optional

import numpy as np

from agent.cards import Card
from agent.policy._features import (
    deadwood, hand_deadwood, melds_containing,
    deadwood_after_discard, risk_term, position_cost,
)
from agent.policy.discard import discard_score, best_discard
from agent.policy.draw import (
    stock_candidates, value_take, value_stock, should_take_discard,
)
from agent.policy.knock import (
    knock_discard, can_knock, layoff_reduce,
    sample_opp_hands, knock_distribution, should_knock,
)

__all__ = [
    "ProbabilisticPolicy",
    "deadwood", "hand_deadwood", "melds_containing",
    "deadwood_after_discard", "risk_term", "position_cost",
    "discard_score", "best_discard",
    "stock_candidates", "value_take", "value_stock", "should_take_discard",
    "knock_discard", "can_knock", "layoff_reduce",
    "sample_opp_hands", "knock_distribution", "should_knock",
]


class ProbabilisticPolicy:
    """
    The Bayesian baseline policy. Hyperparameters:
      alpha  : weight on discard risk R(d) (0 = pure greedy deadwood)
      gamma  : info-leak penalty for taking the discard face-up
      kappa  : opportunity cost of playing on, the knock threshold
      knock_samples : opponent-hand samples per knock evaluation
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.0, kappa: float = 0.0,
                 knock_samples: int = 2000, seed: Optional[int] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa
        self.knock_samples = knock_samples
        self.rng = np.random.default_rng(seed)

    def choose_draw(self, hand: List[Card], bs, discard_top: Card) -> str:
        """Return 'discard' to take the face-up top, else 'stock'."""
        take = should_take_discard(hand, discard_top, bs, self.alpha, self.gamma)
        return "discard" if take else "stock"

    def choose_discard(self, hand: List[Card], bs) -> Card:
        """Pick the discard from an 11-card hand."""
        return best_discard(hand, bs, self.alpha)

    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]:
        """Full knock distribution (EV, P(undercut), ...) or None if illegal."""
        return knock_distribution(hand, bs, self.knock_samples, self.rng)

    def should_knock(self, hand: List[Card], bs) -> bool:
        """Whether to knock from an 11-card hand."""
        return should_knock(hand, bs, self.kappa, self.knock_samples, self.rng)
