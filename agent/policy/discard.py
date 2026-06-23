"""
Discard policy.

score(d) = -(D(d) + alpha * R(d)),  discard = argmax score.

D(d) protects our own melds for free (alpha=0 is pure greedy deadwood
minimisation). R(d) prices the risk of feeding the opponent. alpha trades
the two off; alpha=0 never sacrifices any deadwood.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from agent.cards import Card
from agent.policy._features import deadwood_after_discard, risk_term


def discard_score(d: Card, hand: List[Card], bs, alpha: float) -> Tuple[float, int, float]:
    """Return (score, D, R) for discarding `d` from `hand`."""
    D = deadwood_after_discard(d, hand)
    R = risk_term(d, bs)
    return -(D + alpha * R), D, R


def best_discard(hand: List[Card], bs, alpha: float = 0.1,
                 forbidden: Optional[Card] = None) -> Card:
    """The card maximising the discard score (ties broken by hand order).

    `forbidden` excludes a card from consideration — used to honour the rule
    that the card just taken from the discard pile can't be thrown straight back.
    """
    candidates = [c for c in hand if c != forbidden] or list(hand)
    return max(candidates, key=lambda d: discard_score(d, hand, bs, alpha)[0])
