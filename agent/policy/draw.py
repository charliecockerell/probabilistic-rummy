"""
Draw policy.

Each turn we choose the known top of the discard pile or an unknown card
from stock, then discard back to 10.

  V_take  = position_cost(hand + [top], forbidden=top)   # can't re-discard top
  V_stock = sum_c P(top stock = c) * position_cost(hand + [c])

Take the discard iff  V_take + info_penalty < V_stock, where the info penalty
prices the leak from taking face-up (the opponent learns the card is useful
to us).
"""

from __future__ import annotations
from typing import List

from agent.cards import Card, make_deck
from agent.policy._features import position_cost


def stock_candidates(bs) -> List[Card]:
    """Cards that could still be sitting in the stock (0 < p < 1)."""
    return [c for c in make_deck() if 0.0 < bs.prob(c) < 1.0]


def value_take(hand: List[Card], top: Card, bs, alpha: float) -> float:
    """V_take: best position cost after taking the face-up `top`."""
    cost, _ = position_cost(hand + [top], bs, alpha, forbidden=top)
    return cost


def value_stock(hand: List[Card], bs, alpha: float) -> float:
    """V_stock: belief-weighted expected position cost of a stock draw."""
    cand = stock_candidates(bs)
    Z = sum(1.0 - bs.prob(c) for c in cand)
    if Z <= 0:
        return float('inf')
    return sum(((1.0 - bs.prob(c)) / Z) * position_cost(hand + [c], bs, alpha)[0]
               for c in cand)


def should_take_discard(hand: List[Card], top: Card, bs,
                        alpha: float = 0.1, gamma: float = 0.0) -> bool:
    """True iff taking the discard top beats drawing from stock."""
    return value_take(hand, top, bs, alpha) + gamma * top.value < value_stock(hand, bs, alpha)  # info penalty is placeholder here to be tuned.
