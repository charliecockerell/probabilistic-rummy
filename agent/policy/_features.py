"""
Shared meld/deadwood features used by the discard and draw policies.

These read only the BeliefState public API (`bs.prob`) plus the card
primitives in `agent.cards`. No game-engine state, no inference internals.
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Optional, Tuple

from agent.cards import Card, SUITS, RANKS, RANK_INDEX
from agent.policy._meldcache import best_melds


def deadwood(cards: List[Card]) -> int:
    """Total point value of a list of (unmelded) cards."""
    return sum(c.value for c in cards)


def hand_deadwood(hand: List[Card]) -> int:
    """Deadwood of a hand under its best meld decomposition."""
    _, dw = best_melds(hand)
    return deadwood(dw)


def melds_containing(d: Card) -> List[Tuple[Card, ...]]:
    """
    Every concrete 3-card meld `d` could complete, as the tuple of the
    other cards needed.

    Set melds : d plus two other suits of the same rank  -> C(3,2)=3 pairs.
    Run melds : d as the low / middle / high card of a 3-run in its own suit.
    """
    melds: List[Tuple[Card, ...]] = [
        (x, y) for x, y in combinations([Card(d.rank, s) for s in SUITS if s != d.suit], 2)
    ]
    ri = RANK_INDEX[d.rank]
    for start in (ri - 2, ri - 1, ri):
        if start < 0 or start + 2 >= len(RANKS):
            continue
        run = [Card(RANKS[start + k], d.suit) for k in range(3)]
        melds.append(tuple(c for c in run if c != d))
    return melds


def deadwood_after_discard(d: Card, hand: List[Card]) -> int:
    """D(d): our deadwood if we discard `d` (lower is better)."""
    _, dw = best_melds([c for c in hand if c != d])
    return deadwood(dw)


def risk_term(d: Card, bs) -> float:
    """
    R(d): expected points conceded if the opponent melds `d`.

    Independence approximation on the belief marginals: the chance the
    opponent holds both partners of a meld is taken as p(x)*p(y), weighted
    by the points that meld is worth, summed over every meld `d` completes.
    Conservative: since Sum p = 10, partners are negatively correlated, so
    p(x)p(y) overstates the joint and thus overstates the risk.
    """
    return sum(bs.prob(x) * bs.prob(y) * (d.value + x.value + y.value)
               for x, y in melds_containing(d))


def position_cost(hand: List[Card], bs, alpha: float,
                  forbidden: Optional[Card] = None) -> Tuple[float, Optional[Card]]:
    """
    Best discard score reachable from an 11-card hand: min_d D(d)+alpha*R(d).

    `forbidden` excludes a card from being discarded (e.g. the gin rule that
    you may not re-discard the card you just drew face-up).

    Returns (cost, discard).
    """
    best: Tuple[float, Optional[Card]] = (float('inf'), None)
    for d in hand:
        if forbidden is not None and d == forbidden:
            continue
        c = deadwood_after_discard(d, hand) + alpha * risk_term(d, bs)
        if c < best[0]:
            best = (c, d)
    return best
