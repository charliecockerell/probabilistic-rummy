"""
Knock policy.

Knocking is an EV comparison, not a position cost.
We decide while holding 11 cards (drew, about to discard): pick the discard
minimising kept deadwood k; knocking is legal iff k <= 10.

The opponent's deadwood O is modelled as a distribution. 
We sample opponent hands consistent with the belief (Madow systematic sampling),
score each with the real engine rules (find_best_melds + layoff onto our melds), and average.

knock_distribution returns the EV and P(undercut); should_knock thresholds the
EV against kappa, a flat opportunity cost of playing on.
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np

from agent.cards import Card, make_deck, is_set, is_run
from agent.policy._features import deadwood
from agent.policy._meldcache import best_melds

_DECK = make_deck()


def knock_discard(hand: List[Card]) -> Tuple[int, Optional[Card], list]:
    """
    From an 11-card hand, choose the discard minimising kept deadwood.
    Returns (k, discard, our_melds) for the best 10-card keep.
    """
    best: Tuple[int, Optional[Card], list] = (10 ** 9, None, [])
    for d in hand:
        kept = [c for c in hand if c is not d]
        melds, dw = best_melds(kept)
        k = deadwood(dw)
        if k < best[0]:
            best = (k, d, melds)
    return best


def can_knock(hand: List[Card]) -> bool:
    """True iff the 11-card hand can legally knock (best kept deadwood <= 10)."""
    return knock_discard(hand)[0] <= 10


def layoff_reduce(opp_deadwood: List[Card], our_melds: list) -> int:
    """
    Opponent sheds deadwood onto our melds (same fixpoint loop as game.py).
    Returns the opponent's post-layoff deadwood value.
    """
    remaining = list(opp_deadwood)
    melds = [list(m) for m in our_melds]
    changed = True
    while changed:
        changed = False
        for c in remaining[:]:
            for m in melds:
                if is_set(m + [c]) or is_run(m + [c]):
                    m.append(c)
                    remaining.remove(c)
                    changed = True
                    break
            if changed:
                break
    return deadwood(remaining)


def sample_opp_hands(bs, n_samples: int, rng: np.random.Generator) -> List[List[Card]]:
    """
    Sample opponent hands consistent with the belief marginals via Madow
    systematic sampling: exactly the right size every draw, reproducing each
    card's inclusion probability p(c).
    """
    p = np.array([bs.prob(c) for c in _DECK])
    certain = p >= 1 - 1e-9
    uncertain = (p > 1e-12) & (~certain)
    ui = np.where(uncertain)[0]
    pi = p[ui]
    n_slots = int(round(pi.sum()))          # == 10 - #certain
    cert_cards = [_DECK[i] for i in np.where(certain)[0]]
    if n_slots <= 0 or pi.sum() <= 0:
        return [list(cert_cards) for _ in range(n_samples)]
    # Madow systematic sampling requires the inclusion probabilities to sum
    # exactly to the number of slots. Belief marginals can drift off an exact
    # integer (round() above absorbs the drift into n_slots), so rescale pi to
    # sum to n_slots; otherwise the top sample points overrun C and searchsorted
    # returns len(ui), indexing out of bounds.
    pi = pi * (n_slots / pi.sum())
    C = np.cumsum(pi)
    base = np.arange(n_slots)
    starts = rng.random(n_samples)
    last = len(ui) - 1
    hands = []
    for s in range(n_samples):
        sel = np.searchsorted(C, starts[s] + base, side='left')
        np.clip(sel, 0, last, out=sel)     # float-safety at the top boundary
        hands.append(cert_cards + [_DECK[ui[j]] for j in sel])
    return hands


def knock_distribution(hand: List[Card], bs, n_samples: int = 2000,
                       rng: Optional[np.random.Generator] = None) -> Optional[dict]:
    """
    Distribution of the knock outcome from an 11-card hand.

    Returns None if the hand cannot legally knock, else a dict with:
      k         : our kept deadwood
      discard   : the card to discard before knocking
      ev        : expected signed points (our perspective)
      undercut  : P(o* <= k)
      ostars    : np.array of post-layoff opponent deadwood per sample
    """
    if rng is None:
        rng = np.random.default_rng()
    k, discard, our_melds = knock_discard(hand)
    if k > 10:
        return None
    gains, ostars = [], []
    undercut = 0
    for opp in sample_opp_hands(bs, n_samples, rng):
        _, o_dw = best_melds(opp)
        o_raw = deadwood(o_dw)
        if k == 0:
            gain, ostar = o_raw + 25, o_raw          # gin: opponent cannot lay off
        else:
            ostar = layoff_reduce(o_dw, our_melds)
            if ostar > k:
                gain = ostar - k
            else:
                gain = -((k - ostar) + 25)
                undercut += 1
        gains.append(gain)
        ostars.append(ostar)
    gains = np.array(gains)
    return dict(k=k, discard=discard, ev=float(gains.mean()),
                undercut=undercut / n_samples, ostars=np.array(ostars))


def should_knock(hand: List[Card], bs, kappa: float = 0.0,
                 n_samples: int = 2000, rng: Optional[np.random.Generator] = None) -> bool:
    """Knock iff the expected knock value beats the opportunity cost kappa."""
    r = knock_distribution(hand, bs, n_samples, rng)
    return r is not None and r['ev'] > kappa
