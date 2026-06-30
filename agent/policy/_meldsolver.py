"""
Fast optimal meld solver for the policy layer.

`cards.find_best_melds` is exhaustive: for every `remaining` set it enumerates
*all* index-combinations of every size 3..n and tests each for set/run-ness.
Profiling probabilistic self-play shows that single function dominates runtime
(the knock Monte-Carlo calls it once per sampled opponent hand, and those hands
are mostly unique so the lru_cache misses). At ~25M is_set/is_run calls for a
handful of games it throttles training-game generation.

This computes the same minimum deadwood far more cheaply:
  1. enumerate only the *valid* melds present in the hand (sets by rank, runs by
     suit) — a handful, not every subset;
  2. pick the disjoint subset maximising melded value (= minimising deadwood) by
     branch-and-bound over those candidates.

`cards.py` stays untouched (the engine keeps the authoritative exhaustive version
for end-of-hand scoring); only the policy layer routes through here, via
`_meldcache`. Differential-tested for deadwood-total equality against the
exhaustive solver. The exact meld *decomposition* among equally-optimal options
may differ, but it is always valid and optimal, which is all callers rely on.
"""

from __future__ import annotations
from itertools import combinations
from typing import List, Tuple

from agent.cards import Card


def _candidate_melds(cards: List[Card]) -> List[Tuple[Card, ...]]:
    """Every valid set and run (length >= 3) wholly within `cards`."""
    melds: List[Tuple[Card, ...]] = []

    by_rank: dict = {}
    for c in cards:
        by_rank.setdefault(c.rank, []).append(c)
    for group in by_rank.values():
        if len(group) >= 3:
            melds.extend(combinations(group, 3))
            if len(group) >= 4:
                melds.extend(combinations(group, 4))

    by_suit: dict = {}
    for c in cards:
        by_suit.setdefault(c.suit, []).append(c)
    for group in by_suit.values():
        group.sort(key=lambda c: c.rank_idx)
        n = len(group)
        for i in range(n):
            for j in range(i + 2, n):                       # length j - i + 1 >= 3
                seq = group[i:j + 1]
                if seq[-1].rank_idx - seq[0].rank_idx == j - i:   # contiguous
                    melds.append(tuple(seq))
                else:
                    break                                   # gap: no longer run from i
    return melds


def fast_best_melds(hand: List[Card]) -> Tuple[List[List[Card]], List[Card]]:
    """Return (melds, deadwood_cards) minimising deadwood value. Drop-in for
    cards.find_best_melds; same minimum deadwood, a valid optimal decomposition."""
    cards = list(hand)
    melds = _candidate_melds(cards)
    if not melds:
        return [], cards

    pos = {c: i for i, c in enumerate(cards)}
    masks, vals = [], []
    for m in melds:
        mask = 0
        v = 0
        for c in m:
            mask |= 1 << pos[c]
            v += c.value
        masks.append(mask)
        vals.append(v)

    n_melds = len(masks)
    # Suffix bound: max melded value still reachable from index k onward (loose,
    # ignores disjointness) — prunes branches that can't beat the best so far.
    suffix = [0] * (n_melds + 1)
    for k in range(n_melds - 1, -1, -1):
        suffix[k] = suffix[k + 1] + vals[k]

    best = {"val": 0, "melds": []}

    def dfs(start: int, used: int, val: int, chosen: list) -> None:
        if val > best["val"]:
            best["val"] = val
            best["melds"] = chosen[:]
        for k in range(start, n_melds):
            if val + suffix[k] <= best["val"]:
                break                                       # can't improve
            if masks[k] & used == 0:
                chosen.append(k)
                dfs(k + 1, used | masks[k], val + vals[k], chosen)
                chosen.pop()

    dfs(0, 0, 0, [])

    chosen_masks = 0
    out_melds = []
    for k in best["melds"]:
        out_melds.append(list(melds[k]))
        chosen_masks |= masks[k]
    remaining = [c for i, c in enumerate(cards) if not (chosen_masks >> i) & 1]
    return out_melds, remaining
