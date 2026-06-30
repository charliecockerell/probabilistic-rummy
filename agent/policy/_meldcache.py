"""
Memoised find_best_melds for the policy layer.

find_best_melds is exhaustive backtracking over <= 11 cards and is the dominant
cost in eval/sweeps: knock EV calls it once per Monte-Carlo opponent sample, and
discard scoring calls it once per candidate every turn. Sampled opponent hands
and mid-turn hands repeat heavily — and every sweep cell replays the same deals —
so a result cache keyed on the hand turns most of those calls into dict lookups.
The cache is exact: identical input, identical output, no approximation.

Lives in the policy layer, not the engine: cards.py stays untouched and only
policy code routes through here. A hand never contains duplicate cards, so the
frozenset key is lossless, and find_best_melds is order-independent.

Callers treat the return value as READ-ONLY (all current ones immediately reduce
it to a deadwood total). Do not mutate the returned lists — they are shared.
"""

from __future__ import annotations
from functools import lru_cache
from typing import FrozenSet, List, Tuple

from agent.cards import Card
from agent.policy._meldsolver import fast_best_melds


@lru_cache(maxsize=None)
def _cached(key: FrozenSet[Card]) -> Tuple[list, list]:
    # fast_best_melds returns the same minimum deadwood as cards.find_best_melds
    # (differential-tested); ~180x cheaper, which is what unblocks fast game gen.
    return fast_best_melds(list(key))


def best_melds(hand: List[Card]) -> Tuple[list, list]:
    """Cached find_best_melds(hand) -> (melds, deadwood_cards). Read-only result."""
    return _cached(frozenset(hand))


def clear_meld_cache() -> None:
    """Drop the cache (e.g. between heavy sweeps to bound memory)."""
    _cached.cache_clear()


def meld_cache_info():
    """functools cache_info() — hits/misses/currsize, for profiling."""
    return _cached.cache_info()
