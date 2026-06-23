from dataclasses import dataclass
from typing import List, Tuple
from itertools import combinations

SUITS = ['H', 'D', 'C', 'S']
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
RANK_VALUES = {r: min(i + 1, 10) for i, r in enumerate(RANKS)}
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    def __str__(self):
        return f"{self.rank}{self.suit}"

    @property
    def value(self):
        return RANK_VALUES[self.rank]

    @property
    def rank_idx(self):
        return RANK_INDEX[self.rank]


def make_deck() -> List[Card]:
    return [Card(r, s) for s in SUITS for r in RANKS]


def is_set(cards: List[Card]) -> bool:
    if len(cards) < 3:
        return False
    return len({c.rank for c in cards}) == 1 and len({c.suit for c in cards}) == len(cards)


def is_run(cards: List[Card]) -> bool:
    if len(cards) < 3:
        return False
    if len({c.suit for c in cards}) != 1:
        return False
    idxs = sorted(c.rank_idx for c in cards)
    return idxs == list(range(idxs[0], idxs[0] + len(idxs)))


def deadwood_value(cards: List[Card]) -> int:
    return sum(c.value for c in cards)


def find_best_melds(hand: List[Card]) -> Tuple[List[List[Card]], List[Card]]:
    """Return (melds, deadwood_cards) minimising deadwood value via exhaustive search."""
    best = {'dw': deadwood_value(hand), 'melds': [], 'remaining': list(hand)}

    def search(remaining: List[Card], melds: List[List[Card]]):
        dw = deadwood_value(remaining)
        if dw < best['dw']:
            best['dw'] = dw
            best['melds'] = [m[:] for m in melds]
            best['remaining'] = remaining[:]

        for size in range(3, len(remaining) + 1):
            for combo in combinations(range(len(remaining)), size):
                cards = [remaining[i] for i in combo]
                if is_set(cards) or is_run(cards):
                    rest = [remaining[i] for i in range(len(remaining)) if i not in combo]
                    melds.append(cards)
                    search(rest, melds)
                    melds.pop()

    search(list(hand), [])
    return best['melds'], best['remaining']
