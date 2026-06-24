"""
Eval-only baseline opponents.

`GreedyPolicy` (in agent/bot.py) minimises its own deadwood and knocks the
moment it is legal. It is strong on fundamentals but *belief-illegible*: it
does not collect toward melds, so its discards carry little of the meld-intent
signal the BeliefState's soft inference is built to read.

`MeldSeekerPolicy` here is the complementary caricature: it actively builds
melds. It protects any card already in a meld, holds cards with meld potential
(pairs, near-runs) even when they carry deadwood, and discards the most
*isolated* card. Its draws and discards therefore telegraph what it is
collecting — exactly the generative behaviour the inference assumes.

Neither is a rational opponent (a real player both manages deadwood and builds
melds, and mixes the two). They bracket the spectrum: greedy is the
model-mismatched lower bound, the meld-seeker the model-matched best case. Use
them to characterise where the belief helps, not to pick production parameters.
"""

from __future__ import annotations
from typing import List, Optional

from agent.cards import Card, RANK_INDEX, find_best_melds
from agent.policy import can_knock, knock_discard
from agent.eval.sim import _should_draw_discard


def _meld_potential(card: Card, hand: List[Card]) -> int:
    """How meld-connected `card` is within `hand`: same-rank partners (toward a
    set) plus same-suit neighbours within two ranks (toward a run). Higher means
    more worth keeping; isolated cards score 0."""
    ri = RANK_INDEX[card.rank]
    set_partners = sum(1 for c in hand if c is not card and c.rank == card.rank)
    run_partners = sum(1 for c in hand if c is not card and c.suit == card.suit
                       and 0 < abs(RANK_INDEX[c.rank] - ri) <= 2)
    return set_partners + run_partners


class MeldSeekerPolicy:
    """Belief-legible meld builder (ignores the belief itself)."""

    def choose_draw(self, hand: List[Card], bs, discard_top: Optional[Card]) -> str:
        if discard_top is None:
            return "stock"
        melds, _ = find_best_melds(hand + [discard_top])
        completes_meld = any(discard_top in m for m in melds)
        if completes_meld or _meld_potential(discard_top, hand) >= 2:
            return "discard"
        return "stock"

    def choose_discard(self, hand: List[Card], bs, forbidden: Optional[Card] = None) -> Card:
        melds, _ = find_best_melds(hand)
        melded = {c for m in melds for c in m}
        cands = [c for c in hand if c not in melded and c != forbidden]
        if not cands:                                  # whole hand melds; shed anything legal
            cands = [c for c in hand if c != forbidden] or list(hand)
        # Drop the most isolated card; break ties by shedding the most points.
        return min(cands, key=lambda c: (_meld_potential(c, hand), -c.value))

    def should_knock(self, hand: List[Card], bs) -> bool:
        return can_knock(hand)

    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]:
        k, discard, _ = knock_discard(hand)
        return None if k > 10 else {"k": k, "discard": discard}


class RationalOpponent:
    """Between the two caricatures: deadwood-driven like greedy, but reluctant to
    throw away meld potential like the meld-seeker. Draws on the union of both
    rules (take the upcard if it lowers deadwood OR builds a meld); discards to
    minimise resulting deadwood plus `w` times the meld potential it would shed,
    so it sheds isolated high cards and holds developing melds. `w=0` recovers
    greedy; large `w` approaches the meld-seeker. Still belief-illegible -- it
    does not model the opponent, so it won't punish an info leak (gamma needs a
    belief-using opponent, e.g. prob-vs-prob, to bite)."""

    def __init__(self, w: float = 2.0):
        self.w = w

    def choose_draw(self, hand: List[Card], bs, discard_top: Optional[Card]) -> str:
        if discard_top is None:
            return "stock"
        if _should_draw_discard(hand, discard_top):            # greedy: lowers deadwood
            return "discard"
        melds, _ = find_best_melds(hand + [discard_top])       # meld: completes/extends
        if any(discard_top in m for m in melds) or _meld_potential(discard_top, hand) >= 2:
            return "discard"
        return "stock"

    def choose_discard(self, hand: List[Card], bs, forbidden: Optional[Card] = None) -> Card:
        cands = [c for c in hand if c != forbidden] or list(hand)

        def score(c: Card) -> float:
            _, dw = find_best_melds([x for x in hand if x is not c])
            resulting_deadwood = sum(x.value for x in dw)
            return resulting_deadwood + self.w * _meld_potential(c, hand)

        return min(cands, key=score)

    def should_knock(self, hand: List[Card], bs) -> bool:
        return can_knock(hand)

    def knock_ev(self, hand: List[Card], bs) -> Optional[dict]:
        k, discard, _ = knock_discard(hand)
        return None if k > 10 else {"k": k, "discard": discard}
