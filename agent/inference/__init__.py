"""
Belief state over the opponent's hand in gin rummy.

At every point in the game we maintain p[card] = P(card ∈ opponent_hand)
for all 52 cards, updated after each observable event.

Invariant: sum(p.values()) == opponent_hand_size (always 10, except
transiently 11 between a stock draw and the subsequent discard).
"""

from __future__ import annotations
from typing import Dict, List, Set
from agent.cards import Card, make_deck, RANKS, RANK_INDEX


class BeliefState:
    def __init__(self, own_hand: List[Card], face_up_discard: Card):
        """
        Initialise beliefs at the start of a hand.

        own_hand        : the 10 cards you were dealt
        face_up_discard : the card placed face-up to start the discard pile

        Dead cards (own hand + face-up discard) get P = 0.
        All other 41 cards share the uniform prior 10/41.
        """
        all_cards = make_deck()

        dead = set(own_hand) | {face_up_discard}
        unknown_count = 52 - len(dead)   # 41
        prior = 10 / unknown_count        # 10/41

        self._p: Dict[Card, float] = {}
        for card in all_cards:
            self._p[card] = 0.0 if card in dead else prior

        self._own_hand: set = set(own_hand)
        self._discard_pile: List[Card] = [face_up_discard]

        # How many cards have left the stock so far (both players' stock draws).
        # Stock starts at 31: 52 - 10 (us) - 10 (them) - 1 (face-up discard).
        self._stock_draws: int = 0

    # ------------------------------------------------------------------
    # Public query
    # ------------------------------------------------------------------

    def prob(self, card: Card) -> float:
        """P(card ∈ opponent_hand)."""
        return self._p[card]

    def beliefs(self) -> Dict[Card, float]:
        """Full probability table (copy)."""
        return dict(self._p)

    @property
    def stock_size(self) -> int:
        """Current number of cards remaining in the stock."""
        return 31 - self._stock_draws

    @property
    def hand_size_belief(self) -> float:
        """Sum of all P values — should always equal 10."""
        return sum(self._p.values())

    # ------------------------------------------------------------------
    # Observation methods
    # ------------------------------------------------------------------

    def observe_opponent_draw_discard(self, card: Card) -> None:
        """
        Opponent drew `card` from the top of the discard pile.
        We see exactly which card it is: P = 1.
        """
        self._p[card] = 1.0
        self._remove_from_discard(card)

    def observe_opponent_discard(self, card: Card) -> None:
        """
        Opponent discarded `card` (use after observe_opponent_draw_discard,
        or after observe_opponent_draw_stock).
        Sets P = 0 and adds card to discard pile.
        """
        self._p[card] = 0.0
        self._discard_pile.append(card)
        self._renormalise(10)

    def observe_opponent_draw_stock(self) -> None:
        """
        Opponent drew from the stock (card unknown).
        Only update cards that are actually capable of being in the stock pile.
        """
        S = self.stock_size
        if S <= 0:
            return
            
        # Cards in our hand or already visible in discard pile cannot be drawn
        known_dead = self._own_hand | set(self._discard_pile)

        for card in self._p:
            if card in known_dead:
                continue
            # Apply the update rule to truly hidden cards
            self._p[card] = self._p[card] + (1.0 - self._p[card]) / S
            
        self._stock_draws += 1


    def observe_stock_draw_then_discard(self, discarded: Card) -> None:
        """
        Opponent drew from stock then discarded `discarded`.
        """
        self.observe_opponent_draw_stock()   # sum → 11
        self._p[discarded] = 0.0
        self._discard_pile.append(discarded)
        self._renormalise(10)                # scale remaining unknowns back to sum = 10
                
    def observe_own_draw_stock(self, card: Card) -> None:
        """You drew `card` from the stock — it's in your hand, P = 0.
        Renormalise because we've learned this card was not in the opponent's hand."""
        self._p[card] = 0.0
        self._own_hand.add(card)
        self._stock_draws += 1
        self._renormalise(10.0)

    def observe_own_draw_discard(self, card: Card) -> None:
        """You drew `card` from the discard pile — P = 0."""
        self._p[card] = 0.0
        self._own_hand.add(card)
        self._remove_from_discard(card)

    def observe_own_discard(self, card: Card) -> None:
        """You discarded `card` — P was already 0, now on discard pile."""
        self._p[card] = 0.0
        self._own_hand.discard(card)
        self._discard_pile.append(card)

    # ------------------------------------------------------------------
    # Bayesian signal methods
    # ------------------------------------------------------------------

    def observe_opponent_pass_discard(self, card: Card, mu: float = 0.4) -> None:
        """
        Opponent saw `card` face-up on the discard pile and chose NOT to take it.
        Signal: `card` is not useful to them → downweight its meld partners.

        mu : downweight factor in (0, 1). Smaller = stronger signal.
        """
        self._downweight_meld_partners(card, mu)
        self._renormalise(10.0)

    def observe_stock_draw_then_discard_bayesian(
        self, discarded: Card, lambda_: float = 0.4
    ) -> None:
        """
        Opponent drew from stock then discarded `discarded` — with Bayesian signal.
        Signal: `discarded` was not useful → downweight its meld partners.

        lambda_ : downweight factor in (0, 1). Smaller = stronger signal.
        """
        self.observe_opponent_draw_stock()   # sum → 11
        self._p[discarded] = 0.0
        self._discard_pile.append(discarded)
        self._downweight_meld_partners(discarded, lambda_)
        self._renormalise(10.0)

    def observe_opponent_discard_bayesian(
        self, discarded: Card, lambda_: float = 0.4
    ) -> None:
        """
        Opponent discarded `discarded` after drawing from the discard pile
        (i.e. no stock-draw step). Applies the discard signal only.

        Use this after observe_opponent_draw_discard_bayesian() to complete the turn.
        """
        self._p[discarded] = 0.0
        self._discard_pile.append(discarded)
        self._downweight_meld_partners(discarded, lambda_)
        self._renormalise(10.0)

    def observe_opponent_draw_discard_bayesian(
        self, card: Card, nu: float = 2.0
    ) -> None:
        """
        Opponent picked up `card` from the discard pile — with Bayesian signal.
        Signal: `card` IS useful to them → upweight its meld partners.
        P(card) is set to 1 (certain); meld partners get scaled up by nu then
        the whole distribution is renormalised to restore the invariant.

        nu : upweight factor > 1. Larger = stronger signal.
        """
        self._p[card] = 1.0
        self._remove_from_discard(card)
        self._upweight_meld_partners(card, nu)
        # _renormalise correctly preserves all P=1 certainties (including this
        # card and any previously certain cards) and scales only uncertain cards.
        self._renormalise(10.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meld_partners(self, card: Card) -> Set[Card]:
        """
        Return the set of cards that could form a meld with `card`.

        Set partners : all other cards of the same rank.
        Run partners : cards of the same suit within 2 rank steps in either
                       direction (any two of them can form a 3-card run with card).
        """
        partners: Set[Card] = set()
        ri = RANK_INDEX[card.rank]

        for c in self._p:
            if c == card:
                continue
            # Set partner: same rank, different suit
            if c.rank == card.rank:
                partners.add(c)
                continue
            # Run partner: same suit, rank within 2 steps
            if c.suit == card.suit and abs(RANK_INDEX[c.rank] - ri) <= 2:
                partners.add(c)

        return partners

    def _downweight_meld_partners(self, card: Card, factor: float) -> None:
        """Multiply each meld partner's probability by `factor`.
        Skips cards with P=0 (dead) or P=1 (certain) — hard observations are immutable."""
        for partner in self._meld_partners(card):
            p = self._p[partner]
            if p == 0.0 or p == 1.0:
                continue
            self._p[partner] = p * factor

    def _upweight_meld_partners(self, card: Card, factor: float) -> None:
        """Multiply each meld partner's probability by `factor`, capped at 1.
        Skips cards with P=0 (dead) or P=1 (certain) — hard observations are immutable."""
        for partner in self._meld_partners(card):
            p = self._p[partner]
            if p == 0.0 or p == 1.0:
                continue
            self._p[partner] = min(1.0, p * factor)

    def _renormalise(self, target: float) -> None:
        """Scale uncertain cards (0 < P < 1) so the full distribution sums to `target`.
        Cards with P=0 (dead) and P=1 (certain) are never touched — they are hard
        observations and must remain exact."""
        certain_mass = sum(p for p in self._p.values() if p == 1.0)
        uncertain_total = sum(p for p in self._p.values() if 0.0 < p < 1.0)
        target_uncertain = target - certain_mass
        if uncertain_total == 0 or target_uncertain <= 0:
            return
        scale = target_uncertain / uncertain_total
        for card in self._p:
            if 0.0 < self._p[card] < 1.0:
                self._p[card] *= scale

    def _remove_from_discard(self, card: Card) -> None:
        try:
            self._discard_pile.remove(card)
        except ValueError:
            pass
