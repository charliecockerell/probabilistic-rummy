"""
Random-play game simulator.

Generates a sequence of observable events from player 0's perspective
by simulating a gin rummy game where both players act randomly.

Usage
-----
    from agent.eval.sim import simulate_game, Event

    events, game = simulate_game(n_turns=20, seed=42)
    for e in events:
        print(e)

Event types
-----------
    opp_pass_discard   card  opponent saw top-of-discard and drew from stock instead
    opp_draw_discard   card  opponent picked up the top-of-discard card
    opp_draw_stock     —     opponent drew from stock (card unknown to us)
    opp_discard        card  opponent discarded this card
    own_draw_stock     card  we drew this card from stock
    own_draw_discard   card  we drew this card from discard
    own_discard        card  we discarded this card
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional

from agent.cards import Card, make_deck, find_best_melds
from agent.game import GameState


@dataclass
class Event:
    type: str
    card: Optional[Card] = None

    def __repr__(self):
        return f"Event({self.type!r}, {self.card})"


@dataclass
class SimResult:
    events: List[Event]
    own_hand: List[Card]          # final hand at end of game
    opp_hand: List[Card]          # final hand — ground truth, not visible during play
    face_up: Card
    n_turns: int                  # actual turns played (may be < requested if game ends)
    game_over: bool
    starting_own_hand: List[Card] # hand at deal time — use this to initialise BeliefState
    starting_opp_hand: List[Card] # opponent's hand at deal time


def _random_discard(hand: List[Card]) -> Card:
    """Discard the card that leaves the least deadwood (greedy). Falls back to random."""
    best_card = None
    best_dw = float('inf')
    for card in hand:
        remaining = [c for c in hand if c != card]
        _, dw_cards = find_best_melds(remaining)
        dw = sum(c.value for c in dw_cards)
        if dw < best_dw:
            best_dw = dw
            best_card = card
    return best_card or random.choice(hand)


def _should_draw_discard(hand: List[Card], top: Card) -> bool:
    """
    Return True if taking `top` from the discard reduces our deadwood.
    Simple greedy check: does adding `top` and discarding our worst card
    improve our position?
    """
    _, current_dw = find_best_melds(hand)
    current_val = sum(c.value for c in current_dw)

    candidate = hand + [top]
    discard = _random_discard(candidate)
    new_hand = [c for c in candidate if c != discard]
    _, new_dw = find_best_melds(new_hand)
    new_val = sum(c.value for c in new_dw)

    return new_val < current_val


def simulate_game(n_turns: int = 20, seed: Optional[int] = None) -> SimResult:
    """
    Simulate up to `n_turns` complete turns (one turn = one player draws + discards).
    Both players use a simple greedy policy: take the discard if it reduces deadwood,
    otherwise draw from stock; then discard the card that minimises remaining deadwood.

    Returns a SimResult containing the event list and ground-truth hand info.

    Parameters
    ----------
    n_turns : int
        Maximum number of half-turns to simulate (each draw+discard = 1 turn).
        Total events will be roughly 3–4 × n_turns.
    seed : int or None
        Random seed for reproducibility.
    """
    if seed is not None:
        random.seed(seed)

    # Deal manually so we can capture ground-truth hands
    deck = make_deck()
    random.shuffle(deck)
    own_hand:  List[Card] = list(deck[:10])
    opp_hand:  List[Card] = list(deck[10:20])
    stock:     List[Card] = list(deck[21:])
    face_up:   Card       = deck[20]
    discard_pile: List[Card] = [face_up]

    # Snapshot the starting hands before the game mutates them
    starting_own_hand: List[Card] = list(own_hand)
    starting_opp_hand: List[Card] = list(opp_hand)

    events: List[Event] = []
    game_over = False

    for turn in range(n_turns):
        # --- Determine whose turn it is (player 0 = us, player 1 = opp) ---
        is_our_turn = (turn % 2 == 0)
        hand = own_hand if is_our_turn else opp_hand

        if not stock and len(discard_pile) <= 1:
            game_over = True
            break

        top = discard_pile[-1]

        # --- Draw phase ---
        if _should_draw_discard(hand, top):
            drawn = discard_pile.pop()
            hand.append(drawn)
            if is_our_turn:
                events.append(Event("own_draw_discard", drawn))
            else:
                events.append(Event("opp_draw_discard", drawn))
        else:
            # Passed on the discard
            if not is_our_turn:
                events.append(Event("opp_pass_discard", top))

            if not stock:
                # Reshuffle discard (keep top)
                new_top = discard_pile.pop()
                stock.extend(discard_pile)
                random.shuffle(stock)
                discard_pile = [new_top]

            drawn = stock.pop()
            hand.append(drawn)
            if is_our_turn:
                events.append(Event("own_draw_stock", drawn))
            else:
                events.append(Event("opp_draw_stock"))   # card hidden from us

        # --- Discard phase ---
        card_to_discard = _random_discard(hand)
        hand.remove(card_to_discard)
        discard_pile.append(card_to_discard)

        if is_our_turn:
            events.append(Event("own_discard", card_to_discard))
        else:
            events.append(Event("opp_discard", card_to_discard))

        # Simple knock check: if deadwood <= 10, end game
        _, dw_cards = find_best_melds(hand)
        if sum(c.value for c in dw_cards) <= 10:
            game_over = True
            break

    return SimResult(
        events=events,
        own_hand=own_hand,
        opp_hand=opp_hand,
        face_up=face_up,
        n_turns=turn + 1,
        game_over=game_over,
        starting_own_hand=starting_own_hand,
        starting_opp_hand=starting_opp_hand,
    )
