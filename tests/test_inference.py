"""
Unit tests for agent.inference.BeliefState.

Tests cover:
  - Initialisation (prior values, dead cards, invariant)
  - Each observation method (correct P updates, invariant preserved)
  - Dead cards remain zero through all update types
  - Multi-turn invariant stability
"""

import random
import pytest
from agent.cards import make_deck, Card
from agent.inference import BeliefState

TOL = 1e-9  # floating-point tolerance for sum checks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_state():
    """Standard starting position with a fixed seed."""
    random.seed(42)
    deck = make_deck()
    random.shuffle(deck)
    own_hand = deck[:10]
    face_up  = deck[20]
    return BeliefState(own_hand, face_up), own_hand, face_up, deck


@pytest.fixture
def unknown_cards(fresh_state):
    bs, own_hand, face_up, deck = fresh_state
    dead = set(own_hand) | {face_up}
    return [c for c in make_deck() if c not in dead]


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_sum_equals_ten(self, fresh_state):
        bs, *_ = fresh_state
        assert abs(bs.hand_size_belief - 10.0) < TOL

    def test_own_hand_cards_are_zero(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        for card in own_hand:
            assert bs.prob(card) == 0.0, f"{card} should have P=0"

    def test_face_up_discard_is_zero(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        assert bs.prob(face_up) == 0.0

    def test_unknown_cards_have_uniform_prior(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        expected = 10 / 41
        for card in unknown_cards:
            assert abs(bs.prob(card) - expected) < TOL, \
                f"{card} prior should be {expected:.6f}, got {bs.prob(card):.6f}"

    def test_stock_size_initial(self, fresh_state):
        bs, *_ = fresh_state
        assert bs.stock_size == 31

    def test_all_probs_non_negative(self, fresh_state):
        bs, *_ = fresh_state
        for card, p in bs.beliefs().items():
            assert p >= 0.0, f"{card} has negative probability {p}"


# ---------------------------------------------------------------------------
# 2. observe_opponent_draw_discard
# ---------------------------------------------------------------------------

class TestDrawDiscard:
    def test_drawn_card_becomes_one(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        bs.observe_opponent_draw_discard(face_up)
        assert bs.prob(face_up) == 1.0

    def test_sum_becomes_eleven_after_draw(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        bs.observe_opponent_draw_discard(face_up)
        assert abs(bs.hand_size_belief - 11.0) < TOL

    def test_sum_returns_to_ten_after_discard(self, fresh_state, unknown_cards):
        bs, own_hand, face_up, deck = fresh_state
        bs.observe_opponent_draw_discard(face_up)
        their_discard = unknown_cards[0]
        bs.observe_opponent_discard(their_discard)
        assert abs(bs.hand_size_belief - 10.0) < TOL

    def test_discarded_card_becomes_zero(self, fresh_state, unknown_cards):
        bs, own_hand, face_up, deck = fresh_state
        bs.observe_opponent_draw_discard(face_up)
        their_discard = unknown_cards[0]
        bs.observe_opponent_discard(their_discard)
        assert bs.prob(their_discard) == 0.0


# ---------------------------------------------------------------------------
# 3. observe_stock_draw_then_discard
# ---------------------------------------------------------------------------

class TestStockDraw:
    def test_sum_preserved_after_stock_draw(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        bs.observe_stock_draw_then_discard(unknown_cards[0])
        assert abs(bs.hand_size_belief - 10.0) < TOL

    def test_discarded_card_is_zero(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        discarded = unknown_cards[0]
        bs.observe_stock_draw_then_discard(discarded)
        assert bs.prob(discarded) == 0.0

    def test_stock_size_decrements(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        bs.observe_stock_draw_then_discard(unknown_cards[0])
        assert bs.stock_size == 30

    def test_unknown_card_probs_increase_before_discard(self, fresh_state, unknown_cards):
        """After a stock draw (before discard), every unknown card's P should rise."""
        bs, own_hand, face_up, deck = fresh_state
        p_before = {c: bs.prob(c) for c in unknown_cards}
        bs.observe_opponent_draw_stock()
        for card in unknown_cards:
            assert bs.prob(card) >= p_before[card] - TOL, \
                f"{card} P decreased after stock draw"

    def test_dead_cards_unaffected_by_stock_draw(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        unknown_non_discard = [c for c in make_deck()
                                if c not in set(own_hand) and c != face_up]
        bs.observe_stock_draw_then_discard(unknown_non_discard[0])
        for card in own_hand:
            assert bs.prob(card) == 0.0
        assert bs.prob(face_up) == 0.0


# ---------------------------------------------------------------------------
# 4. observe_own_draw_stock / observe_own_draw_discard / observe_own_discard
# ---------------------------------------------------------------------------

class TestOwnActions:
    def test_own_stock_draw_zeroes_card(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        drawn = unknown_cards[5]
        bs.observe_own_draw_stock(drawn)
        assert bs.prob(drawn) == 0.0

    def test_own_stock_draw_decrements_stock(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        bs.observe_own_draw_stock(unknown_cards[5])
        assert bs.stock_size == 30

    def test_own_stock_draw_preserves_sum(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        bs.observe_own_draw_stock(unknown_cards[5])
        assert abs(bs.hand_size_belief - 10.0) < TOL

    def test_own_discard_pile_draw_zeroes_card(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        bs.observe_own_draw_discard(face_up)
        assert bs.prob(face_up) == 0.0

    def test_own_discard_zeroes_card(self, fresh_state):
        bs, own_hand, face_up, deck = fresh_state
        card = own_hand[0]
        bs.observe_own_discard(card)
        assert bs.prob(card) == 0.0


# ---------------------------------------------------------------------------
# 5. Multi-turn invariant
# ---------------------------------------------------------------------------

class TestMultiTurn:
    def test_invariant_holds_across_many_turns(self, fresh_state, unknown_cards):
        """Sum should equal 10 after every complete turn."""
        bs, *_ = fresh_state
        for i in range(15):
            bs.observe_stock_draw_then_discard(unknown_cards[i])
            assert abs(bs.hand_size_belief - 10.0) < TOL, \
                f"Invariant broken after turn {i+1}: sum={bs.hand_size_belief}"

    def test_no_negative_probs_across_many_turns(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        for i in range(15):
            bs.observe_stock_draw_then_discard(unknown_cards[i])
            for card, p in bs.beliefs().items():
                assert p >= -TOL, f"{card} went negative (p={p}) at turn {i+1}"

    def test_dead_cards_stay_zero_across_many_turns(self, fresh_state, unknown_cards):
        bs, own_hand, face_up, deck = fresh_state
        for i in range(15):
            bs.observe_stock_draw_then_discard(unknown_cards[i])
        for card in own_hand:
            assert bs.prob(card) == 0.0
        assert bs.prob(face_up) == 0.0

    def test_probs_never_exceed_one(self, fresh_state, unknown_cards):
        bs, *_ = fresh_state
        # Force a certain card then keep playing
        bs.observe_opponent_draw_discard(fresh_state[2])  # face_up → P=1
        for i in range(10):
            bs.observe_stock_draw_then_discard(unknown_cards[i])
        for card, p in bs.beliefs().items():
            assert p <= 1.0 + TOL, f"{card} exceeded 1.0 (p={p})"
