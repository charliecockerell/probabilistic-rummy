"""
Unit tests for agent.policy.

  Discard
    - alpha=0 never sacrifices deadwood (greedy-optimal keep)
    - never discards a card that completes one of our own melds when a
      strictly-cheaper discard exists
    - R(d) = 0 when the opponent provably holds no partners

  Draw
    - value_take(alpha=0) < current deadwood reduces to sim.py's greedy
      _should_draw_discard

  Knock
    - can_knock matches the deadwood<=10 rule
    - knock_distribution returns a well-formed EV and a probability undercut
    - a gin hand (k=0) is strongly +EV
    - under a blind belief, should_knock fires exactly on legal hands
    - a near-gin opponent raises the undercut probability vs a blind belief
"""

import random
import numpy as np
import pytest

from agent.cards import Card, make_deck, find_best_melds
from agent.inference import BeliefState
from agent.eval.sim import _should_draw_discard
from agent.policy import (
    ProbabilisticPolicy,
    best_discard, discard_score, risk_term, deadwood_after_discard,
    value_take, hand_deadwood,
    knock_discard, can_knock, knock_distribution, should_knock,
)


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------

def test_discard_alpha0_never_sacrifices_deadwood():
    """At alpha=0 the chosen discard must achieve the minimum possible D."""
    random.seed(1)
    for _ in range(200):
        deck = make_deck(); random.shuffle(deck)
        hand, face_up = deck[:11], deck[11]
        bs = BeliefState(list(hand), face_up)
        d = best_discard(hand, bs, alpha=0.0)
        min_D = min(deadwood_after_discard(c, hand) for c in hand)
        assert deadwood_after_discard(d, hand) == min_D


def test_discard_does_not_break_own_meld_for_free():
    """
    With a clear meld and obvious deadwood, the discard should be a deadwood
    card.
    """
    hand = [Card('3','H'), Card('4','H'), Card('5','H'),
            Card('7','D'), Card('8','D'), Card('9','D'),
            Card('2','C'), Card('2','H'),
            Card('K','S'), Card('Q','D'), Card('6','S')]
    bs = BeliefState(list(hand), Card('A','C'))   # no opponent signal
    meld_cards = {c for m in find_best_melds(hand)[0] for c in m}
    d = best_discard(hand, bs, alpha=0.1)
    assert d not in meld_cards


def test_risk_zero_when_no_partners_available():
    """If every partner of d is in our own hand (P=0), R(d) must be 0."""
    # Hold all of 6C's partners: set (6H,6D,6S) and run neighbours (4C,5C,7C,8C).
    hand = [Card('6','C'), Card('6','H'), Card('6','D'), Card('6','S'),
            Card('4','C'), Card('5','C'), Card('7','C'), Card('8','C'),
            Card('K','S'), Card('Q','D'), Card('J','H')]
    bs = BeliefState(list(hand), Card('A','C'))
    # Every set/run partner of 6C is dead (in our hand), so risk is exactly 0.
    assert risk_term(Card('6','C'), bs) == pytest.approx(0.0)


def test_discard_score_decomposition():
    """score == -(D + alpha*R) by construction."""
    hand = [Card('3','H'), Card('4','H'), Card('5','H'),
            Card('7','D'), Card('8','D'), Card('9','D'),
            Card('2','C'), Card('2','H'), Card('K','S'), Card('Q','D'), Card('6','S')]
    bs = BeliefState(list(hand), Card('A','C'))
    bs.observe_opponent_draw_discard(Card('J','S'))   # opp shows a spade interest
    for d in hand:
        s, D, R = discard_score(d, hand, bs, alpha=0.3)
        assert s == pytest.approx(-(D + 0.3 * R))


# ---------------------------------------------------------------------------
# Draw
# ---------------------------------------------------------------------------

def test_draw_reduces_to_greedy_at_alpha0():
    """value_take(alpha=0) < current deadwood must match sim.py's greedy rule."""
    random.seed(5)
    N = 300
    agree = 0
    for _ in range(N):
        deck = make_deck(); random.shuffle(deck)
        h, fu = deck[:10], deck[10]
        bs = BeliefState(list(h), fu)
        ours = value_take(h, fu, bs, alpha=0.0) < hand_deadwood(h)
        if ours == _should_draw_discard(h, fu):
            agree += 1
    assert agree == N


# ---------------------------------------------------------------------------
# Knock
# ---------------------------------------------------------------------------

def _legal_hand():
    """11 cards: two runs + a set + low filler; clearly knockable."""
    return [Card('7','H'), Card('8','H'), Card('9','H'),
            Card('5','C'), Card('5','D'), Card('5','S'),
            Card('A','C'), Card('2','D'), Card('3','S'), Card('4','H'),
            Card('K','S')]


def test_can_knock_matches_deadwood_rule():
    hand = _legal_hand()
    k, discard, _ = knock_discard(hand)
    assert k <= 10 and can_knock(hand)
    assert discard == Card('K','S')      # dropping the king leaves the lowest deadwood

    # A hand with no melds and high cards cannot knock.
    bad = [Card('K','S'), Card('Q','D'), Card('9','C'), Card('7','H'), Card('5','S'),
           Card('J','H'), Card('K','D'), Card('Q','C'), Card('9','S'), Card('7','D'),
           Card('K','H')]
    assert not can_knock(bad)


def test_knock_distribution_wellformed():
    hand = _legal_hand()
    own = [c for c in hand if c is not hand[-1]]
    bs = BeliefState(own, Card('K','S'))
    r = knock_distribution(hand, bs, n_samples=1500, rng=np.random.default_rng(0))
    assert r is not None
    assert r['k'] == 10
    assert 0.0 <= r['undercut'] <= 1.0
    assert len(r['ostars']) == 1500
    # Illegal hand returns None.
    bad = [Card('K','S'), Card('Q','D'), Card('9','C'), Card('7','H'), Card('5','S'),
           Card('J','H'), Card('K','D'), Card('Q','C'), Card('9','S'), Card('7','D'),
           Card('K','H')]
    assert knock_distribution(bad, bs, n_samples=100, rng=np.random.default_rng(0)) is None


def test_gin_hand_is_strongly_positive():
    """A k=0 (gin) hand: EV must include the 25 bonus and be large."""
    # A 4-run + a 3-run + a 3-set covers all 10 kept cards; discard the filler -> gin.
    hand = [Card('7','H'), Card('8','H'), Card('9','H'), Card('10','H'),
            Card('4','C'), Card('5','C'), Card('6','C'),
            Card('2','D'), Card('2','H'), Card('2','S'),
            Card('K','S')]
    k, _, _ = knock_discard(hand)
    own = [c for c in hand if c is not hand[-1]]
    bs = BeliefState(own, Card('Q','S'))
    r = knock_distribution(hand, bs, n_samples=1000, rng=np.random.default_rng(0))
    assert k == 0
    assert r['ev'] > 25.0          # gin bonus plus opponent deadwood
    assert r['undercut'] == 0.0    # gin can never be undercut


def test_knock_reduces_to_first_legal_under_blind_belief():
    """Blind belief + kappa=0 -> knock exactly when the hand is legal."""
    set_melds = [[Card(r, s) for s in tri]
                 for r in [c.rank for c in make_deck()[:13]]
                 for tri in [('H','D','C'), ('H','D','S'), ('H','C','S'), ('D','C','S')]]
    run_melds = []
    deck = make_deck()
    from agent.cards import RANKS, SUITS
    run_melds = [[Card(RANKS[i+j], s) for j in range(3)]
                 for s in SUITS for i in range(len(RANKS) - 2)]
    all_melds = set_melds + run_melds

    rng = np.random.default_rng(3)
    n = 40
    for _ in range(n):
        order = rng.permutation(len(all_melds))
        chosen, used = [], set()
        for mi in order:
            m = all_melds[mi]
            if used.isdisjoint(m):
                chosen.append(m); used.update(m)
            if len(chosen) == 3:
                break
        fillers = [c for c in deck if c not in used]
        rng.shuffle(fillers)
        hand = [c for mm in chosen for c in mm] + fillers[:2]   # 11 cards
        assert can_knock(hand)
        bs = BeliefState(hand[:10], hand[10])
        assert should_knock(hand, bs, kappa=0.0, n_samples=120, rng=rng)


def test_near_gin_opponent_raises_undercut():
    """A belief that the opponent is collecting low runs lifts undercut risk."""
    hand = _legal_hand()
    own = [c for c in hand if c is not hand[-1]]

    blind = BeliefState(own, Card('K','S'))
    r_blind = knock_distribution(hand, blind, n_samples=2000, rng=np.random.default_rng(1))

    near = BeliefState(own, Card('K','S'))
    for c in [Card('2','D'), Card('3','D'), Card('4','D'),
              Card('2','C'), Card('3','C'), Card('4','C')]:
        near.observe_opponent_draw_discard_bayesian(c, nu=4.0)
        near.observe_opponent_discard(Card('Q','H'))
    r_near = knock_distribution(hand, near, n_samples=2000, rng=np.random.default_rng(2))

    assert r_near['undercut'] > r_blind['undercut']
    assert r_near['ev'] < r_blind['ev']


# ---------------------------------------------------------------------------
# Policy facade
# ---------------------------------------------------------------------------

def test_policy_facade_dispatches():
    policy = ProbabilisticPolicy(alpha=0.1, gamma=0.0, kappa=0.0,
                                 knock_samples=500, seed=0)
    hand11 = _legal_hand()
    own = [c for c in hand11 if c is not hand11[-1]]
    bs = BeliefState(own, Card('K','S'))

    assert policy.choose_draw(own, bs, Card('6','H')) in ("discard", "stock")
    assert policy.choose_discard(hand11, bs) in hand11
    assert isinstance(policy.should_knock(hand11, bs), bool)
    assert policy.knock_ev(hand11, bs)['k'] == 10
