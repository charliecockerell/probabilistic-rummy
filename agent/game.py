from typing import List, Optional
import random

from agent.cards import Card, make_deck, is_set, is_run, deadwood_value, find_best_melds


class GameState:
    def __init__(self):
        self.scores = [0, 0]
        self.reset()

    def reset(self):
        deck = make_deck()
        random.shuffle(deck)
        self.hands: List[List[Card]] = [deck[:10], deck[10:20]]
        self.stock: List[Card] = deck[20:-1]
        self.discard_pile: List[Card] = [deck[-1]]
        self.current_player: int = 0
        self.phase: str = 'draw'
        self.drawn_card: Optional[Card] = None
        self.game_over: bool = False
        self.winner: Optional[int] = None
        self.knock_info: Optional[dict] = None
        self.turn_count: int = 0
        self.message: str = "Player 1's turn — draw a card"

    def draw(self, source: str) -> dict:
        if self.game_over:
            return {'ok': False, 'error': 'Game is over'}
        if self.phase != 'draw':
            return {'ok': False, 'error': 'Not time to draw'}

        p = self.current_player
        if source == 'discard':
            if not self.discard_pile:
                return {'ok': False, 'error': 'Discard pile is empty'}
            card = self.discard_pile.pop()
        elif source == 'stock':
            if not self.stock:
                if len(self.discard_pile) <= 1:
                    self._declare_draw()
                    return {'ok': True}
                top = self.discard_pile.pop()
                self.stock = self.discard_pile[:]
                random.shuffle(self.stock)
                self.discard_pile = [top]
            card = self.stock.pop()
        else:
            return {'ok': False, 'error': 'Invalid source'}

        self.hands[p].append(card)
        self.drawn_card = card
        self.phase = 'discard'
        src = 'the discard' if source == 'discard' else 'the stock'
        self.message = f"Player {p + 1} drew from {src} — select a card to discard or knock"
        return {'ok': True}

    def discard(self, card_str: str) -> dict:
        if self.game_over:
            return {'ok': False, 'error': 'Game is over'}
        if self.phase != 'discard':
            return {'ok': False, 'error': 'Draw a card first'}

        p = self.current_player
        card = self._find_in_hand(p, card_str)
        if card is None:
            return {'ok': False, 'error': f'{card_str} not in hand'}

        self.hands[p].remove(card)
        self.discard_pile.append(card)
        self.drawn_card = None
        self.turn_count += 1
        self.current_player = 1 - p
        self.phase = 'draw'
        self.message = f"Player {self.current_player + 1}'s turn — draw a card"
        return {'ok': True}

    def knock(self, card_str: str) -> dict:
        if self.game_over:
            return {'ok': False, 'error': 'Game is over'}
        if self.phase != 'discard':
            return {'ok': False, 'error': 'Draw a card first'}

        p = self.current_player
        card = self._find_in_hand(p, card_str)
        if card is None:
            return {'ok': False, 'error': f'{card_str} not in hand'}

        test_hand = [c for c in self.hands[p] if c is not card]
        _, dw_cards = find_best_melds(test_hand)
        dw = deadwood_value(dw_cards)
        if dw > 10:
            return {'ok': False, 'error': f'Deadwood is {dw} — need ≤ 10 to knock'}

        self.hands[p].remove(card)
        self.discard_pile.append(card)
        self.drawn_card = None
        self._resolve_knock(p)
        return {'ok': True}

    def _resolve_knock(self, knocker: int):
        opp = 1 - knocker
        k_melds, k_dw = find_best_melds(self.hands[knocker])
        k_dw_val = deadwood_value(k_dw)
        is_gin = k_dw_val == 0

        o_melds, o_dw = find_best_melds(self.hands[opp])

        if not is_gin:
            remaining = list(o_dw)
            changed = True
            while changed:
                changed = False
                for card in remaining[:]:
                    for meld in k_melds:
                        extended = meld + [card]
                        if is_set(extended) or is_run(extended):
                            meld.append(card)
                            remaining.remove(card)
                            changed = True
                            break
            o_dw = remaining

        o_dw_val = deadwood_value(o_dw)

        if is_gin:
            score = o_dw_val + 25
            self.scores[knocker] += score
            self.winner = knocker
            result = 'gin'
        elif o_dw_val <= k_dw_val:
            score = (k_dw_val - o_dw_val) + 25
            self.scores[opp] += score
            self.winner = opp
            result = 'undercut'
        else:
            score = o_dw_val - k_dw_val
            self.scores[knocker] += score
            self.winner = knocker
            result = 'knock'

        self.game_over = True
        self.knock_info = {
            'knocker': knocker,
            'knocker_melds': [[str(c) for c in m] for m in k_melds],
            'knocker_deadwood': [str(c) for c in k_dw],
            'knocker_dw_value': k_dw_val,
            'opponent_melds': [[str(c) for c in m] for m in o_melds],
            'opponent_deadwood': [str(c) for c in o_dw],
            'opponent_dw_value': o_dw_val,
            'result_type': result,
            'score': score,
        }

        w = f"Player {self.winner + 1}"
        if result == 'gin':
            self.message = f"GIN! {w} wins {score} points!"
        elif result == 'undercut':
            self.message = f"UNDERCUT! Player {opp + 1} undercuts — {w} wins {score} points!"
        else:
            self.message = f"Player {knocker + 1} knocks — {w} wins {score} points!"

    def _declare_draw(self):
        self.game_over = True
        self.winner = None
        self.message = "No cards left — hand is a draw!"

    def _find_in_hand(self, player: int, card_str: str) -> Optional[Card]:
        for c in self.hands[player]:
            if str(c) == card_str:
                return c
        return None

    def to_dict(self) -> dict:
        return {
            'current_player': self.current_player,
            'phase': self.phase,
            'hands': [[str(c) for c in h] for h in self.hands],
            'discard_top': str(self.discard_pile[-1]) if self.discard_pile else None,
            'discard_count': len(self.discard_pile),
            'stock_count': len(self.stock),
            'game_over': self.game_over,
            'winner': self.winner,
            'scores': self.scores,
            'message': self.message,
            'knock_info': self.knock_info,
            'turn_count': self.turn_count,
            'drawn_card': str(self.drawn_card) if self.drawn_card else None,
        }
