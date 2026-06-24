import os
from flask import Flask, jsonify, request, send_from_directory

from agent.cards import Card, find_best_melds
from agent.game import GameState
from agent.bot import BotPlayer

STATIC = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, static_folder=STATIC, static_url_path='')

HUMAN_SEAT = 0
BOT_SEAT = 1

# Knock EV is Monte-Carlo per decision; the eval default (high) is too slow for
# interactive play and makes the bot's turn lag for seconds. Keep it modest here
# so the bot responds promptly. (See notebook note: the per-decision sampling is
# the main perf bottleneck and is worth rethinking — caching / analytic approx.)
INTERACTIVE_KNOCK_SAMPLES = 120

_state = GameState()
_bot = None          # BotPlayer or None (None => hot-seat, two humans)
_opponent = 'human'


def _card_from_str(card_str: str) -> Card:
    return Card(card_str[:-1], card_str[-1])


def _own_melds(player: int) -> dict:
    """Best meld/deadwood split for one player's current hand (own-hand view only)."""
    melds, deadwood = find_best_melds(_state.hands[player])
    return {
        'current_melds': [[str(c) for c in m] for m in melds],
        'current_deadwood': [str(c) for c in deadwood],
        'current_deadwood_value': sum(c.value for c in deadwood),
    }


def _payload(extra: dict | None = None) -> dict:
    d = _state.to_dict()
    d['opponent'] = _opponent
    d['opponent_name'] = _bot.name if _bot else 'Player 2'
    d['human_seat'] = HUMAN_SEAT

    # Own-melds view: split the active player's hand into melds vs deadwood so the
    # UI can highlight them. Only ever computed for the current player (no leak of
    # the opponent's hand beyond what the engine already exposes for hot-seat).
    if not _state.game_over:
        d['own_view'] = _own_melds(_state.current_player)

    # Belief histogram: when playing the probabilistic bot, surface its belief
    # P(card in your hand) over all 52 cards so the UI can render a heatmap.
    if _bot is not None:
        d['belief_sum'] = _bot.belief_sum()
        if _bot.bs is not None:
            d['belief'] = {str(c): round(p, 4) for c, p in _bot.bs.beliefs().items()}

    if extra:
        d.update(extra)
    return d


def _run_bot_turns():
    """Let the bot play until it is the human's turn again (or the game ends)."""
    last_msg = None
    while _bot is not None and not _state.game_over and _state.current_player == BOT_SEAT:
        last_msg = _bot.take_turn(_state)
    if last_msg and not _state.game_over:
        _state.message = f"{last_msg}  Your turn — draw a card."


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/state')
def get_state():
    return jsonify(_payload())


@app.route('/api/new_game', methods=['POST'])
def new_game():
    global _bot, _opponent
    data = request.get_json(silent=True) or {}
    _opponent = data.get('opponent', 'human')

    _state.reset()
    if _opponent in ('greedy', 'probabilistic'):
        _bot = BotPlayer(BOT_SEAT, _opponent, seed=data.get('seed'),
                         knock_samples=INTERACTIVE_KNOCK_SAMPLES)
        _bot.start(_state.hands[BOT_SEAT], _state.discard_pile[-1])
    else:
        _bot = None
    return jsonify(_payload())


@app.route('/api/draw', methods=['POST'])
def draw():
    data = request.get_json(force=True)
    source = data.get('source', 'stock')
    top = _state.discard_pile[-1] if _state.discard_pile else None

    result = _state.draw(source)
    if result.get('ok') and _bot is not None:
        _bot.saw_opponent_draw(source, top)
    return jsonify(_payload(result))


@app.route('/api/discard', methods=['POST'])
def discard():
    data = request.get_json(force=True)
    card_str = data.get('card', '')

    result = _state.discard(card_str)
    if result.get('ok') and _bot is not None:
        _bot.saw_opponent_discard(_card_from_str(card_str))
        _run_bot_turns()
    return jsonify(_payload(result))


@app.route('/api/knock', methods=['POST'])
def knock():
    data = request.get_json(force=True)
    result = _state.knock(data.get('card', ''))
    return jsonify(_payload(result))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
