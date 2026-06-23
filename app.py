import os
from flask import Flask, jsonify, request, send_from_directory
from agent.game import GameState

STATIC = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, static_folder=STATIC, static_url_path='')

_state = GameState()


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/state')
def get_state():
    return jsonify(_state.to_dict())


@app.route('/api/new_game', methods=['POST'])
def new_game():
    _state.reset()
    return jsonify(_state.to_dict())


@app.route('/api/draw', methods=['POST'])
def draw():
    data = request.get_json(force=True)
    result = _state.draw(data.get('source', 'stock'))
    return jsonify({**result, **_state.to_dict()})


@app.route('/api/discard', methods=['POST'])
def discard():
    data = request.get_json(force=True)
    result = _state.discard(data.get('card', ''))
    return jsonify({**result, **_state.to_dict()})


@app.route('/api/knock', methods=['POST'])
def knock():
    data = request.get_json(force=True)
    result = _state.knock(data.get('card', ''))
    return jsonify({**result, **_state.to_dict()})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
