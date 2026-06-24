# probabilistic-rummy

A gin rummy engine and a probabilistic agent that maintains a Bayesian belief
state over the opponent's hand; built alongside a statistical
evaluation harness so every design choice is measured.

Development choices and maths documented in `notebooks/`.

## Architecture

```
agent/cards.py, agent/game.py   game engine (complete)
agent/inference/                belief state P(card in opponent hand)
agent/policy/                   action selection given belief + own hand
agent/eval/                     simulation, opponents, metrics
```

## Key findings

- **The inference works** (calibration above prior; reads a meld-builder better).
- **The baseline ties `greedy`** — win ~0.515 over 2000 games (p=0.17). Greedy is
  strong on fundamentals and belief-illegible, so the soft inference the policy
  leans on has little to exploit.
- **Parameters confirmed**: `alpha` is opponent-dependent and
  win-neutral vs greedy; `kappa` is inert then harmful; `gamma` (info-leak
  penalty) is a confirmed no-op.

Chosen parameters: `alpha=0.1, gamma=0.0, kappa=0.0, mu=0.4, nu=2.0, lam=0.4`.

## Running it

```bash
pip install -r requirements.txt

python app.py                       # interactive UI / hot-seat, http://localhost:5001
pytest                              # inference tests
```

## Next

A search policy (determinized expectimax, then information-set MCTS) over the
validated belief.