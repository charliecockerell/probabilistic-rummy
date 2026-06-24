# Next model — trade-offs in quant-signal terms

## Where we are
The belief is validated (calibration climbs from the 0.24 prior toward ~0.5, faster against a meld-builder). The current **one-ply EV policy** ties greedy (win ~0.515 at n=2000, p=0.17), beats meld-seekers significantly, and lands ~0.62 vs the mixed `rational` opponent (n small). Knob tuning is exhausted: `alpha` is opponent-dependent and win-neutral vs greedy, `kappa` is inert-then-harmful, `gamma` is untestable without a modelling opponent.

**The signal we're now chasing** is the sequential / strategic value the current *myopic* policy throws away: multi-turn setup, knock-timing option value, and opponent-adaptive defence. Framing the candidates as signals: expected edge (alpha), the variance/sample cost to *measure* that edge, attribution clarity, overfitting/robustness, and build cost.

## Comparison

| Model | Signal source (the alpha) | Edge ceiling | Variance & sample cost to measure | Attribution | Overfit risk | Build cost |
|---|---|---|---|---|---|---|
| Analytic 1-ply fixes | state-dependent `kappa` (option value of playing on), `gamma` via self-play, late-game `alpha` ramp | low (~1–3pp) | low — cheap to run, low estimator variance | high (each knob isolatable) | low | low |
| Determinized expectimax | 1–2 ply sequencing the myopic policy ignores; reuses the belief you already sample | moderate | medium — ~10–50× current per-move cost, but same game-count to detect | high (depth/width are interpretable) | low (no opponent learning) | medium |
| Information-set MCTS | deep planning + implicit opponent-response in the tree; knock-timing option value | **highest** | high — many determinizations × rollouts/move; slow eval makes small edges hard to detect | medium (rollout policy + UCB confound) | medium (sensitive to belief calibration, which is opponent-dependent) | high |
| RL (self-play net) | end-to-end learned policy + value; could exploit patterns none of the above hand-code | high but unproven | **very high** — training variance, sparse terminal reward, large sample needs | low (black box) | high (overfits self-play / specific opponent — the caricature trap in extremis) | high |

## Read in quant terms
- **Best signal-per-unit-variance-per-unit-compute (the "Sharpe"): determinized expectimax.** It targets the largest myopic blind spot, reuses the validated belief and the Madow sampler + meld cache, generalises across opponents, and stays cheap enough to *measure* its edge at the game counts we already use. It's also the clean test of whether multi-ply lookahead signal even exists before paying for the heavy version.
- **Highest ceiling but worst measurability: ISMCTS.** This is where a real edge over a competent opponent lives, but the per-move cost slows eval (fewer games/sec → you need more wall-clock to resolve a small edge), and its quality is tied to belief calibration, which is itself opponent-dependent. Do it *after* expectimax shows the lookahead signal is real.
- **Worst risk-adjusted right now: RL.** Highest variance, weakest attribution, strongest overfitting to the training distribution, and the project guidance is explicit: no ML before the probabilistic baseline is solid — and parity-with-greedy says the *policy* isn't solid yet. Defer.
- **Free bps first: analytic fixes.** State-dependent `kappa` and a self-play `gamma` test are low-cost, low-variance, fully attributable. Harvest them regardless, but they won't beat a strong opponent on their own.

## Recommended order
1. **Harvest the cheap, attributable signal**: test `gamma` in prob-vs-prob self-play (the only setting where the leak is punished); make `kappa` state-dependent (price the option value of playing on instead of a flat threshold).
2. **Build determinized expectimax** over the belief — the best signal/effort, reuses everything validated, generalises across opponents.
3. **Then ISMCTS**, only once lookahead has proven it pays.
4. **Defer RL** until a search policy beats a competent opponent.

## Measurement discipline (applies to every model above)
- Evaluate against the **`rational`** opponent (and ideally a spectrum from greedy→meld_seeker), never a single caricature — a single-opponent optimum doesn't generalise.
- Use **common random numbers** (shared deals, `alternate_seats=True`) and report **score margin**, not just win rate: the signed point margin is continuous and carries more information per game than the binary outcome, so its test has more power to resolve a small edge.
- Budget games for the effect size: resolving a ~1.5pp win-rate edge needs ~8–9k games; the margin metric and CRN reduce that.
