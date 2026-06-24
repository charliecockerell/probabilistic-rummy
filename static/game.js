// ── State ────────────────────────────────────────────────────────────────────
let state = null;
let selectedCard = null;
let handRevealed = false;
let mode = 'human';                 // 'human' (hot-seat) | 'greedy' | 'probabilistic'
let beliefVisible = false;          // belief histogram toggle (probabilistic bot only)
const isBot = () => mode !== 'human';
const hasBelief = () => mode === 'probabilistic';

// ── API ──────────────────────────────────────────────────────────────────────
let busy = false;                   // a request is in flight — block new actions

async function api(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(`/api/${endpoint}`, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

// Run a network action with a single-flight lock. Blocks concurrent clicks
// (e.g. while the bot is thinking) and, on any failure, re-syncs the client
// from the server so the UI can never get wedged out of phase.
async function act(fn) {
    if (busy) return;
    busy = true;
    setBusyUI(true);
    try {
        await fn();
    } catch (e) {
        try { state = await api('state'); } catch (_) {}
        showMsg('Connection hiccup — resynced with the table.', true);
    } finally {
        busy = false;
        setBusyUI(false);
        render();
    }
}

function setBusyUI(on) {
    document.body.classList.toggle('busy', on);
    if (on) el('message').textContent = 'Working…';
    updateButtons();
}

// ── Actions ──────────────────────────────────────────────────────────────────
async function newGame() {
    mode = el('opponent-select').value;
    await act(async () => {
        state = await api('new_game', 'POST', { opponent: mode });
        selectedCard = null;
        handRevealed = isBot();      // bot mode: human's view is always up, no pass screen
    });
}

async function handleStockClick() {
    if (busy || !state || state.phase !== 'draw' || state.game_over || !handRevealed) return;
    await act(async () => {
        state = await api('draw', 'POST', { source: 'stock' });
        selectedCard = null;
    });
}

async function handleDiscardClick() {
    if (busy || !state || state.phase !== 'draw' || state.game_over || !handRevealed) return;
    if (!state.discard_top) return;
    await act(async () => {
        state = await api('draw', 'POST', { source: 'discard' });
        selectedCard = null;
    });
}

async function doDiscard() {
    if (busy || !selectedCard) return;
    await act(async () => {
        const resp = await api('discard', 'POST', { card: selectedCard });
        if (!resp.ok) { showMsg(resp.error, true); return; }
        state = resp;
        selectedCard = null;
        if (!state.game_over) handRevealed = isBot();   // hot-seat hides for the pass; bot stays revealed
    });
}

async function doKnock() {
    if (busy || !selectedCard) return;
    await act(async () => {
        const resp = await api('knock', 'POST', { card: selectedCard });
        if (!resp.ok) { showMsg(resp.error, true); return; }
        state = resp;
        selectedCard = null;
    });
}

function revealHand() {
    handRevealed = true;
    render();
}

function toggleBelief() {
    beliefVisible = !beliefVisible;
    render();
}

// ── Card helpers ─────────────────────────────────────────────────────────────
const SUIT_SYM = { H: '♥', D: '♦', C: '♣', S: '♠' };
const RED = new Set(['H', 'D']);

function parseCard(str) {
    return str.startsWith('10') ? ['10', str[2]] : [str[0], str[1]];
}

function makeCard(cardStr, { faceDown = false, selected = false, inMeld = false,
    deadwood = false, drawn = false, clickable = false, noHover = false } = {}) {

    const el = document.createElement('div');
    const cls = ['card'];
    if (faceDown) cls.push('face-down');
    if (selected) cls.push('selected');
    if (inMeld) cls.push('in-meld');
    if (deadwood) cls.push('deadwood-card');
    if (drawn && !selected) cls.push('drawn');
    if (noHover) cls.push('no-hover');
    el.className = cls.join(' ');

    if (faceDown) return el;

    const [rank, suit] = parseCard(cardStr);
    if (RED.has(suit)) el.classList.add('red');
    const sym = SUIT_SYM[suit];

    el.innerHTML = `
        <div class="c-tl"><span class="c-rank">${rank}</span><span class="c-suit-sm">${sym}</span></div>
        <span class="c-suit-lg">${sym}</span>
        <div class="c-br"><span class="c-rank">${rank}</span><span class="c-suit-sm">${sym}</span></div>
    `;

    if (clickable) el.addEventListener('click', () => onCardClick(cardStr));
    return el;
}

// ── Interaction ───────────────────────────────────────────────────────────────
function onCardClick(cardStr) {
    if (busy || !state || state.phase !== 'discard' || state.game_over || !handRevealed) return;
    selectedCard = selectedCard === cardStr ? null : cardStr;
    renderCurrentHand();
    updateButtons();
    updateDeadwoodHint();
}

// ── Render ───────────────────────────────────────────────────────────────────
function render() {
    if (!state) return;
    updateBeliefButton();

    if (state.game_over) {
        hide('pass-overlay');
        showGameOver();
        return;
    }

    hide('gameover-overlay');

    if (!handRevealed && !isBot()) {
        showPassScreen();
        return;
    }

    hide('pass-overlay');
    renderGame();
}

function renderGame() {
    const p = state.current_player;
    const opp = 1 - p;

    // Scores
    el('score-p1').textContent = state.scores[0];
    el('score-p2').textContent = state.scores[1];

    // Labels
    el('label-current').textContent = isBot() ? 'You' : `Player ${p + 1}`;
    el('label-current').className = 'player-label active';
    el('label-opponent').textContent = isBot()
        ? (state.opponent_name || 'Bot')
        : `Player ${opp + 1}`;
    el('label-opponent').className = 'player-label';

    // Opponent hand (face down)
    const oppHandEl = el('hand-opponent');
    oppHandEl.innerHTML = '';
    state.hands[opp].forEach(() => oppHandEl.appendChild(makeCard(null, { faceDown: true })));

    // Stock
    el('stock-count').textContent = state.stock_count;
    el('stock-pile').className = 'pile' + (state.phase === 'draw' ? ' can-draw' : '');

    // Discard top
    rebuildDiscardTop();
    el('discard-pile').className = 'pile' + (state.phase === 'draw' && state.discard_top ? ' can-draw' : '');

    // Current hand + UI
    renderCurrentHand();
    updateButtons();
    updateDeadwoodHint();
    renderBelief();
    showMsg(state.message);
}

function renderCurrentHand() {
    const p = state.current_player;
    const handEl = el('hand-current');
    handEl.innerHTML = '';

    // Own-melds view: server splits the active hand into melds vs deadwood.
    const view = state.own_view || {};
    const meldCards = new Set((view.current_melds || []).flat());

    state.hands[p].forEach(c => {
        handEl.appendChild(makeCard(c, {
            selected: c === selectedCard,
            drawn: c === state.drawn_card,
            inMeld: meldCards.has(c),
            deadwood: meldCards.size > 0 && !meldCards.has(c),
            clickable: true,
        }));
    });
}

// ── Belief histogram (probabilistic bot) ───────────────────────────────────────
const RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];
const SUITS = ['S', 'H', 'D', 'C'];

function updateBeliefButton() {
    const btn = el('btn-belief');
    btn.style.display = hasBelief() ? '' : 'none';
    btn.textContent = beliefVisible ? 'Hide belief' : 'Show belief';
}

function renderBelief() {
    updateBeliefButton();
    const panel = el('belief-panel');
    if (!hasBelief() || !beliefVisible || !state || !state.belief) {
        hide('belief-panel');
        return;
    }
    show('belief-panel');

    const sum = state.belief_sum;
    el('belief-sum').textContent = sum != null ? `Σ = ${sum.toFixed(2)}` : '';

    const grid = el('belief-grid');
    grid.innerHTML = '';

    // Header row: blank corner + rank labels
    grid.appendChild(beliefLabel(''));
    RANKS.forEach(r => grid.appendChild(beliefLabel(r)));

    SUITS.forEach(s => {
        grid.appendChild(beliefLabel(SUIT_SYM[s], RED.has(s)));
        RANKS.forEach(r => {
            const cardStr = `${r}${s}`;
            const p = state.belief[cardStr] ?? 0;
            grid.appendChild(beliefCell(p));
        });
    });
}

function beliefLabel(text, red = false) {
    const d = document.createElement('div');
    d.className = 'belief-axis' + (red ? ' red' : '');
    d.textContent = text;
    return d;
}

function beliefCell(p) {
    const d = document.createElement('div');
    d.className = 'belief-cell';
    if (p >= 0.999) d.classList.add('certain');       // known held (P=1)
    else if (p <= 0.001) d.classList.add('dead');     // impossible (P=0)
    // accent fill scaled by probability
    d.style.background = `rgba(0, 170, 204, ${Math.min(1, p).toFixed(3)})`;
    d.textContent = p >= 0.005 ? p.toFixed(2).slice(1) : '';  // ".42" style, blank if ~0
    d.title = p.toFixed(3);
    return d;
}

function rebuildDiscardTop() {
    const old = el('discard-top');
    let card;
    if (state.discard_top) {
        card = makeCard(state.discard_top, { noHover: true });
    } else {
        card = document.createElement('div');
        card.className = 'card placeholder';
        card.innerHTML = '<span class="placeholder-text">—</span>';
    }
    card.id = 'discard-top';
    old.replaceWith(card);
}

function updateButtons() {
    const canAct = !busy && state && state.phase === 'discard' && selectedCard !== null && handRevealed;
    el('btn-discard').disabled = !canAct;
    el('btn-knock').disabled = !canAct;
}

function updateDeadwoodHint() {
    const hint = el('deadwood-hint');
    if (!state || !handRevealed || !state.own_view) {
        hint.textContent = '';
        return;
    }
    const view = state.own_view;
    const dw = view.current_deadwood_value;
    const meldCount = (view.current_melds || []).length;
    const meldTxt = meldCount === 1 ? '1 meld' : `${meldCount} melds`;
    let txt = `Deadwood ${dw} · ${meldTxt}`;
    if (state.phase === 'discard') {
        txt += dw <= 10 ? ' — you can knock' : ' — knock needs deadwood ≤ 10';
    }
    hint.textContent = txt;
}

// ── Pass screen ───────────────────────────────────────────────────────────────
function showPassScreen() {
    const p = state.current_player;
    el('pass-title').textContent = `Player ${p + 1}'s Turn`;
    el('pass-player').textContent = `Player ${p + 1}`;
    show('pass-overlay');
    hide('gameover-overlay');
}

// ── Game over screen ──────────────────────────────────────────────────────────
function showGameOver() {
    el('score-p1').textContent = state.scores[0];
    el('score-p2').textContent = state.scores[1];

    const ki = state.knock_info;
    let title = 'Hand Over';
    if (state.winner !== null) title = `Player ${state.winner + 1} Wins!`;
    el('go-title').textContent = title;
    el('go-result').textContent = state.message;
    el('go-scores').textContent = `Scores — Player 1: ${state.scores[0]}  ·  Player 2: ${state.scores[1]}`;

    const handsEl = el('go-hands');
    handsEl.innerHTML = '';
    if (ki) {
        handsEl.appendChild(buildGoPlayer(ki.knocker, ki.knocker_melds, ki.knocker_deadwood, ki.knocker_dw_value, true));
        handsEl.appendChild(buildGoPlayer(1 - ki.knocker, ki.opponent_melds, ki.opponent_deadwood, ki.opponent_dw_value, false));
    }

    show('gameover-overlay');
}

function buildGoPlayer(idx, melds, deadwood, dwVal, isKnocker) {
    const div = document.createElement('div');

    const lbl = document.createElement('div');
    lbl.className = 'go-player-label';
    lbl.textContent = `Player ${idx + 1}${isKnocker ? ' (knocked)' : ''}  —  Deadwood: ${dwVal}`;
    div.appendChild(lbl);

    if (melds.length > 0) {
        const sl = document.createElement('div');
        sl.className = 'go-section-label';
        sl.textContent = 'Melds';
        div.appendChild(sl);
        melds.forEach(meld => {
            const row = document.createElement('div');
            row.className = 'go-row';
            meld.forEach(c => row.appendChild(makeCard(c, { inMeld: true, noHover: true })));
            div.appendChild(row);
        });
    }

    if (deadwood.length > 0) {
        const sl = document.createElement('div');
        sl.className = 'go-section-label';
        sl.textContent = 'Deadwood';
        div.appendChild(sl);
        const row = document.createElement('div');
        row.className = 'go-row';
        deadwood.forEach(c => row.appendChild(makeCard(c, { deadwood: true, noHover: true })));
        div.appendChild(row);
    }

    return div;
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function el(id) { return document.getElementById(id); }
function show(id) { el(id).classList.remove('hidden'); }
function hide(id) { el(id).classList.add('hidden'); }

function showMsg(text, isError = false) {
    const m = el('message');
    m.textContent = text;
    m.className = 'message' + (isError ? ' error' : '');
    if (isError) setTimeout(() => { m.className = 'message'; m.textContent = state?.message ?? ''; }, 3000);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function init() {
    // Start straight into a game against the selected opponent (default: bot).
    await newGame();
}

init();
