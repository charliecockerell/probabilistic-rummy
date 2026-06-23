// ── State ────────────────────────────────────────────────────────────────────
let state = null;
let selectedCard = null;
let handRevealed = false;

// ── API ──────────────────────────────────────────────────────────────────────
async function api(endpoint, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(`/api/${endpoint}`, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
}

// ── Actions ──────────────────────────────────────────────────────────────────
async function newGame() {
    state = await api('new_game', 'POST');
    selectedCard = null;
    handRevealed = false;
    render();
}

async function handleStockClick() {
    if (!state || state.phase !== 'draw' || state.game_over || !handRevealed) return;
    state = await api('draw', 'POST', { source: 'stock' });
    selectedCard = null;
    render();
}

async function handleDiscardClick() {
    if (!state || state.phase !== 'draw' || state.game_over || !handRevealed) return;
    if (!state.discard_top) return;
    state = await api('draw', 'POST', { source: 'discard' });
    selectedCard = null;
    render();
}

async function doDiscard() {
    if (!selectedCard) return;
    const resp = await api('discard', 'POST', { card: selectedCard });
    if (!resp.ok) { showMsg(resp.error, true); return; }
    state = resp;
    selectedCard = null;
    if (!state.game_over) handRevealed = false;
    render();
}

async function doKnock() {
    if (!selectedCard) return;
    const resp = await api('knock', 'POST', { card: selectedCard });
    if (!resp.ok) { showMsg(resp.error, true); return; }
    state = resp;
    selectedCard = null;
    render();
}

function revealHand() {
    handRevealed = true;
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
    if (!state || state.phase !== 'discard' || state.game_over || !handRevealed) return;
    selectedCard = selectedCard === cardStr ? null : cardStr;
    renderCurrentHand();
    updateButtons();
    updateDeadwoodHint();
}

// ── Render ───────────────────────────────────────────────────────────────────
function render() {
    if (!state) return;

    if (state.game_over) {
        hide('pass-overlay');
        showGameOver();
        return;
    }

    hide('gameover-overlay');

    if (!handRevealed) {
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
    el('label-current').textContent = `Player ${p + 1}`;
    el('label-current').className = 'player-label active';
    el('label-opponent').textContent = `Player ${opp + 1}`;
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
    showMsg(state.message);
}

function renderCurrentHand() {
    const p = state.current_player;
    const handEl = el('hand-current');
    handEl.innerHTML = '';
    state.hands[p].forEach(c => {
        handEl.appendChild(makeCard(c, {
            selected: c === selectedCard,
            drawn: c === state.drawn_card,
            clickable: true,
        }));
    });
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
    const canAct = state && state.phase === 'discard' && selectedCard !== null && handRevealed;
    el('btn-discard').disabled = !canAct;
    el('btn-knock').disabled = !canAct;
}

function updateDeadwoodHint() {
    const hint = el('deadwood-hint');
    if (!state || !handRevealed || state.phase !== 'discard') {
        hint.textContent = '';
        return;
    }
    // Show deadwood of current hand (11 cards) minus selected card
    const p = state.current_player;
    if (!selectedCard) {
        hint.textContent = '';
        return;
    }
    // Compute deadwood client-side (rough: sum of unmatched card values)
    // We just show a prompt to the player
    hint.textContent = `Discarding ${selectedCard} — knock if deadwood ≤ 10`;
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
    state = await api('state');
    handRevealed = false;
    render();
}

init();
