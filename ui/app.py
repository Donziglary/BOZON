# ==============================================================================
# BLOCK 1: IMPORTS AND CONFIGURATION
# ==============================================================================
import time
from pathlib import Path
import chess
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = BASE_DIR / "weights" / "bozon_v0.2_baseline.pt"


# ==============================================================================
# BLOCK 2: BOARD ENCODING AND MOVE TOKENIZATION
# ==============================================================================
def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Converts a chess.Board object into a (14, 8, 8) float32 numpy array."""
    tensor = np.zeros((14, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        channel = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        tensor[channel, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    for move in board.legal_moves:
        to_sq = move.to_square
        tensor[13, 7 - (to_sq // 8), to_sq % 8] = 1.0

    return tensor


def encode_move(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


# ==============================================================================
# BLOCK 3: BOZON v0.2 MODEL ARCHITECTURE AND DIAGNOSTIC LOADING
# ==============================================================================
class BozonBaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(14, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 4096)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_head(self.conv_block(x))


model = BozonBaselineNet().to(DEVICE)
WEIGHTS_LOADED = False
WEIGHT_CHECKSUM = 0.0

if WEIGHTS_PATH.exists():
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    WEIGHTS_LOADED = True
    WEIGHT_CHECKSUM = float(next(model.parameters()).sum().item())
    print("=" * 70)
    print(f"[SUCCESS] Loaded trained weights from: {WEIGHTS_PATH}")
    print(f"[DIAGNOSTIC] First Layer Checksum: {WEIGHT_CHECKSUM:.6f}")
    print("=" * 70)
else:
    WEIGHT_CHECKSUM = float(next(model.parameters()).sum().item())
    print("=" * 70)
    print(f"[CRITICAL WARNING] Weights NOT FOUND at: {WEIGHTS_PATH}")
    print("[CRITICAL WARNING] Model is running with UNTRAINED RANDOM weights!")
    print("=" * 70)

model.eval()


# ==============================================================================
# BLOCK 4: VERIFIED INFERENCE ENGINE WITH TELEMETRY LOGGING
# ==============================================================================
def predict_moves(fen: str, top_k: int = 3) -> dict:
    """Runs neural forward pass and logs telemetry data to verify computation."""
    start_time = time.perf_counter()
    board = chess.Board(fen)

    if board.is_game_over():
        return {"game_over": True, "result": board.result(), "predictions": []}

    state_tensor = torch.tensor(
        board_to_tensor(board), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(state_tensor).squeeze(0)

    legal_moves = list(board.legal_moves)
    legal_mask = torch.full((4096,), float("-inf"), device=DEVICE)
    legal_moves_dict = {}

    for move in legal_moves:
        idx = encode_move(move)
        legal_mask[idx] = 0.0
        # Prefer Queen promotion if multiple promotions map to same square
        if idx not in legal_moves_dict or move.promotion == chess.QUEEN:
            legal_moves_dict[idx] = move

    masked_logits = logits + legal_mask
    probs = torch.softmax(masked_logits, dim=0)

    # Calculate uniform random baseline probability for comparison
    random_baseline_prob = round(100.0 / len(legal_moves), 2)

    k = min(top_k, len(legal_moves_dict))
    top_probs, top_indices = torch.topk(probs, k)

    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        move_obj = legal_moves_dict[idx.item()]
        raw_logit = round(logits[idx.item()].item(), 3)
        conf = round(prob.item() * 100, 2)
        predictions.append({
            "rank": rank,
            "uci": move_obj.uci(),
            "san": board.san(move_obj),
            "from": chess.square_name(move_obj.from_square),
            "to": chess.square_name(move_obj.to_square),
            "confidence": conf,
            "raw_logit": raw_logit
        })

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Print live proof of computation in VS Code terminal
    best = predictions[0]
    print(
        f"[INFERENCE] Turn: {'W' if board.turn else 'B'} | "
        f"Legal Moves: {len(legal_moves)} (Random Baseline: {random_baseline_prob}%) | "
        f"Top-1: {best['san']} ({best['confidence']}%, Logit: {best['raw_logit']}) | "
        f"Time: {latency_ms} ms"
    )

    return {
        "game_over": False,
        "turn": "White" if board.turn else "Black",
        "latency_ms": latency_ms,
        "legal_count": len(legal_moves),
        "random_baseline": random_baseline_prob,
        "weights_loaded": WEIGHTS_LOADED,
        "checksum": round(WEIGHT_CHECKSUM, 4),
        "predictions": predictions
    }


# ==============================================================================
# BLOCK 5: FRONTEND UI WITH DIAGNOSTIC TELEMETRY BAR
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BOZON v0.2 - Verified Neural Interface</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        :root {
            --bg-dark: #12151a;
            --card-bg: #1e232b;
            --accent: #10b981;
            --warn: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        body {
            margin: 0;
            padding: 20px;
            font-family: 'Segoe UI', system-ui, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .telemetry-bar {
            background: #181c23;
            border: 1px solid #2d3542;
            padding: 10px 18px;
            border-radius: 8px;
            margin-bottom: 18px;
            display: flex;
            gap: 20px;
            font-size: 0.85rem;
            font-family: monospace;
            max-width: 855px;
            width: 100%;
            box-sizing: border-box;
            justify-content: space-between;
        }
        .status-ok { color: var(--accent); font-weight: bold; }
        .status-err { color: var(--warn); font-weight: bold; }
        .boards-wrapper {
            display: flex;
            gap: 35px;
            flex-wrap: wrap;
            justify-content: center;
            max-width: 1100px;
            width: 100%;
        }
        .board-card {
            background: var(--card-bg);
            padding: 18px;
            border-radius: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
            width: 400px;
        }
        .board-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            font-weight: 600;
            font-size: 0.95rem;
        }
        .badge {
            background: #2d3542;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            color: var(--accent);
        }
        .board-box {
            position: relative;
            width: 400px;
            height: 400px;
        }
        #arrowOverlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 400px;
            height: 400px;
            pointer-events: none;
            z-index: 20;
        }
        .controls-panel {
            background: var(--card-bg);
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            max-width: 855px;
            width: 100%;
            box-sizing: border-box;
        }
        .fen-bar { display: flex; gap: 10px; margin-bottom: 15px; }
        .fen-bar input {
            flex: 1;
            background: #12151a;
            border: 1px solid #374151;
            color: var(--text-main);
            padding: 10px 12px;
            border-radius: 6px;
            font-family: monospace;
        }
        .btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
        button {
            background: #2d3542;
            color: var(--text-main);
            border: 1px solid #3f4a5a;
            padding: 9px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }
        button.primary { background: var(--accent); color: #06281e; border: none; }
        .predictions-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .pred-card {
            background: #12151a;
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid #4b5563;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .pred-card.rank-1 { border-left-color: #10b981; }
        .pred-card.rank-2 { border-left-color: #3b82f6; }
        .pred-card.rank-3 { border-left-color: #f59e0b; }
        .move-san { font-size: 1.15rem; font-weight: 700; }
        .move-meta { font-size: 0.75rem; color: var(--text-muted); font-family: monospace; }
        .confidence { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
    </style>
</head>
<body>

    <h2 style="margin: 0 0 10px 0;">BOZON v0.2 • Neural Engine Diagnostic & Tester</h2>

    <!-- Live Engine Telemetry Bar -->
    <div class="telemetry-bar">
        <span id="weightStatus">Weights: Checking...</span>
        <span id="latencyInfo">Latency: -- ms</span>
        <span id="baselineInfo">Random Chance: --%</span>
    </div>

    <div class="boards-wrapper">
        <div class="board-card">
            <div class="board-header">
                <span>1. Match Board (Drag Pieces)</span>
                <span class="badge" id="turnBadge">White to Move</span>
            </div>
            <div class="board-box">
                <div id="playBoard" style="width: 400px"></div>
            </div>
        </div>

        <div class="board-card">
            <div class="board-header">
                <span>2. BOZON Neural Output</span>
                <span class="badge" id="BestMoveBadge">Ready</span>
            </div>
            <div class="board-box">
                <div id="predBoard" style="width: 400px"></div>
                <svg id="arrowOverlay" viewBox="0 0 400 400"></svg>
            </div>
        </div>
    </div>

    <div class="controls-panel">
        <div class="fen-bar">
            <input type="text" id="fenInput" placeholder="Paste FEN string here...">
            <button onclick="loadFenFromInput()">Load FEN</button>
        </div>

        <div class="btn-group">
            <button class="primary" onclick="requestPrediction()">⚡ Calculate Move</button>
            <button onclick="playBestMove()">▶ Apply Best Move</button>
            <button onclick="undoLastMove()">↩ Undo</button>
            <button onclick="toggleTurn()">🔄 Switch Turn (W/B)</button>
            <button onclick="flipBoards()">⇅ Flip Boards</button>
            <button onclick="resetGame()">⏮ Reset</button>
        </div>

        <div class="predictions-list" id="predictionsContainer"></div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.3/chess.min.js"></script>

    <script>
        var game = new Chess();
        var playBoard = null;
        var predBoard = null;
        var currentOrientation = 'white';
        var latestBestUci = null;

        function onDragStart(source, piece) {
            if (game.game_over()) return false;
            if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
                (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
                return false;
            }
        }

        function onDrop(source, target) {
            var move = game.move({ from: source, to: target, promotion: 'q' });
            if (move === null) return 'snapback';
            syncBoardsAndPredict();
        }

        function onSnapEnd() { playBoard.position(game.fen()); }

        function syncBoardsAndPredict() {
            var fen = game.fen();
            playBoard.position(fen);
            predBoard.position(fen);
            document.getElementById('fenInput').value = fen;
            document.getElementById('turnBadge').innerText = (game.turn() === 'w' ? 'White' : 'Black') + ' to Move';
            requestPrediction();
        }

        function squareToPixel(square) {
            var file = square.charCodeAt(0) - 'a'.charCodeAt(0);
            var rank = parseInt(square[1]) - 1;
            var sqSize = 400 / 8;
            if (currentOrientation === 'white') {
                return { x: (file + 0.5) * sqSize, y: (7 - rank + 0.5) * sqSize };
            } else {
                return { x: (7 - file + 0.5) * sqSize, y: (rank + 0.5) * sqSize };
            }
        }

        function drawPredictionArrows(predictions) {
            var svg = document.getElementById('arrowOverlay');
            svg.innerHTML = '';
            var colors = ['#10b981', '#3b82f6', '#f59e0b'];
            var opacities = ['0.92', '0.65', '0.45'];
            var widths = [9, 6, 4];

            for (var i = predictions.length - 1; i >= 0; i--) {
                var pred = predictions[i];
                var start = squareToPixel(pred.from);
                var end = squareToPixel(pred.to);
                var color = colors[i] || '#10b981';

                var dx = end.x - start.x;
                var dy = end.y - start.y;
                var len = Math.sqrt(dx * dx + dy * dy);
                if (len === 0) continue;

                var targetX = end.x - (dx / len) * 14;
                var targetY = end.y - (dy / len) * 14;

                var markerId = 'arrow-head-' + i;
                var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
                var marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
                marker.setAttribute('id', markerId);
                marker.setAttribute('markerWidth', '4');
                marker.setAttribute('markerHeight', '4');
                marker.setAttribute('refX', '2');
                marker.setAttribute('refY', '2');
                marker.setAttribute('orient', 'auto');

                var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                path.setAttribute('d', 'M0,0 L4,2 L0,4 Z');
                path.setAttribute('fill', color);
                marker.appendChild(path);
                defs.appendChild(marker);
                svg.appendChild(defs);

                var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', start.x);
                line.setAttribute('y1', start.y);
                line.setAttribute('x2', targetX);
                line.setAttribute('y2', targetY);
                line.setAttribute('stroke', color);
                line.setAttribute('stroke-width', widths[i]);
                line.setAttribute('stroke-linecap', 'round');
                line.setAttribute('opacity', opacities[i]);
                line.setAttribute('marker-end', 'url(#' + markerId + ')');
                svg.appendChild(line);
            }
        }

        function requestPrediction() {
            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fen: game.fen() })
            })
            .then(res => res.json())
            .then(data => {
                var container = document.getElementById('predictionsContainer');
                container.innerHTML = '';

                // Update telemetry header
                var wStat = document.getElementById('weightStatus');
                if (data.weights_loaded) {
                    wStat.innerHTML = 'Weights: <span class="status-ok">LOADED (Sum: ' + data.checksum + ')</span>';
                } else {
                    wStat.innerHTML = 'Weights: <span class="status-err">UNTRAINED (Random!)</span>';
                }
                document.getElementById('latencyInfo').innerText = 'PyTorch Latency: ' + (data.latency_ms || 0) + ' ms';
                document.getElementById('baselineInfo').innerText = 'Legal Moves: ' + (data.legal_count || 0) + ' (Random Chance: ' + (data.random_baseline || 0) + '%)';

                if (data.game_over || data.predictions.length === 0) {
                    document.getElementById('BestMoveBadge').innerText = 'Game Over';
                    document.getElementById('arrowOverlay').innerHTML = '';
                    latestBestUci = null;
                    return;
                }

                latestBestUci = data.predictions[0].uci;
                document.getElementById('BestMoveBadge').innerText = 'Best: ' + data.predictions[0].san + ' (' + data.predictions[0].confidence + '%)';

                drawPredictionArrows(data.predictions);

                data.predictions.forEach(function(p) {
                    var card = document.createElement('div');
                    card.className = 'pred-card rank-' + p.rank;
                    card.innerHTML = `
                        <div>
                            <div class="move-san">#${p.rank} ${p.san}</div>
                            <div class="move-meta">${p.from}→${p.to} | Logit: ${p.raw_logit}</div>
                        </div>
                        <div class="confidence">${p.confidence}%</div>
                    `;
                    container.appendChild(card);
                });
            });
        }

        function playBestMove() {
            if (!latestBestUci) return;
            game.move({ from: latestBestUci.substring(0, 2), to: latestBestUci.substring(2, 4), promotion: 'q' });
            syncBoardsAndPredict();
        }

        function undoLastMove() { game.undo(); syncBoardsAndPredict(); }

        function toggleTurn() {
            var tokens = game.fen().split(' ');
            tokens[1] = (tokens[1] === 'w') ? 'b' : 'w';
            tokens[3] = '-';
            if (game.load(tokens.join(' '))) syncBoardsAndPredict();
        }

        function flipBoards() {
            currentOrientation = (currentOrientation === 'white') ? 'black' : 'white';
            playBoard.orientation(currentOrientation);
            predBoard.orientation(currentOrientation);
            requestPrediction();
        }

        function loadFenFromInput() {
            var fen = document.getElementById('fenInput').value.trim();
            if (game.load(fen)) syncBoardsAndPredict();
            else alert('Invalid FEN string!');
        }

        function resetGame() { game.reset(); syncBoardsAndPredict(); }

        var pieceThemeUrl = 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png';
        playBoard = Chessboard('playBoard', {
            draggable: true,
            position: 'start',
            pieceTheme: pieceThemeUrl,
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd
        });
        predBoard = Chessboard('predBoard', {
            draggable: false,
            position: 'start',
            pieceTheme: pieceThemeUrl
        });

        syncBoardsAndPredict();
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    fen = data.get("fen", chess.STARTING_FEN)
    return jsonify(predict_moves(fen, top_k=3))


if __name__ == "__main__":
    print("[INFO] Starting BOZON Verified UI Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)