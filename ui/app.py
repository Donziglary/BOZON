# ==============================================================================
# BLOCK 1: IMPORTS AND AUTOMATIC WEIGHT FILE DISCOVERY
# ==============================================================================
import time
from pathlib import Path
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.1,
    chess.BISHOP: 3.3,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.5,
    chess.KING: 0.0
}

# Center-control bonus table for active pieces (Knights, Bishops, Queens, Pawns)
CENTER_BONUS = [
    0.00, 0.02, 0.04, 0.05, 0.05, 0.04, 0.02, 0.00,
    0.02, 0.06, 0.10, 0.12, 0.12, 0.10, 0.06, 0.02,
    0.04, 0.10, 0.18, 0.22, 0.22, 0.18, 0.10, 0.04,
    0.05, 0.12, 0.22, 0.28, 0.28, 0.22, 0.12, 0.05,
    0.05, 0.12, 0.22, 0.28, 0.28, 0.22, 0.12, 0.05,
    0.04, 0.10, 0.18, 0.22, 0.22, 0.18, 0.10, 0.04,
    0.02, 0.06, 0.10, 0.12, 0.12, 0.10, 0.06, 0.02,
    0.00, 0.02, 0.04, 0.05, 0.05, 0.04, 0.02, 0.00,
]


def find_latest_weights(weights_dir: Path) -> Path:
    """Finds the newest .pt weight file in weights_dir (prioritizing v0.7 > v0.6 > v0.5)."""
    for preferred in ("bozon_v0.7_rl.pt", "bozon_v0.6_mcts.pt", "bozon_v0.5_seresnet.pt"):
        candidate = weights_dir / preferred
        if candidate.exists():
            return candidate
    pt_files = list(weights_dir.glob("*.pt"))
    if pt_files:
        pt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return pt_files[0]
    return weights_dir / "bozon_v0.7_rl.pt"


WEIGHTS_PATH = find_latest_weights(WEIGHTS_DIR)


# ==============================================================================
# BLOCK 2: DUAL-MODE BOARD & MOVE ENCODING (CANONICAL + LEGACY SUPPORT)
# ==============================================================================
IS_CANONICAL_MODEL = True

def encode_board_state(board: chess.Board) -> np.ndarray:
    tensor = np.zeros((18, 8, 8), dtype=np.float32)

    if IS_CANONICAL_MODEL:
        turn = board.turn
        flip = (turn == chess.BLACK)
        for piece_type in range(1, 7):
            for color in (chess.WHITE, chess.BLACK):
                mask = board.pieces_mask(piece_type, color)
                if not mask:
                    continue
                is_mine = (color == turn)
                ch = (piece_type - 1) + (0 if is_mine else 6)
                while mask:
                    lsb = mask & -mask
                    sq = lsb.bit_length() - 1
                    c_sq = (sq ^ 56) if flip else sq
                    tensor[ch, 7 - (c_sq >> 3), c_sq & 7] = 1.0
                    mask ^= lsb

        if board.is_check():
            tensor[12, :, :] = 1.0

        for move in board.legal_moves:
            to_sq = (move.to_square ^ 56) if flip else move.to_square
            tensor[13, 7 - (to_sq >> 3), to_sq & 7] = 1.0

        my_k = chess.BB_H8 if flip else chess.BB_H1
        my_q = chess.BB_A8 if flip else chess.BB_A1
        op_k = chess.BB_H1 if flip else chess.BB_H8
        op_q = chess.BB_A1 if flip else chess.BB_A8

        if bool(board.castling_rights & my_k):
            tensor[14, :, :] = 1.0
        if bool(board.castling_rights & my_q):
            tensor[15, :, :] = 1.0
        if bool(board.castling_rights & op_k):
            tensor[16, :, :] = 1.0
        if bool(board.castling_rights & op_q):
            tensor[17, :, :] = 1.0
    else:
        for piece_type in range(1, 7):
            for color in (chess.WHITE, chess.BLACK):
                mask = board.pieces_mask(piece_type, color)
                if not mask:
                    continue
                ch = (piece_type - 1) + (0 if color == chess.WHITE else 6)
                while mask:
                    lsb = mask & -mask
                    sq = lsb.bit_length() - 1
                    tensor[ch, 7 - (sq >> 3), sq & 7] = 1.0
                    mask ^= lsb

        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0

        for move in board.legal_moves:
            to_sq = move.to_square
            tensor[13, 7 - (to_sq >> 3), to_sq & 7] = 1.0

        if bool(board.castling_rights & chess.BB_H1):
            tensor[14, :, :] = 1.0
        if bool(board.castling_rights & chess.BB_A1):
            tensor[15, :, :] = 1.0
        if bool(board.castling_rights & chess.BB_H8):
            tensor[16, :, :] = 1.0
        if bool(board.castling_rights & chess.BB_A8):
            tensor[17, :, :] = 1.0

    return tensor


def encode_move_index(move: chess.Move, turn: bool) -> int:
    if IS_CANONICAL_MODEL and turn == chess.BLACK:
        return (move.from_square ^ 56) * 64 + (move.to_square ^ 56)
    return move.from_square * 64 + move.to_square


# ==============================================================================
# BLOCK 3: UNIFIED SE-RESNET ARCHITECTURE LOADER
# ==============================================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SEResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(p=dropout_p)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + residual, inplace=True)


class BozonModelSkip(nn.Module):
    def __init__(self, num_blocks: int = 10, channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[SEResidualBlock(channels, dropout_p=0.1) for _ in range(num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(32 * 8 * 8, 4096)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(512 + 18, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        piece_counts = x[:, :12, :, :].sum(dim=(2, 3)) * 0.125
        meta_features = x[:, 12:, :, :].mean(dim=(2, 3))
        features = self.res_tower(self.stem(x))
        policy_logits = self.policy_head(features)
        v_spatial = self.value_conv(features)
        v_combined = torch.cat([v_spatial, piece_counts, meta_features], dim=1)
        return policy_logits, self.value_mlp(v_combined)


class BozonModelLegacy(nn.Module):
    def __init__(self, num_blocks: int = 10, channels: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[SEResidualBlock(channels, dropout_p=0.1) for _ in range(num_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(32 * 8 * 8, 4096)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        features = self.res_tower(self.stem(x))
        return self.policy_head(features), self.value_head(features)


WEIGHTS_LOADED = False
WEIGHT_CHECKSUM = 0.0
ARCH_LABEL = "Untrained"

if WEIGHTS_PATH.exists():
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "value_conv.0.weight" in state_dict:
        model = BozonModelSkip(num_blocks=10, channels=128).to(DEVICE)
        IS_CANONICAL_MODEL = True
        ARCH_LABEL = f"Canonical RL ({WEIGHTS_PATH.name})"
    else:
        model = BozonModelLegacy(num_blocks=10, channels=128).to(DEVICE)
        IS_CANONICAL_MODEL = False
        ARCH_LABEL = f"Legacy v0.5 ({WEIGHTS_PATH.name})"

    model.load_state_dict(state_dict, strict=False)
    WEIGHTS_LOADED = True
    WEIGHT_CHECKSUM = float(next(model.parameters()).sum().item())
    print("=" * 72)
    print(f"[SUCCESS] Loaded Weights File : {WEIGHTS_PATH}")
    print(f"[SUCCESS] Detected Format     : {ARCH_LABEL}")
    print(f"[DIAGNOSTIC] Layer 1 Checksum : {WEIGHT_CHECKSUM:.6f}")
    print("=" * 72)
else:
    model = BozonModelSkip(num_blocks=10, channels=128).to(DEVICE)
    WEIGHT_CHECKSUM = float(next(model.parameters()).sum().item())
    print("=" * 72)
    print(f"[WARNING] No .pt file found in: {WEIGHTS_DIR}")
    print("=" * 72)

model.eval()


# ==============================================================================
# BLOCK 4: DEPTH-3 ALPHA-BETA + QUIESCENCE SEARCH ENGINE (DECISIVE MATE ENGINE)
# ==============================================================================
def evaluate_static_board(board: chess.Board, side_to_move: bool) -> float:
    """
    Fast static evaluation in pawn units from `side_to_move` perspective.
    Includes Material, Center PST, Passed Pawn Push, and Endgame Mop-Up Mate Drive.
    """
    if board.is_checkmate():
        return -99.0 if board.turn == side_to_move else 99.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    my_score, opp_score = 0.0, 0.0
    opp_color = not side_to_move

    for pt, val in PIECE_VALUES.items():
        if val > 0:
            for sq in board.pieces(pt, side_to_move):
                my_score += val + (CENTER_BONUS[sq] * 0.4)
                if pt == chess.PAWN:
                    rank = chess.square_rank(sq) if side_to_move == chess.WHITE else (7 - chess.square_rank(sq))
                    # Exponential bonus for advancing pawns toward promotion (ranks 5, 6, 7)
                    my_score += (rank ** 1.6) * 0.04
            for sq in board.pieces(pt, opp_color):
                opp_score += val + (CENTER_BONUS[sq] * 0.4)
                if pt == chess.PAWN:
                    rank = chess.square_rank(sq) if opp_color == chess.WHITE else (7 - chess.square_rank(sq))
                    opp_score += (rank ** 1.6) * 0.04

    diff = my_score - opp_score

    # Endgame Checkmate Drive (Mop-Up): push lone enemy King to corner & close distance
    if abs(diff) >= 2.5:
        winning_color = side_to_move if diff > 0 else opp_color
        losing_king = board.king(not winning_color)
        winning_king = board.king(winning_color)
        if losing_king is not None and winning_king is not None:
            lk_r, lk_f = chess.square_rank(losing_king), chess.square_file(losing_king)
            wk_r, wk_f = chess.square_rank(winning_king), chess.square_file(winning_king)
            edge_drive = abs(lk_r - 3.5) + abs(lk_f - 3.5)
            king_proximity = 14.0 - (abs(lk_r - wk_r) + abs(lk_f - wk_f))
            mop_up = (edge_drive * 0.18) + (king_proximity * 0.07)
            diff += mop_up if diff > 0 else -mop_up

    return diff


def mvv_lva_order(board: chess.Board, moves: list) -> list:
    """Orders captures and checks first for ultra-fast Alpha-Beta pruning."""
    def score_move(m: chess.Move) -> float:
        s = 0.0
        if m.promotion == chess.QUEEN:
            s += 9.0
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square)
            attacker = board.piece_type_at(m.from_square)
            v_val = PIECE_VALUES.get(victim, 1.0)
            a_val = PIECE_VALUES.get(attacker, 1.0)
            s += 10.0 + (v_val * 10.0 - a_val)
        return s

    return sorted(moves, key=score_move, reverse=True)


def quiescence_search(board: chess.Board, alpha: float, beta: float, q_depth: int = 4) -> float:
    """
    Resolves all tactical captures & promotions at leaf nodes so the engine
    NEVER fears fake suicide captures and NEVER leaves pieces hanging!
    """
    if board.is_checkmate():
        return -95.0 - q_depth
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    stand_pat = evaluate_static_board(board, board.turn)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if q_depth <= 0:
        return stand_pat

    # Only search captures and queen promotions
    tactical_moves = [
        m for m in board.legal_moves
        if board.is_capture(m) or m.promotion == chess.QUEEN
    ]
    for move in mvv_lva_order(board, tactical_moves):
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, q_depth - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax_alphabeta(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """Negamax Alpha-Beta Search with Mate-Distance scoring and Quiescence extension."""
    if board.is_checkmate():
        return -98.0 - depth  # Prefer faster checkmates!
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth <= 0:
        return quiescence_search(board, alpha, beta, q_depth=4)

    legal = [m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN]
    ordered = mvv_lva_order(board, legal)

    best_val = -999.0
    for move in ordered:
        board.push(move)
        val = -negamax_alphabeta(board, depth - 1, -beta, -alpha)
        board.pop()

        if val > best_val:
            best_val = val
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    return best_val


def predict_engine(fen: str, recent_fens: list, top_k: int = 3) -> dict:
    start_time = time.perf_counter()
    board = chess.Board(fen)

    if board.is_game_over():
        res = board.result()
        w_win = 100.0 if res == "1-0" else (0.0 if res == "0-1" else 50.0)
        return {
            "game_over": True,
            "result": res,
            "white_eval": 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0),
            "white_win_prob": w_win,
            "predictions": []
        }

    # 1. Neural Forward Pass on Root Board
    state_tensor = torch.tensor(
        encode_board_state(board), dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        policy_logits, value_pred = model(state_tensor)
        policy_logits = policy_logits.squeeze(0)
        neural_active_eval = float(value_pred.item())

    static_white = evaluate_static_board(board, chess.WHITE)
    static_eval_white = float(np.tanh(static_white / 4.5))
    neural_white_eval = neural_active_eval if board.turn == chess.WHITE else -neural_active_eval
    white_eval = (0.65 * neural_white_eval + 0.35 * static_eval_white) if IS_CANONICAL_MODEL else static_eval_white
    white_win_prob = round((white_eval + 1.0) * 50.0, 1)

    legal_moves = [m for m in board.legal_moves if not m.promotion or m.promotion == chess.QUEEN]
    legal_mask = torch.full((4096,), float("-inf"), device=DEVICE)
    for move in legal_moves:
        legal_mask[encode_move_index(move, board.turn)] = 0.0

    probs = torch.softmax(policy_logits + legal_mask, dim=0)
    random_baseline = round(100.0 / max(1, len(legal_moves)), 2)

    # 2. Batch Neural Value Evaluation on all 1-Ply Child Boards
    next_tensors = []
    for move in legal_moves:
        board.push(move)
        next_tensors.append(encode_board_state(board))
        board.pop()

    next_batch = torch.tensor(np.array(next_tensors), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, next_values = model(next_batch)
        next_values = next_values.squeeze(1).cpu().numpy()

    # 3. Depth-3 Alpha-Beta + Quiescence Search fused with Neural Policy & Value Priors
    scored_candidates = []
    moving_color = board.turn

    for move_obj, opp_neural_val in zip(legal_moves, next_values):
        idx = encode_move_index(move_obj, moving_color)
        pol_prob = float(probs[idx].item())
        raw_logit = float(policy_logits[idx].item())
        my_neural_eval = float(-opp_neural_val)

        board.push(move_obj)
        if board.is_checkmate():
            ab_raw_score = 100.0  # Immediate Mate-in-1!
        elif board.is_stalemate() or board.is_insufficient_material():
            my_cur_adv = evaluate_static_board(board, moving_color)
            ab_raw_score = -8.0 if my_cur_adv > 1.2 else 0.0
        else:
            # Depth 2 from child = 3 Plies total + 4 Plies Quiescence (7-Ply Tactical Horizon!)
            ab_raw_score = -negamax_alphabeta(board, depth=2, alpha=-105.0, beta=105.0)
            # Repetition penalty when winning so BOZON never loops moves!
            if board.board_fen() in recent_fens and ab_raw_score > 1.0:
                ab_raw_score -= 3.5
        board.pop()

        # Convert Alpha-Beta score into [-1.0, +1.0] and fuse with Neural Network
        if ab_raw_score >= 90.0:
            # Forced Checkmate found (higher ab_raw_score = faster mate!)
            composite_score = ab_raw_score
            final_active_eval = 1.0
        elif ab_raw_score <= -90.0:
            composite_score = ab_raw_score
            final_active_eval = -1.0
        else:
            tactical_squash = float(np.tanh(ab_raw_score / 4.2))
            if IS_CANONICAL_MODEL:
                final_active_eval = 0.70 * tactical_squash + 0.30 * my_neural_eval
            else:
                final_active_eval = tactical_squash
            # Neural policy guides opening & positional tie-breaks; Alpha-Beta rules tactics
            composite_score = (ab_raw_score * 1.2) + (my_neural_eval * 0.8) + (pol_prob * 1.1)

        post_white_eval = final_active_eval if moving_color == chess.WHITE else -final_active_eval
        post_win_prob = round((post_white_eval + 1.0) * 50.0, 1)

        scored_candidates.append({
            "uci": move_obj.uci(),
            "san": board.san(move_obj),
            "from": chess.square_name(move_obj.from_square),
            "to": chess.square_name(move_obj.to_square),
            "confidence": round(pol_prob * 100, 2),
            "raw_logit": round(raw_logit, 2),
            "post_eval": round(post_white_eval, 2),
            "post_win_prob": post_win_prob,
            "composite_score": composite_score
        })

    scored_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    top_predictions = scored_candidates[:top_k]
    for r, item in enumerate(top_predictions, 1):
        item["rank"] = r
        del item["composite_score"]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    best = top_predictions[0]

    print(
        f"[v0.7 ENGINE] Turn: {'W' if board.turn else 'B'} | "
        f"Eval(W): {white_eval:+.2f} ({white_win_prob}%) | "
        f"Best: {best['san']} (Policy: {best['confidence']}%, TargetEval: {best['post_eval']:+.2f}) | "
        f"Time: {latency_ms} ms"
    )

    return {
        "game_over": False,
        "turn": "White" if board.turn else "Black",
        "white_eval": round(white_eval, 2),
        "white_win_prob": white_win_prob,
        "black_win_prob": round(100.0 - white_win_prob, 1),
        "latency_ms": latency_ms,
        "legal_count": len(legal_moves),
        "random_baseline": random_baseline,
        "weights_loaded": WEIGHTS_LOADED,
        "arch_label": ARCH_LABEL,
        "checksum": round(WEIGHT_CHECKSUM, 4),
        "predictions": top_predictions
    }


# ==============================================================================
# BLOCK 5: FRONTEND UI TEMPLATE (DUAL-BOARD NEXTCHESSMOVE STYLE)
# ==============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BOZON v0.7 - RL & Alpha-Beta Quiescence Engine</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css">
    <style>
        :root { --bg-dark: #0f1217; --card-bg: #1a1f29; --accent: #10b981; --blue: #3b82f6; --amber: #f59e0b; --warn: #ef4444; --text-main: #f3f4f6; --text-muted: #9ca3af; }
        body { margin: 0; padding: 18px; font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg-dark); color: var(--text-main); display: flex; flex-direction: column; align-items: center; }
        .header-title { text-align: center; margin-bottom: 12px; }
        .header-title h1 { margin: 0; font-size: 1.65rem; letter-spacing: 0.8px; }
        .header-title span { color: var(--accent); font-size: 0.88rem; font-weight: 600; }
        .telemetry-bar { background: #151921; border: 1px solid #2a3241; padding: 9px 18px; border-radius: 8px; margin-bottom: 18px; display: flex; gap: 18px; font-size: 0.82rem; font-family: monospace; max-width: 930px; width: 100%; box-sizing: border-box; justify-content: space-between; flex-wrap: wrap; }
        .status-ok { color: var(--accent); font-weight: bold; }
        .status-err { color: var(--warn); font-weight: bold; }
        .arena-wrapper { display: flex; gap: 20px; align-items: center; justify-content: center; flex-wrap: wrap; max-width: 1100px; width: 100%; }
        .board-card { background: var(--card-bg); padding: 16px; border-radius: 12px; box-shadow: 0 8px 26px rgba(0,0,0,0.45); width: 400px; }
        .board-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; font-weight: 600; font-size: 0.92rem; }
        .badge { background: #262e3d; padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; color: var(--accent); font-family: monospace; }
        .board-box { position: relative; width: 400px; height: 400px; }
        #arrowOverlay { position: absolute; top: 0; left: 0; width: 400px; height: 400px; pointer-events: none; z-index: 20; }
        .eval-column { display: flex; flex-direction: column; align-items: center; height: 445px; justify-content: space-between; }
        .eval-score-pill { background: #262e3d; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-size: 0.82rem; font-weight: 700; min-width: 48px; text-align: center; }
        .eval-bar-outer { width: 32px; height: 398px; background: #1f242d; border-radius: 6px; overflow: hidden; border: 2px solid #374151; display: flex; flex-direction: column; justify-content: flex-end; position: relative; }
        .eval-bar-white { width: 100%; height: 50%; background: linear-gradient(180deg, #f9fafb 0%, #d1d5db 100%); transition: height 0.35s ease; }
        .eval-midline { position: absolute; top: 50%; left: 0; width: 100%; height: 2px; background: rgba(239, 68, 68, 0.65); z-index: 5; }
        .controls-panel { background: var(--card-bg); margin-top: 20px; padding: 18px 20px; border-radius: 12px; max-width: 930px; width: 100%; box-sizing: border-box; box-shadow: 0 8px 26px rgba(0,0,0,0.45); }
        .win-meter-container { margin-bottom: 16px; }
        .win-meter-labels { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; margin-bottom: 5px; }
        .win-meter-bar { height: 10px; width: 100%; background: #262e3d; border-radius: 5px; overflow: hidden; display: flex; }
        .win-fill-white { height: 100%; width: 50%; background: #f3f4f6; transition: width 0.35s ease; }
        .win-fill-black { height: 100%; flex: 1; background: #374151; }
        .fen-bar { display: flex; gap: 10px; margin-bottom: 14px; }
        .fen-bar input { flex: 1; background: #0f1217; border: 1px solid #374151; color: var(--text-main); padding: 9px 12px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; }
        .btn-group { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 16px; }
        button { background: #262e3d; color: var(--text-main); border: 1px solid #3f4a5a; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.86rem; }
        button:hover { background: #374151; }
        button.primary { background: var(--accent); color: #052e16; border: none; }
        .predictions-list { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
        .pred-card { background: #0f1217; padding: 12px 14px; border-radius: 8px; border-left: 4px solid #4b5563; display: flex; justify-content: space-between; align-items: center; }
        .pred-card.rank-1 { border-left-color: var(--accent); }
        .pred-card.rank-2 { border-left-color: var(--blue); }
        .pred-card.rank-3 { border-left-color: var(--amber); }
        .move-san { font-size: 1.15rem; font-weight: 700; }
        .move-meta { font-size: 0.75rem; color: var(--text-muted); font-family: monospace; margin-top: 3px; }
        .right-metrics { text-align: right; }
        .confidence { font-size: 1.1rem; font-weight: 700; color: var(--accent); }
        .post-eval { font-size: 0.76rem; font-family: monospace; color: #d1d5db; margin-top: 2px; }
    </style>
</head>
<body>

    <div class="header-title">
        <h1>BOZON v0.7 • RL NEURAL + ALPHA-BETA QUIESCENCE ENGINE</h1>
        <span>3-Ply Alpha-Beta + 4-Ply Quiescence Capture Resolution • Fast Checkmate Drive</span>
    </div>

    <div class="telemetry-bar">
        <span id="weightStatus">Weights: Checking...</span>
        <span id="tensorInfo">Model: Detecting...</span>
        <span id="latencyInfo">Latency: -- ms</span>
        <span id="baselineInfo">Legal Moves: --</span>
    </div>

    <div class="arena-wrapper">
        <div class="board-card">
            <div class="board-header">
                <span>1. Match Board (Drag Pieces)</span>
                <span class="badge" id="turnBadge">White to Move</span>
            </div>
            <div class="board-box">
                <div id="playBoard" style="width: 400px"></div>
            </div>
        </div>

        <div class="eval-column">
            <div class="eval-score-pill" id="evalScoreTop">0.00</div>
            <div class="eval-bar-outer">
                <div class="eval-midline"></div>
                <div class="eval-bar-white" id="evalBarWhite"></div>
            </div>
            <div class="eval-score-pill" id="evalScoreBottom">50%</div>
        </div>

        <div class="board-card">
            <div class="board-header">
                <span>2. BOZON v0.7 RL + Quiescence Output</span>
                <span class="badge" id="BestMoveBadge">Ready</span>
            </div>
            <div class="board-box">
                <div id="predBoard" style="width: 400px"></div>
                <svg id="arrowOverlay" viewBox="0 0 400 400"></svg>
            </div>
        </div>
    </div>

    <div class="controls-panel">
        <div class="win-meter-container">
            <div class="win-meter-labels">
                <span id="whiteWinLabel">White Win Chance: 50.0%</span>
                <span id="evalSummaryLabel">Value Head: 0.00 (Balanced)</span>
                <span id="blackWinLabel">Black Win Chance: 50.0%</span>
            </div>
            <div class="win-meter-bar">
                <div class="win-fill-white" id="winFillWhite"></div>
                <div class="win-fill-black"></div>
            </div>
        </div>

        <div class="fen-bar">
            <input type="text" id="fenInput" placeholder="Paste FEN string here...">
            <button onclick="loadFenFromInput()">Load FEN</button>
        </div>

        <div class="btn-group">
            <button class="primary" onclick="requestPrediction()">⚡ Calculate Best Move</button>
            <button onclick="playBestMove()">▶ Apply Best Move</button>
            <button onclick="undoLastMove()">↩ Undo</button>
            <button onclick="toggleTurn()">🔄 Switch Turn (W/B)</button>
            <button onclick="flipBoards()">⇅ Flip Boards</button>
            <button onclick="resetGame()">⏮ Reset Board</button>
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
        var recentBoardFens = [];

        function recordFenHistory() {
            var bFen = game.fen().split(' ')[0];
            recentBoardFens.push(bFen);
            if (recentBoardFens.length > 20) recentBoardFens.shift();
        }

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
            recordFenHistory();
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
            var opacities = ['0.94', '0.68', '0.48'];
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

        function updateEvaluationWidgets(whiteEval, whiteWinProb) {
            var blackWinProb = (100.0 - whiteWinProb).toFixed(1);
            var formattedEval = (whiteEval >= 0 ? '+' : '') + whiteEval.toFixed(2);

            document.getElementById('evalBarWhite').style.height = whiteWinProb + '%';
            document.getElementById('evalScoreTop').innerText = formattedEval;
            document.getElementById('evalScoreBottom').innerText = whiteWinProb.toFixed(0) + '%';

            document.getElementById('winFillWhite').style.width = whiteWinProb + '%';
            document.getElementById('whiteWinLabel').innerText = 'White Win Chance: ' + whiteWinProb.toFixed(1) + '%';
            document.getElementById('blackWinLabel').innerText = 'Black Win Chance: ' + blackWinProb + '%';

            var advantageText = 'Balanced';
            if (whiteEval > 0.15) advantageText = 'White Advantage';
            else if (whiteEval < -0.15) advantageText = 'Black Advantage';
            document.getElementById('evalSummaryLabel').innerText = 'Value Head: ' + formattedEval + ' (' + advantageText + ')';
        }

        function requestPrediction() {
            fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fen: game.fen(), recent_fens: recentBoardFens })
            })
            .then(res => res.json())
            .then(data => {
                var container = document.getElementById('predictionsContainer');
                container.innerHTML = '';

                var wStat = document.getElementById('weightStatus');
                if (data.weights_loaded) {
                    wStat.innerHTML = 'Weights: <span class="status-ok">LOADED (Sum: ' + data.checksum + ')</span>';
                } else {
                    wStat.innerHTML = 'Weights: <span class="status-err">NOT FOUND (Random!)</span>';
                }
                document.getElementById('tensorInfo').innerText = 'Arch: ' + (data.arch_label || 'Unknown');
                document.getElementById('latencyInfo').innerText = '3-Ply+Q Search: ' + (data.latency_ms || 0) + ' ms';
                document.getElementById('baselineInfo').innerText = 'Legal Moves: ' + (data.legal_count || 0);

                updateEvaluationWidgets(data.white_eval || 0.0, data.white_win_prob || 50.0);

                if (data.game_over || !data.predictions || data.predictions.length === 0) {
                    document.getElementById('BestMoveBadge').innerText = 'Game Over';
                    document.getElementById('arrowOverlay').innerHTML = '';
                    latestBestUci = null;
                    return;
                }

                latestBestUci = data.predictions[0].uci;
                document.getElementById('BestMoveBadge').innerText =
                    'Best: ' + data.predictions[0].san + ' (Eval: ' + (data.predictions[0].post_eval >= 0 ? '+' : '') + data.predictions[0].post_eval + ')';

                drawPredictionArrows(data.predictions);

                data.predictions.forEach(function(p) {
                    var postSign = p.post_eval >= 0 ? '+' : '';
                    var card = document.createElement('div');
                    card.className = 'pred-card rank-' + p.rank;
                    card.innerHTML = `
                        <div>
                            <div class="move-san">#${p.rank} ${p.san}</div>
                            <div class="move-meta">${p.from}→${p.to} | Policy: ${p.confidence}%</div>
                        </div>
                        <div class="right-metrics">
                            <div class="confidence">${postSign}${p.post_eval}</div>
                            <div class="post-eval">Win Chance: ${p.post_win_prob}% W</div>
                        </div>
                    `;
                    container.appendChild(card);
                });
            });
        }

        function playBestMove() {
            if (!latestBestUci) return;
            game.move({ from: latestBestUci.substring(0, 2), to: latestBestUci.substring(2, 4), promotion: 'q' });
            recordFenHistory();
            syncBoardsAndPredict();
        }

        function undoLastMove() {
            game.undo();
            recentBoardFens.pop();
            syncBoardsAndPredict();
        }

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
            if (game.load(fen)) {
                recentBoardFens = [];
                recordFenHistory();
                syncBoardsAndPredict();
            } else {
                alert('Invalid FEN string!');
            }
        }

        function resetGame() {
            game.reset();
            recentBoardFens = [];
            recordFenHistory();
            syncBoardsAndPredict();
        }

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

        recordFenHistory();
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
    recent_fens = data.get("recent_fens", [])
    try:
        return jsonify(predict_engine(fen, recent_fens, top_k=3))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("[INFO] Starting BOZON v0.7 RL + Alpha-Beta Quiescence Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)