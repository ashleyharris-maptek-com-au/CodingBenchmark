"""
Test 55: Chess Bot (C++ Implementation)

The AI must write a C++ chess engine that accepts a FEN position and a time
limit on stdin and returns the best move in a custom disambiguated algebraic
notation on stdout.  The engine is compiled once and then invoked repeatedly
-- once per move -- across three tiers of difficulty:

  Subpasses 0-4:   Trivial tactics (hanging pieces, mate-in-1).
  Subpasses 5-19:  Classic puzzles (mate-in-2/3, forks, pins).
  Subpasses 20-29: Full games vs Stockfish at increasing ELO (100-2500).
"""

from solver_utils import GradeCache
from native_compiler import CppCompiler, CompilationError, ExecutionError, describe_this_pc

import chess
import chess.engine
import chess.svg
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

title = "Can your write a chess bot in C++?"

skip = True

# ---------------------------------------------------------------------------
# Move notation helpers
# ---------------------------------------------------------------------------

PIECE_CHAR = {
  chess.PAWN: "p",
  chess.KNIGHT: "N",
  chess.BISHOP: "B",
  chess.ROOK: "R",
  chess.QUEEN: "Q",
  chess.KING: "K",
}

PROMO_CHAR = {
  chess.QUEEN: "Q",
  chess.ROOK: "R",
  chess.BISHOP: "B",
  chess.KNIGHT: "N",
}


def move_to_disambig(board: chess.Board, move: chess.Move) -> str:
  """Convert a python-chess Move to our disambiguated algebraic notation."""
  if board.is_kingside_castling(move):
    suffix = ""
    board.push(move)
    if board.is_check():
      suffix = "#" if board.is_checkmate() else "+"
    board.pop()
    return "O-O" + suffix
  if board.is_queenside_castling(move):
    suffix = ""
    board.push(move)
    if board.is_check():
      suffix = "#" if board.is_checkmate() else "+"
    board.pop()
    return "O-O-O" + suffix

  piece = board.piece_at(move.from_square)
  pc = PIECE_CHAR.get(piece.piece_type, "?") if piece else "?"
  src = chess.square_name(move.from_square)
  dst = chess.square_name(move.to_square)
  cap = "x" if board.is_capture(move) else ""
  promo = ""
  if move.promotion:
    promo = "/" + PROMO_CHAR.get(move.promotion, "Q")
  ep = ""
  if board.is_en_passant(move):
    ep = "ep"

  suffix = ""
  board.push(move)
  if board.is_check():
    suffix = "#" if board.is_checkmate() else "+"
  board.pop()

  return f"{pc}{src}{cap}{dst}{promo}{ep}{suffix}"


def disambig_to_move(board: chess.Board, notation: str) -> Optional[chess.Move]:
  """Parse our disambiguated algebraic notation back to a python-chess Move.
  Returns None if the notation doesn't match any legal move."""
  notation = notation.strip()
  if not notation:
    return None
  # Try matching against all legal moves
  for legal in board.legal_moves:
    if move_to_disambig(board, legal) == notation:
      return legal
  # Fallback: strip check/mate suffixes and try again
  stripped = notation.rstrip("+#")
  if stripped != notation:
    for legal in board.legal_moves:
      canon = move_to_disambig(board, legal).rstrip("+#")
      if canon == stripped:
        return legal
  return None


# ---------------------------------------------------------------------------
# Stockfish discovery / download
# ---------------------------------------------------------------------------

_STOCKFISH_DIR = Path(tempfile.gettempdir()) / "codingbenchmark_stockfish"


def _find_stockfish() -> Optional[str]:
  """Find stockfish binary, downloading if necessary."""
  # 1. Check PATH
  sf = shutil.which("stockfish")
  if sf:
    return sf

  # 2. Check our temp dir
  if os.name == "nt":
    cached = _STOCKFISH_DIR / "stockfish.exe"
  else:
    cached = _STOCKFISH_DIR / "stockfish"
  if cached.exists():
    return str(cached)

  # 3. Try to download
  try:
    _STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
      url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.zip"
    else:
      url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar"

    print(f"  Downloading Stockfish from {url} ...")
    zip_path = _STOCKFISH_DIR / "stockfish_download"
    urllib.request.urlretrieve(url, str(zip_path))

    if os.name == "nt":
      with zipfile.ZipFile(str(zip_path), "r") as zf:
        for name in zf.namelist():
          if name.endswith("stockfish-windows-x86-64-avx2.exe"):
            data = zf.read(name)
            cached.write_bytes(data)
            break
    else:
      import tarfile
      with tarfile.open(str(zip_path), "r") as tf:
        for member in tf.getmembers():
          if "stockfish" in member.name and member.isfile():
            data = tf.extractfile(member).read()
            cached.write_bytes(data)
            os.chmod(str(cached), 0o755)
            break

    zip_path.unlink(missing_ok=True)
    if cached.exists():
      print(f"  Stockfish downloaded to {cached}")
      return str(cached)
  except Exception as e:
    print(f"  Warning: Could not download Stockfish: {e}")

  return None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

prompt = f"""
Write a C++ chess bot that for any given chess position, returns the best move to make (or the best
move you can find before the timeout).

You receive the chess position as a string in FEN notation, and you must return the move in
disambiguated algebraic notation (including the starting square, piece type, and destination square).

See:
- https://www.chess.com/terms/fen-chess

Here is an example FEN representing the starting position with white to move:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

Here is an example disambiguated algebraic move:
```
pe2e4
```
This represents the king's pawn opening - moving the pawn from e2 to e4.

**Move notation rules:**
- Piece letter (lowercase p for pawn, uppercase N/B/R/Q/K for others) + source square + destination square.
- Use "x" between source and destination for captures: `pe4xd5`
- Promotion: append `/` and the piece letter, e.g. `pe7e8/Q`
- Castling: `O-O` (kingside) or `O-O-O` (queenside).
- En passant: append `ep`, e.g. `pe5xd6ep`
- Check: append `+`, e.g. `Bb4c5+`
- Checkmate: append `#`, e.g. `Qd1h5#`

**Input format (stdin, two lines):**
```
timeout_in_seconds
fen_string
```

**Output format (stdout, one line):**
```
move_in_disambiguated_algebraic
```

timeout_in_seconds will be at least 5. Your code will be compiled with full optimisations and
run on a machine with multiple cores, so threading is encouraged.

**Environment:**
{describe_this_pc()}

**C++ Compiler:**
{CppCompiler("test_engine").describe()}

Be sure that any deviation from the C++ standard library is supported by the given compiler,
as referencing the wrong intrinsics or non-standard header (like 'bits/stdc++.h') could fail your submission.

Write complete, compilable C++ code with a main() function.
"""

structure = {
  "type": "object",
  "properties": {
    "reasoning": {
      "type": "string",
      "description": "Explain your chess engine approach (search algorithm, evaluation, etc.)"
    },
    "cpp_code": {
      "type": "string",
      "description": "Complete C++ code with main() function"
    }
  },
  "required": ["reasoning", "cpp_code"],
  "additionalProperties": False
}

extraGradeAnswerRuns = list(range(1, 30))


# ---------------------------------------------------------------------------
# Subpass definitions
# ---------------------------------------------------------------------------

# Subpasses 0-4: Trivial tactics -- obvious best move
TRIVIAL_POSITIONS = [
  {
    "fen": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
    "name": "Fool's mate - Qh4#",
    "best_moves": ["Qd8h4#"],
    "desc": "Black can deliver immediate checkmate with Qh4#.",
  },
  {
    "fen": "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",
    "name": "Back rank mate in 1",
    "best_moves": ["Re1e8#"],
    "desc": "White can deliver back-rank checkmate with Re8#.",
  },
  {
    "fen": "4k3/8/8/8/3q4/4P3/8/4K3 w - - 0 1",
    "name": "Capture the queen",
    "best_moves": ["pe3xd4"],
    "desc": "Black's queen is hanging on d4 — pawn takes queen.",
  },
  {
    "fen": "4k3/8/8/8/8/2r5/3P4/4K3 w - - 0 1",
    "name": "Capture the rook",
    "best_moves": ["pd2xc3"],
    "desc": "Black's rook is hanging on c3 — pawn takes rook.",
  },
  {
    "fen": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "name": "Scholar's mate - Qxf7#",
    "best_moves": ["Qh5xf7#"],
    "desc": "White can deliver Scholar's mate with Qxf7#.",
  },
]

# Subpasses 5-19: Known chess puzzles (mate in 2-3, tactical themes)
# These are classic, well-known puzzles from public domain sources.
PUZZLES = [
  {
    "fen": "2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPPP1P/RNB1K2R w KQq - 0 1",
    "name": "Mate in 2 - Queen f3",
    "solution_moves": ["Qf3a3#"],
    "desc": "White to play and mate in two.",
    "max_moves": 2,
  },
  {
    "fen": "r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/PP2B1PP/RN2K2R w KQkq - 0 1",
    "name": "Mate in 1 - Back rank",
    "solution_moves": [],
    "desc": "White to play and mate in one with Qd7#.",
    "max_moves": 1,
  },
  {
    "fen": "6k1/pp4p1/2p5/2bp4/8/P5Pb/1P3rrP/2BRRK2 b - - 0 1",
    "name": "Mate in 2 - Double rook",
    "solution_moves": [],
    "desc": "Black to play and mate in two moves.",
    "max_moves": 3,
  },
  {
    "fen": "r1b1kb1r/pppp1ppp/5q2/4n3/3KP3/2N3PN/PPP4P/R1BQ1B1R b kq - 0 1",
    "name": "Mate in 2 - King hunt",
    "solution_moves": [],
    "desc": "Black to play, find the forcing sequence.",
    "max_moves": 3,
  },
  {
    "fen": "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1",
    "name": "Mate in 2 - Rook + Bishop",
    "solution_moves": [],
    "desc": "White to play and mate in two.",
    "max_moves": 3,
  },
  {
    "fen": "3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1",
    "name": "Mate in 2 - Knight sacrifice",
    "solution_moves": ["Qd5g8+"],
    "desc": "White to play and force mate.",
    "max_moves": 3,
  },
  {
    "fen": "r1bq2r1/b4pk1/p1pp1p2/1p2pP2/1P2P1PB/3P4/1PPQ4/2KR3R w - - 0 1",
    "name": "Mate in 2 - h file attack",
    "solution_moves": ["Qd2h2"],
    "desc": "White to play and force mate.",
    "max_moves": 3,
  },
  {
    "fen": "2r1nrk1/p4p1p/1p2p1pQ/nPqbRN2/8/P2B4/1BP2PPP/3R2K1 w - - 0 1",
    "name": "Mate in 3 - Exchange sacrifice",
    "solution_moves": ["Nf5e7+"],
    "desc": "White to play and force mate in three.",
    "max_moves": 5,
  },
  {
    "fen": "r1bqr3/ppp1B1kp/1b4p1/n2B4/3NQ3/2P5/PP2B1PP/R3K2R w KQ - 0 1",
    "name": "Mate in 3 - Bishop pair",
    "solution_moves": ["Qe4f5"],
    "desc": "White to play and force mate.",
    "max_moves": 5,
  },
  {
    "fen": "rnbk1b1r/ppqpnQ1p/4p1p1/2p1N1B1/4N3/8/PPP1PPPP/R3KB1R w KQ - 0 1",
    "name": "Mate in 3 - Discovered check",
    "solution_moves": [],
    "desc": "White to play and force mate.",
    "max_moves": 5,
  },
  {
    "fen": "4Rnk1/pr3ppp/1p3q2/5NQ1/2p5/8/PP3PPP/6K1 w - - 0 1",
    "name": "Mate in 3 - Rook + Knight",
    "solution_moves": ["Qg5g6"],
    "desc": "White to play and force mate in three.",
    "max_moves": 5,
  },
  {
    "fen": "3r2k1/1pp2ppp/p1pb4/8/1q1N1Q2/4P3/rPP2PPP/1K1R3R w - - 0 1",
    "name": "Zwischenzug tactic",
    "solution_moves": ["Nd4e6"],
    "desc": "White to play and find the intermediate move.",
    "max_moves": 3,
  },
  {
    "fen": "r4rk1/ppq2pp1/2p2n1p/4N1N1/3pP3/3P3P/PPP2QP1/R3R1K1 w - - 0 1",
    "name": "Double knight attack",
    "solution_moves": ["Ng5h7"],
    "desc": "White to play and win material or force mate.",
    "max_moves": 5,
  },
  {
    "fen": "r2qk2r/ppp1bppp/2np1n2/1B2p1B1/4P1b1/2NP1N2/PPP2PPP/R2QK2R w KQkq - 0 1",
    "name": "Pin exploitation",
    "solution_moves": [],
    "desc": "Complex middlegame position - find the best continuation.",
    "max_moves": 3,
  },
  {
    "fen": "r2r2k1/pp2bppp/2p1pn2/q7/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 1",
    "name": "Positional squeeze",
    "solution_moves": [],
    "desc": "Find the best plan in this IQP position.",
    "max_moves": 3,
  },
]

# Subpasses 20-29: Full games vs Stockfish at increasing ELO
STOCKFISH_GAMES = [
  {"elo": 100,  "time_per_move": 5,  "name": "Stockfish ELO 100 (absolute beginner)"},
  {"elo": 400,  "time_per_move": 5,  "name": "Stockfish ELO 400 (casual player)"},
  {"elo": 700,  "time_per_move": 5,  "name": "Stockfish ELO 700 (club beginner)"},
  {"elo": 1000, "time_per_move": 10, "name": "Stockfish ELO 1000 (intermediate)"},
  {"elo": 1200, "time_per_move": 10, "name": "Stockfish ELO 1200 (club player)"},
  {"elo": 1500, "time_per_move": 15, "name": "Stockfish ELO 1500 (strong club)"},
  {"elo": 1800, "time_per_move": 20, "name": "Stockfish ELO 1800 (expert)"},
  {"elo": 2000, "time_per_move": 30, "name": "Stockfish ELO 2000 (candidate master)"},
  {"elo": 2200, "time_per_move": 45, "name": "Stockfish ELO 2200 (master)"},
  {"elo": 2500, "time_per_move": 60, "name": "Stockfish ELO 2500 (grandmaster)"},
  {"elo": 2800, "time_per_move": 60, "name": "Stockfish ELO 2800 (Magnus Carlsen)"},
  {"elo": 3000, "time_per_move": 120, "name": "Stockfish ELO 3000 (Epic chess engine)"},
  {"elo": 3190, "time_per_move": 300, "name": "Stockfish ELO 3190 (Stockfish 18 highest limitable)"},
  {"elo": 3600, "time_per_move": 900, "name": "Stockfish ELO 3600 (Stockfish 18 full power)"},
]

# Cache for reports
_REPORT_CACHE: Dict[Tuple[str, int], dict] = {}
_grade_cache = GradeCache("test_55")


# ---------------------------------------------------------------------------
# Engine runner - invoke the C++ bot for a single position
# ---------------------------------------------------------------------------

def _run_engine_once(exe_path: str, fen: str, timeout: float) -> Tuple[Optional[str], str]:
  """Run the C++ engine for one position. Returns (move_string, error_or_empty)."""
  try:
    input_data = f"{int(timeout)}\n{fen}\n"
    proc = subprocess.run(
      [exe_path],
      input=input_data,
      capture_output=True,
      text=True,
      timeout=timeout + 5,
    )
    if proc.returncode != 0:
      return None, f"Engine crashed (exit code {proc.returncode}): {proc.stderr[:200]}"
    output = proc.stdout.strip()
    if not output:
      return None, "Engine produced no output"
    move_str = output.split("\n")[0].strip()
    return move_str, ""
  except subprocess.TimeoutExpired:
    return None, f"Engine timed out after {timeout+5:.0f}s"
  except Exception as e:
    return None, f"Engine error: {str(e)[:200]}"


# ---------------------------------------------------------------------------
# gradeAnswer
# ---------------------------------------------------------------------------

def gradeAnswer(result, subPass, aiEngineName):
  if not result or "cpp_code" not in result:
    return 0.0, "No C++ code provided"

  code = result["cpp_code"]

  # --- Compile ---
  compiler = CppCompiler(aiEngineName)
  if not compiler.find_compiler():
    return 0.0, "No C++ compiler found"

  try:
    exe_path = str(compiler.compile(code))
  except CompilationError as e:
    return 0.0, f"Compilation error: {str(e)[:500]}"

  # --- Subpasses 0-4: trivial tactics ---
  if subPass < 5:
    return _grade_trivial(exe_path, subPass, aiEngineName)

  # --- Subpasses 5-19: puzzles ---
  elif subPass < 20:
    return _grade_puzzle(exe_path, subPass, aiEngineName)

  # --- Subpasses 20-29: full games vs Stockfish ---
  else:
    return _grade_stockfish_game(exe_path, subPass, aiEngineName)


def _grade_trivial(exe_path: str, subPass: int, aiEngineName: str):
  """Grade a trivial tactic position (subpass 0-4)."""
  pos = TRIVIAL_POSITIONS[subPass]
  board = chess.Board(pos["fen"])

  move_str, err = _run_engine_once(exe_path, pos["fen"], timeout=10)
  if err:
    _REPORT_CACHE[(aiEngineName, subPass)] = {
      "type": "trivial", "pos": pos, "board_fen": pos["fen"],
      "engine_move": None, "error": err, "score": 0.0,
      "moves": [],
    }
    return 0.0, f"[{pos['name']}] {err}"

  # Parse the move
  parsed = disambig_to_move(board, move_str)
  if parsed is None:
    _REPORT_CACHE[(aiEngineName, subPass)] = {
      "type": "trivial", "pos": pos, "board_fen": pos["fen"],
      "engine_move": move_str, "error": f"Invalid move: {move_str}", "score": 0.0,
      "moves": [],
    }
    return 0.0, f"[{pos['name']}] Invalid move: {move_str}"

  # Check against best moves (if specified)
  if pos["best_moves"]:
    if move_str.rstrip("+#") in [m.rstrip("+#") for m in pos["best_moves"]]:
      score = 1.0
    else:
      score = 0.5  # Legal move but not the best
  else:
    score = 1.0  # Any legal move is fine

  # Use Stockfish to evaluate the position after the move (if available)
  sf_path = _find_stockfish()
  sf_eval = None
  if sf_path:
    try:
      engine = chess.engine.SimpleEngine.popen_uci(sf_path)
      board.push(parsed)
      info = engine.analyse(board, chess.engine.Limit(time=1.0))
      sf_eval = info.get("score")
      board.pop()
      engine.quit()
    except Exception:
      pass

  _REPORT_CACHE[(aiEngineName, subPass)] = {
    "type": "trivial", "pos": pos, "board_fen": pos["fen"],
    "engine_move": move_str, "error": "", "score": score,
    "sf_eval": str(sf_eval) if sf_eval else None,
    "moves": [(pos["fen"], move_str, True)],
  }

  detail = f"[{pos['name']}] Move: {move_str}"
  if sf_eval:
    detail += f" (eval: {sf_eval})"
  return score, detail


def _grade_puzzle(exe_path: str, subPass: int, aiEngineName: str):
  """Grade a puzzle position (subpass 5-19)."""
  puz = PUZZLES[subPass - 5]
  board = chess.Board(puz["fen"])
  moves_played = []
  sf_path = _find_stockfish()
  sf_engine = None

  if sf_path:
    try:
      sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception:
      sf_engine = None

  score = 0.0
  detail = ""
  total_half_moves = puz["max_moves"]
  correct_moves = 0

  try:
    for half_move in range(total_half_moves):
      if board.is_game_over():
        break

      if board.turn == (chess.WHITE if puz["fen"].split()[1] == "w" else chess.BLACK):
        # Tested engine's turn
        move_str, err = _run_engine_once(exe_path, board.fen(), timeout=15)
        if err:
          moves_played.append((board.fen(), None, False))
          detail = f"[{puz['name']}] Engine error on move {half_move+1}: {err}"
          break
        parsed = disambig_to_move(board, move_str)
        if parsed is None:
          moves_played.append((board.fen(), move_str, False))
          detail = f"[{puz['name']}] Invalid move on move {half_move+1}: {move_str}"
          break
        moves_played.append((board.fen(), move_str, True))
        board.push(parsed)
        correct_moves += 1
      else:
        # Opponent's turn - use Stockfish or solution moves
        if sf_engine:
          try:
            sf_result = sf_engine.play(board, chess.engine.Limit(time=0.5))
            opp_move = sf_result.move
          except Exception:
            opp_move = list(board.legal_moves)[0]
        else:
          opp_move = list(board.legal_moves)[0]
        opp_str = move_to_disambig(board, opp_move)
        moves_played.append((board.fen(), opp_str, True))
        board.push(opp_move)

    if board.is_checkmate():
      score = 1.0
      detail = f"[{puz['name']}] Checkmate found!"
    elif correct_moves > 0:
      score = correct_moves / max(1, (total_half_moves + 1) // 2)
      score = min(score, 0.9)  # Cap at 0.9 unless checkmate
      if not detail:
        detail = f"[{puz['name']}] {correct_moves} correct moves, no mate"
    else:
      if not detail:
        detail = f"[{puz['name']}] No correct moves"

  except Exception as e:
    detail = f"[{puz['name']}] Error: {str(e)[:200]}"
  finally:
    if sf_engine:
      try:
        sf_engine.quit()
      except Exception:
        pass

  _REPORT_CACHE[(aiEngineName, subPass)] = {
    "type": "puzzle", "puzzle": puz, "board_fen": puz["fen"],
    "moves": moves_played, "score": score,
    "final_fen": board.fen(),
  }
  return score, detail


def _grade_stockfish_game(exe_path: str, subPass: int, aiEngineName: str):
  """Grade a full game vs Stockfish (subpass 20-29)."""
  game_cfg = STOCKFISH_GAMES[subPass - 20]
  sf_path = _find_stockfish()
  if not sf_path:
    return 0.0, f"[{game_cfg['name']}] Stockfish not found - cannot grade"

  board = chess.Board()
  moves_played = []
  # Tested engine plays white
  max_game_moves = 200  # 200 half-moves = 100 full moves
  error = ""

  target_elo = game_cfg["elo"]
  # Stockfish minimum UCI_Elo is 1320 (Skill Level 0).  For lower targets
  # we run at Skill Level 0 and mix in random legal moves to simulate
  # weaker play.  random_move_prob is the chance of playing randomly.
  MIN_SF_ELO = 1320
  if target_elo >= MIN_SF_ELO:
    random_move_prob = 0.0
    sf_elo = target_elo
  else:
    random_move_prob = 1.0 - target_elo / MIN_SF_ELO
    sf_elo = MIN_SF_ELO

  try:
    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
 
    if sf_elo > 3190:
      sf_engine.configure({"UCI_LimitStrength": False, "Threads": min(32, os.cpu_count())})
    else:
      sf_engine.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})
  except Exception as e:
    return 0.0, f"[{game_cfg['name']}] Failed to start Stockfish: {str(e)[:200]}"

  try:
    for half_move in range(max_game_moves):
      if board.is_game_over():
        break

      if board.turn == chess.WHITE:
        # Tested engine's turn
        move_str, err = _run_engine_once(
          exe_path, board.fen(), timeout=game_cfg["time_per_move"])
        if err:
          moves_played.append((board.fen(), None, False))
          error = f"Engine error on move {half_move//2 + 1}: {err}"
          break
        parsed = disambig_to_move(board, move_str)
        if parsed is None:
          moves_played.append((board.fen(), move_str, False))
          error = f"Invalid move on move {half_move//2 + 1}: {move_str}"
          break
        moves_played.append((board.fen(), move_str, True))
        board.push(parsed)
      else:
        # Stockfish's turn (possibly weakened with random moves)
        try:
          if random_move_prob > 0 and random.random() < random_move_prob:
            legal = list(board.legal_moves)
            sf_move = random.choice(legal)
          else:
            sf_result = sf_engine.play(board, chess.engine.Limit(time=1.0))
            sf_move = sf_result.move
          sf_str = move_to_disambig(board, sf_move)
          moves_played.append((board.fen(), sf_str, True))
          board.push(sf_move)
        except Exception as e:
          error = f"Stockfish error: {str(e)[:200]}"
          break

  except Exception as e:
    error = f"Game error: {str(e)[:200]}"
  finally:
    try:
      sf_engine.quit()
    except Exception:
      pass

  # Score the game
  outcome = board.outcome()
  if error:
    score = 0.0
    detail = f"[{game_cfg['name']}] {error}"
  elif outcome is None:
    # Game didn't finish (draw by move limit)
    score = 0.3
    detail = f"[{game_cfg['name']}] Draw (move limit) after {len(moves_played)} half-moves"
  elif outcome.winner == chess.WHITE:
    score = 1.0
    detail = f"[{game_cfg['name']}] WIN as White! ({outcome.termination.name})"
  elif outcome.winner == chess.BLACK:
    score = 0.0
    detail = f"[{game_cfg['name']}] LOSS ({outcome.termination.name})"
  else:
    # Draw
    score = 0.4
    detail = f"[{game_cfg['name']}] Draw ({outcome.termination.name})"

  _REPORT_CACHE[(aiEngineName, subPass)] = {
    "type": "game", "game_cfg": game_cfg, "board_fen": chess.STARTING_FEN,
    "moves": moves_played, "score": score,
    "final_fen": board.fen(), "error": error,
    "outcome": outcome.termination.name if outcome else "incomplete",
  }
  return score, detail


# ---------------------------------------------------------------------------
# resultToNiceReport - chess board visualization with move playback
# ---------------------------------------------------------------------------

_BTN_STYLE = ("padding:4px 10px;border:1px solid #334155;border-radius:4px;"
              "background:#1e293b;color:#e2e8f0;cursor:pointer;font-size:14px;")


def _board_svg(board: chess.Board, last_move: str = None, size: int = 350) -> str:
  """Generate an SVG of the board position."""
  try:
    return chess.svg.board(board, size=size)
  except Exception:
    return "<div style='color:#ef4444;'>SVG generation failed</div>"


def _build_report_html(report, board_states, moves, score, error):
  """Build the HTML report with JS playback controls.

  Uses string concatenation instead of f-strings to avoid brace-escaping
  headaches with embedded JavaScript.
  """
  # Determine title and status
  rtype = report["type"]
  if rtype == "trivial":
    title_text = escape(report["pos"]["name"])
  elif rtype == "puzzle":
    title_text = escape(report["puzzle"]["name"])
  else:
    title_text = escape(report["game_cfg"]["name"])

  if score >= 1.0:
    sc, status = "#22c55e", "PASS"
  elif score > 0:
    sc, status = "#f59e0b", "PARTIAL"
  else:
    sc, status = "#ef4444", "FAIL"

  frames_json = json.dumps([
    {"svg": s["svg"], "label": s["label"]} for s in board_states
  ])

  uid = "chess_" + str(id(report))

  error_html = ""
  if error:
    error_html = (
      "<div style='color:#ef4444;font-size:12px;margin-top:6px;'>"
      + escape(str(error)) + "</div>"
    )

  last_idx = len(board_states) - 1
  score_pct = f"{score:.0%}"

  # Build HTML with JS using plain string concat to avoid brace issues
  parts = []
  parts.append("<div style='margin:10px 0;padding:14px;border:1px solid #1f2937;")
  parts.append("            border-radius:8px;background:#0f172a;'>")
  parts.append(f"  <div style='font-weight:600;color:#e2e8f0;font-size:14px;margin-bottom:2px;'>")
  parts.append(f"    {title_text}</div>")
  parts.append(f"  <div style='font-size:12px;color:#64748b;margin-bottom:10px;'>")
  parts.append(f"    {len(moves)} moves &middot;")
  parts.append(f"    <span style='color:{sc};font-weight:700;'>{status} ({score_pct})</span></div>")
  parts.append(f"  {error_html}")
  parts.append(f"  <div id='{uid}_board' style='width:360px;height:360px;margin:8px 0;'></div>")
  parts.append(f"  <div style='display:flex;align-items:center;gap:8px;margin-top:6px;'>")
  parts.append(f"    <button onclick='{uid}_go(0)' style='{_BTN_STYLE}'>&#x23EE;</button>")
  parts.append(f"    <button onclick='{uid}_step(-1)' style='{_BTN_STYLE}'>&larr;</button>")
  parts.append(f"    <button onclick='{uid}_toggle()' id='{uid}_play' style='{_BTN_STYLE}'>&#x25B6;</button>")
  parts.append(f"    <button onclick='{uid}_step(1)' style='{_BTN_STYLE}'>&rarr;</button>")
  parts.append(f"    <button onclick='{uid}_go({last_idx})' style='{_BTN_STYLE}'>&#x23ED;</button>")
  parts.append(f"    <input type='range' min='0' max='{last_idx}' value='0'")
  parts.append(f"           id='{uid}_slider' oninput='{uid}_go(+this.value)'")
  parts.append(f"           style='flex:1;accent-color:#3b82f6;'/>")
  parts.append(f"    <span id='{uid}_lbl' style='font-size:11px;color:#94a3b8;min-width:80px;'>Start</span>")
  parts.append(f"  </div>")

  # JavaScript - use a separate string to keep braces clean
  js = """
  <script>
  (function(){
    var frames = """ + frames_json + """;
    var idx = 0;
    var playing = false;
    var timer = null;
    var boardEl = document.getElementById('""" + uid + """_board');
    var slider = document.getElementById('""" + uid + """_slider');
    var lbl = document.getElementById('""" + uid + """_lbl');
    var playBtn = document.getElementById('""" + uid + """_play');
    function show(i){
      idx = Math.max(0, Math.min(i, frames.length-1));
      boardEl.innerHTML = frames[idx].svg;
      slider.value = idx;
      lbl.textContent = frames[idx].label;
    }
    window.""" + uid + """_go = function(i){ playing=false; clearInterval(timer); playBtn.innerHTML="&#x25B6;"; show(+i); };
    window.""" + uid + """_step = function(d){ playing=false; clearInterval(timer); playBtn.innerHTML="&#x25B6;"; show(idx+d); };
    window.""" + uid + """_toggle = function(){
      playing = !playing;
      if(playing){
        playBtn.innerHTML="&#x23F8;";
        timer = setInterval(function(){
          if(idx >= frames.length-1){ playing=false; clearInterval(timer); playBtn.innerHTML="&#x25B6;"; return; }
          show(idx+1);
        }, 800);
      } else {
        clearInterval(timer);
        playBtn.innerHTML="&#x25B6;";
      }
    };
    show(0);
  })();
  </script>
  """
  parts.append(js)
  parts.append("</div>")

  return "\n".join(parts)


def resultToNiceReport(result, subPass, aiEngineName):
  report = _REPORT_CACHE.get((aiEngineName, subPass))
  if not report:
    return "<div style='color:#94a3b8;'>No visualization data captured</div>"

  moves = report.get("moves", [])
  score = report.get("score", 0)
  error = report.get("error", "")

  # Build the board states for playback
  board_states = []
  initial_fen = report.get("board_fen", chess.STARTING_FEN)
  board = chess.Board(initial_fen)
  board_states.append({
    "fen": initial_fen,
    "svg": _board_svg(board),
    "label": "Start",
  })

  for i, (fen, move_str, valid) in enumerate(moves):
    b = chess.Board(fen)
    if move_str and valid:
      parsed = disambig_to_move(b, move_str)
      if parsed:
        b.push(parsed)
    label = f"{i+1}. {move_str or '?'}"
    if not valid:
      label += " (invalid)"
    board_states.append({
      "fen": b.fen(),
      "svg": _board_svg(b),
      "label": label,
    })

  htmlHeader = "FEN:<input type='text' id='fen' value='" + initial_fen + "'><br>"

  if subPass == 0:
    if "cpp_code" in result:
      code = result["cpp_code"]
      code_escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
      htmlHeader += f"<details><summary>View Code ({len(code)} chars)</summary><pre>{code_escaped}</pre></details><br>"


  return htmlHeader + _build_report_html(report, board_states, moves, score, error)

def setup():
  _find_stockfish()

highLevelSummary = """
<p>Write a C++ chess engine that, given any board position in FEN notation, returns
the best move it can find within a time limit. The engine is compiled once and then
called repeatedly &mdash; once per move &mdash; as it faces increasingly difficult
challenges.</p>
<p>Early subpasses test trivial tactics (hanging pieces, mate in one). Middle
subpasses present classic chess puzzles requiring short tactical sequences. Later
subpasses pit the engine against Stockfish at rising ELO levels, from absolute
beginner (100) to grandmaster strength (2500).</p>
"""
