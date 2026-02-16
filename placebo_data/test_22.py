"""Test 22: Tetris - Placebo responses for all control types."""


def get_response(model_name, subpass):
  """Return (result_dict, reasoning_string) for the given control type."""
  if model_name == 'naive':
    return _naive(subpass)
  elif model_name == 'naive-optimised':
    return _naive_optimised(subpass)
  elif model_name == 'best-published':
    return _best_published(subpass)
  elif model_name == 'random':
    return _random(subpass)
  elif model_name == 'human':
    return _human(subpass)
  return None, ''


def _naive(subpass):
  reasoning = 'Using heuristic evaluation for Tetris placement: 1) For each piece, try all 4 rotations and all valid columns. 2) Simulate placing piece and evaluate resulting board state. 3) Score = 0.76*lines - 0.51*height - 0.36*holes - 0.18*bumpiness. 4) Pick placement with highest score. 5) Track board state and clear completed lines after each piece.'
  code = 'using System;\nusing System.Collections.Generic;\nusing System.Linq;\n\nclass Tetris {\n    static int width, height;\n    static bool[,] board;\n    \n    static readonly Dictionary<char, int[,][]> PIECES = new Dictionary<char, int[,][]> {\n        {\'I\', new[] {\n            new int[,] {{0,0},{0,1},{0,2},{0,3}},\n            new int[,] {{0,0},{1,0},{2,0},{3,0}},\n            new int[,] {{0,0},{0,1},{0,2},{0,3}},\n            new int[,] {{0,0},{1,0},{2,0},{3,0}}\n        }},\n        {\'O\', new[] {\n            new int[,] {{0,0},{0,1},{1,0},{1,1}},\n            new int[,] {{0,0},{0,1},{1,0},{1,1}},\n            new int[,] {{0,0},{0,1},{1,0},{1,1}},\n            new int[,] {{0,0},{0,1},{1,0},{1,1}}\n        }},\n        {\'T\', new[] {\n            new int[,] {{0,0},{0,1},{0,2},{1,1}},\n            new int[,] {{0,0},{1,0},{2,0},{1,1}},\n            new int[,] {{1,0},{1,1},{1,2},{0,1}},\n            new int[,] {{0,1},{1,0},{1,1},{2,1}}\n        }},\n        {\'S\', new[] {\n            new int[,] {{0,0},{0,1},{1,1},{1,2}},\n            new int[,] {{0,1},{1,0},{1,1},{2,0}},\n            new int[,] {{0,0},{0,1},{1,1},{1,2}},\n            new int[,] {{0,1},{1,0},{1,1},{2,0}}\n        }},\n        {\'Z\', new[] {\n            new int[,] {{0,1},{0,2},{1,0},{1,1}},\n            new int[,] {{0,0},{1,0},{1,1},{2,1}},\n            new int[,] {{0,1},{0,2},{1,0},{1,1}},\n            new int[,] {{0,0},{1,0},{1,1},{2,1}}\n        }},\n        {\'J\', new[] {\n            new int[,] {{0,0},{1,0},{1,1},{1,2}},\n            new int[,] {{0,0},{0,1},{1,0},{2,0}},\n            new int[,] {{0,0},{0,1},{0,2},{1,2}},\n            new int[,] {{0,1},{1,1},{2,0},{2,1}}\n        }},\n        {\'L\', new[] {\n            new int[,] {{0,2},{1,0},{1,1},{1,2}},\n            new int[,] {{0,0},{1,0},{2,0},{2,1}},\n            new int[,] {{0,0},{0,1},{0,2},{1,0}},\n            new int[,] {{0,0},{0,1},{1,1},{2,1}}\n        }}\n    };\n    \n    static int GetPieceWidth(char piece, int rot) {\n        var coords = PIECES[piece][rot % 4];\n        int max = 0;\n        for (int i = 0; i < 4; i++) max = Math.Max(max, coords[i, 1]);\n        return max + 1;\n    }\n    \n    static int GetColumnHeight(int col) {\n        for (int r = height - 1; r >= 0; r--)\n            if (board[r, col]) return r + 1;\n        return 0;\n    }\n    \n    static int FindLandingRow(char piece, int rot, int col) {\n        var coords = PIECES[piece][rot % 4];\n        for (int startRow = height - 1; startRow >= 0; startRow--) {\n            bool canPlace = true;\n            for (int i = 0; i < 4 && canPlace; i++) {\n                int r = startRow + coords[i, 0];\n                int c = col + coords[i, 1];\n                if (c < 0 || c >= width) canPlace = false;\n                else if (r >= 0 && r < height && board[r, c]) canPlace = false;\n            }\n            if (!canPlace) return startRow + 1;\n        }\n        return 0;\n    }\n    \n    static double Evaluate(char piece, int rot, int col) {\n        var coords = PIECES[piece][rot % 4];\n        int pw = GetPieceWidth(piece, rot);\n        if (col < 0 || col + pw > width) return double.MinValue;\n        \n        int landRow = FindLandingRow(piece, rot, col);\n        \n        // Simulate placement\n        var tempBoard = (bool[,])board.Clone();\n        int maxRow = 0;\n        for (int i = 0; i < 4; i++) {\n            int r = landRow + coords[i, 0];\n            int c = col + coords[i, 1];\n            if (r >= height) return double.MinValue;\n            tempBoard[r, c] = true;\n            maxRow = Math.Max(maxRow, r);\n        }\n        \n        // Count lines cleared\n        int lines = 0;\n        for (int r = 0; r < height; r++) {\n            bool full = true;\n            for (int c = 0; c < width && full; c++)\n                if (!tempBoard[r, c]) full = false;\n            if (full) lines++;\n        }\n        \n        // Calculate aggregate height\n        int aggHeight = 0;\n        for (int c = 0; c < width; c++) {\n            for (int r = height - 1; r >= 0; r--) {\n                if (tempBoard[r, c]) { aggHeight += r + 1; break; }\n            }\n        }\n        \n        // Count holes\n        int holes = 0;\n        for (int c = 0; c < width; c++) {\n            bool foundBlock = false;\n            for (int r = height - 1; r >= 0; r--) {\n                if (tempBoard[r, c]) foundBlock = true;\n                else if (foundBlock) holes++;\n            }\n        }\n        \n        // Calculate bumpiness\n        int bumpiness = 0;\n        int[] colHeights = new int[width];\n        for (int c = 0; c < width; c++) {\n            for (int r = height - 1; r >= 0; r--) {\n                if (tempBoard[r, c]) { colHeights[c] = r + 1; break; }\n            }\n        }\n        for (int c = 0; c < width - 1; c++)\n            bumpiness += Math.Abs(colHeights[c] - colHeights[c + 1]);\n        \n        return 0.76 * lines - 0.51 * aggHeight - 0.36 * holes - 0.18 * bumpiness;\n    }\n    \n    static (int rot, int col) FindBestMove(char piece) {\n        double bestScore = double.MinValue;\n        int bestRot = 0, bestCol = 0;\n        \n        for (int rot = 0; rot < 4; rot++) {\n            int pw = GetPieceWidth(piece, rot);\n            for (int col = 0; col <= width - pw; col++) {\n                double score = Evaluate(piece, rot, col);\n                if (score > bestScore) {\n                    bestScore = score;\n                    bestRot = rot;\n                    bestCol = col;\n                }\n            }\n        }\n        return (bestRot, bestCol);\n    }\n    \n    static void PlacePiece(char piece, int rot, int col) {\n        var coords = PIECES[piece][rot % 4];\n        int landRow = FindLandingRow(piece, rot, col);\n        for (int i = 0; i < 4; i++) {\n            int r = landRow + coords[i, 0];\n            int c = col + coords[i, 1];\n            if (r < height) board[r, c] = true;\n        }\n        // Clear lines\n        for (int r = 0; r < height; ) {\n            bool full = true;\n            for (int c = 0; c < width && full; c++)\n                if (!board[r, c]) full = false;\n            if (full) {\n                for (int rr = r; rr < height - 1; rr++)\n                    for (int c = 0; c < width; c++)\n                        board[rr, c] = board[rr + 1, c];\n                for (int c = 0; c < width; c++)\n                    board[height - 1, c] = false;\n            } else r++;\n        }\n    }\n    \n    static void Main() {\n        var parts = Console.ReadLine().Split();\n        width = int.Parse(parts[0]);\n        height = int.Parse(parts[1]);\n        int numPieces = int.Parse(parts[2]);\n        \n        board = new bool[height, width];\n        \n        for (int i = 0; i < numPieces; i++) {\n            string line = Console.ReadLine();\n            if (string.IsNullOrEmpty(line)) break;\n            char piece = line.Trim()[0];\n            \n            var (rot, col) = FindBestMove(piece);\n            Console.WriteLine($"{rot} {col}");\n            Console.Out.Flush();\n            \n            PlacePiece(piece, rot, col);\n            \n            string result = Console.ReadLine();\n            if (result != null && result.StartsWith("gameover")) break;\n        }\n    }\n}\n'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C#
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: El-Tetris (Thiery & Scherrer 2009, ICGA Journal 32(1):3-11). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement El-Tetris'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        var rng = new Random(42);\n        Console.ReadLine();\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Tetris. Fill in the TODOs.'
  code = '// TODO: Human attempt at Tetris\nusing System;\nusing System.Collections.Generic;\nusing System.Linq;\nclass Program {\n    static void Main() {\n        // TODO: Parse input\n        // TODO: Implement solution\n        // TODO: Output result\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
