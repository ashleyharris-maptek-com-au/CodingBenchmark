"""Test 21: N-D Snake Game - Placebo responses for all control types."""


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
  reasoning = 'Using greedy pathfinding for N-dimensional snake: 1) Parse game setup: dimensions, bounds, food, obstacles. 2) Each turn, find nearest food using Euclidean distance. 3) Generate all 2*D possible moves (each axis, each direction). 4) Filter to valid moves (not wall, obstacle, or self). 5) Pick move that minimizes distance to target food. 6) If stuck, pick random valid move. 7) Track snake body and update after each move.'
  code = '#include <iostream>\n#include <vector>\n#include <set>\n#include <queue>\n#include <cmath>\n#include <algorithm>\n#include <random>\nusing namespace std;\n\nint D, maxTurns;\nvector<int> bounds;\nvector<vector<int>> food;\nset<vector<int>> obstacles;\ndeque<vector<int>> snake;\n\nmt19937 rng(42);\n\ndouble distance(const vector<int>& a, const vector<int>& b) {\n    double sum = 0;\n    for (int i = 0; i < D; i++) {\n        sum += (a[i] - b[i]) * (a[i] - b[i]);\n    }\n    return sqrt(sum);\n}\n\nbool isValid(const vector<int>& pos) {\n    for (int i = 0; i < D; i++) {\n        if (pos[i] < 0 || pos[i] >= bounds[i]) return false;\n    }\n    if (obstacles.count(pos)) return false;\n    // Check snake body (except tail which will move)\n    for (size_t i = 0; i + 1 < snake.size(); i++) {\n        if (snake[i] == pos) return false;\n    }\n    return true;\n}\n\npair<int, int> findMove() {\n    vector<int> head = snake.front();\n    \n    // Find nearest food\n    vector<int>* target = nullptr;\n    double minDist = 1e18;\n    for (auto& f : food) {\n        double d = distance(head, f);\n        if (d < minDist) {\n            minDist = d;\n            target = &f;\n        }\n    }\n    \n    if (!target) {\n        // No food, just survive - pick random valid move\n        vector<pair<int,int>> moves;\n        for (int axis = 0; axis < D; axis++) {\n            for (int dir : {-1, 1}) {\n                vector<int> newPos = head;\n                newPos[axis] += dir;\n                if (isValid(newPos)) {\n                    moves.push_back({axis, dir});\n                }\n            }\n        }\n        if (moves.empty()) return {0, 1}; // No valid move, will die\n        return moves[rng() % moves.size()];\n    }\n    \n    // Greedy: move toward target\n    vector<pair<double, pair<int,int>>> candidates;\n    for (int axis = 0; axis < D; axis++) {\n        for (int dir : {-1, 1}) {\n            vector<int> newPos = head;\n            newPos[axis] += dir;\n            if (isValid(newPos)) {\n                double newDist = distance(newPos, *target);\n                candidates.push_back({newDist, {axis, dir}});\n            }\n        }\n    }\n    \n    if (candidates.empty()) {\n        // No valid moves, try any move\n        return {0, 1};\n    }\n    \n    // Sort by distance to target\n    sort(candidates.begin(), candidates.end());\n    return candidates[0].second;\n}\n\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    \n    cin >> D >> maxTurns;\n    bounds.resize(D);\n    for (int i = 0; i < D; i++) cin >> bounds[i];\n    \n    int snakeLen;\n    cin >> snakeLen;\n    vector<int> headPos(D);\n    for (int i = 0; i < D; i++) cin >> headPos[i];\n    \n    // Initialize snake\n    for (int i = 0; i < snakeLen; i++) {\n        vector<int> pos = headPos;\n        pos[0] -= i;\n        if (pos[0] >= 0) snake.push_back(pos);\n    }\n    \n    int F;\n    cin >> F;\n    food.resize(F);\n    for (int i = 0; i < F; i++) {\n        food[i].resize(D);\n        for (int j = 0; j < D; j++) cin >> food[i][j];\n    }\n    \n    int O;\n    cin >> O;\n    for (int i = 0; i < O; i++) {\n        vector<int> pos(D);\n        for (int j = 0; j < D; j++) cin >> pos[j];\n        obstacles.insert(pos);\n    }\n    \n    // Play game\n    for (int turn = 0; turn < maxTurns && !food.empty(); turn++) {\n        auto [axis, dir] = findMove();\n        cout << axis << " " << dir << "\\n";\n        cout.flush();\n        \n        // Update snake position\n        vector<int> newHead = snake.front();\n        newHead[axis] += dir;\n        snake.push_front(newHead);\n        \n        // Check if ate food\n        bool ate = false;\n        for (auto it = food.begin(); it != food.end(); ++it) {\n            if (*it == newHead) {\n                food.erase(it);\n                ate = true;\n                break;\n            }\n        }\n        if (!ate) {\n            snake.pop_back();\n        }\n    }\n    \n    return 0;\n}\n'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: A* with flood-fill heuristic (game-specific). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement A* with flood-fill heuristic'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for N-D Snake Game. Fill in the TODOs.'
  code = '// TODO: Human attempt at N-D Snake Game\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
