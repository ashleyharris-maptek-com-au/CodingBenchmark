"""Test 1: TSP - Placebo responses for all control types."""


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
  reasoning = 'Nearest-neighbor heuristic in C++. O(n^2), no optimisation. Times out on large inputs.'
  code = """
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;
int main() {
    int n; cin >> n;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    vector<bool> vis(n, false);
    vis[0] = true;
    int cur = 0;
    cout << 0;
    for (int step = 1; step < n; step++) {
        int best = -1; double bd = 1e18;
        for (int j = 0; j < n; j++) {
            if (!vis[j]) {
                double d = sqrt((x[cur]-x[j])*(x[cur]-x[j])+(y[cur]-y[j])*(y[cur]-y[j]));
                if (d < bd) { bd = d; best = j; }
            }
        }
        vis[best] = true; cur = best;
        cout << " " << best;
    }
    cout << endl;
}
"""
  return code, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  # Not much points because it scores 0 due to edge crossings.
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lin-Kernighan heuristic (Lin & Kernighan 1973, Operations Research 21(2):498-516). "
    "Implements a Lin-Kernighan-inspired tour improvement: nearest-neighbor initialization, "
    "candidate lists, iterative 2-opt swaps, and a scalable fallback ordering for huge N."
  )
  code = r'''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

using namespace std;

static inline double dist2(const vector<double>& x, const vector<double>& y, int a, int b) {
    double dx = x[a] - x[b];
    double dy = y[a] - y[b];
    return dx * dx + dy * dy;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        cin >> x[i] >> y[i];
    }
    if (n <= 0) return 0;
    if (n == 1) {
        cout << 0 << "\n";
        return 0;
    }

    // For very large instances, use a fast deterministic ordering.
    if (n > 2000) {
        vector<int> idx;
        idx.reserve(n - 1);
        for (int i = 1; i < n; ++i) idx.push_back(i);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            if (x[a] == x[b]) return y[a] < y[b];
            return x[a] < x[b];
        });
        cout << 0;
        for (int v : idx) cout << ' ' << v;
        cout << '\n';
        return 0;
    }

    // Nearest-neighbor initialization
    vector<int> tour;
    tour.reserve(n);
    vector<char> used(n, 0);
    int cur = 0;
    tour.push_back(cur);
    used[cur] = 1;
    for (int step = 1; step < n; ++step) {
        int best = -1;
        double bd = 1e100;
        for (int j = 0; j < n; ++j) {
            if (!used[j]) {
                double d = dist2(x, y, cur, j);
                if (d < bd) {
                    bd = d;
                    best = j;
                }
            }
        }
        if (best < 0) break;
        used[best] = 1;
        tour.push_back(best);
        cur = best;
    }

    // Candidate lists (nearest neighbors)
    const int K = min(20, n - 1);
    vector<vector<int>> cand(n);
    for (int i = 0; i < n; ++i) {
        vector<pair<double,int>> dists;
        dists.reserve(n - 1);
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            dists.push_back({dist2(x, y, i, j), j});
        }
        nth_element(dists.begin(), dists.begin() + K, dists.end(),
                    [](const auto& a, const auto& b){ return a.first < b.first; });
        dists.resize(K);
        sort(dists.begin(), dists.end(),
             [](const auto& a, const auto& b){ return a.first < b.first; });
        cand[i].reserve(K);
        for (auto &p : dists) cand[i].push_back(p.second);
    }

    // Position lookup for 2-opt
    vector<int> pos(n, 0);
    for (int i = 0; i < n; ++i) pos[tour[i]] = i;

    auto reverse_segment = [&](int l, int r) {
        while (l < r) {
            swap(tour[l], tour[r]);
            pos[tour[l]] = l;
            pos[tour[r]] = r;
            ++l; --r;
        }
    };

    // Lin-Kernighan-inspired local search: iterative 2-opt on candidate edges
    const int MAX_ITERS = 50;
    for (int iter = 0; iter < MAX_ITERS; ++iter) {
        bool improved = false;
        for (int i = 0; i < n - 1 && !improved; ++i) {
            int a = tour[i];
            int b = tour[i + 1];
            for (int c : cand[a]) {
                int j = pos[c];
                if (j <= i + 1 || j >= n) continue;
                int d = tour[j + 1 < n ? j + 1 : 0];
                if (a == c || a == d || b == c || b == d) continue;
                double before = dist2(x, y, a, b) + dist2(x, y, c, d);
                double after  = dist2(x, y, a, c) + dist2(x, y, b, d);
                if (after + 1e-12 < before) {
                    reverse_segment(i + 1, j);
                    improved = true;
                    break;
                }
            }
        }
        if (!improved) break;
    }

    cout << tour[0];
    for (int i = 1; i < n; ++i) cout << ' ' << tour[i];
    cout << '\n';
    return 0;
}
'''
  return code, reasoning


def _random(subpass):
  reasoning = 'Random: seeded pseudorandom permutation starting at city 0 (seed 42).'
  code = """
#include <iostream>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;
int main() {
    mt19937 rng(42);
    int n; if (!(cin >> n)) return 0;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    if (n <= 0) return 0;
    vector<int> order;
    order.reserve(max(0, n-1));
    for (int i = 1; i < n; ++i) order.push_back(i);
    shuffle(order.begin(), order.end(), rng);
    cout << 0;
    for (int v : order) cout << ' ' << v;
    cout << endl;
    return 0;
}
"""
  return code, reasoning


def _human(subpass):
  reasoning = (
    'Human baseline: greedy nearest-neighbor tour, untangle crossings by reversing the '
    'shortest segment, then iterate local 7-chain exhaustive reorders (5! permutations) '
    'until ~298s or <2s remaining.'
  )
  code = r'''
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

static inline double dist2(const vector<double>& x, const vector<double>& y, int a, int b) {
    double dx = x[a] - x[b];
    double dy = y[a] - y[b];
    return dx * dx + dy * dy;
}

static inline long double orient(const pair<double,double>& a,
                                 const pair<double,double>& b,
                                 const pair<double,double>& c) {
    return (long double)(b.first - a.first) * (c.second - a.second)
         - (long double)(b.second - a.second) * (c.first - a.first);
}

static inline bool segments_cross(const pair<double,double>& a,
                                  const pair<double,double>& b,
                                  const pair<double,double>& c,
                                  const pair<double,double>& d) {
    long double o1 = orient(a, b, c);
    long double o2 = orient(a, b, d);
    long double o3 = orient(c, d, a);
    long double o4 = orient(c, d, b);
    const long double eps = 1e-12L;
    if (fabsl(o1) < eps || fabsl(o2) < eps || fabsl(o3) < eps || fabsl(o4) < eps) {
        return false; // ignore colinear/near-colinear cases
    }
    return (o1 > 0) != (o2 > 0) && (o3 > 0) != (o4 > 0);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) cin >> x[i] >> y[i];
    if (n <= 0) return 0;
    if (n == 1) {
        cout << 0 << '\n';
        return 0;
    }

    auto t0 = chrono::steady_clock::now();
    auto elapsed = [&]() {
        chrono::duration<double> d = chrono::steady_clock::now() - t0;
        return d.count();
    };
    auto time_left = [&]() {
        return 300.0 - elapsed();
    };

    // Greedy nearest-neighbor tour
    vector<int> tour;
    tour.reserve(n);
    vector<char> used(n, 0);
    int cur = 0;
    tour.push_back(cur);
    used[cur] = 1;
    for (int step = 1; step < n; ++step) {
        int best = -1;
        double bd = 1e100;
        for (int j = 0; j < n; ++j) {
            if (!used[j]) {
                double d = dist2(x, y, cur, j);
                if (d < bd) { bd = d; best = j; }
            }
        }
        if (best < 0) break;
        used[best] = 1;
        tour.push_back(best);
        cur = best;
    }

    // Untangle crossings: reverse shortest crossing segment (2-opt)
    auto untanglePass = [&]() {
      if (n >= 4) {
          bool improved = true;
          while (improved && time_left() > 2.0 && elapsed() < 298.0) {
              improved = false;
              int best_i = -1, best_j = -1;
              int best_len = n + 1;
              for (int i = 0; i < n - 2; ++i) {
                  if (time_left() <= 2.0 || elapsed() >= 298.0) break;
                  int a = tour[i];
                  int b = tour[i + 1];
                  pair<double,double> A{x[a], y[a]};
                  pair<double,double> B{x[b], y[b]};
                  for (int j = i + 2; j < n - 1; ++j) {
                      int c = tour[j];
                      int d = tour[j + 1];
                      pair<double,double> C{x[c], y[c]};
                      pair<double,double> D{x[d], y[d]};
                      if (segments_cross(A, B, C, D)) {
                          int len = j - i;
                          if (len < best_len) {
                              best_len = len;
                              best_i = i;
                              best_j = j;
                          }
                      }
                  }
              }
              if (best_i >= 0 && best_j >= 0) {
                  reverse(tour.begin() + best_i + 1, tour.begin() + best_j + 1);
                  improved = true;
              }
          }
      }
    };

    untanglePass();

    // Local chain optimization with escalating window size.
    // Start at chain_size=7 (5 interior nodes permuted = 5!).
    // After n random samples, increase chain_size by 1.
    // Handles wrap-around: tour is treated as a cycle.
    // If chain_size reaches n, do one exhaustive pass and stop.
    if (n >= 7) {
        mt19937 rng(42);
        int chain_size = 7;
        while (time_left() > chain_size*chain_size && elapsed() < 298.0 && chain_size <= n) {
            int interior = chain_size - 2; // nodes to permute
            int samples = (chain_size >= n) ? 1 : n;
            uniform_int_distribution<int> pick(0, n - 1);
            for (int s = 0; s < samples; ++s) {
                if (time_left() <= chain_size || elapsed() >= 298.0) goto done;

                int start_idx = (chain_size >= n) ? 0 : pick(rng);

                // Gather chain indices (cyclic)
                vector<int> chain(chain_size);
                for (int k = 0; k < chain_size; ++k)
                    chain[k] = (start_idx + k) % n;

                // Fixed endpoints: first and last of the chain
                int anchor_l = tour[chain[0]];
                int anchor_r = tour[chain[chain_size - 1]];

                // Interior nodes to permute
                vector<int> mid(interior);
                for (int k = 0; k < interior; ++k)
                    mid[k] = tour[chain[k + 1]];

                // Current cost of this chain segment
                double cur_cost = 0.0;
                cur_cost += dist2(x, y, anchor_l, mid[0]);
                for (int k = 0; k + 1 < interior; ++k)
                    cur_cost += dist2(x, y, mid[k], mid[k + 1]);
                cur_cost += dist2(x, y, mid[interior - 1], anchor_r);

                // Try all permutations of interior nodes
                vector<int> perm(mid);
                sort(perm.begin(), perm.end());
                double best_cost = 1e100;
                vector<int> best_perm(interior);
                do {
                    double cost = dist2(x, y, anchor_l, perm[0]);
                    for (int k = 0; k + 1 < interior; ++k)
                        cost += dist2(x, y, perm[k], perm[k + 1]);
                    cost += dist2(x, y, perm[interior - 1], anchor_r);
                    if (cost < best_cost) {
                        best_cost = cost;
                        for (int k = 0; k < interior; ++k) best_perm[k] = perm[k];
                    }
                } while (next_permutation(perm.begin(), perm.end()));

                if (best_cost + 1e-12 < cur_cost) {
                    for (int k = 0; k < interior; ++k)
                        tour[chain[k + 1]] = best_perm[k];
                }
            }
            if (chain_size >= n) goto done;
            ++chain_size;
            untanglePass();
        }
    }
    untanglePass();
    done:

    cout << tour[0];
    for (int i = 1; i < n; ++i) cout << ' ' << tour[i];
    cout << '\n';
    return 0;
}
'''
  return code, reasoning
