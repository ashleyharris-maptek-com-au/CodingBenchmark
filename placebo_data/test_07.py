"""Test 7: CSG Union - Placebo responses for all control types."""


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
  reasoning = 'Naive mesh concatenation for CSG union in C++. Not a true boolean - just merges vertices and faces.'
  code = '#include <iostream>\n#include <vector>\nusing namespace std;\nint main() {\n    int nv1, nf1; cin >> nv1 >> nf1;\n    vector<double> vx, vy, vz;\n    for (int i = 0; i < nv1; i++) { double x,y,z; cin>>x>>y>>z; vx.push_back(x); vy.push_back(y); vz.push_back(z); }\n    vector<int> fa, fb, fc;\n    for (int i = 0; i < nf1; i++) { int a,b,c; cin>>a>>b>>c; fa.push_back(a); fb.push_back(b); fc.push_back(c); }\n    int nv2, nf2; cin >> nv2 >> nf2;\n    int off = nv1;\n    for (int i = 0; i < nv2; i++) { double x,y,z; cin>>x>>y>>z; vx.push_back(x); vy.push_back(y); vz.push_back(z); }\n    for (int i = 0; i < nf2; i++) { int a,b,c; cin>>a>>b>>c; fa.push_back(a+off); fb.push_back(b+off); fc.push_back(c+off); }\n    cout << vx.size() << " " << fa.size() << endl;\n    for (size_t i = 0; i < vx.size(); i++) cout << vx[i] << " " << vy[i] << " " << vz[i] << "\\n";\n    for (size_t i = 0; i < fa.size(); i++) cout << fa[i] << " " << fb[i] << " " << fc[i] << "\\n";\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: BSP-tree CSG (Naylor et al. 1990, SIGGRAPH 24(4):115-124). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement BSP-tree CSG'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for CSG Union. Fill in the TODOs.'
  code = '// TODO: Human attempt at CSG Union\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
