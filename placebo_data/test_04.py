"""Test 4: Tetrahedron Packing - Placebo responses for all control types."""


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
  reasoning = 'Grid-based tetrahedron packing in C++. Axis-aligned grid, no rotation. Times out on dense packing.'
  code = '#include <iostream>\n#include <cmath>\n#include <vector>\nusing namespace std;\nint main() {\n    int nv; double edge;\n    cin >> nv;\n    vector<double> vx(nv), vy(nv), vz(nv);\n    for (int i = 0; i < nv; i++) cin >> vx[i] >> vy[i] >> vz[i];\n    cin >> edge;\n    double minx=1e18,miny=1e18,minz=1e18,maxx=-1e18,maxy=-1e18,maxz=-1e18;\n    for (int i = 0; i < nv; i++) {\n        minx=min(minx,vx[i]); miny=min(miny,vy[i]); minz=min(minz,vz[i]);\n        maxx=max(maxx,vx[i]); maxy=max(maxy,vy[i]); maxz=max(maxz,vz[i]);\n    }\n    double sp = edge * 1.1;\n    int count = 0;\n    for (double x=minx+edge; x<maxx-edge; x+=sp)\n        for (double y=miny+edge; y<maxy-edge; y+=sp)\n            for (double z=minz+edge; z<maxz-edge; z+=sp)\n                count++;\n    cout << count << endl;\n    for (double x=minx+edge; x<maxx-edge; x+=sp)\n        for (double y=miny+edge; y<maxy-edge; y+=sp)\n            for (double z=minz+edge; z<maxz-edge; z+=sp)\n                cout << x << " " << y << " " << z << " 0 0 0" << endl;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _naive_optimised(subpass):
  # TODO: Same algorithm as naive but hyper-optimised for C++
  # (parallel, SIMD, small data types, register packing, etc.)
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Torquato-Jiao dense packing (Torquato & Jiao 2009, Physical Review E 80(4):041104). "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Torquato-Jiao dense packing'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: pseudorandom output with fixed seed 42.'
  code = '#include <iostream>\n#include <random>\nusing namespace std;\nint main() {\n    mt19937 rng(42);\n    string line;\n    getline(cin, line);\n    cout << "0" << endl;\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for Tetrahedron Packing. Fill in the TODOs.'
  code = '// TODO: Human attempt at Tetrahedron Packing\n#include <iostream>\n#include <vector>\n#include <algorithm>\nusing namespace std;\nint main() {\n    ios_base::sync_with_stdio(false);\n    cin.tie(nullptr);\n    // TODO: Parse input\n    // TODO: Implement solution\n    // TODO: Output result\n    return 0;\n}'
  return {"reasoning": reasoning, 'cpp_code': code}, reasoning
