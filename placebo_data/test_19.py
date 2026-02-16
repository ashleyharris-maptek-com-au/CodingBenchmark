"""Test 19: 3D Voxel Mining - Placebo responses for all control types."""


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


_NAIVE_CODE = r"""using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

class Program
{
    static int XDim, YDim, ZDim;
    static byte[] voxels;
    static int[] heightmap;

    static int Idx(int x, int y, int z) => x * YDim * ZDim + y * ZDim + z;

    static void DecodeByte(byte b, out int mass, out int value)
    {
        mass = b & 0xF;
        value = b >> 4;
    }

    static bool InBounds(int x, int y) => x >= 0 && x < XDim && y >= 0 && y < YDim;

    static bool IsExposed(int x, int y, int z)
    {
        if (z + 1 >= ZDim || voxels[Idx(x, y, z + 1)] == 0) return true;
        if (x - 1 < 0 || voxels[Idx(x - 1, y, z)] == 0) return true;
        if (x + 1 >= XDim || voxels[Idx(x + 1, y, z)] == 0) return true;
        if (y - 1 < 0 || voxels[Idx(x, y - 1, z)] == 0) return true;
        if (y + 1 >= YDim || voxels[Idx(x, y + 1, z)] == 0) return true;
        return false;
    }

    static void RebuildHeightmap()
    {
        heightmap = new int[XDim * YDim];
        for (int x = 0; x < XDim; x++)
            for (int y = 0; y < YDim; y++)
            {
                int h = 0;
                for (int z = ZDim - 1; z >= 0; z--)
                {
                    if (voxels[Idx(x, y, z)] != 0) { h = z + 1; break; }
                }
                heightmap[x * YDim + y] = h;
            }
    }

    static int H(int x, int y) => heightmap[x * YDim + y];

    // Find a 4-connected path from (sx,sy) to (tx,ty) respecting slope<=1
    // Uses BFS on the XY grid
    static List<int[]> FindPath(int sx, int sy, int tx, int ty)
    {
        if (sx == tx && sy == ty) return new List<int[]>();
        var prev = new Dictionary<long, long>();
        var queue = new Queue<long>();
        long Key(int a, int b) => (long)a * YDim + b;
        queue.Enqueue(Key(sx, sy));
        prev[Key(sx, sy)] = -1;
        int[] dx = {-1, 1, 0, 0};
        int[] dy = {0, 0, -1, 1};
        while (queue.Count > 0)
        {
            long cur = queue.Dequeue();
            int cx = (int)(cur / YDim);
            int cy = (int)(cur % YDim);
            if (cx == tx && cy == ty)
            {
                // Reconstruct
                var path = new List<int[]>();
                long k = Key(tx, ty);
                while (k != Key(sx, sy))
                {
                    int px = (int)(k / YDim);
                    int py = (int)(k % YDim);
                    path.Add(new int[]{px, py});
                    k = prev[k];
                }
                path.Reverse();
                return path;
            }
            for (int d = 0; d < 4; d++)
            {
                int nx = cx + dx[d];
                int ny = cy + dy[d];
                if (!InBounds(nx, ny)) continue;
                long nk = Key(nx, ny);
                if (prev.ContainsKey(nk)) continue;
                if (Math.Abs(H(nx, ny) - H(cx, cy)) <= 1)
                {
                    prev[nk] = Key(cx, cy);
                    queue.Enqueue(nk);
                }
            }
        }
        return null; // no path found
    }

    // Find the shallowest ore voxel (smallest depth below surface)
    static bool FindShallowestOre(out int ox, out int oy, out int oz)
    {
        ox = oy = oz = -1;
        int bestDepth = int.MaxValue;
        for (int x = 0; x < XDim; x++)
            for (int y = 0; y < YDim; y++)
            {
                int h = H(x, y);
                for (int z = h - 1; z >= 0; z--)
                {
                    byte b = voxels[Idx(x, y, z)];
                    if (b == 0) continue;
                    int mass, val;
                    DecodeByte(b, out mass, out val);
                    if (val > 0)
                    {
                        int depth = h - 1 - z;
                        if (depth < bestDepth)
                        {
                            bestDepth = depth;
                            ox = x; oy = y; oz = z;
                        }
                        break; // found ore in this column, no need to go deeper
                    }
                }
            }
        return ox >= 0;
    }

    // Dig the top voxel at (x,y) and dump it. Returns the operation string or null.
    static string DigTop(int x, int y, int dumpX, int dumpY)
    {
        int h = H(x, y);
        if (h <= 0) return null;
        int z = h - 1;
        byte b = voxels[Idx(x, y, z)];
        if (b == 0) return null;

        int mass, val;
        DecodeByte(b, out mass, out val);

        // Remove voxel
        voxels[Idx(x, y, z)] = 0;
        heightmap[x * YDim + y] = z; // update height

        bool toPlant = (dumpX == 0 && dumpY == 0 && val > 0);

        // Find path
        var path = FindPath(x, y, dumpX, dumpY);
        if (path == null)
        {
            // Can't reach destination - dump adjacent instead
            // Try to dump on self (just remove it effectively)
            return null;
        }

        // Build operation string
        // Format: N digX digY digZ step1X step1Y ... destX destY
        var sb = new StringBuilder();
        // path includes all intermediate + destination
        // We need: intermediate steps (path minus last) + dest
        int nSteps = path.Count - 1;
        if (nSteps < 0) nSteps = 0;

        if (path.Count == 0)
        {
            // Same position
            sb.Append("0 ");
            sb.Append(x); sb.Append(' ');
            sb.Append(y); sb.Append(' ');
            sb.Append(z); sb.Append(' ');
            sb.Append(dumpX); sb.Append(' ');
            sb.Append(dumpY);
        }
        else
        {
            sb.Append(path.Count - 1); sb.Append(' ');
            sb.Append(x); sb.Append(' ');
            sb.Append(y); sb.Append(' ');
            sb.Append(z); sb.Append(' ');
            for (int i = 0; i < path.Count - 1; i++)
            {
                sb.Append(path[i][0]); sb.Append(' ');
                sb.Append(path[i][1]); sb.Append(' ');
            }
            var last = path[path.Count - 1];
            sb.Append(last[0]); sb.Append(' ');
            sb.Append(last[1]);
        }

        // If not going to plant, dump raises ground at destination
        if (!toPlant)
        {
            int dh = H(dumpX, dumpY);
            if (dh < ZDim)
            {
                voxels[Idx(dumpX, dumpY, dh)] = b;
                heightmap[dumpX * YDim + dumpY] = dh + 1;
            }
        }

        return sb.ToString();
    }

    static void Main()
    {
        // Read header
        string header = "";
        int ch;
        while ((ch = Console.In.Read()) != -1 && ch != '\n')
            header += (char)ch;

        var parts = header.Trim().Split(' ');
        XDim = int.Parse(parts[0]);
        YDim = int.Parse(parts[1]);
        ZDim = int.Parse(parts[2]);

        // Read voxel bytes from stdin
        int totalVoxels = XDim * YDim * ZDim;
        voxels = new byte[totalVoxels];
        using (var stdin = Console.OpenStandardInput())
        {
            int read = 0;
            while (read < totalVoxels)
            {
                int n = stdin.Read(voxels, read, totalVoxels - read);
                if (n <= 0) break;
                read += n;
            }
        }

        RebuildHeightmap();

        var ops = new List<string>();
        int maxOps = Math.Min(XDim * YDim * ZDim, 50000);

        // Strategy: find shallowest ore, dig straight down to it from surface,
        // dumping waste far from plant. Deliver ore to plant.
        for (int iter = 0; iter < maxOps; iter++)
        {
            int ox, oy, oz;
            if (!FindShallowestOre(out ox, out oy, out oz)) break;

            int curH = H(ox, oy);
            if (curH <= 0) break;

            // Dig from top down to the ore voxel
            while (H(ox, oy) > oz)
            {
                int z = H(ox, oy) - 1;
                byte b = voxels[Idx(ox, oy, z)];
                if (b == 0) break;

                int mass, val;
                DecodeByte(b, out mass, out val);

                if (z == oz && val > 0)
                {
                    // This is the ore - send to plant
                    string op = DigTop(ox, oy, 0, 0);
                    if (op != null) ops.Add(op);
                    else break;
                }
                else
                {
                    // Waste - dump far from plant, shrinking distance to avoid spike
                    // Pick dump location on far edge
                    int dumpX = XDim - 1;
                    int dumpY = YDim - 1;
                    // Shrink toward centre to avoid making an unclimbable pile
                    int spread = ops.Count % Math.Max(1, (XDim + YDim) / 2);
                    dumpX = Math.Max(XDim / 2, XDim - 1 - spread % (XDim / 2 + 1));
                    dumpY = Math.Max(YDim / 2, YDim - 1 - spread / (XDim / 2 + 1) % (YDim / 2 + 1));

                    string op = DigTop(ox, oy, dumpX, dumpY);
                    if (op != null) ops.Add(op);
                    else break; // can't path - give up on this column
                }

                if (ops.Count >= maxOps) break;
            }
            if (ops.Count >= maxOps) break;
        }

        // Output
        Console.WriteLine(ops.Count);
        foreach (var op in ops)
            Console.WriteLine(op);
    }
}
"""


def _naive(subpass):
  reasoning = (
    'Naive greedy voxel mining in C#: '
    '1) Find the shallowest ore voxel (closest to surface). '
    '2) Dig straight down from the surface to reach it, removing each voxel from top. '
    '3) Dump waste rock far from the processing plant, spreading dumps to avoid spikes. '
    '4) Deliver ore to the processing plant at (0,0). '
    '5) Repeat until no more ore is reachable or op limit hit. '
    'Uses BFS pathfinding for each haul respecting slope constraint.'
  )
  return {"reasoning": reasoning, 'csharp_code': _NAIVE_CODE}, reasoning


def _naive_optimised(subpass):
  return _naive(subpass)


def _best_published(subpass):
  reasoning = (
    "Best published: Lerchs-Grossmann or pseudoflow pit optimisation. "
    "TODO: Full implementation pending."
  )
  code = '// TODO: Implement Lerchs-Grossmann pit optimisation'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _random(subpass):
  reasoning = 'Random: reads input and outputs zero operations.'
  code = 'using System;\nclass Program {\n    static void Main() {\n        int ch; while ((ch = Console.In.Read()) != -1 && ch != (int)\'\\n\') {}\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning


def _human(subpass):
  reasoning = 'Human starting point for 3D Voxel Mining. Fill in the TODOs.'
  code = '// TODO: Human attempt at 3D Voxel Mining\nusing System;\nusing System.Collections.Generic;\nusing System.IO;\nclass Program {\n    static void Main() {\n        // TODO: Parse binary voxel input (header line + raw bytes)\n        // TODO: Find valuable ore bodies\n        // TODO: Plan dig operations respecting exposure and slope constraints\n        // TODO: Output operations\n        Console.WriteLine("0");\n    }\n}'
  return {"reasoning": reasoning, 'csharp_code': code}, reasoning
