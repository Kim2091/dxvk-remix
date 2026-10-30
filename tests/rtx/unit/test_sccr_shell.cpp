/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

// Checks the shell adjacency build (sccr_shell_adjacency.comp.slang, copied below on the
// CPU; the GPU's atomics run one thread at a time here) on small meshes: a cube with split vertices per face (every
// edge welds, the face edges across split copies, corner directions along the diagonals), the same cube without an
// index buffer, an open cylinder with a UV seam (seam welded, rims open), a fin of three triangles on one edge
// (non-manifold), two triangles with inconsistent winding (flipped), and a two-sided sheet (degenerate directions).
// Every neighbour relation must be symmetric. Hard corners: the boundary kinds; the weld grid
// (weldEpsilon) on a cube whose split copies drift in the last bits; the fold guard's limits (phase 5 of the build) on
// cube corners, a subdivided cube's edges, a concave L (an open floor and wall, and a closed L-shaped block), a thin slab
// (low-poly: its corners cap the depth below half the thickness, so the relief cannot punch through; subdivided: the
// interior of its big faces is unlimited, so a depth above the thickness punches through there, risk R5) and the
// cylinder; and, for every mesh, that prisms built with depths clamped to the limits stay valid (the march's own
// determinant test) at any material depth. The CPU copy of the build is rtx_fork_sccr_shell_cpu.cpp; keep it in step
// with the shader and sccr_shell_args.h.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../../src/dxvk/rtx_render/rtx_fork_sccr_shell_cpu.h"

namespace {

using namespace dxvk::sccr_shell_cpu;

int g_failures = 0;

void check(bool condition, const std::string& what) {
  if (!condition) {
    ++g_failures;
    std::cerr << "FAILED: " << what << std::endl;
  }
}

// Every neighbour relation points back, and the shared edge's ends are the same positions in both triangles (with
// exactPositions; a grid weld joins positions that differ in their last bits).
void checkSymmetric(const Mesh& mesh, const Result& r, const std::string& name, bool exactPositions = true) {
  const auto vertexOf = [&](uint32_t t, uint32_t k) { return mesh.indices.empty() ? t * 3 + k : mesh.indices[t * 3 + k]; };
  const auto samePosition = [&](uint32_t a, uint32_t b) {
    const Float3 pa = mesh.positions[a];
    const Float3 pb = mesh.positions[b];
    return pa.x == pb.x && pa.y == pb.y && pa.z == pb.z;
  };
  for (uint32_t t = 0; t < r.triangles; ++t) {
    for (uint32_t e = 0; e < 3; ++e) {
      const uint32_t word = r.neighbour(t, e);
      if (isBoundary(word)) {
        continue;
      }
      const uint32_t n = word & kNeighbourTriangleMask;
      const uint32_t s = word >> kNeighbourSlotShift;
      check(n < r.triangles && s < 3, name + ": neighbour in range");
      if (n >= r.triangles || s >= 3) {
        continue;
      }
      const uint32_t back = r.neighbour(n, s);
      check(back == (t | (e << kNeighbourSlotShift)), name + ": neighbour " + std::to_string(t) + "/" + std::to_string(e) + " symmetric");
      // Our edge runs (e + 1) -> (e + 2); the neighbour's runs the other way.
      check(!exactPositions ||
            (samePosition(vertexOf(t, (e + 1) % 3), vertexOf(n, (s + 2) % 3)) &&
             samePosition(vertexOf(t, (e + 2) % 3), vertexOf(n, (s + 1) % 3))), name + ": shared edge ends match");
    }
  }
}

// The fold guard against the march (sccrShellPrismLoad and sccrShellJacobianInverse with an identity transform): at
// material depth D every vertex's shell vector is its welded direction times min(D, its limit) (against the direction:
// orientation +1; along it: -1). Every triangle the build did not skip (directions in front of its plane) must give a
// prism whose shell vector stays on the valid side of the triangle at every depth, with the march's margin
// (kSccrShellMinCosine), on a grid of points. Returns the smallest such cosine.
float worstPrismCosine(const Mesh& mesh, const Result& r, float depth, float orientation) {
  float worst = kInfinity;
  for (uint32_t t = 0; t < r.triangles; ++t) {
    Float3 p[3];
    Float3 n[3];
    Float3 sh[3];
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t v = mesh.indices.empty() ? t * 3 + k : mesh.indices[t * 3 + k];
      p[k] = mesh.positions[v];
      n[k] = decodeDirection(r.vertexWord(v, 0));
      const float limit = orientation > 0.0f ? r.depthLimits[v].first : r.depthLimits[v].second;
      sh[k] = n[k] * (orientation * std::min(depth, limit));
    }
    const Float3 e1 = p[1] - p[0];
    const Float3 e2 = p[2] - p[0];
    const Float3 normal = cross(e1, e2);
    if (dot(n[0], normal) <= 1e-3f * length(normal) || dot(n[1], normal) <= 1e-3f * length(normal) ||
        dot(n[2], normal) <= 1e-3f * length(normal)) {
      continue;
    }
    const Float3 f1 = sh[1] - sh[0];
    const Float3 f2 = sh[2] - sh[0];
    const uint32_t steps = 6;
    for (uint32_t i = 0; i <= steps; ++i) {
      for (uint32_t j = 0; i + j <= steps; ++j) {
        for (uint32_t k = 0; k <= steps; ++k) {
          const float b1 = float(i) / steps;
          const float b2 = float(j) / steps;
          const float h = float(k) / steps;
          const Float3 shell = sh[0] + f1 * b1 + f2 * b2;
          const Float3 r01 = cross(e1 - f1 * h, e2 - f2 * h);
          // sccrShellJacobianInverse: det = dot(-shell, r01), valid when det * detSign (= -orientation) is positive.
          const float det = dot(shell * -1.0f, r01);
          const float denominator = length(r01) * length(shell);
          worst = std::min(worst, denominator > 0.0f ? -orientation * det / denominator : -1.0f);
        }
      }
    }
  }
  return worst;
}

void checkFoldGuard(const Mesh& mesh, const Result& r, const std::string& name) {
  for (float depth : { 0.05f, 0.3f, 1.0f, 10.0f }) {
    for (float orientation : { 1.0f, -1.0f }) {
      const float worst = worstPrismCosine(mesh, r, depth, orientation);
      check(worst > 1e-3f, name + ": prisms valid at depth " + std::to_string(depth) + ", orientation " +
            std::to_string(int(orientation)) + " (worst cosine " + std::to_string(worst) + ")");
    }
  }
}

// A unit cube with 4 split vertices per face, two triangles per face, outward winding (counter-clockwise from outside).
Mesh makeCube(bool indexed) {
  struct Face { Float3 corner, u, v; };
  const Face faces[6] = {
    { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },   // +x
    { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },   // -x
    { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } },   // +y
    { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },   // -y
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },   // +z
    { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } },   // -z
  };
  Mesh mesh;
  for (const Face& f : faces) {
    const uint32_t base = uint32_t(mesh.positions.size());
    const auto at = [&](float a, float b) {
      return Float3 { f.corner.x + a * f.u.x + b * f.v.x, f.corner.y + a * f.u.y + b * f.v.y, f.corner.z + a * f.u.z + b * f.v.z };
    };
    mesh.positions.push_back(at(0, 0));
    mesh.positions.push_back(at(1, 0));
    mesh.positions.push_back(at(1, 1));
    mesh.positions.push_back(at(0, 1));
    const uint32_t quad[6] = { base, base + 1, base + 2, base, base + 2, base + 3 };
    mesh.indices.insert(mesh.indices.end(), quad, quad + 6);
  }
  if (!indexed) {
    Mesh flat;
    for (uint32_t i : mesh.indices) {
      flat.positions.push_back(mesh.positions[i]);
    }
    return flat;
  }
  return mesh;
}

void testCube(bool indexed) {
  const std::string name = indexed ? "cube" : "cube (non-indexed)";
  const Mesh mesh = makeCube(indexed);
  // Check the winding: every face normal points away from the centre.
  for (uint32_t t = 0; t < mesh.triangleCount(); ++t) {
    const auto p = [&](uint32_t k) { return mesh.positions[mesh.indices.empty() ? t * 3 + k : mesh.indices[t * 3 + k]]; };
    const Float3 c = cross(p(1) - p(0), p(2) - p(0));
    const Float3 toFace = p(0) - Float3 { 0.5f, 0.5f, 0.5f };
    check(dot(c, toFace) > 0.0f, name + ": test mesh winds outwards");
  }
  const Result r = build(mesh);
  check((r.flags() & 0xFFu) == kVersion, name + ": version");
  check(r.boundaryEdges() == 0, name + ": no boundary edges (got " + std::to_string(r.boundaryEdges()) + ")");
  check((r.flags() & (kFlagHasBoundary | kFlagWeldOverflow | kFlagNonManifold | kFlagFlippedWinding | kFlagDegenerateDirections)) == 0,
        name + ": no problem flags");
  checkSymmetric(mesh, r, name);
  for (uint32_t v = 0; v < r.vertices; ++v) {
    const Float3 p = mesh.positions[v];
    const Float3 expected { p.x - 0.5f, p.y - 0.5f, p.z - 0.5f };
    const Float3 d = decodeDirection(r.vertexWord(v, 0));
    const float cosine = dot(d, expected) / length(expected);
    check(cosine > 0.9999f, name + ": corner direction along the diagonal at vertex " + std::to_string(v));
    // Moved in along the diagonals, a face shrinks evenly by 2 d / sqrt(3) and collapses to a point at d = sqrt(3) / 2
    // (its normal only touches 0 there: the margin catches it); it keeps a quarter of its area, the margin, at half its
    // size: d = sqrt(3) / 4. Moved out it only grows.
    check(std::fabs(r.depthLimits[v].first - 0.4330127f) < 1e-5f, name + ": fold limit at a corner");
    check(r.depthLimits[v].second == kNoLimit, name + ": no fold limit outwards");
  }
  check((r.flags() & kFlagFoldsClamped) != 0, name + ": folds-clamped flag");
  check((r.flags() >> kWeldModeShift & 0xFu) == 0, name + ": exact weld mode");
  checkFoldGuard(mesh, r, name);
  // Every corner position has exactly one canonical copy.
  uint32_t canonical = 0;
  for (uint32_t v = 0; v < r.vertices; ++v) {
    canonical += (r.vertexWord(v, 1) & kVertexFlagCanonical) != 0 ? 1u : 0u;
  }
  check(canonical == 8, name + ": 8 canonical vertices (got " + std::to_string(canonical) + ")");
}

// An open cylinder: segments x 2 rings, the seam column duplicated (UV seam), outward winding.
void testCylinder() {
  const uint32_t segments = 12;
  Mesh mesh;
  for (uint32_t ring = 0; ring < 2; ++ring) {
    for (uint32_t s = 0; s <= segments; ++s) {
      // The last column repeats the first exactly (same bits), as a seam's split copies do.
      const uint32_t column = s % segments;
      const float angle = 2.0f * kPi * float(column) / float(segments);
      mesh.positions.push_back({ std::cos(angle), std::sin(angle), float(ring) });
    }
  }
  const uint32_t row = segments + 1;
  for (uint32_t s = 0; s < segments; ++s) {
    const uint32_t a = s, b = s + 1, c = row + s + 1, d = row + s;
    const uint32_t quad[6] = { a, b, c, a, c, d };
    mesh.indices.insert(mesh.indices.end(), quad, quad + 6);
  }
  const Result r = build(mesh);
  checkSymmetric(mesh, r, "cylinder");
  check(r.boundaryEdges() == 2 * segments, "cylinder: only the two rims are open (got " + std::to_string(r.boundaryEdges()) + ")");
  check((r.flags() & kFlagHasBoundary) != 0, "cylinder: boundary flag");
  check((r.flags() & (kFlagNonManifold | kFlagFlippedWinding | kFlagDegenerateDirections | kFlagWeldOverflow)) == 0,
        "cylinder: no other problem flags");
  // The seam: the first quad's second triangle (a, c, d) has column 0 (d -> a) at edge slot 1; across the split copies
  // it faces the last quad's first triangle (a, b, c), whose column (b -> c) is edge slot 0.
  check(r.neighbour(1, 1) == ((2 * (segments - 1)) | (0u << kNeighbourSlotShift)), "cylinder: the UV seam welds");
  // Directions point outwards, horizontally.
  for (uint32_t v = 0; v < r.vertices; ++v) {
    const Float3 p = mesh.positions[v];
    const Float3 d = decodeDirection(r.vertexWord(v, 0));
    check(dot(d, Float3 { p.x, p.y, 0.0f }) > 0.99f && std::fabs(d.z) < 0.75f, "cylinder: outward direction at vertex " + std::to_string(v));
    // Moved in along the radial directions, the polygon shrinks to a point at the radius (1); the quads only narrow
    // (their height stays), so they keep a quarter of their area at d = 0.75. Within 1e-3: the welded directions come
    // from a fixed-point sum (1e-4 of a corner's weight per unit, truncated), which tilts neighbouring directions by up
    // to about 1e-4 radians each and moves the limit by about 2e-4 (0.74984 here); the limit is stored as a half float,
    // whose step near 0.75 is 4.9e-4, anyway.
    check(std::fabs(r.depthLimits[v].first - 0.75f) < 1e-3f, "cylinder: fold limit at three quarters of the radius at vertex " + std::to_string(v));
    check(r.depthLimits[v].second == kNoLimit, "cylinder: no fold limit outwards");
  }
  checkFoldGuard(mesh, r, "cylinder");
}

void testFin() {
  // Three triangles on the edge (0,0,0)-(1,0,0).
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, -1, 0 } };
  mesh.indices = { 0, 1, 2, 1, 0, 3, 1, 0, 4 };
  const Result r = build(mesh);
  check((r.flags() & kFlagNonManifold) != 0, "fin: non-manifold flag");
  check(r.neighbour(0, 2) == kNeighbourNonManifold && r.neighbour(1, 2) == kNeighbourNonManifold &&
        r.neighbour(2, 2) == kNeighbourNonManifold, "fin: the shared edge is a non-manifold boundary in all three triangles");
  checkSymmetric(mesh, r, "fin");
}

void testFlipped() {
  // Two triangles sharing (0,0,0)-(1,0,0), both running from the first to the second.
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 } };
  mesh.indices = { 0, 1, 2, 0, 1, 3 };
  const Result r = build(mesh);
  check((r.flags() & kFlagFlippedWinding) != 0, "flipped: flag");
  check(r.neighbour(0, 2) == kNeighbourFlipped && r.neighbour(1, 2) == kNeighbourFlipped, "flipped: the shared edge is a flipped boundary");
  check(r.boundaryEdges() == 6, "flipped: every edge a boundary");
  check(r.neighbour(0, 0) == kNeighbourNone && r.neighbour(1, 1) == kNeighbourNone, "flipped: the outer edges open");
}

void testTwoSidedSheet() {
  // A quad and its back face on the same positions: the welded directions cancel.
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 } };
  mesh.indices = { 0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2 };
  const Result r = build(mesh);
  check((r.flags() & kFlagDegenerateDirections) != 0, "sheet: degenerate directions flag");
  for (uint32_t v = 0; v < r.vertices; ++v) {
    check((r.vertexWord(v, 1) & kVertexFlagDegenerate) != 0, "sheet: vertex " + std::to_string(v) + " degenerate");
    const Float3 d = decodeDirection(r.vertexWord(v, 0));
    check(std::fabs(std::fabs(d.z) - 1.0f) < 1e-3f, "sheet: fallback direction is a face normal");
  }
  // The diagonal carries four triangles; each outer edge welds a front and a back triangle, but its ends are
  // degenerate, so it is a boundary of that kind.
  check((r.flags() & kFlagNonManifold) != 0, "sheet: diagonal non-manifold");
  check(r.neighbour(0, 1) == kNeighbourNonManifold, "sheet: the diagonal is a non-manifold boundary");
  check(r.neighbour(0, 2) == kNeighbourDegenerate && r.neighbour(0, 0) == kNeighbourDegenerate &&
        r.neighbour(2, 1) == kNeighbourDegenerate, "sheet: the outer edges are degenerate boundaries");
  check(r.boundaryEdges() == 12, "sheet: no edge welds");
  checkSymmetric(mesh, r, "sheet");
}

// A cube whose split copies drift in the last bits (as exported meshes sometimes do) only welds with a grid.
void testWeldEpsilon() {
  Mesh mesh = makeCube(true);
  // Every split copy after the first of a position moves by a few float steps.
  for (uint32_t v = 0; v < mesh.positions.size(); ++v) {
    Float3& p = mesh.positions[v];
    const float drift = 1e-7f * float(1 + v % 3);
    p = { p.x + (v % 2 ? drift : -drift), p.y + drift, p.z - drift };
  }
  const Result exact = build(mesh);
  check(exact.boundaryEdges() > 0, "weld epsilon: drifted copies do not weld exactly (" + std::to_string(exact.boundaryEdges()) + " open edges)");
  // weldEpsilon 1e-5 of the bounding diagonal (sqrt(3)): the copies round to one grid point, the corners stay apart.
  const float cell = 1e-5f * std::sqrt(3.0f);
  const Result grid = build(mesh, cell);
  check(grid.boundaryEdges() == 0, "weld epsilon: the grid welds every edge (got " + std::to_string(grid.boundaryEdges()) + " open)");
  check((grid.flags() >> kWeldModeShift & 0xFu) == kWeldGrid, "weld epsilon: grid weld mode");
  checkSymmetric(mesh, grid, "weld epsilon", false);
}

// Six faces of a box [0, size.x] x [0, size.y] x [0, size.z], each split into sub x sub quads, split vertices per face,
// outward winding.
Mesh makeBox(Float3 size, uint32_t sub) {
  struct Face { Float3 corner, u, v; };
  const Face faces[6] = {
    { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
    { { 0, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 } },
    { { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 0 } },
    { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 1 } },
    { { 0, 0, 1 }, { 1, 0, 0 }, { 0, 1, 0 } },
    { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } },
  };
  Mesh mesh;
  for (const Face& f : faces) {
    const uint32_t base = uint32_t(mesh.positions.size());
    for (uint32_t i = 0; i <= sub; ++i) {
      for (uint32_t j = 0; j <= sub; ++j) {
        const float a = float(i) / float(sub);
        const float b = float(j) / float(sub);
        mesh.positions.push_back({ (f.corner.x + a * f.u.x + b * f.v.x) * size.x, (f.corner.y + a * f.u.y + b * f.v.y) * size.y,
                                   (f.corner.z + a * f.u.z + b * f.v.z) * size.z });
      }
    }
    for (uint32_t i = 0; i < sub; ++i) {
      for (uint32_t j = 0; j < sub; ++j) {
        const uint32_t a = base + i * (sub + 1) + j;
        const uint32_t b = base + (i + 1) * (sub + 1) + j;
        const uint32_t quad[6] = { a, b, b + 1, a, b + 1, a + 1 };
        mesh.indices.insert(mesh.indices.end(), quad, quad + 6);
      }
    }
  }
  return mesh;
}

// A subdivided cube (4 x 4 quads per face, spacing 0.25). Along an edge the triangles between the edge's
// vertices (directions at 45 degrees, at the corners along the diagonal) and the face's next row (the face normal)
// tip over once the edge vertices have moved in by about the spacing, which limits the edge vertices and that next row
// to below the spacing (0.14 to 0.19 here); the vertex in the middle of each face, whose triangles all have parallel
// directions, keeps no limit.
void testSubdividedCube() {
  const Mesh mesh = makeBox({ 1, 1, 1 }, 4);
  const Result r = build(mesh);
  check(r.boundaryEdges() == 0, "subdivided cube: closed");
  uint32_t unlimited = 0;
  for (uint32_t v = 0; v < r.vertices; ++v) {
    const Float3 p = mesh.positions[v];
    const uint32_t onFaces = (p.x == 0.0f || p.x == 1.0f ? 1u : 0u) + (p.y == 0.0f || p.y == 1.0f ? 1u : 0u) +
                             (p.z == 0.0f || p.z == 1.0f ? 1u : 0u);
    const float limit = r.depthLimits[v].first;
    if (onFaces == 1) {
      // Inside a face: unlimited unless it is in the row next to an edge (a coordinate at 0.25 or 0.75).
      const auto nearEdge = [](float c) { return std::fabs(c - 0.25f) < 1e-6f || std::fabs(c - 0.75f) < 1e-6f; };
      const bool nextToEdge = nearEdge(p.x) || nearEdge(p.y) || nearEdge(p.z);
      if (!nextToEdge) {
        check(limit == kNoLimit, "subdivided cube: face interior unlimited");
        ++unlimited;
      } else {
        check(limit > 0.1f && limit < kNoLimit, "subdivided cube: next to an edge, limited");
      }
    } else {
      check(limit > 0.1f && limit < 0.25f, "subdivided cube: on an edge, limited below the spacing (" + std::to_string(limit) + ")");
    }
  }
  check(unlimited > 0, "subdivided cube: some unlimited vertices");
  checkFoldGuard(mesh, r, "subdivided cube");
}

// A concave corner. An open floor (z = 0, normal +z) and wall (x = 1, normal -x) meeting at a concave edge
// along y, split vertices on the edge. Moved in (against the welded directions, into the corner's solid), the faces
// separate: no limit. Moved out, the floor's edge vertices (direction (-1, 0, 1) / sqrt(2)) run into the floor, whose
// corner heights fall as 1 - d (0 at d = 1): limit 0.75 everywhere (a quarter left).
void testConcaveL() {
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 1, 0, 1 }, { 1, 1, 1 }, { 1, 1, 0 } };
  mesh.indices = { 0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7 };
  const Result r = build(mesh);
  checkSymmetric(mesh, r, "concave L");
  // The floor and the wall weld across the concave edge (floor edge slot 0 of triangle 0, wall triangle 3).
  check(!isBoundary(r.neighbour(0, 0)) && (r.neighbour(0, 0) & kNeighbourTriangleMask) == 3, "concave L: the corner welds");
  for (uint32_t v = 0; v < r.vertices; ++v) {
    check(r.depthLimits[v].first == kNoLimit, "concave L: no limit inwards at vertex " + std::to_string(v));
    check(std::fabs(r.depthLimits[v].second - 0.75f) < 1e-4f, "concave L: limit 0.75 outwards at vertex " + std::to_string(v));
  }
  checkFoldGuard(mesh, r, "concave L");
}

// A closed L-shaped block (an L in xz extruded along y, split vertices per face, every face edge conforming):
// its reflex edge is concave, its other edges convex. Everything welds; every vertex has a finite limit inwards
// (convex edges and the cap corners); the prisms stay valid at any depth.
void testClosedL() {
  const float outline[8][2] = { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 1, 2 }, { 1, 1 }, { 2, 1 }, { 2, 0 }, { 1, 0 } };
  Mesh mesh;
  const auto addQuad = [&](Float3 a, Float3 b, Float3 c, Float3 d) {
    // Wound outwards: flip when the normal points into the block.
    const uint32_t base = uint32_t(mesh.positions.size());
    mesh.positions.insert(mesh.positions.end(), { a, b, c, d });
    const Float3 n = cross(b - a, c - a);
    const Float3 centre = (a + b + c + d) * 0.25f;
    const Float3 probe = centre + n * (1e-3f / length(n));
    const bool inside = probe.y > 0.0f && probe.y < 1.0f &&
      ((probe.x > 0.0f && probe.x < 1.0f && probe.z > 0.0f && probe.z < 2.0f) ||
       (probe.x >= 1.0f && probe.x < 2.0f && probe.z > 0.0f && probe.z < 1.0f));
    const uint32_t quad[6] = { base, base + 1, base + 2, base, base + 2, base + 3 };
    const uint32_t flipped[6] = { base, base + 2, base + 1, base, base + 3, base + 2 };
    mesh.indices.insert(mesh.indices.end(), inside ? flipped : quad, (inside ? flipped : quad) + 6);
  };
  for (uint32_t k = 0; k < 8; ++k) {
    const float* a = outline[k];
    const float* c = outline[(k + 1) % 8];
    addQuad({ a[0], 0, a[1] }, { c[0], 0, c[1] }, { c[0], 1, c[1] }, { a[0], 1, a[1] });
  }
  const float caps[3][4][2] = {
    { { 0, 0 }, { 1, 0 }, { 1, 1 }, { 0, 1 } }, { { 0, 1 }, { 1, 1 }, { 1, 2 }, { 0, 2 } }, { { 1, 0 }, { 2, 0 }, { 2, 1 }, { 1, 1 } } };
  for (float y : { 0.0f, 1.0f }) {
    for (const auto& q : caps) {
      addQuad({ q[0][0], y, q[0][1] }, { q[1][0], y, q[1][1] }, { q[2][0], y, q[2][1] }, { q[3][0], y, q[3][1] });
    }
  }
  const Result r = build(mesh);
  check(r.boundaryEdges() == 0, "closed L: closed (" + std::to_string(r.boundaryEdges()) + " boundary edges)");
  checkSymmetric(mesh, r, "closed L");
  for (uint32_t v = 0; v < r.vertices; ++v) {
    check(r.depthLimits[v].first > 0.0f && r.depthLimits[v].first < kNoLimit, "closed L: limited inwards at vertex " + std::to_string(v));
  }
  checkFoldGuard(mesh, r, "closed L");
}

// Thin slabs (1 x 1 x 0.1). Low-poly (one quad per face): every vertex is a corner, whose direction is the
// box diagonal; the thin side faces shrink by u = 2 d / sqrt(3) along and 10 u across, and keep a quarter of their area
// at (1 - u)(1 - 10 u) = 1/4, u = 0.0730, d = 0.0632. Every vertex gets that limit; the top and bottom shells then sink
// 2 d / sqrt(3) = 0.073 together, less than the thickness: the relief cannot punch through. Subdivided (4 x 4 per
// face): the vertex in the middle of each big face has no limit, so a depth above the thickness punches through there
// (risk R5, documented: the fold guard is local).
void testThinSlab() {
  {
    const Mesh mesh = makeBox({ 1, 1, 0.1f }, 1);
    const Result r = build(mesh);
    for (uint32_t v = 0; v < r.vertices; ++v) {
      check(std::fabs(r.depthLimits[v].first - 0.0632462f) < 1e-5f, "slab: corner limit (" + std::to_string(r.depthLimits[v].first) + ")");
      check(2.0f * r.depthLimits[v].first / std::sqrt(3.0f) < 0.1f, "slab: the shells of both sides do not meet");
    }
    checkFoldGuard(mesh, r, "slab");
  }
  {
    const Mesh mesh = makeBox({ 1, 1, 0.1f }, 4);
    const Result r = build(mesh);
    uint32_t unlimitedTop = 0;
    for (uint32_t v = 0; v < r.vertices; ++v) {
      const Float3 p = mesh.positions[v];
      if (p.z == 0.1f && p.x > 0.3f && p.x < 0.7f && p.y > 0.3f && p.y < 0.7f) {
        unlimitedTop += r.depthLimits[v].first == kNoLimit ? 1u : 0u;
      }
    }
    check(unlimitedTop > 0, "subdivided slab: the middle of the top face is unlimited (punch-through possible, R5)");
    checkFoldGuard(mesh, r, "subdivided slab");
  }
}

void run() {
  testCube(true);
  testCube(false);
  testCylinder();
  testFin();
  testFlipped();
  testTwoSidedSheet();
  testWeldEpsilon();
  testSubdividedCube();
  testConcaveL();
  testClosedL();
  testThinSlab();
  if (g_failures > 0) {
    throw std::runtime_error(std::to_string(g_failures) + " check(s) failed");
  }
  std::cout << "test_sccr_shell: all checks passed" << std::endl;
}

} // namespace

int main() {
  try {
    run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
