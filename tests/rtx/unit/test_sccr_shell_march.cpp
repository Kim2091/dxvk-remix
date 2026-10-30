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

// Checks the prism march (sccr_shell_march.slangh, copied below on the CPU) against
// independent references on small meshes:
// - the prism map, its Newton inverse (round trip) and the continuity of neighbouring prisms on their shared side;
// - a flat quad: the march equals plain POM (the flat-plane march in UV) for the same ray and field, in linear steps and
//   with conservative cones, also where the ray crosses the quad's diagonal into the other triangle;
// - a bent strip (two triangles at an angle), a cylinder segment (curved welded directions, several crossings at a
//   grazing view), mirrored UVs and a UV seam (split vertices, offset UVs): the march against a brute-force reference
//   that walks the ray in tiny steps, finds the prism containing each point by a cold-started solve in every triangle,
//   and compares the field in that prism's own UVs;
// - at a UV seam whose field jumps, the march stops at the seam (the wall the jump makes), as the reference does;
// - carved silhouettes: an outer exit over a convex ridge and a boundary exit at an open edge, with their
//   along-ray distances (the exits sccrShellResolve shades as plain SCCR);
// - the fold guard: a unit cube at a depth (1) deeper than its half size. With the guard the depth is capped at
//   the corners' limits and a steep ray meets the relief in the top face, as the reference does; without it the folded
//   prisms let the ray walk through the cube and out of its bottom.
// - shell shadows: the shell marker's float round trip; the outward shadow march from a carved point (found
//   again from its texture coordinates) against a brute-force walk of the light ray: a pit occluded by its rim after a
//   crossing, the same pit lit under a steep light, a level field lit through the top of a neighbour (exit distance),
//   and a concave crease whose far slope occludes a low light.
// - 12.14: a smoothed low-poly boulder (box-projected UVs, welded seams) at one and two triangle sizes of depth, viewed
//   at grazing angles all around its outline: no limits with the default budget (32 prisms), bounded steps and
//   crossings, outer exits only where the ray really leaves the mesh, carved rays never deeper than the relief, limits
//   never shaded where they stopped (and with the old 8 prisms, rare once rising rays count as exits), and six grazing
//   rays against the brute-force walk.
// The shell data (welded directions, adjacency, fold limits) is built here by brute force with the conventions of
// sccr_shell_args.h, not by the GPU kernel's copy (test_sccr_shell.cpp checks that). Object space is world space here
// (identity instance transform), so D_obj = D. The step LOD is off (it needs a pixel footprint). Keep the copy in step
// with the shader.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
// Optional: the triangles a march walks (entry first, then every prism crossed into).
std::vector<uint32_t>* g_marchPath = nullptr;

void check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "FAILED: " << what << std::endl;
    ++g_failures;
  }
}

struct Float2 {
  float x, y;
};

struct Float3 {
  float x, y, z;
};

Float2 operator+(Float2 a, Float2 b) { return { a.x + b.x, a.y + b.y }; }
Float2 operator-(Float2 a, Float2 b) { return { a.x - b.x, a.y - b.y }; }
Float2 operator*(Float2 a, float s) { return { a.x * s, a.y * s }; }
float length(Float2 a) { return std::sqrt(a.x * a.x + a.y * a.y); }

Float3 operator+(Float3 a, Float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Float3 operator-(Float3 a, Float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Float3 operator-(Float3 a) { return { -a.x, -a.y, -a.z }; }
Float3 operator*(Float3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
float dot(Float3 a, Float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Float3 cross(Float3 a, Float3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
float length(Float3 a) { return std::sqrt(dot(a, a)); }
Float3 normalize(Float3 a) { return a * (1.0f / length(a)); }
float component(Float3 v, uint32_t i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }
float saturate(float v) { return std::min(std::max(v, 0.0f), 1.0f); }
float lerp(float a, float b, float w) { return a + (b - a) * w; }
Float3 lerp(Float3 a, Float3 b, float w) { return a + (b - a) * w; }

// sccr_shell_args.h
constexpr uint32_t kNeighbourNone = 0xFFFFFFFFu;
constexpr uint32_t kNeighbourNonManifold = 0x3FFFFFFFu;
constexpr uint32_t kNeighbourFlipped = 0x7FFFFFFFu;
constexpr uint32_t kNeighbourDegenerate = 0xBFFFFFFFu;
constexpr uint32_t kNeighbourTriangleMask = 0x3FFFFFFFu;
constexpr uint32_t kNeighbourSlotShift = 30;
constexpr float kNoLimit = 65504.0f;
constexpr float kFoldMargin = 0.75f;
constexpr float kFoldMinCosine = 1e-3f;

bool isBoundary(uint32_t word) { return (word & kNeighbourTriangleMask) == kNeighbourTriangleMask; }

// sccr_shell_march.slangh
constexpr uint32_t kResultFailed = 3;
constexpr uint32_t kResultHit = 4;
constexpr uint32_t kResultBoundary = 5;
constexpr uint32_t kResultOuter = 6;
constexpr uint32_t kResultLimit = 7;
constexpr uint32_t kResultSolve = 8;
constexpr uint32_t kFaceTop = 3;
constexpr uint32_t kFaceBottom = 4;
constexpr uint32_t kFaceNone = 5;
constexpr float kBaryEpsilon = 1e-4f;
constexpr float kDepthEpsilon = 1e-4f;
constexpr float kConeSafety = 1.25f;
constexpr uint32_t kNewtonIterations = 3;
constexpr float kNewtonTolerance = 1e-5f;
constexpr float kNewtonAccept = 1e-3f;
constexpr float kMinCosine = 1e-3f;
constexpr float kMaxStepEdges = 0.5f;
constexpr float kMinRate = 0.05f;
constexpr float kExitOvershoot = 1e-3f;
constexpr float kExitNudge = 1e-5f;
constexpr uint32_t kStepsPerCrossing = 4;
constexpr float kUvContinuity = 2e-3f;
constexpr uint32_t kBoundaryOpen = 0;
constexpr uint32_t kBoundaryBlocked = 1;
constexpr uint32_t kBoundaryNonManifold = 2;
constexpr uint32_t kBoundaryFlipped = 3;
constexpr uint32_t kBoundaryDegenerate = 4;

uint32_t boundaryKind(uint32_t word) {
  return word == kNeighbourNonManifold ? kBoundaryNonManifold :
         word == kNeighbourFlipped ? kBoundaryFlipped :
         word == kNeighbourDegenerate ? kBoundaryDegenerate : kBoundaryOpen;
}

// The constants the shader reads (rtx.displacement defaults; shell.maxPrisms / maxSteps defaults).
struct Params {
  uint32_t maxIterations = 64;
  uint32_t coneSteps = 16;
  uint32_t binarySteps = 5;
  uint32_t maxPrisms = 32;
  uint32_t maxSteps = 128;
  // Shell shadows: rtx.displacement.coneShadowSteps and shell.maxShadowPrisms defaults.
  uint32_t coneShadowSteps = 8;
  uint32_t maxShadowPrisms = 4;
};

struct Mesh {
  std::vector<Float3> positions;
  std::vector<Float2> texcoords;
  std::vector<uint32_t> indices;

  uint32_t triangleCount() const { return uint32_t(indices.size() / 3); }
};

// What the shader reads from the shell buffer: neighbour words, welded directions (out of the mesh) and the fold
// limits (against and along the direction, as the kernel's phase 5 computes them), and whether the fold guard is on
// (SCCR_SHELL_MARCH_FLAG_FOLD_GUARD).
struct ShellData {
  std::vector<uint32_t> neighbours;
  std::vector<Float3> directions;
  std::vector<Float2> limits;
  bool foldGuard = true;
};

// sccrShellSmallestPositiveRoot (sccr_shell_adjacency.comp.slang).
float smallestPositiveRoot(float a, float b, float c) {
  const float infinity = std::numeric_limits<float>::infinity();
  if (a == 0.0f) {
    return b < 0.0f ? -c / b : infinity;
  }
  const float discriminant = b * b - 4.0f * a * c;
  if (discriminant < 0.0f) {
    return infinity;
  }
  const float root = std::sqrt(discriminant);
  const float q = -0.5f * (b + (b >= 0.0f ? root : -root));
  float smallest = infinity;
  if (q / a > 0.0f) {
    smallest = q / a;
  }
  if (q != 0.0f && c / q > 0.0f && c / q < smallest) {
    smallest = c / q;
  }
  return smallest;
}

// Brute force with the shell data's conventions: welded id = the smallest vertex index at the same position (exact),
// angle-weighted face normals summed per welded position, edge slot e opposite vertex e running from (e + 1) % 3 to
// (e + 2) % 3, a neighbour only across an edge shared by exactly two triangles running along it in opposite directions.
ShellData buildShell(const Mesh& mesh) {
  const uint32_t vertexCount = uint32_t(mesh.positions.size());
  const uint32_t triangleCount = mesh.triangleCount();
  std::vector<uint32_t> weld(vertexCount);
  for (uint32_t v = 0; v < vertexCount; ++v) {
    weld[v] = v;
    for (uint32_t u = 0; u < v; ++u) {
      const Float3 a = mesh.positions[u];
      const Float3 b = mesh.positions[v];
      if (a.x == b.x && a.y == b.y && a.z == b.z) {
        weld[v] = u;
        break;
      }
    }
  }
  std::vector<Float3> sums(vertexCount, Float3 { 0.0f, 0.0f, 0.0f });
  for (uint32_t t = 0; t < triangleCount; ++t) {
    const uint32_t* v = &mesh.indices[3 * t];
    const Float3 n = normalize(cross(mesh.positions[v[1]] - mesh.positions[v[0]], mesh.positions[v[2]] - mesh.positions[v[0]]));
    for (uint32_t k = 0; k < 3; ++k) {
      const Float3 a = normalize(mesh.positions[v[(k + 1) % 3]] - mesh.positions[v[k]]);
      const Float3 b = normalize(mesh.positions[v[(k + 2) % 3]] - mesh.positions[v[k]]);
      const float angle = std::acos(std::min(std::max(dot(a, b), -1.0f), 1.0f));
      sums[weld[v[k]]] = sums[weld[v[k]]] + n * angle;
    }
  }
  ShellData data;
  data.directions.resize(vertexCount);
  for (uint32_t v = 0; v < vertexCount; ++v) {
    data.directions[v] = normalize(sums[weld[v]]);
  }
  // The fold limits (the kernel's phase 5), per welded position.
  std::vector<Float2> weldLimits(vertexCount, Float2 { kNoLimit, kNoLimit });
  for (uint32_t t = 0; t < triangleCount; ++t) {
    const uint32_t* v = &mesh.indices[3 * t];
    const Float3 e1 = mesh.positions[v[1]] - mesh.positions[v[0]];
    const Float3 e2 = mesh.positions[v[2]] - mesh.positions[v[0]];
    const Float3 normal = cross(e1, e2);
    const Float3 n[3] = { data.directions[v[0]], data.directions[v[1]], data.directions[v[2]] };
    const Float3 g1 = n[1] - n[0];
    const Float3 g2 = n[2] - n[0];
    const Float3 linear = cross(e1, g2) + cross(g1, e2);
    const Float3 quadratic = cross(g1, g2);
    float against = std::numeric_limits<float>::infinity();
    float along = std::numeric_limits<float>::infinity();
    bool skip = false;
    for (uint32_t k = 0; k < 3; ++k) {
      const float a = dot(n[k], normal);
      if (!(a > kFoldMinCosine * length(normal))) {
        skip = true;
        break;
      }
      against = std::min(against, smallestPositiveRoot(dot(n[k], quadratic), -dot(n[k], linear), kFoldMargin * a));
      along = std::min(along, smallestPositiveRoot(dot(n[k], quadratic), dot(n[k], linear), kFoldMargin * a));
    }
    if (skip) {
      continue;
    }
    for (uint32_t k = 0; k < 3; ++k) {
      Float2& limit = weldLimits[weld[v[k]]];
      limit = { std::min(limit.x, against), std::min(limit.y, along) };
    }
  }
  data.limits.resize(vertexCount);
  for (uint32_t v = 0; v < vertexCount; ++v) {
    data.limits[v] = weldLimits[weld[v]];
  }
  data.neighbours.assign(3 * size_t(triangleCount), kNeighbourNone);
  for (uint32_t t = 0; t < triangleCount; ++t) {
    for (uint32_t e = 0; e < 3; ++e) {
      const uint32_t from = weld[mesh.indices[3 * t + (e + 1) % 3]];
      const uint32_t to = weld[mesh.indices[3 * t + (e + 2) % 3]];
      uint32_t reverse = 0;
      uint32_t same = 0;
      uint32_t word = kNeighbourNone;
      for (uint32_t t2 = 0; t2 < triangleCount; ++t2) {
        for (uint32_t e2 = 0; e2 < 3; ++e2) {
          if (t2 == t && e2 == e) {
            continue;
          }
          const uint32_t from2 = weld[mesh.indices[3 * t2 + (e2 + 1) % 3]];
          const uint32_t to2 = weld[mesh.indices[3 * t2 + (e2 + 2) % 3]];
          if (from2 == to && to2 == from) {
            ++reverse;
            word = t2 | (e2 << kNeighbourSlotShift);
          } else if (from2 == from && to2 == to) {
            ++same;
          }
        }
      }
      data.neighbours[3 * t + e] = reverse == 1 && same == 0 ? word : kNeighbourNone;
    }
  }
  return data;
}

// A height field: depth in [0, 1] at uv (0 top, 1 bottom) and its cone ratio (UV per unit of depth, 0: none).
struct Field {
  std::function<float(Float2)> depth;
  float cone = 0.0f;
};

struct Prism {
  uint32_t triangle = 0;
  Float3 p0 {}, e1 {}, e2 {}, s0 {}, f1 {}, f2 {};
  Float2 uv0 {}, duv1 {}, duv2 {};
  float maxStep = 0.0f;
  float size = 0.0f;

  Float3 position(Float3 x) const { return e1 * x.x + e2 * x.y - (s0 + f1 * x.x + f2 * x.y) * x.z; }
  Float2 textureCoordinates(Float3 x) const { return uv0 + duv1 * x.x + duv2 * x.y; }
};

struct Jacobian {
  Float3 row0 {}, row1 {}, row2 {};
  Float3 mul(Float3 v) const { return { dot(row0, v), dot(row1, v), dot(row2, v) }; }
};

Float3 barycentrics(Float3 x) { return { 1.0f - x.x - x.y, x.x, x.y }; }

Float3 clampToPrism(Float3 x) {
  Float3 b = barycentrics(x);
  b = { std::max(b.x, 0.0f), std::max(b.y, 0.0f), std::max(b.z, 0.0f) };
  const float sum = b.x + b.y + b.z;
  return sum > 0.0f ? Float3 { b.y / sum, b.z / sum, saturate(x.z) } : Float3 { 1.0f / 3.0f, 1.0f / 3.0f, saturate(x.z) };
}

// sccrShellPrismLoad with an identity instance transform (the fold guard's limits as floats, not half floats).
bool loadPrism(const Mesh& mesh, const ShellData& data, uint32_t tri, float worldDepth, float orientation, Prism& prism) {
  if (tri >= mesh.triangleCount()) {
    return false;
  }
  Float3 p[3];
  Float2 uv[3];
  Float3 s[3];
  for (uint32_t k = 0; k < 3; ++k) {
    const uint32_t v = mesh.indices[3 * tri + k];
    p[k] = mesh.positions[v];
    uv[k] = mesh.texcoords[v];
    const Float3 direction = data.directions[v] * orientation;
    float objectDepth = worldDepth / length(direction);
    if (data.foldGuard) {
      objectDepth = std::min(objectDepth, orientation > 0.0f ? data.limits[v].x : data.limits[v].y);
    }
    s[k] = direction * objectDepth;
  }
  prism.triangle = tri;
  prism.p0 = p[0];
  prism.e1 = p[1] - p[0];
  prism.e2 = p[2] - p[0];
  prism.s0 = s[0];
  prism.f1 = s[1] - s[0];
  prism.f2 = s[2] - s[0];
  prism.uv0 = uv[0];
  prism.duv1 = uv[1] - uv[0];
  prism.duv2 = uv[2] - uv[0];
  const float l01 = length(prism.e1);
  const float l02 = length(prism.e2);
  const float l12 = length(p[2] - p[1]);
  prism.maxStep = kMaxStepEdges * std::min(std::min(l01, l02), l12);
  prism.size = std::max(std::max(l01, l02), l12);
  return prism.maxStep > 0.0f;
}

bool jacobianInverse(const Prism& prism, Float3 x, float detSign, Jacobian& inverse) {
  const Float3 shell = prism.s0 + prism.f1 * x.x + prism.f2 * x.y;
  const Float3 c0 = prism.e1 - prism.f1 * x.z;
  const Float3 c1 = prism.e2 - prism.f2 * x.z;
  const Float3 c2 = -shell;
  const Float3 r01 = cross(c0, c1);
  const float det = dot(c2, r01);
  const bool valid = det * detSign > kMinCosine * length(r01) * length(shell);
  const float invDet = valid ? 1.0f / det : 0.0f;
  inverse.row0 = cross(c1, c2) * invDet;
  inverse.row1 = cross(c2, c0) * invDet;
  inverse.row2 = r01 * invDet;
  return valid;
}

bool solve(const Prism& prism, Float3 target, float detSign, Float3& x, Jacobian& inverse, uint32_t iterations = kNewtonIterations) {
  float lastStep = 1.0f;
  for (uint32_t i = 0; i < iterations; ++i) {
    if (!jacobianInverse(prism, x, detSign, inverse)) {
      return false;
    }
    const Float3 dx = inverse.mul(prism.position(x) - target);
    x = x - dx;
    lastStep = std::max(std::max(std::fabs(dx.x), std::fabs(dx.y)), std::fabs(dx.z));
    if (lastStep < kNewtonTolerance) {
      break;
    }
  }
  return lastStep < kNewtonAccept && std::isfinite(x.x) && std::isfinite(x.y) && std::isfinite(x.z);
}

bool outside(Float3 x, bool exitLimited, uint32_t exitFace, uint32_t& face) {
  const Float3 b = barycentrics(x);
  face = kFaceNone;
  float least = -kBaryEpsilon;
  if (b.x < least) { least = b.x; face = 0; }
  if (b.y < least) { least = b.y; face = 1; }
  if (b.z < least) { face = 2; }
  if (face == kFaceNone && x.z < -kDepthEpsilon) {
    face = kFaceTop;
  }
  if (face == kFaceNone && exitLimited) {
    const bool reached = exitFace == kFaceTop ? x.z < kDepthEpsilon : component(b, exitFace) < kBaryEpsilon;
    if (reached) {
      face = exitFace;
    }
  }
  return face != kFaceNone;
}

Float3 carry(Float3 x, uint32_t edgeSlot, uint32_t neighbourSlot) {
  const Float3 b = barycentrics(x);
  const float from = std::max(component(b, (edgeSlot + 1) % 3), 0.0f);
  const float to = std::max(component(b, (edgeSlot + 2) % 3), 0.0f);
  const float sum = from + to;
  const float fromShare = sum > 0.0f ? from / sum : 0.5f;
  const float toShare = 1.0f - fromShare;
  const Float3 carried =
    neighbourSlot == 0 ? Float3 { 0.0f, toShare, fromShare } :
    neighbourSlot == 1 ? Float3 { fromShare, 0.0f, toShare } :
                         Float3 { toShare, fromShare, 0.0f };
  return { carried.y, carried.z, x.z };
}

struct Result {
  // sccrShellMarch's return value: the result is used (false: plain SCCR runs).
  bool used = false;
  uint32_t reason = kResultFailed;
  uint32_t steps = 0;
  uint32_t crossings = 0;
  uint32_t triangle = 0;
  Float3 x {};
  Float2 uv {};
  float t = 0.0f;
  uint32_t boundary = kBoundaryOpen;
};

void setResult(Result& result, uint32_t reason, const Prism& prism, Float3 x, float t) {
  const Float3 clamped = clampToPrism(x);
  result.used = reason == kResultHit || reason == kResultBoundary || reason == kResultOuter;
  result.reason = reason;
  result.triangle = prism.triangle;
  result.x = clamped;
  result.uv = prism.textureCoordinates(clamped);
  result.t = t;
}

void refine(const Field& field, const Prism& prism, Float3 entryRelative, Float3 rayDirection, float detSign, const Params& params,
            float aboveT, Float3 aboveX, float aboveGap, float belowT, Float3 belowX, float belowGap,
            float& hitT, Float3& hitX, uint32_t& steps) {
  const uint32_t bisections = params.binarySteps > 0 ? params.binarySteps - 1 : 0;
  const bool falsePosition = params.binarySteps > 0;
  const uint32_t refinements = bisections + (falsePosition ? 1 : 0);
  for (uint32_t j = 0; j < refinements; ++j) {
    const float weight = j < bisections ? 0.5f : saturate(aboveGap / std::max(aboveGap - belowGap, 1e-6f));
    const float middleT = lerp(aboveT, belowT, weight);
    Float3 middleX = lerp(aboveX, belowX, weight);
    Float3 solved = middleX;
    Jacobian unused;
    if (solve(prism, entryRelative + rayDirection * middleT, detSign, solved, unused)) {
      middleX = solved;
    }
    middleX.z = std::min(middleX.z, 1.0f);
    const float middleGap = field.depth(prism.textureCoordinates(middleX)) - middleX.z;
    ++steps;
    if (middleGap > 0.0f) {
      aboveT = middleT; aboveX = middleX; aboveGap = middleGap;
    } else {
      belowT = middleT; belowX = middleX; belowGap = middleGap;
    }
  }
  const float weight = saturate(aboveGap / std::max(aboveGap - belowGap, 1e-6f));
  hitT = lerp(aboveT, belowT, weight);
  hitX = lerp(aboveX, belowX, weight);
}

// sccrShellLimit: out of steps or prisms while rising above the field is an outer exit, otherwise LIMIT (plain SCCR).
void limit(Result& result, const Prism& prism, Float3 x, float t, float rateDepth, float gap) {
  setResult(result, rateDepth < 0.0f && gap > 0.0f ? kResultOuter : kResultLimit, prism, x, t);
}

// Field samples the march took (instrumentation, not in the shader).
uint32_t g_fieldSamples = 0;

// sccrShellMarch with an identity instance transform and the step LOD off. rayDirection: unit, world (= object).
Result march(const Mesh& mesh, const ShellData& data, const Field& field, uint32_t entryTriangle, Float2 entryB12,
             Float3 rayDirection, float worldDepth, const Params& params) {
  Result result;
  result.triangle = entryTriangle;
  Prism prism;
  if (!loadPrism(mesh, data, entryTriangle, worldDepth, 1.0f, prism)) {
    return result;
  }
  const float orientation = dot(rayDirection, cross(prism.e1, prism.e2)) < 0.0f ? 1.0f : -1.0f;
  if (orientation < 0.0f && !loadPrism(mesh, data, entryTriangle, worldDepth, orientation, prism)) {
    return result;
  }
  const float detSign = -orientation;
  Float3 x { entryB12.x, entryB12.y, 0.0f };
  if (g_marchPath) {
    g_marchPath->assign(1, entryTriangle);
  }
  const Float3 entryPoint = prism.p0 + prism.position(x);
  Float3 entryRelative = entryPoint - prism.p0;
  Jacobian inverse;
  if (!jacobianInverse(prism, x, detSign, inverse)) {
    return result;
  }
  Float3 rate = inverse.mul(rayDirection);
  if (!(rate.z > 0.0f)) {
    return result;
  }
  const float cosView = saturate(rate.z * worldDepth);
  const float numSamples = lerp(float(params.maxIterations), params.maxIterations / 4.0f, cosView);
  const float minStepDepth = 1.0f / numSamples;
  const float rateFloor = kMinRate / worldDepth;
  const float objectMaxStepScale = 1.0f / length(rayDirection);
  const uint32_t maxPrisms = params.maxPrisms;
  const uint32_t maxSteps = std::max(std::min(params.maxSteps,
    params.coneSteps + uint32_t(std::ceil(numSamples)) + 1 + kStepsPerCrossing * maxPrisms), 1u);

  float t = 0.0f;
  uint32_t coneStepsLeft = params.coneSteps;
  bool hasAbove = false;
  float aboveT = 0.0f;
  Float3 aboveX = x;
  float aboveGap = 0.0f;
  float depth = 0.0f;
  float gap = 0.0f;
  bool pending = false;
  bool pendingFromStep = false;
  uint32_t pendingFace = kFaceNone;
  Float2 sampleUv {};
  bool keepSample = false;
  const uint32_t iterationLimit = maxSteps + maxPrisms + 2;
  for (uint32_t iteration = 0; iteration < iterationLimit; ++iteration) {
    if (!pending || pendingFromStep) {
      if (!keepSample) {
        sampleUv = prism.textureCoordinates(x);
        depth = field.depth(sampleUv);
        ++g_fieldSamples;
      }
      keepSample = false;
      gap = depth - x.z;
      if (gap <= 0.0f) {
        float hitT = t;
        Float3 hitX = x;
        if (hasAbove) {
          refine(field, prism, entryRelative, rayDirection, detSign, params, aboveT, aboveX, aboveGap, t, x, gap, hitT, hitX, result.steps);
        }
        setResult(result, kResultHit, prism, hitX, hitT);
        return result;
      }
    }
    if (pending) {
      pending = false;
      if (pendingFace == kFaceTop) {
        setResult(result, kResultOuter, prism, x, t);
        return result;
      }
      if (result.crossings >= maxPrisms || result.steps >= maxSteps) {
        limit(result, prism, x, t, rate.z, gap);
        return result;
      }
      const bool crossingFromStep = pendingFromStep;
      const uint32_t word = data.neighbours[3 * prism.triangle + pendingFace];
      const bool boundaryEdge = isBoundary(word);
      Prism next;
      if (boundaryEdge || !loadPrism(mesh, data, word & kNeighbourTriangleMask, worldDepth, orientation, next)) {
        setResult(result, kResultBoundary, prism, x, t);
        result.boundary = boundaryEdge ? boundaryKind(word) : kBoundaryBlocked;
        return result;
      }
      Float3 nextX = carry(x, pendingFace, word >> kNeighbourSlotShift);
      const Float3 nextEntryRelative = entryPoint - next.p0;
      Jacobian nextInverse;
      if (!solve(next, nextEntryRelative + rayDirection * t, detSign, nextX, nextInverse)) {
        setResult(result, kResultSolve, prism, x, t);
        return result;
      }
      prism = next;
      if (g_marchPath) {
        g_marchPath->push_back(next.triangle);
      }
      x = nextX;
      x.z = std::min(x.z, 1.0f);
      inverse = nextInverse;
      entryRelative = nextEntryRelative;
      rate = inverse.mul(rayDirection);
      ++result.crossings;
      ++result.steps;
      coneStepsLeft = params.coneSteps;
      hasAbove = false;
      pending = outside(x, false, kFaceNone, pendingFace);
      pendingFromStep = false;
      const float uvExtent = std::max(length(prism.duv1), length(prism.duv2));
      keepSample = crossingFromStep && !pending && length(prism.textureCoordinates(x) - sampleUv) <= kUvContinuity * uvExtent;
      continue;
    }
    if (result.steps >= maxSteps) {
      limit(result, prism, x, t, rate.z, gap);
      return result;
    }
    const float a = rate.z;
    const float uvRate = length(prism.duv1 * rate.x + prism.duv2 * rate.y);
    float dt = minStepDepth / std::max(a, rateFloor);
    bool coneStep = false;
    if (coneStepsLeft > 0 && field.cone > 0.0f) {
      dt = std::max(dt, gap / (kConeSafety * (std::max(a, 0.0f) + uvRate / field.cone)));
      coneStep = true;
    }
    dt = std::min(dt, prism.maxStep * objectMaxStepScale);
    const Float3 b = barycentrics(x);
    const Float3 db { -(rate.x + rate.y), rate.x, rate.y };
    float exitT = 1e30f;
    uint32_t exitFace = kFaceNone;
    for (uint32_t i = 0; i < 3; ++i) {
      const float rateI = component(db, i);
      if (rateI < 0.0f) {
        const float faceT = std::max(component(b, i), 0.0f) / -rateI;
        if (faceT < exitT) { exitT = faceT; exitFace = i; }
      }
    }
    if (a < 0.0f) {
      const float faceT = std::max(x.z, 0.0f) / -a;
      if (faceT < exitT) { exitT = faceT; exitFace = kFaceTop; }
    } else if (a > 0.0f) {
      const float faceT = std::max(1.0f - x.z, 0.0f) / a;
      if (faceT < exitT) { exitT = faceT; exitFace = kFaceBottom; }
    }
    bool exitLimited = false;
    if (exitT <= dt) {
      dt = exitT * (1.0f + kExitOvershoot) + kExitNudge * prism.size * objectMaxStepScale;
      exitLimited = exitFace < 3 || exitFace == kFaceTop;
    }
    hasAbove = true;
    aboveT = t;
    aboveX = x;
    aboveGap = gap;
    t += dt;
    Float3 nextX = x + rate * dt;
    if (!solve(prism, entryRelative + rayDirection * t, detSign, nextX, inverse)) {
      setResult(result, kResultSolve, prism, aboveX, aboveT);
      return result;
    }
    x = nextX;
    x.z = std::min(x.z, 1.0f);
    rate = inverse.mul(rayDirection);
    ++result.steps;
    if (coneStep) {
      --coneStepsLeft;
    }
    pending = outside(x, exitLimited, exitFace, pendingFace);
    pendingFromStep = true;
  }
  limit(result, prism, x, t, rate.z, gap);
  return result;
}

// ---------------------------------------------------------------------------------------------------------------------
// The brute-force reference.

struct Location {
  bool inside = false;
  uint32_t triangle = 0;
  Float3 x {};
  Float2 uv {};
};

// The prism containing point (world = object), by a cold-started solve in every triangle; the most interior one where
// two contain it (a shared side).
Location locate(const Mesh& mesh, const ShellData& data, float worldDepth, float orientation, Float3 point) {
  Location best;
  float bestMargin = -std::numeric_limits<float>::infinity();
  for (uint32_t tri = 0; tri < mesh.triangleCount(); ++tri) {
    // A prism lies within its triangle's bounding sphere grown by D (shell vectors are at most D long): skip the rest.
    const Float3 corner0 = mesh.positions[mesh.indices[3 * tri]];
    const Float3 corner1 = mesh.positions[mesh.indices[3 * tri + 1]];
    const Float3 corner2 = mesh.positions[mesh.indices[3 * tri + 2]];
    const Float3 centre = (corner0 + corner1 + corner2) * (1.0f / 3.0f);
    const float radius = std::max(std::max(length(corner0 - centre), length(corner1 - centre)), length(corner2 - centre));
    if (length(point - centre) > radius + 1.5f * worldDepth) {
      continue;
    }
    Prism prism;
    if (!loadPrism(mesh, data, tri, worldDepth, orientation, prism)) {
      continue;
    }
    Float3 x { 1.0f / 3.0f, 1.0f / 3.0f, 0.5f };
    Jacobian inverse;
    if (!solve(prism, point - prism.p0, -orientation, x, inverse, 40)) {
      continue;
    }
    const Float3 b = barycentrics(x);
    const float margin = std::min(std::min(std::min(b.x, b.y), b.z), std::min(x.z, 1.0f - x.z) + 1.0f);
    if (b.x < -1e-5f || b.y < -1e-5f || b.z < -1e-5f || x.z < -1e-5f || x.z > 1.0f + 1e-5f) {
      continue;
    }
    if (margin > bestMargin) {
      bestMargin = margin;
      best.inside = true;
      best.triangle = tri;
      best.x = x;
      best.uv = prism.textureCoordinates(x);
    }
  }
  return best;
}

struct Reference {
  bool hit = false;
  float t = 0.0f;
  Location location;
};

// Walks the ray from the entry point in steps of stepT until the first point at or below the field (in the containing
// prism's UVs), then bisects. No hit when the ray leaves the volume first.
Reference reference(const Mesh& mesh, const ShellData& data, const Field& field, float worldDepth, float orientation,
                    Float3 entryPoint, Float3 rayDirection, float stepT, float maxT) {
  Reference result;
  float previousT = 0.0f;
  for (float t = 0.0f; t <= maxT; t += stepT) {
    const Location location = locate(mesh, data, worldDepth, orientation, entryPoint + rayDirection * t);
    if (!location.inside) {
      return result;
    }
    if (field.depth(location.uv) - location.x.z <= 0.0f) {
      float lowT = previousT;
      float highT = t;
      for (int i = 0; i < 40 && t > 0.0f; ++i) {
        const float middleT = 0.5f * (lowT + highT);
        const Location middle = locate(mesh, data, worldDepth, orientation, entryPoint + rayDirection * middleT);
        if (middle.inside && field.depth(middle.uv) - middle.x.z > 0.0f) {
          lowT = middleT;
        } else {
          highT = middleT;
        }
      }
      result.hit = true;
      result.t = highT;
      result.location = locate(mesh, data, worldDepth, orientation, entryPoint + rayDirection * highT);
      return result;
    }
    previousT = t;
  }
  return result;
}

float entryOrientation(const Mesh& mesh, uint32_t tri, Float3 rayDirection) {
  const Float3 p0 = mesh.positions[mesh.indices[3 * tri]];
  const Float3 p1 = mesh.positions[mesh.indices[3 * tri + 1]];
  const Float3 p2 = mesh.positions[mesh.indices[3 * tri + 2]];
  return dot(rayDirection, cross(p1 - p0, p2 - p0)) < 0.0f ? 1.0f : -1.0f;
}

Float3 entryPointOf(const Mesh& mesh, uint32_t tri, Float2 b12) {
  const Float3 p0 = mesh.positions[mesh.indices[3 * tri]];
  const Float3 p1 = mesh.positions[mesh.indices[3 * tri + 1]];
  const Float3 p2 = mesh.positions[mesh.indices[3 * tri + 2]];
  return p0 + (p1 - p0) * b12.x + (p2 - p0) * b12.y;
}

// The march against the reference: both hit, at the same distance and texture coordinates.
void compareWithReference(const std::string& name, const Mesh& mesh, const Field& field, uint32_t tri, Float2 b12,
                          Float3 rayDirection, float worldDepth, uint32_t minCrossings, float toleranceT, float toleranceUv) {
  const ShellData data = buildShell(mesh);
  Params params;
  const Result result = march(mesh, data, field, tri, b12, rayDirection, worldDepth, params);
  const Reference ref = reference(mesh, data, field, worldDepth, entryOrientation(mesh, tri, rayDirection),
                                  entryPointOf(mesh, tri, b12), rayDirection, worldDepth / 4000.0f, 20.0f * worldDepth);
  check(ref.hit, name + ": the reference hits the field");
  check(result.reason == kResultHit, name + ": the march hits the field (reason " + std::to_string(result.reason) + ")");
  check(result.crossings >= minCrossings,
        name + ": at least " + std::to_string(minCrossings) + " crossing(s), got " + std::to_string(result.crossings));
  if (ref.hit && result.reason == kResultHit) {
    check(std::fabs(result.t - ref.t) <= toleranceT,
          name + ": distance " + std::to_string(result.t) + " against " + std::to_string(ref.t));
    check(length(result.uv - ref.location.uv) <= toleranceUv,
          name + ": texture coordinates (" + std::to_string(result.uv.x) + ", " + std::to_string(result.uv.y) +
          ") against (" + std::to_string(ref.location.uv.x) + ", " + std::to_string(ref.location.uv.y) + ")");
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Meshes and fields.

// The unit square in z = 0, UV = xy, two triangles wound towards +z (diagonal from (1,0) to (0,1)).
Mesh flatQuad() {
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
  mesh.texcoords = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
  mesh.indices = { 0, 1, 2, 2, 1, 3 };
  return mesh;
}

// Two triangles folded along their shared diagonal: the second's far corner lies `lift` above the first's plane
// (positive: a concave valley, negative: a convex ridge). Planar UVs (continuous).
Mesh bentStrip(float lift) {
  Mesh mesh;
  mesh.positions = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 1, 1, lift } };
  mesh.texcoords = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
  mesh.indices = { 0, 1, 2, 2, 1, 3 };
  return mesh;
}

// A cylinder segment of radius 1 around the y axis, 12 quads over 90 degrees, outward winding; u = arc length, v = y.
Mesh cylinderSegment() {
  Mesh mesh;
  const uint32_t segments = 12;
  const float span = 0.5f * 3.14159265f;
  for (uint32_t k = 0; k <= segments; ++k) {
    const float angle = span * float(k) / float(segments) - 0.5f * span;
    for (uint32_t j = 0; j < 2; ++j) {
      mesh.positions.push_back({ std::sin(angle), float(j), std::cos(angle) });
      mesh.texcoords.push_back({ angle, float(j) });
    }
  }
  for (uint32_t k = 0; k < segments; ++k) {
    const uint32_t a = 2 * k;
    const uint32_t b = 2 * k + 1;
    const uint32_t c = 2 * (k + 1);
    const uint32_t d = 2 * (k + 1) + 1;
    // Outward (+radial) normals: cross(c - a, b - a) points out.
    mesh.indices.insert(mesh.indices.end(), { a, c, b, b, c, d });
  }
  return mesh;
}

// The unit square in two halves split at x = 0.5 (four triangles). mirrored: the right half's u is 1 - x (UVs
// continuous at the split, mirrored; shared vertices). Otherwise (seam) the right half has its own vertices on the
// split, with UVs offset by seamOffset.
Mesh splitQuad(bool mirrored, Float2 seamOffset) {
  Mesh mesh;
  // Left half: vertices 0-3.
  const float xs[3] = { 0.0f, 0.5f, 1.0f };
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      mesh.positions.push_back({ xs[i], float(j), 0.0f });
      mesh.texcoords.push_back({ xs[i], float(j) });
    }
  }
  // 0 (0,0), 1 (0,1), 2 (0.5,0), 3 (0.5,1)
  mesh.indices.insert(mesh.indices.end(), { 0, 2, 1, 1, 2, 3 });
  if (mirrored) {
    mesh.positions.push_back({ 1.0f, 0.0f, 0.0f });
    mesh.texcoords.push_back({ 0.0f, 0.0f });
    mesh.positions.push_back({ 1.0f, 1.0f, 0.0f });
    mesh.texcoords.push_back({ 0.0f, 1.0f });
    // 4 (1,0), 5 (1,1); the split vertices 2 and 3 have u = 0.5 on both sides.
    mesh.indices.insert(mesh.indices.end(), { 2, 4, 3, 3, 4, 5 });
  } else {
    for (uint32_t i = 1; i < 3; ++i) {
      for (uint32_t j = 0; j < 2; ++j) {
        mesh.positions.push_back({ xs[i], float(j), 0.0f });
        mesh.texcoords.push_back(Float2 { xs[i], float(j) } + seamOffset);
      }
    }
    // 4 (0.5,0), 5 (0.5,1), 6 (1,0), 7 (1,1): split copies of 2 and 3 on the seam.
    mesh.indices.insert(mesh.indices.end(), { 4, 6, 5, 5, 6, 7 });
  }
  return mesh;
}

// A smooth field, |grad| <= 0.3 * sqrt(5^2 + 4^2) < 2: conservative cone ratio 0.5.
Field smoothField(bool cones) {
  Field field;
  field.depth = [](Float2 uv) { return 0.45f + 0.3f * std::sin(5.0f * uv.x + 1.0f) * std::cos(4.0f * uv.y + 0.5f); };
  field.cone = cones ? 0.5f : 0.0f;
  return field;
}

// ---------------------------------------------------------------------------------------------------------------------
// Tests.

void testSolveRoundTrip(const std::string& name, const Mesh& mesh, float worldDepth) {
  const ShellData data = buildShell(mesh);
  uint32_t seed = 12345;
  const auto random = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return float(seed >> 8) / float(1u << 24);
  };
  for (uint32_t tri = 0; tri < mesh.triangleCount(); ++tri) {
    Prism prism;
    check(loadPrism(mesh, data, tri, worldDepth, 1.0f, prism), name + ": prism loads");
    for (int i = 0; i < 20; ++i) {
      float b1 = random();
      float b2 = random();
      if (b1 + b2 > 1.0f) {
        b1 = 1.0f - b1;
        b2 = 1.0f - b2;
      }
      const Float3 truth { b1, b2, random() };
      // A warm start a step away, as the march has it.
      Float3 x { truth.x + 0.02f, truth.y - 0.015f, truth.z + 0.03f };
      Jacobian inverse;
      const bool solved = solve(prism, prism.position(truth), -1.0f, x, inverse);
      check(solved, name + ": solve converges");
      check(std::fabs(x.x - truth.x) < 1e-4f && std::fabs(x.y - truth.y) < 1e-4f && std::fabs(x.z - truth.z) < 1e-4f,
            name + ": solve recovers the prism coordinates");
    }
  }
}

void testSharedSides(const std::string& name, const Mesh& mesh, float worldDepth) {
  const ShellData data = buildShell(mesh);
  uint32_t shared = 0;
  for (uint32_t tri = 0; tri < mesh.triangleCount(); ++tri) {
    Prism prism;
    loadPrism(mesh, data, tri, worldDepth, 1.0f, prism);
    for (uint32_t e = 0; e < 3; ++e) {
      const uint32_t word = data.neighbours[3 * tri + e];
      if (isBoundary(word)) {
        continue;
      }
      ++shared;
      Prism next;
      loadPrism(mesh, data, word & kNeighbourTriangleMask, worldDepth, 1.0f, next);
      for (float along : { 0.1f, 0.5f, 0.85f }) {
        for (float h : { 0.0f, 0.4f, 1.0f }) {
          // On side e: b_e = 0, the other two along the edge.
          float b[3] = { 0.0f, 0.0f, 0.0f };
          b[(e + 1) % 3] = 1.0f - along;
          b[(e + 2) % 3] = along;
          const Float3 x { b[1], b[2], h };
          const Float3 carried = carry(x, e, word >> kNeighbourSlotShift);
          const Float3 here = prism.p0 + prism.position(x);
          const Float3 there = next.p0 + next.position(carried);
          check(length(here - there) < 1e-5f, name + ": shared side " + std::to_string(tri) + "/" + std::to_string(e) + " matches");
        }
      }
    }
  }
  check(shared > 0, name + ": has shared sides");
}

// The flat quad against plain POM: in a flat prism with parallel shell directions the march is the flat-plane march.
void testFlatQuadEqualsPom(bool cones) {
  const std::string name = std::string("flat quad") + (cones ? " (cones)" : " (linear)");
  const Mesh mesh = flatQuad();
  const ShellData data = buildShell(mesh);
  const Field field = smoothField(cones);
  const float worldDepth = 0.2f;
  // Entry in triangle 0 near the diagonal, heading across it into triangle 1, about 60 degrees from the normal.
  const Float2 b12 { 0.35f, 0.45f };
  const Float3 rayDirection = normalize(Float3 { 0.7f, 0.6f, -0.55f });
  const Result result = march(mesh, data, field, 0, b12, rayDirection, worldDepth, Params {});
  // Plain POM: uv(h) = uv0 + dirUv h, dirUv = dir.xy / -dir.z * D (UV = world here), first h with d(uv(h)) <= h.
  const Float2 uv0 { 0.35f, 0.45f };
  const Float2 dirUv = Float2 { rayDirection.x, rayDirection.y } * (worldDepth / -rayDirection.z);
  float low = 0.0f;
  float high = -1.0f;
  for (float h = 0.0f; h <= 1.0f; h += 1e-5f) {
    if (field.depth(uv0 + dirUv * h) - h <= 0.0f) {
      high = h;
      break;
    }
    low = h;
  }
  check(high >= 0.0f, name + ": plain POM reference hits");
  for (int i = 0; i < 30; ++i) {
    const float middle = 0.5f * (low + high);
    if (field.depth(uv0 + dirUv * middle) - middle > 0.0f) {
      low = middle;
    } else {
      high = middle;
    }
  }
  const Float2 pomUv = uv0 + dirUv * high;
  const float pomT = high * worldDepth / -rayDirection.z;
  check(result.reason == kResultHit, name + ": hit (reason " + std::to_string(result.reason) + ")");
  check(result.crossings >= 1, name + ": crossed the diagonal");
  check(length(result.uv - pomUv) < 1e-3f, name + ": texture coordinates equal plain POM's (" +
        std::to_string(result.uv.x) + ", " + std::to_string(result.uv.y) + ") against (" + std::to_string(pomUv.x) + ", " +
        std::to_string(pomUv.y) + ")");
  check(std::fabs(result.t - pomT) < 1e-3f * worldDepth + 1e-5f, name + ": distance equals plain POM's");
  check(std::fabs(result.x.z - high) < 2e-3f, name + ": depth equals plain POM's");
}

void testBentStrip() {
  for (float lift : { 0.4f, -0.4f }) {
    const std::string name = lift > 0.0f ? "bent strip (concave)" : "bent strip (convex)";
    const Mesh mesh = bentStrip(lift);
    testSolveRoundTrip(name, mesh, 0.15f);
    testSharedSides(name, mesh, 0.15f);
    for (bool cones : { false, true }) {
      const Field field = smoothField(cones);
      if (lift > 0.0f) {
        // A grazing ray from triangle 0 across the crease into the second triangle, which rises towards it.
        compareWithReference(name + (cones ? " (cones)" : ""), mesh, field, 0, { 0.4f, 0.35f },
                             normalize(Float3 { 0.7f, 0.6f, -0.2f }), 0.15f, 1, 2e-3f * 0.15f, 2e-3f);
      } else {
        // Over the ridge: steep enough to keep going in after the crease, entering close to it.
        compareWithReference(name + (cones ? " (cones)" : ""), mesh, field, 0, { 0.5f, 0.45f },
                             normalize(Float3 { 0.6f, 0.5f, -0.62f }), 0.15f, 1, 2e-3f * 0.15f, 2e-3f);
      }
    }
  }
}

void testCylinder() {
  const Mesh mesh = cylinderSegment();
  const float worldDepth = 0.2f;
  testSolveRoundTrip("cylinder", mesh, worldDepth);
  testSharedSides("cylinder", mesh, worldDepth);
  // A grazing ray along the circumference (about 60 degrees from the normal), running into the convex surface along a
  // chord: it crosses several quads (two triangles each) before it reaches the relief. Its deepest point lies about
  // 0.13 below the surface (h 0.67), below the field everywhere.
  Field field;
  field.depth = [](Float2 uv) { return 0.5f + 0.1f * std::sin(9.0f * uv.x) * std::cos(3.0f * uv.y); };
  // Entry on quad 1 (triangles 2 and 3).
  const uint32_t tri = 2;
  const Float2 b12 { 0.3f, 0.3f };
  const Float3 entry = entryPointOf(mesh, tri, b12);
  const Float3 normal = normalize(Float3 { entry.x, 0.0f, entry.z });
  const Float3 tangent { normal.z, 0.0f, -normal.x };
  const Float3 rayDirection = normalize(tangent * 0.85f - normal * 0.5f + Float3 { 0.0f, 0.12f, 0.0f });
  compareWithReference("cylinder", mesh, field, tri, b12, rayDirection, worldDepth, 2, 2e-3f * worldDepth, 3e-3f);
}

void testMirroredUvs() {
  const Mesh mesh = splitQuad(true, { 0.0f, 0.0f });
  testSharedSides("mirrored", mesh, 0.2f);
  // An asymmetric field: mirroring matters. Deeper than the ray gets on the left half; it meets the ray in the right
  // (mirrored) half, where u runs back from 0.5 to 0.
  Field field;
  field.depth = [](Float2 uv) { return 0.5f + 0.25f * std::sin(7.0f * uv.x + 0.3f) + 0.1f * uv.y; };
  // From the left half (entry (0.3, 0.3)) at a grazing angle across the left diagonal, the split at x = 0.5 and the
  // right half's diagonal.
  compareWithReference("mirrored", mesh, field, 0, { 0.6f, 0.3f }, normalize(Float3 { 0.9f, 0.2f, -0.2f }), 0.2f, 2,
                       2e-3f * 0.2f, 2e-3f);
}

void testUvSeam() {
  const Mesh mesh = splitQuad(false, { 0.37f, 0.11f });
  testSharedSides("seam", mesh, 0.2f);
  {
    // Deeper than the ray gets before the seam; it meets the ray beyond, in the right half's own (offset) UVs.
    const Field field = smoothField(false);
    compareWithReference("seam", mesh, field, 0, { 0.6f, 0.3f }, normalize(Float3 { 0.9f, 0.2f, -0.2f }), 0.2f, 2,
                         2e-3f * 0.2f, 2e-3f);
  }
  {
    // A wall: deep (0.95) for u < 0.85 (the left half), shallow (0.05) beyond (the right half, whose u starts at
    // 0.87). The ray reaches the seam (x = 0.5) at about h 0.56, below the right half's field, so it stops at the seam.
    Field field;
    field.depth = [](Float2 uv) { return uv.x < 0.85f ? 0.95f : 0.05f; };
    const ShellData data = buildShell(mesh);
    const Float3 rayDirection = normalize(Float3 { 0.8f, 0.1f, -0.6f });
    // Entry (0.35, 0.25) in triangle 0; the ray crosses the left half's diagonal, then the seam.
    const Float2 b12 { 0.7f, 0.25f };
    const Result result = march(mesh, data, field, 0, b12, rayDirection, 0.2f, Params {});
    const Reference ref = reference(mesh, data, field, 0.2f, entryOrientation(mesh, 0, rayDirection),
                                    entryPointOf(mesh, 0, b12), rayDirection, 0.2f / 4000.0f, 4.0f);
    check(result.reason == kResultHit && ref.hit, "seam wall: both hit");
    check(result.crossings == 2, "seam wall: two crossings (diagonal, seam), got " + std::to_string(result.crossings));
    const Float3 hitPoint = entryPointOf(mesh, 0, b12) + rayDirection * result.t;
    check(std::fabs(hitPoint.x - 0.5f) < 1e-3f, "seam wall: the hit lies on the seam (x = " + std::to_string(hitPoint.x) + ")");
    check(std::fabs(result.t - ref.t) < 1e-3f, "seam wall: distance matches the reference");
    check(result.uv.x >= 0.87f - 1e-3f, "seam wall: texture coordinates from the right half's UVs");
  }
}

void testBoundaryExit() {
  // The flat quad, a field that never comes up (depth 1 everywhere): a ray towards the open edge leaves the volume
  // through it before it reaches the bottom.
  const Mesh mesh = flatQuad();
  const ShellData data = buildShell(mesh);
  Field field;
  field.depth = [](Float2) { return 1.0f; };
  const Float2 b12 { 0.05f, 0.1f };
  const Float3 rayDirection = normalize(Float3 { -0.9f, 0.0f, -0.3f });
  const float worldDepth = 0.2f;
  const Result result = march(mesh, data, field, 0, b12, rayDirection, worldDepth, Params {});
  check(result.reason == kResultBoundary, "boundary: stops at the open edge (reason " + std::to_string(result.reason) + ")");
  check(result.boundary == kBoundaryOpen, "boundary: an open edge");
  check(result.x.x + result.x.y < 1.0f && clampToPrism(result.x).z > 0.0f, "boundary: stop point inside the prism");
  // The exit lies on the open edge (x = 0).
  const Float3 entry = entryPointOf(mesh, 0, b12);
  const float exitT = -entry.x / rayDirection.x;
  check(std::fabs(result.t - exitT) < 1e-3f * worldDepth, "boundary: exit distance at the edge");
}

void testOuterExit() {
  // Carved silhouettes: the convex ridge, a field that never comes up: a shallow ray goes in on the first triangle and leaves
  // through the top of the second, which falls away faster than the ray goes down (a silhouette).
  const Mesh mesh = bentStrip(-0.4f);
  const ShellData data = buildShell(mesh);
  Field field;
  field.depth = [](Float2) { return 1.0f; };
  const float worldDepth = 0.15f;
  const Float2 b12 { 0.4f, 0.35f };
  const Float3 entry = entryPointOf(mesh, 0, b12);
  for (const Float3 direction : { Float3 { 0.7f, 0.6f, -0.1f }, Float3 { 0.7f, 0.6f, -0.15f }, Float3 { 0.6f, 0.6f, -0.05f } }) {
    const Float3 rayDirection = normalize(direction);
    const Result result = march(mesh, data, field, 0, b12, rayDirection, worldDepth, Params {});
    check(result.reason == kResultOuter, "outer: leaves through the top (reason " + std::to_string(result.reason) + ")");
    check(result.triangle == 1 && result.crossings == 1, "outer: in the second triangle");
    // The true exit: where the ray meets the second triangle's plane.
    const Float3 p0 = mesh.positions[mesh.indices[3]];
    const Float3 n = normalize(cross(mesh.positions[mesh.indices[4]] - p0, mesh.positions[mesh.indices[5]] - p0));
    const float exitT = dot(p0 - entry, n) / dot(rayDirection, n);
    check(std::fabs(result.t - exitT) < 1e-3f * worldDepth, "outer: exit distance " + std::to_string(result.t) + " against " + std::to_string(exitT));
  }
}

// A unit cube with split vertices per face (two triangles each, outward winding) and per-face UVs (the face's own
// coordinates), as test_sccr_shell.cpp's: the top face (+z) is triangles 8 and 9.
Mesh unitCube() {
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
    for (const Float2 ab : { Float2 { 0, 0 }, Float2 { 1, 0 }, Float2 { 1, 1 }, Float2 { 0, 1 } }) {
      mesh.positions.push_back(f.corner + f.u * ab.x + f.v * ab.y);
      mesh.texcoords.push_back(ab);
    }
    mesh.indices.insert(mesh.indices.end(), { base, base + 1, base + 2, base, base + 2, base + 3 });
  }
  return mesh;
}

void testFoldGuard() {
  // The fold guard. At D = 1 the cube's face prisms (corner directions along the diagonals) shrink to a point at
  // h = sqrt(3) / 2 and fold beyond; the guard caps every corner at sqrt(3) / 4. A steep ray into the middle of the top
  // face, against a field at depth 0.95.
  const Mesh mesh = unitCube();
  Field field;
  field.depth = [](Float2) { return 0.95f; };
  const uint32_t tri = 8;
  const Float2 b12 { 0.05f, 0.47f };  // (0.52, 0.47) on the top face
  for (const Float3 direction : { Float3 { 0.05f, 0.02f, -1.0f }, Float3 { 0.3f, 0.1f, -1.0f } }) {
    const Float3 rayDirection = normalize(direction);
    compareWithReference("fold guard", mesh, field, tri, b12, rayDirection, 1.0f, 0, 2e-3f, 2e-3f);
    ShellData data = buildShell(mesh);
    check(std::fabs(data.limits[0].x - 0.4330127f) < 1e-5f, "fold guard: corner limit sqrt(3) / 4");
    data.foldGuard = false;
    const Result unguarded = march(mesh, data, field, tri, b12, rayDirection, 1.0f, Params {});
    const Float3 end = entryPointOf(mesh, tri, b12) + rayDirection * unguarded.t;
    // What the guard prevents: through the folded prisms the walk reaches the cube's bottom face and leaves there.
    check(unguarded.reason == kResultOuter && end.z < 0.01f,
          "fold guard off: the ray walks out of the bottom (reason " + std::to_string(unguarded.reason) + ", z " + std::to_string(end.z) + ")");
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Shell shadows: the shadow march (sccrShellCarvedPointLoad, sccrShellShadowMarch) with an identity instance transform, the
// step LOD off and every cone counted as conservative (CoarseOnly).

struct ShadowResult {
  bool found = false;  // a carved point (the shader's march returned true)
  uint32_t reason = kResultFailed;
  float visibility = 1.0f;
  uint32_t steps = 0;
  uint32_t crossings = 0;
  float rampStep = 0.0f;
  float progressRate = 0.0f;
  float t = 0.0f;
  // 12.14 (sccrShellShadowExit): where an OUTER march left the shell, and that prism's outward normal.
  bool exited = false;
  Float3 exitPosition {};
  Float3 exitNormal {};
};

void shadowExit(ShadowResult& shadow, const Prism& prism, Float3 position) {
  shadow.exited = true;
  shadow.exitPosition = position;
  const Float3 triangleNormal = cross(prism.e1, prism.e2);
  const Float3 middleShell = prism.s0 + (prism.f1 + prism.f2) * (1.0f / 3.0f);
  shadow.exitNormal = dot(triangleNormal, middleShell) >= 0.0f ? triangleNormal : -triangleNormal;
}

// The carved point: barycentrics from the texture coordinates by the triangle's affine UV map, h = d(uv).
bool carvedPoint(const Mesh& mesh, const ShellData& data, const Field& field, uint32_t tri, Float2 uv, float worldDepth,
                 float orientation, Prism& prism, Float3& x) {
  if (!loadPrism(mesh, data, tri, worldDepth, orientation, prism)) {
    return false;
  }
  const float det = prism.duv1.x * prism.duv2.y - prism.duv1.y * prism.duv2.x;
  if (!(std::fabs(det) > 1e-6f * length(prism.duv1) * length(prism.duv2))) {
    return false;
  }
  const Float2 d = uv - prism.uv0;
  const float b1 = (d.x * prism.duv2.y - d.y * prism.duv2.x) / det;
  const float b2 = (prism.duv1.x * d.y - prism.duv1.y * d.x) / det;
  x = clampToPrism(Float3 { b1, b2, field.depth(uv) });
  return true;
}

// lightDirection: unit, world (= object). orientation: the primary march's.
ShadowResult shadowMarch(const Mesh& mesh, const ShellData& data, const Field& field, uint32_t tri, Float2 uv,
                         Float3 lightDirection, float worldDepth, float orientation, const Params& params) {
  ShadowResult shadow;
  Prism prism;
  Float3 x;
  if (!carvedPoint(mesh, data, field, tri, uv, worldDepth, orientation, prism, x)) {
    return shadow;
  }
  shadow.found = true;
  const float texelDepth = x.z;
  if (!(texelDepth > 0.0f)) {
    shadow.reason = kResultOuter;
    shadowExit(shadow, prism, prism.p0 + prism.position(x));
    return shadow;
  }
  const float detSign = -orientation;
  const Float3 rayDirection = lightDirection;
  const Float3 entryPoint = prism.p0 + prism.position(x);
  Float3 entryRelative = entryPoint - prism.p0;
  Jacobian inverse;
  if (!jacobianInverse(prism, x, detSign, inverse)) {
    return shadow;
  }
  Float3 rate = inverse.mul(rayDirection);
  if (!(rate.z < 0.0f)) {
    return shadow;
  }
  const float cosLight = saturate(-rate.z * worldDepth);
  const float numSamples = lerp(float(params.maxIterations), params.maxIterations / 4.0f, cosLight);
  const float minStepDepth = 1.0f / numSamples;
  const float rateFloor = kMinRate / worldDepth;
  const float objectMaxStepScale = 1.0f / length(rayDirection);
  const float rampStep = std::min(minStepDepth, texelDepth);
  const float progressRate = -rate.z;
  const float occludedT = 1.5f * rampStep / progressRate;
  shadow.rampStep = rampStep;
  shadow.progressRate = progressRate;
  const uint32_t maxPrisms = params.maxShadowPrisms;
  const uint32_t maxSteps = std::max(std::min(params.maxSteps,
    params.coneShadowSteps + uint32_t(std::ceil(texelDepth / rampStep)) + 1 + kStepsPerCrossing * maxPrisms), 1u);

  float t = 0.0f;
  uint32_t coneStepsLeft = params.coneShadowSteps;
  float visibility = 1.0f;
  float gap = 0.0f;
  bool sampleHere = false;
  bool hasBelow = false;
  bool belowAcross = false;
  float belowT = 0.0f;
  Float3 belowX = x;
  float belowGap = 0.0f;
  bool pending = false;
  bool pendingFromStep = false;
  uint32_t pendingFace = kFaceNone;
  const auto finish = [&](uint32_t reason, float value) {
    shadow.reason = reason;
    shadow.visibility = saturate(value);
    shadow.t = t;
    return shadow;
  };
  const uint32_t iterationLimit = maxSteps + maxPrisms + 2;
  for (uint32_t iteration = 0; iteration < iterationLimit; ++iteration) {
    if (!pending || pendingFromStep) {
      if (sampleHere) {
        gap = field.depth(prism.textureCoordinates(x)) - x.z;
        if (gap <= 0.0f) {
          if (t >= occludedT) {
            return finish(kResultHit, 0.0f);
          }
          hasBelow = true;
          belowAcross = false;
          belowT = t;
          belowX = x;
          belowGap = gap;
        } else if (hasBelow) {
          float leaveT;
          if (belowAcross) {
            leaveT = lerp(belowT, t, saturate(-belowGap / std::max(gap - belowGap, 1e-6f)));
          } else {
            Float3 leaveX;
            refine(field, prism, entryRelative, rayDirection, detSign, params, t, x, gap, belowT, belowX, belowGap,
                   leaveT, leaveX, shadow.steps);
          }
          hasBelow = false;
          visibility = std::min(visibility, saturate(1.5f - leaveT * progressRate / rampStep));
          if (visibility <= 0.0f) {
            return finish(kResultHit, 0.0f);
          }
        }
      }
      sampleHere = true;
    }
    if (pending) {
      pending = false;
      if (pendingFace == kFaceTop) {
        shadowExit(shadow, prism, entryPoint + rayDirection * t);
        return finish(kResultOuter, visibility);
      }
      if (shadow.crossings >= maxPrisms || shadow.steps >= maxSteps) {
        return finish(kResultLimit, visibility);
      }
      const uint32_t word = data.neighbours[3 * prism.triangle + pendingFace];
      Prism next;
      if (isBoundary(word) || !loadPrism(mesh, data, word & kNeighbourTriangleMask, worldDepth, orientation, next)) {
        return finish(kResultBoundary, visibility);
      }
      Float3 nextX = carry(x, pendingFace, word >> kNeighbourSlotShift);
      const Float3 nextEntryRelative = entryPoint - next.p0;
      Jacobian nextInverse;
      if (!solve(next, nextEntryRelative + rayDirection * t, detSign, nextX, nextInverse)) {
        return finish(kResultSolve, visibility);
      }
      prism = next;
      x = nextX;
      x.z = std::min(x.z, 1.0f);
      inverse = nextInverse;
      entryRelative = nextEntryRelative;
      rate = inverse.mul(rayDirection);
      ++shadow.crossings;
      ++shadow.steps;
      coneStepsLeft = params.coneShadowSteps;
      belowAcross = hasBelow;
      pending = outside(x, false, kFaceNone, pendingFace);
      pendingFromStep = false;
      continue;
    }
    if (shadow.steps >= maxSteps) {
      return finish(kResultLimit, visibility);
    }
    const float a = rate.z;
    const float uvRate = length(prism.duv1 * rate.x + prism.duv2 * rate.y);
    float dt = minStepDepth / std::max(std::fabs(a), rateFloor);
    bool coneStep = false;
    if (gap > 0.0f && coneStepsLeft > 0 && field.cone > 0.0f) {
      dt = std::max(dt, gap / (kConeSafety * (std::max(a, 0.0f) + uvRate / field.cone)));
      coneStep = true;
    }
    dt = std::min(dt, prism.maxStep * objectMaxStepScale);
    const Float3 b = barycentrics(x);
    const Float3 db { -(rate.x + rate.y), rate.x, rate.y };
    float exitT = 1e30f;
    uint32_t exitFace = kFaceNone;
    for (uint32_t i = 0; i < 3; ++i) {
      const float rateI = component(db, i);
      if (rateI < 0.0f) {
        const float faceT = std::max(component(b, i), 0.0f) / -rateI;
        if (faceT < exitT) { exitT = faceT; exitFace = i; }
      }
    }
    if (a < 0.0f) {
      const float faceT = std::max(x.z, 0.0f) / -a;
      if (faceT < exitT) { exitT = faceT; exitFace = kFaceTop; }
    } else if (a > 0.0f) {
      const float faceT = std::max(1.0f - x.z, 0.0f) / a;
      if (faceT < exitT) { exitT = faceT; exitFace = kFaceBottom; }
    }
    bool exitLimited = false;
    if (exitT <= dt) {
      dt = exitT * (1.0f + kExitOvershoot) + kExitNudge * prism.size * objectMaxStepScale;
      exitLimited = exitFace < 3 || exitFace == kFaceTop;
    }
    t += dt;
    Float3 nextX = x + rate * dt;
    if (!solve(prism, entryRelative + rayDirection * t, detSign, nextX, inverse)) {
      return finish(kResultSolve, visibility);
    }
    x = nextX;
    x.z = std::min(x.z, 1.0f);
    rate = inverse.mul(rayDirection);
    ++shadow.steps;
    if (coneStep) {
      --coneStepsLeft;
    }
    pending = outside(x, exitLimited, exitFace, pendingFace);
    pendingFromStep = true;
  }
  return finish(kResultLimit, visibility);
}

// The brute-force reference: walks the light ray from the carved point in steps of stepT (locating every point by a
// cold-started solve in every prism) until it leaves the volume, with the shadow march's measure and ramp: 0 at a point
// below the field at distance >= 1.5 ramp steps, the ramp at every point where the ray leaves the field (bisected).
float shadowReference(const Mesh& mesh, const ShellData& data, const Field& field, uint32_t tri, Float2 uv,
                      Float3 lightDirection, float worldDepth, float orientation, float rampStep, float progressRate,
                      float stepT, float maxT) {
  Prism prism;
  Float3 x;
  carvedPoint(mesh, data, field, tri, uv, worldDepth, orientation, prism, x);
  const Float3 start = prism.p0 + prism.position(x);
  const auto below = [&](float t, bool& inside) {
    const Location location = locate(mesh, data, worldDepth, orientation, start + lightDirection * t);
    inside = location.inside;
    return inside && field.depth(location.uv) - location.x.z <= 0.0f;
  };
  float visibility = 1.0f;
  float belowT = -1.0f;
  for (float t = stepT; t <= maxT; t += stepT) {
    bool inside;
    const bool isBelow = below(t, inside);
    if (!inside) {
      return visibility;
    }
    if (isBelow) {
      if (t * progressRate >= 1.5f * rampStep) {
        return 0.0f;
      }
      belowT = t;
    } else if (belowT >= 0.0f) {
      float low = belowT;
      float high = t;
      for (int i = 0; i < 40; ++i) {
        const float middle = 0.5f * (low + high);
        bool middleInside;
        (below(middle, middleInside) ? low : high) = middle;
      }
      visibility = std::min(visibility, saturate(1.5f - high * progressRate / rampStep));
      belowT = -1.0f;
    }
  }
  return visibility;
}

// sccrShellMarkerEncode / sccrShellMarkerDecode (surface_interaction_sccr.slangh): the marker survives a float exactly,
// up to the largest triangle, with its sign; packed normals (1 to 2^22) never read as markers.
void testShellMarker() {
  constexpr float kMarkerBase = 4194306.0f;
  constexpr uint32_t kMaxTriangle = 16777215u - 4194306u;
  for (const uint32_t tri : { 0u, 1u, 12345u, 4194303u, kMaxTriangle - 1, kMaxTriangle }) {
    for (const float orientation : { 1.0f, -1.0f }) {
      const float magnitude = kMarkerBase + float(tri);
      const float marker = orientation < 0.0f ? -magnitude : magnitude;
      const float stored = marker;  // RGBA32F
      const float decodedMagnitude = std::fabs(stored);
      check(decodedMagnitude >= kMarkerBase && decodedMagnitude < 16777216.0f, "marker: recognised");
      check(uint32_t(decodedMagnitude - kMarkerBase) == tri, "marker: triangle " + std::to_string(tri) + " exact");
      check((stored < 0.0f ? -1.0f : 1.0f) == orientation, "marker: orientation");
    }
  }
  check(4194304.0f < kMarkerBase, "marker: above every packed normal (at most 2^22)");
}

// A pit in the flat quad (D = 0.2): depth 0.9 within 0.12 of (0.5, 0.5), 0.1 around it (vertical walls).
Field pitField() {
  Field field;
  field.depth = [](Float2 uv) {
    const float du = uv.x - 0.5f;
    const float dv = uv.y - 0.5f;
    return du * du + dv * dv < 0.12f * 0.12f ? 0.9f : 0.1f;
  };
  return field;
}

void testShadowPit() {
  const Mesh mesh = flatQuad();
  const ShellData data = buildShell(mesh);
  const Field field = pitField();
  const float worldDepth = 0.2f;
  const Float2 uv { 0.42f, 0.45f };  // triangle 0, in the pit
  {
    // Low light across the diagonal: the ray leaves the pit floor, crosses into triangle 1 and runs into the rim's wall
    // at about h 0.37, below the rim's field (0.1).
    const Float3 light = normalize(Float3 { 0.78f, 0.62f, 0.5f });
    const ShadowResult shadow = shadowMarch(mesh, data, field, 0, uv, light, worldDepth, 1.0f, Params {});
    check(shadow.found && shadow.reason == kResultHit && shadow.visibility == 0.0f,
          "shadow pit: occluded by the rim (reason " + std::to_string(shadow.reason) + ")");
    check(shadow.crossings >= 1, "shadow pit: crossed the diagonal on the way out");
    const float reference = shadowReference(mesh, data, field, 0, uv, light, worldDepth, 1.0f, shadow.rampStep,
                                            shadow.progressRate, worldDepth / 4000.0f, 4.0f * worldDepth);
    check(reference == 0.0f, "shadow pit: the reference is occluded");
  }
  {
    // Steep light: the ray rises out of the pit before it reaches a wall: open sky.
    const Float3 light = normalize(Float3 { 0.1f, 0.05f, 1.0f });
    const ShadowResult shadow = shadowMarch(mesh, data, field, 0, uv, light, worldDepth, 1.0f, Params {});
    check(shadow.reason == kResultOuter && shadow.visibility == 1.0f, "shadow pit: lit through the top under open sky");
    const float reference = shadowReference(mesh, data, field, 0, uv, light, worldDepth, 1.0f, shadow.rampStep,
                                            shadow.progressRate, worldDepth / 4000.0f, 4.0f * worldDepth);
    check(reference == 1.0f, "shadow pit: the reference is lit");
  }
}

void testShadowCrossing() {
  // A level field (0.6): a light ray from a texel next to the diagonal crosses into triangle 1 and leaves through the
  // top there, lit, at h0 D / L.z.
  const Mesh mesh = flatQuad();
  const ShellData data = buildShell(mesh);
  Field field;
  field.depth = [](Float2) { return 0.6f; };
  const float worldDepth = 0.2f;
  const Float3 light = normalize(Float3 { 0.8f, 0.7f, 0.6f });
  const ShadowResult shadow = shadowMarch(mesh, data, field, 0, { 0.45f, 0.5f }, light, worldDepth, 1.0f, Params {});
  check(shadow.reason == kResultOuter && shadow.visibility == 1.0f && shadow.crossings == 1,
        "shadow crossing: lit through the top of the neighbour");
  const float exitT = 0.6f * worldDepth / light.z;
  check(std::fabs(shadow.t - exitT) < 2e-3f * worldDepth,
        "shadow crossing: leaves at the top (" + std::to_string(shadow.t) + " against " + std::to_string(exitT) + ")");
  // 12.14: the scene's shadow ray starts at the exit, on the neighbour's top (z = 0) lifted along its outward normal
  // (+z) by kSccrShellShadowExitOffset x D: above the mesh, and the light leaves through that face outwards.
  check(shadow.exited && std::fabs(shadow.exitPosition.z) < 2e-4f * worldDepth,
        "shadow crossing: exit on the top (z " + std::to_string(shadow.exitPosition.z) + ")");
  check(shadow.exited && normalize(shadow.exitNormal).z > 0.9999f && dot(shadow.exitNormal, light) > 0.0f,
        "shadow crossing: exit normal out of the mesh, towards the light");
  const Float3 origin = shadow.exitPosition + normalize(shadow.exitNormal) * (1e-3f * worldDepth);
  check(origin.z > 0.0f, "shadow crossing: the scene's shadow ray starts above the surface");
}

void testShadowCrease() {
  // A concave crease (the second triangle rises 0.8 over the first), level field 0.3, D = 0.15. A low light from the
  // floor towards the slope runs into the slope's relief: in the slope's prisms it goes deeper than the texel (h > h0)
  // while it passes through rock, so only the along-ray measure calls it occluded. A steeper light leaves through the
  // floor's top first.
  const Mesh mesh = bentStrip(0.8f);
  const ShellData data = buildShell(mesh);
  Field field;
  field.depth = [](Float2) { return 0.3f; };
  const float worldDepth = 0.15f;
  struct Case { Float2 uv; Float3 light; bool lit; };
  for (const Case& c : { Case { { 0.4f, 0.35f }, { 0.7f, 0.6f, 0.2f }, false },
                         Case { { 0.45f, 0.4f }, { 0.7f, 0.6f, 0.1f }, false },
                         Case { { 0.4f, 0.4f }, { 0.6f, 0.6f, 0.3f }, true } }) {
    const Float3 light = normalize(c.light);
    const ShadowResult shadow = shadowMarch(mesh, data, field, 0, c.uv, light, worldDepth, 1.0f, Params {});
    const float reference = shadowReference(mesh, data, field, 0, c.uv, light, worldDepth, 1.0f, shadow.rampStep,
                                            shadow.progressRate, worldDepth / 3000.0f, 3.0f * worldDepth);
    check(shadow.visibility == (c.lit ? 1.0f : 0.0f), std::string("shadow crease: ") + (c.lit ? "lit" : "occluded") +
          " (visibility " + std::to_string(shadow.visibility) + ", reason " + std::to_string(shadow.reason) + ")");
    check(reference == shadow.visibility, "shadow crease: matches the reference");
    if (!c.lit) {
      check(shadow.crossings >= 1, "shadow crease: crossed into the slope");
    }
  }
}


// ---------------------------------------------------------------------------------------------------------------------
// A smoothed low-poly boulder: an icosphere (subdivided `levels` times) with a smooth radial bump, outward winding, and
// box-projected UVs (each triangle takes the axis its normal is closest to; vertices split where the axis changes, at
// the same position, so they weld). uvScale UV units per world unit.
Mesh boulder(uint32_t levels, float uvScale) {
  std::vector<Float3> p;
  const float g = 0.5f * (1.0f + std::sqrt(5.0f));
  const float base[12][3] = { { -1, g, 0 }, { 1, g, 0 }, { -1, -g, 0 }, { 1, -g, 0 }, { 0, -1, g }, { 0, 1, g },
                              { 0, -1, -g }, { 0, 1, -g }, { g, 0, -1 }, { g, 0, 1 }, { -g, 0, -1 }, { -g, 0, 1 } };
  for (const auto& v : base) {
    p.push_back(normalize(Float3 { v[0], v[1], v[2] }));
  }
  std::vector<uint32_t> tris = { 0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6,
                                 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7,
                                 9, 8, 1 };
  for (uint32_t level = 0; level < levels; ++level) {
    std::vector<uint32_t> next;
    std::vector<std::pair<uint64_t, uint32_t>> midpoints;
    const auto midpoint = [&](uint32_t a, uint32_t b) {
      const uint64_t key = (uint64_t(std::min(a, b)) << 32) | std::max(a, b);
      for (const auto& m : midpoints) {
        if (m.first == key) {
          return m.second;
        }
      }
      p.push_back(normalize((p[a] + p[b]) * 0.5f));
      midpoints.push_back({ key, uint32_t(p.size() - 1) });
      return uint32_t(p.size() - 1);
    };
    for (size_t t = 0; t < tris.size(); t += 3) {
      const uint32_t a = tris[t], b = tris[t + 1], c = tris[t + 2];
      const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
      next.insert(next.end(), { a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca });
    }
    tris.swap(next);
  }
  for (Float3& v : p) {
    const float r = 1.0f + 0.12f * std::sin(2.0f * v.x + 1.0f) * std::cos(1.5f * v.y) + 0.08f * std::sin(2.5f * v.z + 2.0f);
    v = v * r;
    v.y *= 0.8f;  // a little flattened, like a boulder
  }
  Mesh mesh;
  std::vector<int32_t> split(p.size() * 3, -1);
  for (size_t t = 0; t < tris.size(); t += 3) {
    const Float3 n = cross(p[tris[t + 1]] - p[tris[t]], p[tris[t + 2]] - p[tris[t]]);
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    const uint32_t axis = ax >= ay && ax >= az ? 0 : (ay >= az ? 1 : 2);
    for (uint32_t k = 0; k < 3; ++k) {
      const uint32_t v = tris[t + k];
      int32_t& id = split[3 * v + axis];
      if (id < 0) {
        id = int32_t(mesh.positions.size());
        mesh.positions.push_back(p[v]);
        const Float3 q = p[v] * uvScale;
        mesh.texcoords.push_back(axis == 0 ? Float2 { q.y, q.z } : (axis == 1 ? Float2 { q.x, q.z } : Float2 { q.x, q.y }));
      }
      mesh.indices.push_back(uint32_t(id));
    }
  }
  return mesh;
}

// Moller-Trumbore. front: triangles facing the ray (outward winding against the direction); otherwise those facing
// away. The nearest hit beyond tMin.
bool rayMesh(const Mesh& mesh, Float3 origin, Float3 direction, bool front, float tMin, uint32_t& tri, Float2& b12, float& tHit) {
  tHit = std::numeric_limits<float>::infinity();
  for (uint32_t t = 0; t < mesh.triangleCount(); ++t) {
    const Float3 p0 = mesh.positions[mesh.indices[3 * t]];
    const Float3 e1 = mesh.positions[mesh.indices[3 * t + 1]] - p0;
    const Float3 e2 = mesh.positions[mesh.indices[3 * t + 2]] - p0;
    const float facing = dot(direction, cross(e1, e2));
    if ((front && facing >= 0.0f) || (!front && facing <= 0.0f)) {
      continue;
    }
    const Float3 pv = cross(direction, e2);
    const float det = dot(e1, pv);
    const Float3 tv = origin - p0;
    const float u = dot(tv, pv) / det;
    if (u < 0.0f || u > 1.0f) {
      continue;
    }
    const Float3 qv = cross(tv, e1);
    const float v = dot(direction, qv) / det;
    if (v < 0.0f || u + v > 1.0f) {
      continue;
    }
    const float d = dot(e2, qv) / det;
    if (d > tMin && d < tHit) {
      tHit = d;
      tri = t;
      b12 = { u, v };
    }
  }
  return std::isfinite(tHit);
}

// Distance from point q to triangle (a, b, c) (Ericson, Real-Time Collision Detection 5.1.5).
float pointTriangleDistance(Float3 q, Float3 a, Float3 b, Float3 c) {
  const Float3 ab = b - a, ac = c - a, ap = q - a;
  const float d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0.0f && d2 <= 0.0f) return length(q - a);
  const Float3 bp = q - b;
  const float d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0.0f && d4 <= d3) return length(q - b);
  const float vc = d1 * d4 - d3 * d2;
  if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return length(q - (a + ab * (d1 / (d1 - d3))));
  const Float3 cp = q - c;
  const float d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0.0f && d5 <= d6) return length(q - c);
  const float vb = d5 * d2 - d1 * d6;
  if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return length(q - (a + ac * (d2 / (d2 - d6))));
  const float va = d3 * d6 - d5 * d4;
  if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
    return length(q - (b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))));
  }
  const float denom = 1.0f / (va + vb + vc);
  return length(q - (a + ab * (vb * denom) + ac * (vc * denom)));
}

float meshDistance(const Mesh& mesh, Float3 q) {
  float best = std::numeric_limits<float>::infinity();
  for (uint32_t t = 0; t < mesh.triangleCount(); ++t) {
    best = std::min(best, pointTriangleDistance(q, mesh.positions[mesh.indices[3 * t]], mesh.positions[mesh.indices[3 * t + 1]],
                                                mesh.positions[mesh.indices[3 * t + 2]]));
  }
  return best;
}

// 12.14: grazing views of a smoothed low-poly boulder (many small curved triangles, box-projected UVs with welded
// seams) at strong depth, the in-game case where the old default of 8 prisms ran out all along the outline.
void testBoulderGrazing() {
  const Mesh mesh = boulder(3, 1.0f);
  const ShellData data = buildShell(mesh);
  uint32_t boundarySides = 0;
  for (uint32_t w : data.neighbours) {
    boundarySides += isBoundary(w) ? 1u : 0u;
  }
  check(boundarySides == 0, "boulder: closed, every side welded across the UV seams");
  Field field;
  field.depth = [](Float2 uv) { return 0.5f + 0.35f * std::sin(6.0f * uv.x + 1.0f) * std::cos(5.0f * uv.y + 0.5f); };
  field.cone = 1.0f / (0.35f * std::sqrt(61.0f));
  const Float3 direction = normalize(Float3 { 0.1f, -0.05f, -1.0f });
  // Triangle edges are about 0.15: D = 0.15 is one edge, 0.3 two.
  for (const float worldDepth : { 0.15f, 0.3f }) {
    const std::string name = "boulder D " + std::to_string(worldDepth);
    for (const uint32_t maxPrisms : { 32u, 8u }) {
      Params params;
      params.maxPrisms = maxPrisms;
      uint32_t rays = 0, hits = 0, outer = 0, limits = 0, others = 0, usedLimits = 0, falseOuter = 0, lateOuter = 0;
      uint32_t pingPong = 0, maxSteps = 0, maxCrossings = 0, tooDeep = 0;
      double sumSteps = 0.0;
      float deepest = 0.0f;
      std::vector<uint32_t> path;
      g_marchPath = &path;
      g_fieldSamples = 0;
      for (float py = -1.3f; py <= 1.3f; py += 0.02f) {
        for (float px = -1.3f; px <= 1.3f; px += 0.02f) {
          uint32_t tri; Float2 b12; float tHit;
          if (!rayMesh(mesh, Float3 { px, py, 5.0f }, direction, true, 0.0f, tri, b12, tHit)) {
            continue;
          }
          ++rays;
          const Result r = march(mesh, data, field, tri, b12, direction, worldDepth, params);
          sumSteps += r.steps;
          maxSteps = std::max(maxSteps, r.steps);
          maxCrossings = std::max(maxCrossings, r.crossings);
          for (size_t i = 2; i < path.size(); ++i) {
            if (path[i] == path[i - 2]) {
              ++pingPong;
              break;
            }
          }
          usedLimits += r.reason == kResultLimit && r.used ? 1u : 0u;
          if (r.reason == kResultHit) {
            ++hits;
          } else if (r.reason == kResultLimit) {
            ++limits;
          } else if (r.reason != kResultOuter) {
            ++others;
          }
          if (r.reason != kResultOuter || maxPrisms != 32u) {
            outer += r.reason == kResultOuter ? 1u : 0u;
            continue;
          }
          ++outer;
          // A true exit: where the ray leaves the mesh (the first back face from the entry), within a capped step's
          // overshoot; never before it (a false exit would carve a hole inside the outline).
          const Float3 entry = entryPointOf(mesh, tri, b12);
          uint32_t backTri; Float2 backB12; float tBack;
          if (rayMesh(mesh, entry, direction, false, 1e-5f, backTri, backB12, tBack)) {
            falseOuter += r.t < tBack - 2e-3f * worldDepth ? 1u : 0u;
            lateOuter += r.t > tBack + 1e-2f * worldDepth ? 1u : 0u;
            // The band: a carved ray never went deeper under the mesh than the relief reaches (0.85 D at most here).
            for (uint32_t k = 1; k < 16; ++k) {
              const float depthUnder = meshDistance(mesh, entry + direction * (tBack * float(k) / 16.0f));
              deepest = std::max(deepest, depthUnder / worldDepth);
              tooDeep += depthUnder > 0.9f * worldDepth ? 1u : 0u;
            }
          }
        }
      }
      g_marchPath = nullptr;
      std::cout << "  " << name << ", maxPrisms " << maxPrisms << ": " << rays << " rays, " << hits << " hit, " << outer
                << " outer, " << limits << " limit; steps " << sumSteps / rays << " average, " << maxSteps << " most; "
                << maxCrossings << " crossings most; " << double(g_fieldSamples) / rays << " field samples average; "
                << pingPong << " walks crossed back; deepest carved ray " << deepest << " D" << std::endl;
      check(others == 0, name + ": only hits, outer exits and limits");
      check(usedLimits == 0, name + ": a limit is never shaded where it stopped (plain SCCR)");
      if (maxPrisms == 32u) {
        check(limits == 0, name + ": no limits with the default budget");
        check(maxCrossings <= 24u, name + ": at most 24 prisms crossed, got " + std::to_string(maxCrossings));
        check(maxSteps <= 96u, name + ": at most 96 steps, got " + std::to_string(maxSteps));
        check(outer > rays / 20u, name + ": the outline is carved");
        check(falseOuter == 0, name + ": no outer exit inside the mesh (" + std::to_string(falseOuter) + ")");
        check(lateOuter == 0, name + ": every outer exit at the mesh's own exit (" + std::to_string(lateOuter) + ")");
        check(tooDeep == 0, name + ": no carved ray deeper than the relief reaches");
        check(pingPong * 200u <= rays, name + ": walks that cross straight back stay rare (" + std::to_string(pingPong) + ")");
      } else {
        // The old default: limits remain only where the ray was not rising out of the shell.
        check(limits * 20u <= rays, name + ": with 8 prisms, limits stay rare (" + std::to_string(limits) + ")");
      }
    }
  }

  // Grazing rays just inside the outline against the brute-force walk: the same hit, or both leave the volume.
  const float worldDepth = 0.15f;
  Params params;
  uint32_t compared = 0;
  for (uint32_t row = 0; row < 3 && compared < 6; ++row) {
    const float py = row == 0 ? 0.1f : (row == 1 ? -0.3f : 0.4f);
    uint32_t inRow = 0;
    // The rays lean +x: the outline is near px = 0.6 at z = 5.
    for (float px = 0.20f; px <= 0.90f && inRow < 2; px += 0.004f) {
    uint32_t tri; Float2 b12; float tHit;
    const Float3 origin { px, py, 5.0f };
    if (!rayMesh(mesh, origin, direction, true, 0.0f, tri, b12, tHit)) {
      continue;
    }
    const Float3 p0 = mesh.positions[mesh.indices[3 * tri]];
    const Float3 normal = normalize(cross(mesh.positions[mesh.indices[3 * tri + 1]] - p0, mesh.positions[mesh.indices[3 * tri + 2]] - p0));
    // Grazing: within about 65 degrees of the surface's plane; one ray further in, one at the very edge.
    if (-dot(normal, direction) > (inRow == 0 ? 0.45f : 0.2f)) {
      continue;
    }
    ++compared;
    ++inRow;
    const std::string name = "boulder grazing (" + std::to_string(px) + ", " + std::to_string(py) + ")";
    const Result r = march(mesh, data, field, tri, b12, direction, worldDepth, params);
    const Reference ref = reference(mesh, data, field, worldDepth, entryOrientation(mesh, tri, direction),
                                    entryPointOf(mesh, tri, b12), direction, worldDepth / 1000.0f, 20.0f * worldDepth);
    if (ref.hit) {
      check(r.reason == kResultHit, name + ": the march hits as the reference does (reason " + std::to_string(r.reason) + ")");
      check(r.reason != kResultHit || std::fabs(r.t - ref.t) <= 3e-3f * worldDepth,
            name + ": distance " + std::to_string(r.t) + " against " + std::to_string(ref.t));
    } else {
      check(r.reason == kResultOuter, name + ": leaves through the top as the reference does (reason " + std::to_string(r.reason) + ")");
    }
    std::cout << "  " << name << ": " << (ref.hit ? "hit" : "leaves") << ", march reason " << r.reason << ", "
              << r.crossings << " crossings" << std::endl;
    }
  }
  check(compared == 6, "boulder grazing: six grazing rays compared");
}

void run() {
  testFlatQuadEqualsPom(false);
  testFlatQuadEqualsPom(true);
  testBentStrip();
  testCylinder();
  testMirroredUvs();
  testUvSeam();
  testBoundaryExit();
  testOuterExit();
  testFoldGuard();
  testShellMarker();
  testShadowPit();
  testShadowCrossing();
  testShadowCrease();
  testBoulderGrazing();
  if (g_failures > 0) {
    throw std::runtime_error(std::to_string(g_failures) + " check(s) failed");
  }
  std::cout << "test_sccr_shell_march: all checks passed" << std::endl;
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
