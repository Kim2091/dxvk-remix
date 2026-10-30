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

// Tessellates and displaces one static displaced mesh into real triangles for SCCR real geometry (SccrTessManager in
// rtx_fork_sccr_tess.cpp). sccr_tess_args.h describes the layout, the lattice
// and the displacement rule.
//
// Every lattice point is written once per triangle that contains it, so for a watertight surface the copies on a
// shared edge or corner must come out bit-identical. Each kind of point is therefore computed from inputs both sides
// share, in one fixed order:
// - A corner uses its welded id's position, shell vector and texture coordinates. Every copy of a welded position has
//   the same welded id.
// - A point on a shared edge uses the edge's two ends ordered by welded id, and the texture coordinates of the edge's
//   owner, which is the lower triangle index of the two. With this dominant-edge rule a UV seam shows at worst a ridge
//   in the relief, never a hole.
// - A point on a boundary edge stays on the base surface (h = 0), which every triangle on it computes alike. A boundary
//   edge is open, non-manifold, flipped or degenerate, so it has no single neighbor to agree with.
// Interior points belong to one triangle only. Shading attributes (normal, texture coordinates, color) are the
// triangle's own, interpolated, so hard edges and UV seams shade as on the base mesh.

#include "rtx/pass/sccr_tess/sccr_tess_args.h"
#include "rtx/pass/sccr_shell/sccr_shell_args.h"
#include "rtx/utility/common.slangh"
#include "rtx/utility/packing.slangh"

layout(binding = SCCR_TESS_BINDING_POSITION_RO)
StructuredBuffer<float> PositionData;

layout(binding = SCCR_TESS_BINDING_NORMAL_RO)
StructuredBuffer<float> NormalData;

layout(binding = SCCR_TESS_BINDING_TEXCOORD_RO)
StructuredBuffer<float> TexcoordData;

layout(binding = SCCR_TESS_BINDING_COLOR_RO)
StructuredBuffer<uint> ColorData;

layout(binding = SCCR_TESS_BINDING_INDEX_RO)
StructuredBuffer<uint> IndexData;

layout(binding = SCCR_TESS_BINDING_SHELL_RO)
StructuredBuffer<uint> ShellData;

layout(binding = SCCR_TESS_BINDING_HEIGHT)
Sampler2D<float4> HeightMap;

layout(binding = SCCR_TESS_BINDING_VERTEX_OUTPUT)
RWStructuredBuffer<uint> VertexOutput;

layout(binding = SCCR_TESS_BINDING_INDEX_OUTPUT)
RWStructuredBuffer<uint> IndexOutput;

layout(push_constant)
ConstantBuffer<SccrTessArgs> cb;

uint sccrTessLoadIndex(uint indexIndex)
{
  if ((cb.flags & SCCR_TESS_NO_INDICES) != 0u)
  {
    return indexIndex;
  }
  if ((cb.flags & SCCR_TESS_INDICES_16) == 0u)
  {
    return IndexData[(cb.indexOffset >> 2u) + indexIndex];
  }
  const uint byteOffset = cb.indexOffset + indexIndex * 2u;
  return (IndexData[byteOffset >> 2u] >> ((byteOffset & 2u) * 8u)) & 0xFFFFu;
}

float3 sccrTessLoadPosition(uint vertexIndex)
{
  const uint base = (cb.positionOffset + vertexIndex * cb.positionStride) >> 2u;
  return float3(PositionData[base + 0u], PositionData[base + 1u], PositionData[base + 2u]);
}

float3 sccrTessLoadNormal(uint vertexIndex)
{
  const uint base = (cb.normalOffset + vertexIndex * cb.normalStride) >> 2u;
  if ((cb.flags & SCCR_TESS_NORMAL_ENCODED) != 0u)
  {
    const uint word = asuint(NormalData[base]);
    return unsignedOctahedralToSphereDirection(float2(float(word & 0xFFFFu), float(word >> 16u)) / 65535.0f);
  }
  return float3(NormalData[base + 0u], NormalData[base + 1u], NormalData[base + 2u]);
}

float2 sccrTessLoadTexcoord(uint vertexIndex)
{
  const uint base = (cb.texcoordOffset + vertexIndex * cb.texcoordStride) >> 2u;
  return float2(TexcoordData[base + 0u], TexcoordData[base + 1u]);
}

uint sccrTessLoadColor(uint vertexIndex)
{
  if ((cb.flags & SCCR_TESS_HAS_COLOR) == 0u)
  {
    return 0xFFFFFFFFu;
  }
  return ColorData[(cb.colorOffset + vertexIndex * cb.colorStride) >> 2u];
}

// The inverse of unpackUnorm4x8 (shader_types.slangh), for a color in [0, 1].
uint sccrTessPackUnorm4x8(float4 color)
{
  const uint4 bytes = uint4(round(saturate(color) * 255.0f));
  return bytes.x | (bytes.y << 8u) | (bytes.z << 16u) | (bytes.w << 24u);
}

uint sccrTessWeldedId(uint vertexIndex)
{
  return ShellData[SCCR_SHELL_HEADER_DWORDS + 3u * cb.numTriangles + 2u * cb.numVertices + vertexIndex];
}

uint sccrTessNeighbour(uint baseTriangle, uint edgeSlot)
{
  return ShellData[SCCR_SHELL_HEADER_DWORDS + baseTriangle * 3u + edgeSlot];
}

float3 sccrTessDecodeDirection(uint packedDirection)
{
  const int2 quantized = int2(int(packedDirection << 16u) >> 16, int(packedDirection) >> 16);
  const float2 octahedral = max(float2(quantized) / 32767.0f, float2(-1.0f, -1.0f));
  float3 direction = float3(octahedral, 1.0f - abs(octahedral.x) - abs(octahedral.y));
  const float fold = max(-direction.z, 0.0f);
  direction.x += direction.x >= 0.0f ? -fold : fold;
  direction.y += direction.y >= 0.0f ? -fold : fold;
  const float len = length(direction);
  return len > 0.0f ? direction / len : float3(0.0f, 0.0f, 0.0f);
}

// A welded corner, as every triangle around it sees it.
struct SccrTessCorner
{
  float3 position;  // The welded id's position.
  float3 direction; // The unit welded direction towards the mesh's outside, or 0 for a vertex that did not weld.
  float3 shell;     // The direction times the corner's object depth. The inner shell lies at position - shell.
  float2 heightUv;  // The welded id's texture coordinates.
};

SccrTessCorner sccrTessCornerLoad(uint vertexIndex)
{
  SccrTessCorner corner;
  const uint weldedId = sccrTessWeldedId(vertexIndex);
  const uint safeId = weldedId < cb.numVertices ? weldedId : vertexIndex;
  corner.position = sccrTessLoadPosition(safeId);
  corner.heightUv = sccrTessLoadTexcoord(safeId);

  // The shell words are the same on every copy of a welded position (sccr_shell_args.h).
  const uint wordsBase = SCCR_SHELL_HEADER_DWORDS + 3u * cb.numTriangles + 2u * safeId;
  const uint word0 = ShellData[wordsBase];
  const uint word1 = ShellData[wordsBase + 1u];
  const bool unwelded = (word1 & SCCR_SHELL_VERTEX_FLAG_UNWELDED) != 0u;
  corner.direction = unwelded ? float3(0.0f, 0.0f, 0.0f) : sccrTessDecodeDirection(word0) * cb.orientation;

  // D_obj = D / |M n|, the march's rule, under the scale the mesh is built for.
  const float worldLength = length(cb.objectScale * corner.direction);
  float objectDepth = worldLength > 0.0f ? cb.worldDepth / worldLength : 0.0f;
  if ((cb.flags & SCCR_TESS_FOLD_GUARD) != 0u)
  {
    // The limit for a shell against the welded direction (orientation +1) is the half float in the low bits.
    // The limit for a shell along it is the shortened half float in bits 20-31 (sccrShellVertexDepthLimit).
    const uint halfBits = cb.orientation > 0.0f ?
      (word1 & SCCR_SHELL_VERTEX_LIMIT_MASK) :
      (((word1 >> SCCR_SHELL_VERTEX_OUTWARD_LIMIT_SHIFT) & SCCR_SHELL_VERTEX_OUTWARD_LIMIT_MASK) << 3u);
    objectDepth = min(objectDepth, f16tof32(halfBits));
  }
  if (!(objectDepth > 0.0f) || !isfinite(objectDepth))
  {
    objectDepth = 0.0f;
  }
  corner.shell = corner.direction * objectDepth;
  return corner;
}

// Phong tessellation (Boubekeur and Alexa) projects the point onto each corner's tangent plane, blends the projections
// by the barycentric weights, then mixes the result with the flat point by phongShape. Corners stay put. The welded
// directions serve as the corner normals because they are identical on every copy, so both sides of an edge agree.
float3 sccrTessProject(float3 p, SccrTessCorner corner)
{
  return p - dot(p - corner.position, corner.direction) * corner.direction;
}

// Returns a point on an edge from its two ends, ordered by welded id with lo first, and their weights.
float3 sccrTessEdgePosition(SccrTessCorner lo, SccrTessCorner hi, float wLo, float wHi, out float3 shell)
{
  const float3 flatPoint = lo.position * wLo + hi.position * wHi;
  const float3 curved = sccrTessProject(flatPoint, lo) * wLo + sccrTessProject(flatPoint, hi) * wHi;
  shell = lo.shell * wLo + hi.shell * wHi;
  return flatPoint + (curved - flatPoint) * cb.phongShape;
}

float3 sccrTessInteriorPosition(SccrTessCorner corners[3], float3 b, out float3 shell)
{
  const float3 flatPoint = corners[0].position * b.x + corners[1].position * b.y + corners[2].position * b.z;
  const float3 curved = sccrTessProject(flatPoint, corners[0]) * b.x + sccrTessProject(flatPoint, corners[1]) * b.y +
                        sccrTessProject(flatPoint, corners[2]) * b.z;
  shell = corners[0].shell * b.x + corners[1].shell * b.y + corners[2].shell * b.z;
  return flatPoint + (curved - flatPoint) * cb.phongShape;
}

// Returns the relief depth at uv, which is 0 at the mesh surface and 1 at the inner shell, as in SCCR.
float sccrTessDepth(float2 uv)
{
  const float height = HeightMap.SampleLevel(uv, cb.heightLod).r;
  const float depth = 1.0f - height;
  return isfinite(depth) ? saturate(depth) : 0.0f;
}

// Maps a local index in a triangle's grid to its lattice point (a, b). Row b holds N - b + 1 points (a = 0..N - b).
uint2 sccrTessLatticePoint(uint local, uint n)
{
  uint b = 0u;
  uint rowStart = 0u;
  for (; b < n; ++b)
  {
    const uint rowLength = n - b + 1u;
    if (local < rowStart + rowLength)
    {
      break;
    }
    rowStart += rowLength;
  }
  return uint2(local - rowStart, b);
}

uint sccrTessLatticeIndex(uint a, uint b, uint n)
{
  // The points before row b number the sum of (n + 1 - r) for r < b.
  return b * (n + 1u) - (b * (b - 1u)) / 2u + a;
}

uint sccrTessPointsPerTriangle(uint n)
{
  return (n + 1u) * (n + 2u) / 2u;
}

void sccrTessWriteVertex(uint outputVertex)
{
  const uint n = cb.segments;
  const uint pointsPerTriangle = sccrTessPointsPerTriangle(n);
  const uint baseTriangle = outputVertex / pointsPerTriangle;
  const uint2 ab = sccrTessLatticePoint(outputVertex - baseTriangle * pointsPerTriangle, n);
  // The integer barycentrics give the weights (i0, i1, i2) / n.
  const uint3 integer = uint3(n - ab.x - ab.y, ab.x, ab.y);
  const float rn = 1.0f / float(n);
  const float3 b = float3(integer) * rn;

  uint vertexIndices[3];
  [unroll]
  for (uint k = 0; k < 3; ++k)
  {
    vertexIndices[k] = min(sccrTessLoadIndex(baseTriangle * 3u + k), cb.numVertices - 1u);
  }

  // The shading attributes are the triangle's own.
  float3 normal = float3(0.0f, 0.0f, 0.0f);
  float2 texcoord = float2(0.0f, 0.0f);
  float4 color = float4(0.0f, 0.0f, 0.0f, 0.0f);
  [unroll]
  for (uint k = 0; k < 3; ++k)
  {
    normal += sccrTessLoadNormal(vertexIndices[k]) * b[k];
    texcoord += sccrTessLoadTexcoord(vertexIndices[k]) * b[k];
    color += unpackUnorm4x8(sccrTessLoadColor(vertexIndices[k])) * b[k];
  }
  const float normalLength = length(normal);
  normal = normalLength > 0.0f && isfinite(normalLength) ? normal / normalLength : float3(0.0f, 0.0f, 1.0f);

  // The displaced position depends on the kind of point.
  float3 position;
  float3 shell;
  float depth;
  uint cornerSlot = 3u;
  uint edgeSlot = 3u;
  [unroll]
  for (uint k = 0; k < 3; ++k)
  {
    if (integer[k] == n)
    {
      cornerSlot = k;
    }
    else if (integer[k] == 0u)
    {
      edgeSlot = k;
    }
  }

  if (cornerSlot < 3u)
  {
    const SccrTessCorner corner = sccrTessCornerLoad(vertexIndices[cornerSlot]);
    position = corner.position;
    shell = corner.shell;
    depth = sccrTessDepth(corner.heightUv);
  }
  else if (edgeSlot < 3u)
  {
    // Edge slot e lies opposite vertex e, from (e + 1) % 3 to (e + 2) % 3 (the shell data's convention).
    const uint slotA = (edgeSlot + 1u) % 3u;
    const uint slotB = (edgeSlot + 2u) % 3u;
    const SccrTessCorner cornerA = sccrTessCornerLoad(vertexIndices[slotA]);
    const SccrTessCorner cornerB = sccrTessCornerLoad(vertexIndices[slotB]);
    const uint idA = sccrTessWeldedId(vertexIndices[slotA]);
    const uint idB = sccrTessWeldedId(vertexIndices[slotB]);
    // The ends are ordered by welded id, and each end's weight is an integer count of segments from the other end.
    const bool aIsLo = idA <= idB;
    const uint countHi = aIsLo ? integer[slotB] : integer[slotA];
    const float wHi = float(countHi) * rn;
    const float wLo = float(n - countHi) * rn;
    const SccrTessCorner lo = aIsLo ? cornerA : cornerB;
    const SccrTessCorner hi = aIsLo ? cornerB : cornerA;
    position = sccrTessEdgePosition(lo, hi, wLo, wHi, shell);

    const uint neighbourWord = sccrTessNeighbour(baseTriangle, edgeSlot);
    if (SCCR_SHELL_NEIGHBOUR_IS_BOUNDARY(neighbourWord) || idA == idB)
    {
      // With no single neighbor to agree with, the point stays on the base surface.
      depth = 0.0f;
    }
    else
    {
      // Uses the owner's texture coordinates at the two ends. The neighbor runs along the shared edge the other way,
      // so its (slot + 2) end is this triangle's A end and its (slot + 1) end the B end.
      const uint neighbourTriangle = neighbourWord & SCCR_SHELL_NEIGHBOUR_TRIANGLE_MASK;
      const uint neighbourSlot = neighbourWord >> SCCR_SHELL_NEIGHBOUR_SLOT_SHIFT;
      float2 uvA;
      float2 uvB;
      if (neighbourTriangle < baseTriangle)
      {
        uvA = sccrTessLoadTexcoord(min(sccrTessLoadIndex(neighbourTriangle * 3u + (neighbourSlot + 2u) % 3u), cb.numVertices - 1u));
        uvB = sccrTessLoadTexcoord(min(sccrTessLoadIndex(neighbourTriangle * 3u + (neighbourSlot + 1u) % 3u), cb.numVertices - 1u));
      }
      else
      {
        uvA = sccrTessLoadTexcoord(vertexIndices[slotA]);
        uvB = sccrTessLoadTexcoord(vertexIndices[slotB]);
      }
      const float2 uvLo = aIsLo ? uvA : uvB;
      const float2 uvHi = aIsLo ? uvB : uvA;
      depth = sccrTessDepth(uvLo * wLo + uvHi * wHi);
    }
  }
  else
  {
    SccrTessCorner corners[3];
    [unroll]
    for (uint k = 0; k < 3; ++k)
    {
      corners[k] = sccrTessCornerLoad(vertexIndices[k]);
    }
    position = sccrTessInteriorPosition(corners, b, shell);
    float2 heightUv = float2(0.0f, 0.0f);
    [unroll]
    for (uint k = 0; k < 3; ++k)
    {
      heightUv += sccrTessLoadTexcoord(vertexIndices[k]) * b[k];
    }
    depth = sccrTessDepth(heightUv);
  }

  float3 displaced = position - shell * depth;
  if (!all(isfinite(displaced)))
  {
    displaced = position;
  }

  const uint base = outputVertex * SCCR_TESS_VERTEX_DWORDS;
  VertexOutput[base + 0u] = asuint(displaced.x);
  VertexOutput[base + 1u] = asuint(displaced.y);
  VertexOutput[base + 2u] = asuint(displaced.z);
  VertexOutput[base + 3u] = asuint(normal.x);
  VertexOutput[base + 4u] = asuint(normal.y);
  VertexOutput[base + 5u] = asuint(normal.z);
  VertexOutput[base + 6u] = asuint(texcoord.x);
  VertexOutput[base + 7u] = asuint(texcoord.y);
  VertexOutput[base + 8u] = (cb.flags & SCCR_TESS_HAS_COLOR) != 0u ? sccrTessPackUnorm4x8(color) : 0xFFFFFFFFu;
}

void sccrTessWriteTriangle(uint outputTriangle)
{
  const uint n = cb.segments;
  const uint trianglesPerTriangle = n * n;
  const uint baseTriangle = outputTriangle / trianglesPerTriangle;
  uint local = outputTriangle - baseTriangle * trianglesPerTriangle;

  // Row b holds N - b upward triangles (a = 0..N - b - 1), then N - b - 1 downward ones.
  uint b = 0u;
  for (; b < n; ++b)
  {
    const uint rowTriangles = 2u * (n - b) - 1u;
    if (local < rowTriangles)
    {
      break;
    }
    local -= rowTriangles;
  }
  const uint up = n - b;
  uint3 corners;
  if (local < up)
  {
    const uint a = local;
    corners = uint3(sccrTessLatticeIndex(a, b, n), sccrTessLatticeIndex(a + 1u, b, n), sccrTessLatticeIndex(a, b + 1u, n));
  }
  else
  {
    const uint a = local - up;
    corners = uint3(sccrTessLatticeIndex(a + 1u, b, n), sccrTessLatticeIndex(a + 1u, b + 1u, n),
                    sccrTessLatticeIndex(a, b + 1u, n));
  }
  // The winding matches the base triangle, with a running towards vertex 1 and b towards vertex 2.
  const uint first = baseTriangle * sccrTessPointsPerTriangle(n);
  IndexOutput[outputTriangle * 3u + 0u] = first + corners.x;
  IndexOutput[outputTriangle * 3u + 1u] = first + corners.y;
  IndexOutput[outputTriangle * 3u + 2u] = first + corners.z;
}

[shader("compute")]
[numthreads(SCCR_TESS_GROUP_SIZE, 1, 1)]
void main(uint2 threadId : SV_DispatchThreadID)
{
  const uint threadIndex = threadId.y * cb.rowThreads + threadId.x;
  if (threadId.x >= cb.rowThreads || threadIndex >= cb.threadCount)
  {
    return;
  }

  if (cb.phase == SCCR_TESS_PHASE_VERTICES)
  {
    sccrTessWriteVertex(threadIndex);
  }
  else if (cb.phase == SCCR_TESS_PHASE_INDICES)
  {
    sccrTessWriteTriangle(threadIndex);
  }
}
