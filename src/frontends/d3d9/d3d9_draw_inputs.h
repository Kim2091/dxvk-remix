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
#pragma once

// D3D9 v2: what one draw is made of, as ids, versions and small value hashes.
// Built on the D3D9 thread from bound state. It reads no vertex data, and index data only for refilled index rings
// (kIndexContent).

#include <cstdint>

#include "../translator/resource_stamp.h"
#include "../../util/util_matrix.h"
#include "../../util/xxHash/xxhash.h"

namespace dxvk {
  namespace frontends {
    namespace d3d9 {

      struct D3D9DrawInputs {
        static constexpr uint32_t kMaxStreams = 16;  // caps::MaxStreams

        enum Flag : uint8_t {
          kUserPointer    = 1 << 0,  // DrawPrimitiveUP / DrawIndexedPrimitiveUP: no buffer objects, unversioned
          kProgrammableVS = 1 << 1,
          kVertexCapture  = 1 << 2,  // positions come from vertex shader capture
          kSkinned        = 1 << 3,  // fixed-function vertex blending
          kDynamicSource  = 1 << 4,  // R5: a vertex or index buffer that is D3DUSAGE_DYNAMIC or a refilled ring, or user memory
          // The index buffer is dynamic or a refilled ring, and indices.version is a content hash of the drawn range
          // (with the primitive count) instead of the buffer version. The object key then leaves out the start index and
          // the primitive count, so a batch rewritten at another place in the ring keeps its object, and an unchanged one
          // is a repeat.
          kIndexContent   = 1 << 5,
          // The vertex shader samples a texture. A texture the GPU writes keeps its image and hash, so the draw's output
          // can change while every input here stays the same: it is never a repeat.
          kVertexTextures = 1 << 6,
        };

        struct Stream {
          translator::ResourceRef buffer;  // {D3D9CommonBuffer::remixResourceId, remixContentVersion}
          uint32_t offset = 0;
          uint32_t stride = 0;
        };

        // Identity: which buffers, which range, which shaders.
        Stream streams[kMaxStreams];
        uint32_t streamMask = 0;               // streams with a buffer bound (or the UP slice, stream 0)
        translator::ResourceRef indices;       // id 0: not indexed (or UP indices)
        uint32_t indexBits = 0;                // 0, 16 or 32
        int32_t baseVertex = 0;
        uint32_t startIndex = 0;
        uint32_t primitiveCount = 0;
        uint32_t minVertexIndex = 0;
        uint32_t numVertices = 0;
        uint32_t primitiveType = 0;
        XXH64_hash_t vsBytecode = 0;           // 0 = fixed function
        XXH64_hash_t psBytecode = 0;
        XXH64_hash_t declaration = 0;          // D3D9VertexDecl::GetRemixElementsHash

        // Content that has no per-object version, compared by value.
        XXH64_hash_t vsConstants = 0;          // programmable VS: float / int / bool constants in use
        XXH64_hash_t material = 0;             // LegacyMaterialData::computeIdentityHash (includes the colour texture hashes)
        Matrix4 objectToWorld;
        Matrix4 textureTransform;              // not in the material identity hash
        uint32_t texgenMode = 0;
        // R1: inputs the preparation reads that the record above lacked: the clip
        // planes, the stream frequencies (instance count) and the viewport (minZ: Sky threshold, maxZ: view model).
        XXH64_hash_t extra = 0;
        // R5: fixed-function skinning, the bone matrices the draw uses. 0 means not skinned.
        XXH64_hash_t boneHash = 0;
        // Fixed-function skinning: the centroid of the world positions of the bones the draw uses (the first bone's when
        // the range is unknown), which tells apart draws of one key (two characters of one model) when they change draw
        // order. Not part of any key.
        Vector3 skinPosition;
        // Fixed-function skinning: the largest distance of a used bone from skinPosition (0 with one bone or an unknown
        // range), and the used bone range (skinBoneCount 0 = unknown). Not part of any key.
        float skinRadius = 0.f;
        uint32_t skinFirstBone = 0;
        uint32_t skinBoneCount = 0;

        uint8_t flags = 0;

        // Every input has a version: the draw can be recognised without reading its data.
        bool versioned() const {
          if (flags & kUserPointer) {
            return false;
          }
          for (uint32_t i = 0; i < kMaxStreams; ++i) {
            if ((streamMask & (1u << i)) && streams[i].buffer.version == 0) {
              return false;
            }
          }
          return indices.id == 0 || indices.version != 0;
        }

        // The object this draw draws: same key = same buffers, same range, same shaders (data may differ).
        XXH64_hash_t objectKey() const {
          XXH64_hash_t h = XXH3_64bits(&streamMask, sizeof(streamMask));
          for (uint32_t i = 0; i < kMaxStreams; ++i) {
            if (streamMask & (1u << i)) {
              const uint64_t part[3] = { streams[i].buffer.id, streams[i].offset, streams[i].stride };
              h = XXH3_64bits_withSeed(part, sizeof(part), h);
            }
          }
          const bool indexContent = (flags & kIndexContent) != 0;
          const struct {
            uint64_t indexId;
            uint32_t indexBits;
            int32_t baseVertex;
            uint32_t startIndex, primitiveCount, minVertexIndex, numVertices, primitiveType;
            uint32_t pad;  // explicit, so no indeterminate padding byte is hashed
            XXH64_hash_t vs, decl;
          } range = { indices.id, indexBits, baseVertex, indexContent ? 0u : startIndex, indexContent ? 0u : primitiveCount,
                      minVertexIndex, numVertices, primitiveType, 0u, vsBytecode, declaration };
          static_assert(sizeof(range) == 56, "no implicit padding");
          return XXH3_64bits_withSeed(&range, sizeof(range), h);
        }

        // R5: the identity of draws whose buffers, offsets or counts change from frame to frame (dynamic rings, user-pointer
        // and CPU-written data): what the draw is, not where its data sits (S7's coarse key, moved to the frontend).
        XXH64_hash_t coarseKey() const {
          const struct {
            XXH64_hash_t vs, ps, decl, material;
            uint32_t primitiveType, indexBits, streamMask, pad;
          } k = { vsBytecode, psBytecode, declaration, material, primitiveType, indexBits, streamMask, 0u };
          static_assert(sizeof(k) == 48, "no implicit padding");
          XXH64_hash_t h = XXH3_64bits_withSeed(&k, sizeof(k), 0x5235436f61727365ull);  // "R5Coarse"
          for (uint32_t i = 0; i < kMaxStreams; ++i) {
            if (streamMask & (1u << i)) {
              h = XXH3_64bits_withSeed(&streams[i].stride, sizeof(uint32_t), h);
            }
          }
          return h;
        }

        // What tells apart draws of one key at the same position when the slot is chosen (lookupNearest): the material
        // and the drawn counts, which stay the same from frame to frame for one object.
        XXH64_hash_t contentSignature() const {
          const uint64_t parts[2] = { material, (uint64_t(primitiveCount) << 32) | numVertices };
          return XXH3_64bits(parts, sizeof(parts));
        }

        // The draw without its buffer objects: what is left the same when a device Reset makes the game create its
        // buffers again with the same content.
        XXH64_hash_t resetSignature() const {
          XXH64_hash_t h = XXH3_64bits(&streamMask, sizeof(streamMask));
          for (uint32_t i = 0; i < kMaxStreams; ++i) {
            if (streamMask & (1u << i)) {
              const uint32_t layout[2] = { streams[i].offset, streams[i].stride };
              h = XXH3_64bits_withSeed(layout, sizeof(layout), h);
            }
          }
          const struct {
            uint64_t indexContent;
            uint32_t indexBits;
            int32_t baseVertex;
            uint32_t startIndex, primitiveCount, minVertexIndex, numVertices, primitiveType, texgenMode;
            XXH64_hash_t vs, ps, decl, material;
          } draw = { (flags & kIndexContent) ? indices.version : 0, indexBits, baseVertex, startIndex, primitiveCount,
                     minVertexIndex, numVertices, primitiveType, texgenMode, vsBytecode, psBytecode, declaration, material };
          static_assert(sizeof(draw) == 72, "no implicit padding");
          h = XXH3_64bits_withSeed(&draw, sizeof(draw), h);
          h = XXH3_64bits_withSeed(&textureTransform, sizeof(textureTransform), h);
          return XXH3_64bits_withSeed(&objectToWorld, sizeof(objectToWorld), h) | 1;
        }

        // Everything that can change while the object stays the same. includeVsConstants = false gives the key
        // CameraSplit (V3) aims at: unchanged except for VS constants. includeWorld = false: unchanged except for the
        // world matrix (a move).
        XXH64_hash_t inputsKey(bool includeVsConstants, bool includeWorld = true) const {
          return inputsKeyWith(material, includeVsConstants, includeWorld);
        }

        // A4: the inputs with passMaterial in place of the material identity, which holds the blend and alpha state.
        XXH64_hash_t passKey(XXH64_hash_t passMaterial) const {
          return inputsKeyWith(passMaterial, true, true);
        }

        // The inputs with materialKey as the material identity.
        XXH64_hash_t inputsKeyWith(XXH64_hash_t materialKey, bool includeVsConstants, bool includeWorld) const {
          XXH64_hash_t h = XXH3_64bits(&flags, sizeof(flags));
          for (uint32_t i = 0; i < kMaxStreams; ++i) {
            if (streamMask & (1u << i)) {
              h = XXH3_64bits_withSeed(&streams[i].buffer.version, sizeof(uint64_t), h);
            }
          }
          const XXH64_hash_t parts[7] = { indices.version, psBytecode, materialKey, includeVsConstants ? vsConstants : 0,
                                          texgenMode, extra, boneHash };
          h = XXH3_64bits_withSeed(parts, sizeof(parts), h);
          h = XXH3_64bits_withSeed(&textureTransform, sizeof(textureTransform), h);
          return includeWorld ? XXH3_64bits_withSeed(&objectToWorld, sizeof(objectToWorld), h) : h;
        }
      };

    } // namespace d3d9
  } // namespace frontends
} // namespace dxvk
