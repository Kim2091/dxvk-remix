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

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dxvk {

  // Asset keys of API assets (scene interface spec section 2.1; remixapi_AssetKeyEXT, rtx.scene.assetKeys).
  //
  // A host may give a mesh or material a key that mods, captures and caches use instead of the default (the mesh
  // handle; the material's runtime content hash), plus aliases: the keys an older host or plugin version produced,
  // so mods made against them keep working. Replacements are looked up by key first, then by each alias.
  // CS thread only.
  class AssetResolver {
  public:
    // Bumped whenever the key-derivation rule changes; written into capture metadata.
    static constexpr uint32_t kKeyDerivationVersion = 1;

    struct AssetKey {
      uint64_t key = 0;
      std::vector<uint64_t> aliases;
      std::string name;
    };

    void setMeshKey(uint64_t meshHandle, AssetKey key) { m_meshKeys[meshHandle] = std::move(key); }
    void eraseMeshKey(uint64_t meshHandle) { m_meshKeys.erase(meshHandle); }
    const AssetKey* meshKey(uint64_t meshHandle) const {
      auto found = m_meshKeys.find(meshHandle);
      return found != m_meshKeys.end() ? &found->second : nullptr;
    }

    // Materials are keyed by the runtime content hash of the material the host created (what replacements are
    // looked up with); two materials with equal contents share a key.
    void setMaterialKey(uint64_t contentHash, AssetKey key) { m_materialKeys[contentHash] = std::move(key); }
    const AssetKey* materialKey(uint64_t contentHash) const {
      auto found = m_materialKeys.find(contentHash);
      return found != m_materialKeys.end() ? &found->second : nullptr;
    }

    // The first of key, aliases for which lookup returns a non-null result.
    template<typename Lookup>
    static auto resolve(const AssetKey& key, Lookup&& lookup) -> decltype(lookup(uint64_t(0))) {
      if (key.key != 0) {
        if (auto found = lookup(key.key)) {
          return found;
        }
      }
      for (const uint64_t alias : key.aliases) {
        if (alias != 0) {
          if (auto found = lookup(alias)) {
            return found;
          }
        }
      }
      return nullptr;
    }

    // Whether a set of changed replacement keys touches this asset (its key or an alias).
    static bool touches(const AssetKey& key, const std::unordered_set<uint64_t>& changed) {
      if (changed.count(key.key) != 0) {
        return true;
      }
      for (const uint64_t alias : key.aliases) {
        if (changed.count(alias) != 0) {
          return true;
        }
      }
      return false;
    }

    bool empty() const { return m_meshKeys.empty() && m_materialKeys.empty(); }

  private:
    std::unordered_map<uint64_t, AssetKey> m_meshKeys;
    std::unordered_map<uint64_t, AssetKey> m_materialKeys;
  };

} // namespace dxvk
