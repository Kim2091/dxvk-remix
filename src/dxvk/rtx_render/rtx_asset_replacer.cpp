/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
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

#include "rtx_asset_replacer.h"

#include "dxvk_device.h"
#include "dxvk_context.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_utils.h"
#include "rtx_asset_data_manager.h"

namespace dxvk {

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForMesh(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMeshes())
    return nullptr;

  auto variantInfo = m_variantInfos.find(hash);

  if (variantInfo != m_variantInfos.end()) {
    hash += variantInfo->second.selectedVariant;
  }

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eMesh>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

std::vector<AssetReplacement>* AssetReplacer::getReplacementsForLight(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementLights())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    if (auto replacement = mod->replacements().get<AssetReplacement::eLight>(hash)) {
      return replacement;
    }
  }

  return nullptr;
}

MaterialData* AssetReplacer::getReplacementMaterial(XXH64_hash_t hash) {
  if (!RtxOptions::getEnableReplacementMaterials())
    return nullptr;

  for (auto& mod : m_modManager.mods()) {
    MaterialData* material;
    if (mod->replacements().getObject(hash, material)) {
      return material;
    }
  }

  return nullptr;
}

void AssetReplacer::initialize(const Rc<DxvkContext>& context) {
  for (auto& mod : m_modManager.mods()) {
    mod->load(context);
  }
  updateSecretReplacements();
}

bool AssetReplacer::checkForChanges(const Rc<DxvkContext>& context) {
  ScopedCpuProfileZone();

  bool changed = false;
  for (auto& mod : m_modManager.mods()) {
    changed |= mod->checkForChanges(context);
  }
  if (changed) {
    updateSecretReplacements();
  }
  return changed;
}

bool AssetReplacer::areAllReplacementsLoaded() const {
  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      return false;
    }
  }

  return true;
}

std::vector<Mod::State> AssetReplacer::getReplacementStates() const {
  const auto& mods = m_modManager.mods();
  std::vector<Mod::State> modStates(mods.size());

  for (auto& mod : mods) {
    modStates.emplace_back(mod->state());
  }

  return modStates;
}

void AssetReplacer::updateSecretReplacements() {
  bool updated = false;

  m_variantInfos.clear();
  m_secretReplacements.clear();

  for (auto& mod : m_modManager.mods()) {
    if (mod->state().progressState != Mod::ProgressState::Loaded) {
      continue;
    }

    // Pull secret replacement info
    for (auto& secrets : mod->replacements().secretReplacements()) {
      for (auto& secret : secrets.second) {
        m_secretReplacements[secrets.first].emplace_back(
          SecretReplacement{secret.header,
                            secret.name,
                            secret.description,
                            secret.unlockHash,
                            secret.assetHash,
                            secret.replacementPath,
                            secret.bDisplayBeforeUnlocked,
                            secret.bExclusiveReplacement,
                            secret.variantId
          }
        );

        auto& numVariants = m_variantInfos[secrets.first].numVariants;
        numVariants = std::max(secret.variantId, numVariants);

        updated = true;
      }
    }
  }

  m_bSecretReplacementsUpdated = updated;
}

namespace {
  std::string tostr(const remixapi_MaterialHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
  std::string tostr(const remixapi_MeshHandle& h) {
    static_assert(sizeof h == sizeof uint64_t);
    return std::to_string(reinterpret_cast<uint64_t>(h));
  }
}

void AssetReplacer::makeMaterialWithTexturePreload(DxvkContext& ctx, remixapi_MaterialHandle handle, MaterialData&& data) {
  auto [iter, isNew] = m_extMaterials.emplace(handle, std::move(data));

  if (!isNew) {
    Logger::info("Ignoring repeated material registration (handle=" + tostr(handle) + ") ");
    return;
  }
}

const MaterialData* AssetReplacer::accessExternalMaterial(remixapi_MaterialHandle handle) const {
  auto found = m_extMaterials.find(handle);
  if (found == m_extMaterials.end()) {
    return nullptr;
  }
  return found->second ? &found->second.value() : nullptr;
}

void AssetReplacer::destroyExternalMaterial(remixapi_MaterialHandle handle) {
  m_extMaterials.erase(handle);
}

void AssetReplacer::registerExternalMesh(remixapi_MeshHandle handle, std::vector<RasterGeometry>&& submeshes,
                                         bool refreshGeometry) {
  auto existing = m_extMeshes.find(handle);
  if (existing != m_extMeshes.end()) {
    // In-place geometry refresh for CPU-deformed meshes that keep their
    // topology (FaceGen morphs). Opt-in only: the default remains "ignore the
    // repeated registration", because the ordinary re-create path re-submits
    // byte-identical content (a released-but-not-yet-drained handle being
    // revived), and silently reallocating buffers for that would churn the
    // BLAS for geometry that never moved.
    //
    // Thread safety: every caller reaches this from the render (CS) thread --
    // either inside the EmitCs lambda in remixapi_CreateMesh or via
    // applyPendingMeshCreatesOnCs -- which is the same thread that runs
    // SceneManager::submitExternalDraw. That serialization is what makes
    // replacing the contents safe: no in-flight draw loop can be holding the
    // reference returned by accessExternalMesh while we swap it. The old
    // RasterGeometry's DxvkBuffer refs drop here, but any command list that
    // already recorded them keeps them alive through DXVK's own lifetime
    // tracking. We assign THROUGH the unique_ptr rather than replacing it, so
    // the vector keeps its address (which is why it is heap-held in the first
    // place -- accessExternalMesh hands out a reference to it).
    if (!refreshGeometry || existing->second->size() != submeshes.size()) {
      //Logger::info("Ignoring repeated mesh registration (handle=" + tostr(handle) + ") ");
      return;
    }

    for (size_t i = 0; i < submeshes.size(); i++) {
      const GeometryHashes& prev = (*existing->second)[i].hashes;
      GeometryHashes& next = submeshes[i].hashes;

      // Carry over everything that did NOT change, leaving only VertexPosition
      // holding the fresh value minted by the caller. Two consumers depend on
      // this split:
      //   * DrawCallCache::get buckets on TopologicalHash (Indices |
      //     GeometryDescriptor). Keeping those stable is what makes it hand
      //     back the SAME BlasEntry instead of allocating a new one.
      //   * SceneManager::processGeometryInfo compares Indices (now equal) and
      //     VertexPosition (now different), which lands on kUpdateBVH -- the
      //     path that swaps historyBuffer[0]/[1] and assigns
      //     previousPositionBuffer, i.e. real motion vectors for the morph.
      next[HashComponents::Indices]            = prev[HashComponents::Indices];
      next[HashComponents::GeometryDescriptor] = prev[HashComponents::GeometryDescriptor];
      next[HashComponents::VertexTexcoord]     = prev[HashComponents::VertexTexcoord];
      next[HashComponents::VertexLayout]       = prev[HashComponents::VertexLayout];
      next.precombine();

      submeshes[i].externalMesh = handle;
    }

    *existing->second = std::move(submeshes);
    return;
  }

  // Tag each submesh with the external mesh handle so capture + runtime
  // use the same identity for replacement-lookup parity.
  for (auto& submesh : submeshes) {
    submesh.externalMesh = handle;
  }

  m_extMeshes.emplace(handle, std::make_unique< std::vector<RasterGeometry>>(std::move(submeshes)));
}

const std::vector<RasterGeometry>& AssetReplacer::accessExternalMesh(remixapi_MeshHandle handle) const {
  auto found = m_extMeshes.find(handle);
  if (found == m_extMeshes.end()) {
    static const auto s_empty = std::vector<RasterGeometry> {};
    return s_empty;
  }
  return *found->second;
}

void AssetReplacer::destroyExternalMesh(remixapi_MeshHandle handle) {
  m_extMeshes.erase(handle);
}

} // namespace dxvk

