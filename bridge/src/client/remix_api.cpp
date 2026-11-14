/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cmath>
#include <cstring>
#include <cwchar>
#include <mutex>
#include <string>
#include <unordered_map>

#include "log/log.h"
#include "util_bridgecommand.h"
#include "util_devicecommand.h"
#include "util_remixapi.h"
#include "d3d9_texture.h"
#include "remix_state.h"

using namespace remixapi::util;

namespace remixapi {

namespace {

uint64_t g_appRemixApiVersion = 0;

bool isRemixApiVersionCompatible(uint64_t appVersion) {
  constexpr uint64_t kCompiledVersion = REMIXAPI_VERSION_MAKE(
    REMIXAPI_VERSION_MAJOR,
    REMIXAPI_VERSION_MINOR,
    REMIXAPI_VERSION_PATCH);

  const bool isDevelopment =
    REMIXAPI_VERSION_GET_MAJOR(appVersion) == 0 &&
    REMIXAPI_VERSION_GET_MAJOR(kCompiledVersion) == 0;

  if (isDevelopment) {
    return REMIXAPI_VERSION_GET_MINOR(appVersion) == REMIXAPI_VERSION_GET_MINOR(kCompiledVersion);
  }

  return REMIXAPI_VERSION_GET_MAJOR(appVersion) == REMIXAPI_VERSION_GET_MAJOR(kCompiledVersion) &&
         REMIXAPI_VERSION_GET_MINOR(appVersion) <= REMIXAPI_VERSION_GET_MINOR(kCompiledVersion);
}

}

bool g_bInterfaceInitialized = false;
PFN_remixapi_BridgeCallback g_beginSceneCallback = nullptr;
PFN_remixapi_BridgeCallback g_endSceneCallback = nullptr;
PFN_remixapi_BridgeCallback g_presentCallback = nullptr;

template<typename T>
inline void send(ClientMessage& msg, const T& val) {
  msg.send_data(sizeof(T), &val);
}

template<>
inline void send(ClientMessage& msg, const remixapi_Float3D& vec3) {
  send(msg, vec3.x);
  send(msg, vec3.y);
  send(msg, vec3.z);
}

template<>
inline void send(ClientMessage& msg, const remixapi_Path& path) {
  const auto nonNullPath = (path) ? (path) : L"";
  msg.send_data(wcslen(nonNullPath) * sizeof(wchar_t), nonNullPath);
}

template<>
inline void send(ClientMessage& msg, const char* const& c_str) {
  msg.send_data((uint32_t) strlen(c_str) + 1, c_str);
}

template<typename RemixApiHandleT>
inline void sendHandle(ClientMessage& msg, const Handle<RemixApiHandleT>& handle) {
  msg.send_data(handle.uid);
}

template<>
inline void send(ClientMessage& msg, const Bool& b) {
  uint32_t boolVal = 0x0;
  boolVal |= (uint8_t)b;
  msg.send_data(boolVal);
}

template<typename SerializableT>
auto serializeAndSend(ClientMessage& msg, const SerializableT& serializable) {
  static_assert(is_serializable_v<SerializableT>, "serializeAndSend(...)  may only be called with defined Serializable<T> types");
  msg.send_data(ToRemixApiStructEnum<SerializableT::BaseT>);
  // Serialized in place in the data ring: the same words as send_data(size, data).
  if (uint8_t* const pBlob = msg.begin_data_blob(serializable.size())) {
    serializable.serialize(pBlob);
    msg.end_data_blob();
  }
}


remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMaterial(
  const remixapi_MaterialInfo* info,
  remixapi_MaterialHandle*     out_handle) {

  ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateMaterial);
  assert(info->sType == REMIXAPI_STRUCT_TYPE_MATERIAL_INFO);

  MaterialHandle newHandle;
  {
    ClientMessage c(Commands::RemixApi_CreateMaterial);

    serializeAndSend<serialize::MaterialInfo>(c, *info);

    // For each valid pNext, we will send a true-valued bool to indicate that
    // server must read another extension. If it reads false, it knows that it
    // is done reading.
    // send(c, Bool::True); -> CONTINUE
    // send(c, Bool::False); -> STOP
    const void* infoItr = info;
    while (auto* const pNext = getPNext(infoItr)) {
      infoItr = pNext;
      switch (getSType(pNext)) {
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_EXT:
        {
          auto* pOpaqueMat = static_cast<const remixapi_MaterialInfoOpaqueEXT* const>(infoItr);
          remixapi_MaterialInfoOpaqueEXT opaqueMat = { };
          // Upstream grew the struct at 0.6.5; on the Remix Plus ABI line that is 0.1000.1.
          // 0.1000.0 shares the 0.6.4 layout, so the same 92-byte (x86) prefix copy applies.
          if (g_appRemixApiVersion < REMIXAPI_VERSION_MAKE(0, 1000, 1)) {
            constexpr size_t kOpaqueMaterialInfoSizeAtVersion064 = 92;
            std::memcpy(&opaqueMat, pOpaqueMat, kOpaqueMaterialInfoSizeAtVersion064);
            opaqueMat.enableDlssControlMask = true;
            opaqueMat.dlssControlMaskIntensity = 1.f;
            opaqueMat.dlssControlMaskToneStrength = 1.f;
            opaqueMat.dlssControlMaskStructuralStrength = 1.f;
          } else {
            opaqueMat = *pOpaqueMat;
          }
          send(c, Bool::True); 
          serializeAndSend<serialize::MaterialInfoOpaque>(c, opaqueMat);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_SUBSURFACE_EXT:
        {
          auto* pOpaqueSubsurfaceMat = static_cast<const remixapi_MaterialInfoOpaqueSubsurfaceEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::MaterialInfoOpaqueSubsurface>(c, *pOpaqueSubsurfaceMat);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_TRANSLUCENT_EXT:
        {
          auto* pTranslucentMat = static_cast<const remixapi_MaterialInfoTranslucentEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::MaterialInfoTranslucent>(c, *pTranslucentMat);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_PORTAL_EXT:
        {
          auto* pPortalMat = static_cast<const remixapi_MaterialInfoPortalEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::MaterialInfoPortal>(c, *pPortalMat);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_TERRAIN_LAYERS_EXT:
        {
          // FNV terrain layers (sent once per material; paths are copied by value).
          auto* pTerrainLayers = static_cast<const remixapi_MaterialInfoTerrainLayersEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::MaterialInfoTerrainLayers>(c, *pTerrainLayers);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_MATERIAL_INFO_OPAQUE_ALBEDO_TINT_EXT:
        {
          // Per-material albedo tint (FNV unlit / particle MaterialColor). The FNV wrapper looks for
          // this message's text in the client image to tell a client that forwards the struct.
          static bool s_loggedAlbedoTint = false;
          if (!s_loggedAlbedoTint) {
            s_loggedAlbedoTint = true;
            Logger::info("[remixapi_CreateMaterial] Forwarding MaterialInfoOpaqueAlbedoTintEXT");
          }
          auto* pAlbedoTint = static_cast<const remixapi_MaterialInfoOpaqueAlbedoTintEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::MaterialInfoOpaqueAlbedoTint>(c, *pAlbedoTint);
          break;
        }
        default:
        {
          Logger::warn("[remixapi_CreateMaterial] Unknown sType. Skipping.");
          break;
        }
      }
      infoItr = pNext;
    }
    send(c, Bool::False);
    sendHandle(c, newHandle);
  }

  *out_handle = newHandle;

  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyMaterial(remixapi_MaterialHandle handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DestroyMaterial);
  MaterialHandle materialHandle(handle);
  if(!materialHandle.isValid()) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  {
    ClientMessage c(Commands::RemixApi_DestroyMaterial);
    sendHandle(c, materialHandle);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// The public API represents uploaded textures as a byte range rather than a
// path.  Keep the large payload in its own bridge data-queue entry: unlike a
// DDS filename this reaches the 64-bit runtime before CreateMaterial tries
// to resolve its 0xHASH pseudo-path.
remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateTexture(
  const remixapi_TextureInfo* info,
  remixapi_TextureHandle*     out_handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateTexture);
  if (!info || !out_handle || info->sType != REMIXAPI_STRUCT_TYPE_TEXTURE_INFO ||
      !info->data || info->dataSize == 0 || info->dataSize > 0xFFFFFFFFull) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  // A chained remixapi_TextureInfoHostImportEXT is sent field by field. Other chained structs are not sent.
  const remixapi_TextureInfoHostImportEXT* hostImport = nullptr;
  for (auto* p = static_cast<const remixapi_TextureInfoHostImportEXT*>(info->pNext); p != nullptr;
       p = static_cast<const remixapi_TextureInfoHostImportEXT*>(p->pNext)) {
    if (p->sType == REMIXAPI_STRUCT_TYPE_TEXTURE_INFO_HOST_IMPORT_EXT) {
      hostImport = p;
      break;
    }
  }
  if (hostImport) {
    static bool s_logged = false;
    if (!s_logged) {
      s_logged = true;
      Logger::info("[remixapi_CreateTexture] Forwarding remixapi_TextureInfoHostImportEXT (encoding, vkFormat, flags).");
    }
  }

  TextureHandle newHandle;
  {
    ClientMessage c(Commands::RemixApi_CreateTexture);
    send(c, info->sType);
    send(c, info->hash);
    send(c, info->width);
    send(c, info->height);
    send(c, info->depth);
    send(c, info->mipLevels);
    send(c, info->format);
    send(c, hostImport ? hostImport->encoding : 0u);
    send(c, hostImport ? hostImport->vkFormat : 0u);
    send(c, hostImport ? hostImport->flags : 0u);
    send(c, info->dataSize);
    c.send_data(static_cast<uint32_t>(info->dataSize), info->data);
    sendHandle(c, newHandle);
  }

  *out_handle = newHandle;
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyTexture(remixapi_TextureHandle handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DestroyTexture);
  TextureHandle textureHandle(handle);
  if (!textureHandle.isValid()) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  {
    ClientMessage c(Commands::RemixApi_DestroyTexture);
    sendHandle(c, textureHandle);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// Bridge mesh uid per live runtime hash, so a re-submit of a live hash (a
// MeshInfoRefreshGeometryEXT refresh, or a plain re-create) reuses the uid the
// mesh was created with. The runtime's mesh handle
// IS the hash, so every refresh addresses the same runtime mesh; minting a
// fresh uid per refresh leaked one server handle-map entry each time, and a
// DestroyMesh on any of those uids destroyed the one shared mesh. The server
// just rewrites the same slot (MeshHandle(uid, runtimeHandle)): no wire change.
// Guarded by g_meshUidMutex, which is also held across the send so the
// create/destroy order on the wire matches the map's.
static std::mutex g_meshUidMutex;
static std::unordered_map<uint64_t, uint32_t> g_meshUidByHash;
static std::unordered_map<uint32_t, uint64_t> g_meshHashByUid;

// remix_c.h: the blend arrays hold bonesPerVertex * vertexCount elements and
// their _count fields are those totals. The runtime reads that many indices per
// surface regardless of blendIndices_count, so a short array would be read out
// of bounds on the server; refuse it here instead.
static bool validateSkinning(const remixapi_MeshInfo& info) {
  for (uint32_t i = 0; i < info.surfaces_count; ++i) {
    const auto& surf = info.surfaces_values[i];
    if (!surf.skinning_hasvalue) {
      continue;
    }
    const auto& skin = surf.skinning_value;
    const uint64_t needed = uint64_t(skin.bonesPerVertex) * surf.vertices_count;
    if (skin.bonesPerVertex == 0 ||
        skin.blendWeights_count < needed || skin.blendIndices_count < needed ||
        (skin.blendWeights_count && !skin.blendWeights_values) ||
        (skin.blendIndices_count && !skin.blendIndices_values)) {
      Logger::err(format_string("[remixapi_CreateMesh] Invalid skinning on surface %u: bonesPerVertex=%u "
                                "vertices=%u blendWeights_count=%u blendIndices_count=%u (each must be >= "
                                "bonesPerVertex * vertices)",
                                i, skin.bonesPerVertex, surf.vertices_count,
                                skin.blendWeights_count, skin.blendIndices_count));
      return false;
    }
  }
  return true;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMesh(
  const remixapi_MeshInfo* info,
  remixapi_MeshHandle*     out_handle) {

  ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateMesh);
  if (!info || !out_handle || info->sType != REMIXAPI_STRUCT_TYPE_MESH_INFO) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  if (!validateSkinning(*info)) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  // Two MeshInfo extensions are forwarded. MeshInfoRefreshGeometryEXT carries no data: the server is
  // sent one flag and chains its own. AssetKeyEXT goes as its key and aliases (not its name). The rest
  // of the chain is walked and skipped, so a non-NULL pNext is harmless.
  bool refreshGeometry = false;
  const remixapi_AssetKeyEXT* assetKey = nullptr;
  {
    const void* infoItr = info;
    while (auto* const pNext = getPNext(infoItr)) {
      infoItr = pNext;
      if (getSType(pNext) == REMIXAPI_STRUCT_TYPE_MESH_INFO_REFRESH_GEOMETRY_EXT) {
        refreshGeometry = true;
      } else if (getSType(pNext) == REMIXAPI_STRUCT_TYPE_ASSET_KEY_EXT) {
        // The FNV wrapper looks for this message's text in the client image to tell a client that forwards
        // the struct.
        static bool s_loggedAssetKey = false;
        if (!s_loggedAssetKey) {
          s_loggedAssetKey = true;
          Logger::info("[remixapi_CreateMesh] Forwarding AssetKeyEXT");
        }
        assetKey = static_cast<const remixapi_AssetKeyEXT*>(pNext);
      } else {
        Logger::warn("[remixapi_CreateMesh] Unsupported pNext sType. Skipping.");
      }
    }
  }

  std::lock_guard<std::mutex> lock(g_meshUidMutex);

  // One uid per live hash. A refresh (or a plain re-create, which the runtime
  // drops or re-registers under the same hash-valued handle) of a live hash
  // reuses the uid it was created with.
  uint32_t uid = 0;
  if (info->hash != 0) {
    const auto it = g_meshUidByHash.find(info->hash);
    if (it != g_meshUidByHash.end()) {
      uid = it->second;
    }
  }
  if (uid == 0) {
    MeshHandle fresh;
    uid = fresh.uid;
    if (info->hash != 0) {
      g_meshUidByHash[info->hash] = uid;
      g_meshHashByUid[uid] = info->hash;
    }
  }

  const MeshHandle handle(reinterpret_cast<remixapi_MeshHandle>((uintptr_t) uid));
  {
    ClientMessage c(Commands::RemixApi_CreateMesh);
    serializeAndSend<serialize::MeshInfo>(c, *info);
    send(c, refreshGeometry ? Bool::True : Bool::False);
    // The asset key: a flag, then the key, the alias count and each alias.
    const bool sendKey = assetKey != nullptr && assetKey->key != 0;
    send(c, sendKey ? Bool::True : Bool::False);
    if (sendKey) {
      const uint32_t aliasCount = assetKey->aliases_values != nullptr ? assetKey->aliases_count : 0;
      c.send_data(sizeof(assetKey->key), &assetKey->key);
      c.send_data(aliasCount);
      for (uint32_t i = 0; i < aliasCount; ++i) {
        c.send_data(sizeof(uint64_t), &assetKey->aliases_values[i]);
      }
    }
    sendHandle(c, handle);
  }

  *out_handle = reinterpret_cast<remixapi_MeshHandle>((uintptr_t) uid);

  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// The runtime's batched variant only defers the mesh's registration to the
// next flush point; across the bridge a plain CreateMesh is already async, so
// the slot forwards to it.
remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateMeshBatched(
  const remixapi_MeshInfo* info,
  remixapi_MeshHandle*     out_handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateMeshBatched);
  return remixapi_CreateMesh(info, out_handle);
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyMesh(remixapi_MeshHandle handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DestroyMesh);
  MeshHandle meshHandle(handle);
  if(!meshHandle.isValid()) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  std::lock_guard<std::mutex> lock(g_meshUidMutex);
  const auto rev = g_meshHashByUid.find(meshHandle.uid);
  if (rev != g_meshHashByUid.end()) {
    // The hash is no longer live: the next create of it mints a new uid.
    g_meshUidByHash.erase(rev->second);
    g_meshHashByUid.erase(rev);
  }
  {
    ClientMessage c(Commands::RemixApi_DestroyMesh);
    sendHandle(c, meshHandle);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// Sends one InstanceInfo and its extension chain. Shared by DrawInstance and
// the retained-instance calls, so all of them forward the same extensions.
static void sendInstanceInfo(ClientMessage& c, const remixapi_InstanceInfo* info, const char* fnName) {
  serializeAndSend<serialize::InstanceInfo>(c, *info);

  // For each valid pNext, we will send a true-valued bool to indicate that
  // server must read another extension. If it reads false, it knows that it
  // is done reading.
  // send(c, Bool::True); -> CONTINUE
  // send(c, Bool::False); -> STOP
  const void* infoItr = info;
  while (auto* const pNext = getPNext(infoItr)) {
    infoItr = pNext;
    switch (getSType(pNext)) {
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_OBJECT_PICKING_EXT:
      {
        auto* pObjectPicking = static_cast<const remixapi_InstanceInfoObjectPickingEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoObjectPicking>(c, *pObjectPicking);
        break;
      }
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BLEND_EXT:
      {
        auto* pBlend = static_cast<const remixapi_InstanceInfoBlendEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoBlend>(c, *pBlend);
        break;
      }
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_BONE_TRANSFORMS_EXT:
      {
        auto* pXforms = static_cast<const remixapi_InstanceInfoBoneTransformsEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoTransforms>(c, *pXforms);
        break;
      }
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_PARTICLE_SYSTEM_EXT:
      {
        auto* pParticle = static_cast<const remixapi_InstanceInfoParticleSystemEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoParticleSystem>(c, *pParticle);
        break;
      }
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_MATERIAL_OVERRIDE_EXT:
      {
        // The material goes out as its bridge handle; the server maps it.
        auto* pOverride = static_cast<const remixapi_InstanceInfoMaterialOverrideEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoMaterialOverride>(c, *pOverride);
        break;
      }
      case REMIXAPI_STRUCT_TYPE_INSTANCE_INFO_GPU_INSTANCING_EXT:
      {
        // The FNV wrapper looks for this message's text in the client image to tell a client that forwards
        // the struct.
        static bool s_loggedGpuInstancing = false;
        if (!s_loggedGpuInstancing) {
          s_loggedGpuInstancing = true;
          Logger::info("[remixapi_InstanceInfo] Forwarding InstanceInfoGpuInstancingEXT");
        }
        auto* pInstancing = static_cast<const remixapi_InstanceInfoGpuInstancingEXT* const>(infoItr);
        send(c, Bool::True);
        serializeAndSend<serialize::InstanceInfoGpuInstancing>(c, *pInstancing);
        break;
      }
      default:
      {
        Logger::warn(format_string("[%s] Unknown sType. Skipping.", fnName));
        break;
      }
    }
  }
  send(c, Bool::False);
}

// Retained-instance handles, allocated here so no call waits for the server.
static uint64_t g_nextRetainedHandle = 1;

// Field by field: the x86 and x64 layouts of the struct need not agree.
static void sendRetainedUpdate(ClientMessage& c, const remixapi_RetainedInstanceUpdate& u) {
  c.send_data(sizeof(u.handle), &u.handle);
  c.send_data(sizeof(u.transform.matrix), &u.transform.matrix[0][0]);
  c.send_data(u.flags);
  c.send_data(static_cast<uint32_t>(u.visible));
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawInstance(const remixapi_InstanceInfo* info) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DrawInstance);
  {
    ClientMessage c(Commands::RemixApi_DrawInstance);
    sendInstanceInfo(c, info, "remixapi_DrawInstance");
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_CreateLight(
  const remixapi_LightInfo* info,
  remixapi_LightHandle*     out_handle) {
    
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateLight);
  assert(info->sType == REMIXAPI_STRUCT_TYPE_LIGHT_INFO);

  LightHandle newHandle;
  {
    ClientMessage c(Commands::RemixApi_CreateLight);

    serializeAndSend<serialize::LightInfo>(c, *info);
    
    // For each valid pNext, we will send a true-valued bool to indicate that
    // server must read another extension. If it reads false, it knows that it
    // is done reading.
    // send(c, Bool::True); -> CONTINUE
    // send(c, Bool::False); -> STOP
    const void* infoItr = info;
    while (auto* const pNext = getPNext(infoItr)) {
      infoItr = pNext;
      switch (getSType(infoItr)) {
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_SPHERE_EXT:
        {
          auto* pSphere = static_cast<const remixapi_LightInfoSphereEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoSphere>(c, *pSphere);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_RECT_EXT:
        {
          auto* pRect = static_cast<const remixapi_LightInfoRectEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoRect>(c, *pRect);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISK_EXT:
        {
          auto* pDisk = static_cast<const remixapi_LightInfoDiskEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoDisk>(c, *pDisk);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_CYLINDER_EXT:
        {
          auto* pCylinder = static_cast<const remixapi_LightInfoCylinderEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoCylinder>(c, *pCylinder);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DISTANT_EXT:
        {
          auto* pDistant = static_cast<const remixapi_LightInfoDistantEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoDistant>(c, *pDistant);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_DOME_EXT:
        {
          auto* pDome = static_cast<const remixapi_LightInfoDomeEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoDome>(c, *pDome);
          break;
        }
        case REMIXAPI_STRUCT_TYPE_LIGHT_INFO_USD_EXT:
        {
          auto* pUSD = static_cast<const remixapi_LightInfoUSDEXT* const>(infoItr);
          send(c, Bool::True);
          serializeAndSend<serialize::LightInfoUSD>(c, *pUSD);
          break;
        }
        default:
        {
          Logger::warn("[remixapi_CreateLight] Unknown sType. Skipping.");
          break;
        }
      }
    }
    send(c, Bool::False);
    sendHandle(c, newHandle);
  }

  *out_handle = newHandle;

  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DestroyLight(remixapi_LightHandle handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DestroyLight);
  LightHandle lightHandle(handle);
  if(!lightHandle.isValid()) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  {
    ClientMessage c(Commands::RemixApi_DestroyLight);
    sendHandle(c, lightHandle);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_DrawLightInstance(remixapi_LightHandle handle) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_DrawLightInstance);
  LightHandle lightHandle(handle);
  if(!lightHandle.isValid()) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  {
    ClientMessage c(Commands::RemixApi_DrawLightInstance);
    sendHandle(c, lightHandle);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// The words of one CameraInfoLensEXT in a SetupCamera message: the id's low and high halves, reticle, then
// eyepieceCenter, axis, up (xyz each), eyepieceRadius, objectiveDistance, objectiveRadius, magnification,
// fieldStopDegrees, eyeRelief, tint (xyz), reticleColor (xyz), reticleRadiance and reticleLineWidth as float bits,
// then kind, aperture, apertureAspect (float bits), visionMode, lensFlags and minMagnification (float bits),
// then reticleAccentColor (xyz), reticleAccentRadiance, edgeGlowColor (xyz), edgeGlowStrength, edgeShade,
// edgeTint (xyz), nightTint (xyz) and nightGain as float bits, then markings, referenceMagnification (float bits),
// markingColor (xyz), markingRadiance and markingOpacity as float bits, and parallaxDistance, glareStrength and
// dirtStrength as float bits.
// The words are followed by the reticleTexture path's size in bytes, terminator included, and, when it is not 0, the
// path as UTF-16.
static constexpr uint32_t kCameraLensWords = 58;
// The most lenses one SetupCamera message carries. The runtime traces at most 4 per frame.
static constexpr uint32_t kMaxCameraLenses = 8;

// One async message with a fixed layout, field by field (the x86 and x64 struct layouts need not agree):
// type, view, projection, then ParameterizedEXT and FrameEventsEXT each as a presence word plus its data,
// sent zeroed when absent, then the CameraInfoLensEXT count and kCameraLensWords words per lens.
// Other extensions are skipped.
remixapi_ErrorCode REMIXAPI_CALL remixapi_SetupCamera(const remixapi_CameraInfo* info) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetupCamera);
  if (!info || info->sType != REMIXAPI_STRUCT_TYPE_CAMERA_INFO) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  const remixapi_CameraInfoParameterizedEXT* params = nullptr;
  const remixapi_CameraInfoFrameEventsEXT* events = nullptr;
  const remixapi_CameraInfoLensEXT* lenses[kMaxCameraLenses] = {};
  uint32_t lensCount = 0;
  {
    const void* infoItr = info;
    while (auto* const pNext = getPNext(infoItr)) {
      infoItr = pNext;
      switch (getSType(pNext)) {
        case REMIXAPI_STRUCT_TYPE_CAMERA_INFO_PARAMETERIZED_EXT:
          params = static_cast<const remixapi_CameraInfoParameterizedEXT*>(pNext);
          break;
        case REMIXAPI_STRUCT_TYPE_CAMERA_INFO_FRAME_EVENTS_EXT:
          events = static_cast<const remixapi_CameraInfoFrameEventsEXT*>(pNext);
          break;
        case REMIXAPI_STRUCT_TYPE_CAMERA_INFO_LENS_EXT:
        {
          // Hosts look for the log line's text in this image to learn that lenses are forwarded. Its first
          // parenthesized part says that the kind, aperture, vision mode and flag fields are forwarded too, and its
          // second that the accent, edge and night tint fields are, its third that markings is, and its fourth that
          // referenceMagnification and reticleTexture are, its fifth that the marking color fields are, and its sixth
          // that parallaxDistance, glareStrength and dirtStrength are.
          static bool s_loggedLenses = false;
          if (!s_loggedLenses) {
            s_loggedLenses = true;
            Logger::info("[remixapi_SetupCamera] Forwarding CameraInfoLensEXT (kind, aperture, vision modes) (accent, edge, night tint) (markings) (reticle texture) (marking color) (parallax, glass)");
          }
          if (lensCount < kMaxCameraLenses) {
            lenses[lensCount++] = static_cast<const remixapi_CameraInfoLensEXT*>(pNext);
          } else {
            static bool s_warnedLensCount = false;
            if (!s_warnedLensCount) {
              s_warnedLensCount = true;
              Logger::warn("[remixapi_SetupCamera] More CameraInfoLensEXT than one message carries. Skipping the rest.");
            }
          }
          break;
        }
        default:
          Logger::warn("[remixapi_SetupCamera] Unsupported pNext sType. Skipping.");
          break;
      }
    }
  }
  float paramValues[16] = {};   // position, forward, up, right (xyz each), fovY, aspect, near, far
  if (params) {
    const remixapi_Float3D* const vecs[4] = { &params->position, &params->forward, &params->up, &params->right };
    for (int i = 0; i < 4; ++i) {
      paramValues[i * 3 + 0] = vecs[i]->x;
      paramValues[i * 3 + 1] = vecs[i]->y;
      paramValues[i * 3 + 2] = vecs[i]->z;
    }
    paramValues[12] = params->fovYInDegrees;
    paramValues[13] = params->aspect;
    paramValues[14] = params->nearPlane;
    paramValues[15] = params->farPlane;
  }
  float originShift[3] = {};
  if (events) {
    originShift[0] = events->originShift.x;
    originShift[1] = events->originShift.y;
    originShift[2] = events->originShift.z;
  }
  {
    ClientMessage c(Commands::RemixApi_SetupCamera);
    c.send_data(static_cast<uint32_t>(info->type));
    c.send_data(static_cast<uint32_t>(sizeof(info->view)), &info->view[0][0]);
    c.send_data(static_cast<uint32_t>(sizeof(info->projection)), &info->projection[0][0]);
    c.send_data(params ? 1u : 0u);
    c.send_data(static_cast<uint32_t>(sizeof(paramValues)), paramValues);
    c.send_data(events ? 1u : 0u);
    c.send_data(events ? static_cast<uint32_t>(events->flags) : 0u);
    c.send_data(static_cast<uint32_t>(sizeof(originShift)), originShift);
    c.send_data(lensCount);
    for (uint32_t i = 0; i < lensCount; ++i) {
      const remixapi_CameraInfoLensEXT& lens = *lenses[i];
      const float values[] = {
        lens.eyepieceCenter.x, lens.eyepieceCenter.y, lens.eyepieceCenter.z,
        lens.axis.x, lens.axis.y, lens.axis.z,
        lens.up.x, lens.up.y, lens.up.z,
        lens.eyepieceRadius, lens.objectiveDistance, lens.objectiveRadius, lens.magnification,
        lens.fieldStopDegrees, lens.eyeRelief,
        lens.tint.x, lens.tint.y, lens.tint.z,
        lens.reticleColor.x, lens.reticleColor.y, lens.reticleColor.z,
        lens.reticleRadiance, lens.reticleLineWidth,
      };
      const float values2[] = {
        lens.reticleAccentColor.x, lens.reticleAccentColor.y, lens.reticleAccentColor.z, lens.reticleAccentRadiance,
        lens.edgeGlowColor.x, lens.edgeGlowColor.y, lens.edgeGlowColor.z, lens.edgeGlowStrength, lens.edgeShade,
        lens.edgeTint.x, lens.edgeTint.y, lens.edgeTint.z,
        lens.nightTint.x, lens.nightTint.y, lens.nightTint.z, lens.nightGain,
      };
      const float markingValues[] = {
        lens.markingColor.x, lens.markingColor.y, lens.markingColor.z, lens.markingRadiance, lens.markingOpacity,
        lens.parallaxDistance, lens.glareStrength, lens.dirtStrength,
      };
      static_assert(3 + sizeof(values) / sizeof(float) + 6 + sizeof(values2) / sizeof(float) + 2 +
                    sizeof(markingValues) / sizeof(float) == kCameraLensWords,
                    "The lens layout must fill kCameraLensWords.");
      uint32_t words[kCameraLensWords] = {};
      words[0] = static_cast<uint32_t>(lens.id);
      words[1] = static_cast<uint32_t>(lens.id >> 32);
      words[2] = lens.reticle;
      memcpy(&words[3], values, sizeof(values));
      words[26] = lens.kind;
      words[27] = lens.aperture;
      memcpy(&words[28], &lens.apertureAspect, sizeof(float));
      words[29] = lens.visionMode;
      words[30] = lens.lensFlags;
      memcpy(&words[31], &lens.minMagnification, sizeof(float));
      memcpy(&words[32], values2, sizeof(values2));
      words[48] = lens.markings;
      memcpy(&words[49], &lens.referenceMagnification, sizeof(float));
      memcpy(&words[50], markingValues, sizeof(markingValues));
      c.send_data(static_cast<uint32_t>(sizeof(words)), words);
      const uint32_t pathBytes = lens.reticleTexture != nullptr
        ? static_cast<uint32_t>((wcslen(lens.reticleTexture) + 1) * sizeof(wchar_t))
        : 0u;
      c.send_data(pathBytes);
      if (pathBytes != 0) {
        c.send_data(pathBytes, lens.reticleTexture);
      }
    }
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

// Sends the material's handle uid; uid 0 is a null medium, which clears it.
remixapi_ErrorCode REMIXAPI_CALL remixapi_SetCameraMediumMaterial(const remixapi_CameraMediumInfo* info) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetCameraMediumMaterial);
  if (!info || info->sType != REMIXAPI_STRUCT_TYPE_CAMERA_MEDIUM_INFO) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  uint32_t uid = 0;
  if (info->medium) {
    MaterialHandle materialHandle(info->medium);
    if (!materialHandle.isValid()) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    uid = materialHandle.uid;
  }
  {
    ClientMessage c(Commands::RemixApi_SetCameraMediumMaterial);
    c.send_data(uid);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_SetConfigVariable(const char* var, const char* value) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetConfigVariable);
  if (!var || !value) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  {
    ClientMessage c(Commands::RemixApi_SetConfigVariable);
    send(c, var);
    send(c, value);
  }
  return REMIXAPI_ERROR_CODE_SUCCESS;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_CreateD3D9(
  remixapi_Bool       editorModeEnabled,
  IDirect3D9Ex**      out_pD3D9) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_dxvk_CreateD3D9);
  Logger::err("[remixapi_dxvk_CreateD3D9] Not yet supported. Device used by Remix API defaults to "
                                         "most recently created by client application.");
  *out_pD3D9 = nullptr;
  return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
}

remixapi_ErrorCode REMIXAPI_CALL remixapi_dxvk_RegisterD3D9Device(IDirect3DDevice9Ex* d3d9Device) {
  ASSERT_REMIXAPI_PFN_TYPE(remixapi_dxvk_RegisterD3D9Device);
  if (!d3d9Device) {
    return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }
  Logger::err("[remixapi_dxvk_RegisterD3D9Device] Not yet supported. Device used by Remix API defaults to "
                                                 "most recently created by client application.");
  return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
}

// https://stackoverflow.com/a/27490954
constexpr bool strings_equal(char const * a, char const * b) {
  return *a == *b && (*a == '\0' || strings_equal(a + 1, b + 1));
}

// ---- Host settings ----
// Setting handles are allocated here, so RegisterHostSetting never waits for
// the server; the server maps each one to the runtime's handle. The runtime
// gives a re-registered key the handle it already had, and so does the key
// map below. Whether the server's runtime has the exports is asked once, on
// the first call; without them every call fails and nothing else is sent.
namespace {
std::mutex g_hostSettingsMutex;
int g_hostSettingsSupport = -1;   // -1 = not asked yet
uint32_t g_nextHostSettingHandle = 1;
std::unordered_map<std::string, uint32_t> g_hostSettingHandles;   // key -> client handle

// Called with g_hostSettingsMutex held.
bool hostSettingsSupported() {
  if (g_hostSettingsSupport >= 0) {
    return g_hostSettingsSupport != 0;
  }
  uint32_t supported = 0;
  {
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_QueryHostSettings);
      currentUID = c.get_uid();
    }
    const uint32_t timeoutMs = GlobalOptions::getAckTimeout();
    if (Result::Success == DeviceBridge::waitForCommand(Commands::Bridge_Response, timeoutMs, nullptr, true, currentUID)) {
      supported = DeviceBridge::get_data();
      DeviceBridge::pop_front();
    } else {
      Logger::err("[remixapi host settings] No response from the server to the capability query.");
    }
  }
  g_hostSettingsSupport = supported ? 1 : 0;
  if (supported) {
    Logger::info("[remixapi host settings] The server's runtime exports the host-settings calls.");
  } else {
    Logger::warn("[remixapi host settings] The server's runtime has no host-settings exports; "
                 "every host-settings call fails.");
  }
  return supported != 0;
}

void sendHostString(ClientMessage& c, const char* s) {
  const char* const nonNull = s ? s : "";
  send(c, nonNull);
}
}

extern "C" {

  // remix_c.h declares no function for this slot, so the definition below needs this declaration.
  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_GetVramStats(remixapi_VramStats* out_stats);

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_InitializeLibrary(
    const remixapi_InitializeLibraryInfo* info,
    remixapi_Interface*                   out_result) {

    static_assert(strings_equal(__func__, remixapi::exported_func_name::initRemixApi));
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_InitializeLibrary);
    
    if (!GlobalOptions::getExposeRemixApi()) {
      Logger::err("Remix API is not enabled. This is currently an experimental feature and must be explicitly enabled \
                   in the `bridge.conf`. Please set `exposeRemixApi = True` if you are sure you want it enabled.");
      return REMIXAPI_ERROR_CODE_NOT_INITIALIZED;
    }
    if (!info || info->sType != REMIXAPI_STRUCT_TYPE_INITIALIZE_LIBRARY_INFO) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!out_result) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (!isRemixApiVersionCompatible(info->version)) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }

    auto interf = remixapi_Interface {};
    {
      // interf.Startup = remixapi_Startup;
      // interf.Shutdown = remixapi_Shutdown;
      // interf.Present = remixapi_Present;
      interf.CreateMaterial = remixapi_CreateMaterial;
      interf.DestroyMaterial = remixapi_DestroyMaterial;
      interf.CreateTexture = remixapi_CreateTexture;
      interf.DestroyTexture = remixapi_DestroyTexture;
      interf.CreateMesh = remixapi_CreateMesh;
      interf.CreateMeshBatched = remixapi_CreateMeshBatched;
      interf.DestroyMesh = remixapi_DestroyMesh;
      interf.SetupCamera = remixapi_SetupCamera;
      interf.SetCameraMediumMaterial = remixapi_SetCameraMediumMaterial;
      interf.DrawInstance = remixapi_DrawInstance;
      interf.CreateLight = remixapi_CreateLight;
      interf.DestroyLight = remixapi_DestroyLight;
      interf.DrawLightInstance = remixapi_DrawLightInstance;
      interf.SetConfigVariable = remixapi_SetConfigVariable;
      interf.dxvk_CreateD3D9 = remixapi_dxvk_CreateD3D9;
      interf.dxvk_RegisterD3D9Device = remixapi_dxvk_RegisterD3D9Device;
      // Remix Plus API entry points. dxvk-remix's d3d9.dll implements
      // these for real (rtx_remix_api.cpp:2244+ / 2357+ / 2409+ / 2413+), but
      // they are not yet plumbed through the bridge IPC channel. Without a
      // populated function pointer here the slots are NULL and 32-bit clients
      // crash the moment they call interf.GetUIState() / SetUIState() /
      // AutoInstancePersistentLights() / UpdateLightDefinition(). The stubs
      // below return safe defaults so the bridge stays alive; full IPC
      // forwarding is a follow-up task tracked separately.
      interf.GetUIState                   = remixapi_GetUIState;
      interf.SetUIState                   = remixapi_SetUIState;
      interf.AutoInstancePersistentLights = remixapi_AutoInstancePersistentLights;
      interf.UpdateLightDefinition        = remixapi_UpdateLightDefinition;
      interf.SetGameValue                 = remixapi_SetGameValue;
      interf.GetGameValue                 = remixapi_GetGameValue;
      interf.GetVramStats                 = remixapi_GetVramStats;
      // interf.dxvk_GetExternalSwapchain = remixapi_dxvk_GetExternalSwapchain;
      // interf.dxvk_GetVkImage = remixapi_dxvk_GetVkImage;
      // interf.dxvk_CopyRenderingOutput = remixapi_dxvk_CopyRenderingOutput;
      // interf.dxvk_SetDefaultOutput = remixapi_dxvk_SetDefaultOutput;
      // interf.pick_RequestObjectPicking = remixapi_pick_RequestObjectPicking;
      // interf.pick_HighlightObjects = remixapi_pick_HighlightObjects;
    }

    *out_result = interf;
    remixapi::g_appRemixApiVersion = info->version;
    remixapi::g_bInterfaceInitialized = true;

    return REMIXAPI_ERROR_CODE_SUCCESS;
  }
  
  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_RegisterCallbacks(
    PFN_remixapi_BridgeCallback beginSceneCallback,
    PFN_remixapi_BridgeCallback endSceneCallback,
    PFN_remixapi_BridgeCallback presentCallback) {

    static_assert(strings_equal(__func__, remixapi::exported_func_name::registerCallbacks));
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_RegisterCallbacks);

    remixapi::g_beginSceneCallback = beginSceneCallback;
    remixapi::g_endSceneCallback = endSceneCallback;
    remixapi::g_presentCallback = presentCallback;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---- Remix Plus API entry points (client-side stubs) ----
  // Mirror the surface dxvk-remix exposes (rtx_remix_api.cpp ~2244, ~2357,
  // ~2409, ~2413) so the function pointers in remixapi_Interface aren't NULL
  // for 32-bit clients. These stubs do NOT forward over IPC yet — full
  // plumbing is a follow-up. Each logs once the first time it's called so
  // the user sees that the feature isn't reachable through the bridge.

  // Answered from the client's own copy of the UI state (RemixState, kept in step
  // by the server's UWM_REMIX_UIACTIVE_MSG), so it costs no round trip. The
  // message says only open or closed, so an open menu reports ADVANCED.
  DLLEXPORT remixapi_UIState __stdcall remixapi_GetUIState(void) {
    return RemixState::isUIActive() ? REMIXAPI_UI_STATE_ADVANCED : REMIXAPI_UI_STATE_NONE;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SetUIState(remixapi_UIState state) {
    (void) state;
    static bool warned = false;
    if (!warned) {
      Logger::warn("[remixapi_SetUIState] Bridge stub: no-op. "
                   "Remix UI state changes are not yet plumbed through the bridge.");
      warned = true;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_AutoInstancePersistentLights(void) {
    static bool warned = false;
    if (!warned) {
      Logger::warn("[remixapi_AutoInstancePersistentLights] Bridge stub: no-op. "
                   "Persistent-light auto-instancing is not yet plumbed through the bridge.");
      warned = true;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_UpdateLightDefinition(
    remixapi_LightHandle      handle,
    const remixapi_LightInfo* info) {
    (void) handle;
    (void) info;
    static bool warned = false;
    if (!warned) {
      Logger::warn("[remixapi_UpdateLightDefinition] Bridge stub: no-op. "
                   "Light-definition updates are not yet plumbed through the bridge — "
                   "lights created via CreateLight cannot be updated from 32-bit clients yet.");
      warned = true;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SetGameValue(
    const char* key,
    const char* value) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetGameValue);
    if (key == nullptr || key[0] == '\0') {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (value == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    DeviceBridge::ResponseTransaction responseTransaction;

    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_SetGameValue);
      currentUID = c.get_uid();
      send(c, key);
      send(c, value);
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_SetGameValue", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    DeviceBridge::pop_front();
    return result;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_GetGameValue(
    const char* key,
    char*       out_buffer,
    uint32_t    in_buffer_size,
    uint32_t*   out_actual_size) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_GetGameValue);
    if (key == nullptr || key[0] == '\0') {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (out_actual_size == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (in_buffer_size > 0 && out_buffer == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    DeviceBridge::ResponseTransaction responseTransaction;

    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_GetGameValue);
      currentUID = c.get_uid();
      send(c, key);
      c.send_data(in_buffer_size);
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_GetGameValue", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    const uint32_t actual = DeviceBridge::get_data();
    *out_actual_size = actual;
    if (result == REMIXAPI_ERROR_CODE_SUCCESS && actual > 0 && in_buffer_size >= actual) {
      // Server sent the value bytes when the call succeeded and the caller's buffer was large enough (the exact
      // condition sv/main.cpp RemixApi_GetGameValue uses; any other condition would leave the response data out of step).
      void* value_ptr = nullptr;
      const uint32_t value_size = DeviceBridge::get_data(&value_ptr);
      (void) value_size;
      memcpy(out_buffer, value_ptr, actual);
    }
    DeviceBridge::pop_front();
    return result;
  }

  // The server answers with the result and, on success, kVramStatsWords uint64 words: the twelve uint64 fields of
  // remixapi_VramStats in declaration order, then forkTextureCacheCount widened to 64 bits. Field by field, so the
  // x86 and x64 struct layouts need not agree.
  static constexpr uint32_t kVramStatsWords = 13;

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_GetVramStats(remixapi_VramStats* out_stats) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_GetVramStats);
    if (out_stats == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }

    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_GetVramStats);
      currentUID = c.get_uid();
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_GetVramStats", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      void* words = nullptr;
      const uint32_t size = DeviceBridge::get_data(&words);
      if (size == sizeof(uint64_t) * kVramStatsWords) {
        uint64_t w[kVramStatsWords];
        memcpy(w, words, sizeof(w));
        out_stats->totalAllocatedBytes = w[0];
        out_stats->totalUsedBytes = w[1];
        out_stats->poolRetainedBytes = w[2];
        out_stats->usedReplacementGeometryBytes = w[3];
        out_stats->usedBufferBytes = w[4];
        out_stats->usedAccelerationStructureBytes = w[5];
        out_stats->usedOpacityMicromapBytes = w[6];
        out_stats->usedMaterialTextureBytes = w[7];
        out_stats->usedRenderTargetBytes = w[8];
        out_stats->driverAllocatedBytes = w[9];
        out_stats->driverBudgetBytes = w[10];
        out_stats->forkTextureCacheCount = static_cast<uint32_t>(w[12]);
        // w[11] is reserved for the next uint64 field; the bridge sends it as 0.
      }
    }
    DeviceBridge::pop_front();
    return result;
  }

  // ---- Retained instances (remix_c.h "Retained instances") ----
  // Exported by name only (d3d9_lss.def), exactly like the runtime exports
  // them: a host resolves them with GetProcAddress, and their presence is the
  // capability check. Every call is one async message. The client allocates
  // the retained handle itself, so no call waits for the server; the server
  // maps it to the runtime's handle. A server whose runtime lacks the exports
  // logs once and drops the calls.

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_CreateRetainedInstance(
    const remixapi_InstanceInfo* info,
    uint64_t*                    out_handle) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateRetainedInstance);
    if (!info || !out_handle || info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    const uint64_t handle = g_nextRetainedHandle++;
    {
      ClientMessage c(Commands::RemixApi_CreateRetainedInstance);
      sendInstanceInfo(c, info, "remixapi_CreateRetainedInstance");
      c.send_data(sizeof(handle), &handle);
    }
    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_CreateRetainedInstances(
    const remixapi_InstanceInfo* infos,
    uint32_t                     count,
    uint64_t*                    out_handles) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_CreateRetainedInstances);
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!infos || !out_handles) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    // Same contract as the runtime: nothing is created unless every info is valid.
    for (uint32_t i = 0; i < count; ++i) {
      if (infos[i].sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !infos[i].mesh) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    {
      ClientMessage c(Commands::RemixApi_CreateRetainedInstances);
      c.send_data(count);
      for (uint32_t i = 0; i < count; ++i) {
        const uint64_t handle = g_nextRetainedHandle++;
        sendInstanceInfo(c, &infos[i], "remixapi_CreateRetainedInstances");
        c.send_data(sizeof(handle), &handle);
        out_handles[i] = handle;
      }
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_UpdateRetainedInstance(
    uint64_t                     handle,
    const remixapi_InstanceInfo* info) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_UpdateRetainedInstance);
    if (!handle || handle >= g_nextRetainedHandle || !info ||
        info->sType != REMIXAPI_STRUCT_TYPE_INSTANCE_INFO || !info->mesh) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    {
      ClientMessage c(Commands::RemixApi_UpdateRetainedInstance);
      c.send_data(sizeof(handle), &handle);
      sendInstanceInfo(c, info, "remixapi_UpdateRetainedInstance");
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_UpdateRetainedInstances(
    const remixapi_RetainedInstanceUpdate* updates,
    uint32_t                               count) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_UpdateRetainedInstances);
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!updates) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    for (uint32_t i = 0; i < count; ++i) {
      if (!updates[i].handle || updates[i].handle >= g_nextRetainedHandle) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }
    {
      ClientMessage c(Commands::RemixApi_UpdateRetainedInstances);
      c.send_data(count);
      for (uint32_t i = 0; i < count; ++i) {
        sendRetainedUpdate(c, updates[i]);
      }
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_DestroyRetainedInstances(
    const uint64_t* handles,
    uint32_t        count) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_DestroyRetainedInstances);
    if (count == 0) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }
    if (!handles) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    {
      ClientMessage c(Commands::RemixApi_DestroyRetainedInstances);
      c.send_data(count);
      c.send_data(count * static_cast<uint32_t>(sizeof(uint64_t)), handles);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_DrawRetainedInstances(const float* sceneOrigin) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_DrawRetainedInstances);
    {
      ClientMessage c(Commands::RemixApi_DrawRetainedInstances);
      const float zero[3] = { 0.f, 0.f, 0.f };
      send(c, sceneOrigin ? Bool::True : Bool::False);
      c.send_data(sizeof(zero), sceneOrigin ? sceneOrigin : zero);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---- D3D9 texture import (remix_c.h "remixapi_ImportD3D9Texture") ----
  // Exported by name only (d3d9_lss.def), like the runtime export. The host
  // passes the texture it got from this bridge (the client object); the server
  // resolves it to its own texture on the real device and calls the runtime.
  // Only the object id, hash and flags cross the IPC -- never pixels.
  //
  // Like CreateTexture, the client allocates the texture handle and the call is
  // one async message, ordered in the queue after every lock/unlock that filled
  // the texture and before any CreateMaterial naming it. So that a host can
  // fall back per texture without a round trip, the client applies the
  // runtime's eligibility rules itself, and asks the server once (the first
  // call, synchronously) whether the runtime can import at all. A per-texture
  // failure on the server after that is logged there; the handle stays unknown
  // to the server and a later DestroyTexture of it is dropped.
  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_ImportD3D9Texture(
    IDirect3DBaseTexture9*  texture,
    uint64_t                hash,
    uint32_t                flags,
    uint32_t                encoding,
    remixapi_TextureHandle* out_handle) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_ImportD3D9Texture);
    // Capability of the server's runtime: -1 = not asked yet.
    static int s_serverSupport = -1;
    const bool probe = texture == nullptr && out_handle == nullptr;
    if (out_handle) {
      *out_handle = nullptr;
    }
    if (!probe) {
      constexpr uint32_t kKnownFlags = REMIXAPI_D3D9_TEXTURE_IMPORT_SRGB | REMIXAPI_D3D9_TEXTURE_IMPORT_OPAQUE_ALPHA;
      if (!texture || !out_handle || !hash || (flags & ~kKnownFlags) != 0 ||
          encoding > REMIXAPI_TEXTURE_ENCODING_SMOOTHNESS_IN_GREEN ||
          texture->GetType() != D3DRTYPE_TEXTURE) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
      D3DSURFACE_DESC desc = {};
      if (FAILED(static_cast<IDirect3DTexture9*>(texture)->GetLevelDesc(0, &desc)) ||
          (desc.Pool != D3DPOOL_MANAGED && desc.Pool != D3DPOOL_DEFAULT) ||
          (desc.Usage & (D3DUSAGE_RENDERTARGET | D3DUSAGE_DEPTHSTENCIL | D3DUSAGE_DYNAMIC | D3DUSAGE_AUTOGENMIPMAP)) != 0 ||
          desc.MultiSampleType != D3DMULTISAMPLE_NONE) {
        return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
      }
    }

    if (s_serverSupport < 0) {
      DeviceBridge::ResponseTransaction responseTransaction;
      UID currentUID = 0;
      {
        ClientMessage c(Commands::RemixApi_QueryD3D9TextureImport);
        currentUID = c.get_uid();
      }
      WAIT_FOR_SERVER_RESPONSE("remixapi_ImportD3D9Texture", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
      const auto result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
      DeviceBridge::pop_front();
      s_serverSupport = (result == REMIXAPI_ERROR_CODE_SUCCESS) ? 1 : 0;
      if (s_serverSupport) {
        Logger::info("[remixapi_ImportD3D9Texture] The server's runtime imports D3D9 textures.");
      } else {
        Logger::warn(format_string("[remixapi_ImportD3D9Texture] The server's runtime cannot import D3D9 textures "
                                   "(error %d); every import returns GENERAL_FAILURE.", static_cast<int>(result)));
      }
    }
    if (!s_serverSupport) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (probe) {
      return REMIXAPI_ERROR_CODE_SUCCESS;
    }

    auto* const pLssTexture = bridge_cast<Direct3DTexture9_LSS*>(static_cast<IDirect3DTexture9*>(texture));
    const auto textureId = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pLssTexture->D3D<IDirect3DBaseTexture9>()));
    if (textureId == 0) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    TextureHandle newHandle;
    {
      ClientMessage c(Commands::RemixApi_ImportD3D9Texture);
      c.send_data(textureId);
      send(c, hash);
      send(c, flags);
      send(c, encoding);
      sendHandle(c, newHandle);
    }
    *out_handle = newHandle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  // ---- Host settings (remix_c.h "Host settings") ----
  // Exported by name only (d3d9_lss.def), like the runtime exports. Register,
  // Set and Clear are async messages; GetHostSettingsGeneration,
  // GetHostSettingValue and the bridge-only PollHostSettings wait for the
  // server. A host polls with PollHostSettings: an unchanged generation is
  // answered from the one the server publishes at each Present, and one round
  // trip returns the generation and every value asked for when it moved.

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_RegisterHostSetting(
    const remixapi_HostSettingInfo* info,
    remixapi_HostSettingHandle*     out_handle) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_RegisterHostSetting);
    if (out_handle) {
      *out_handle = 0;
    }
    // The runtime's own checks, so a setting it would refuse never gets a
    // client handle the server can't map.
    if (!info || !out_handle || info->structSize < sizeof(remixapi_HostSettingInfo) || !info->key || !info->key[0] ||
        info->type > REMIXAPI_HOST_SETTING_TYPE_ACTION ||
        (info->type == REMIXAPI_HOST_SETTING_TYPE_ENUM && (!info->enumLabels || info->enumCount == 0))) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    auto [it, inserted] = g_hostSettingHandles.try_emplace(std::string(info->key), 0u);
    if (inserted) {
      it->second = g_nextHostSettingHandle++;
    }
    const uint32_t handle = it->second;
    const uint32_t enumCount = info->type == REMIXAPI_HOST_SETTING_TYPE_ENUM ? info->enumCount : 0u;
    {
      ClientMessage c(Commands::RemixApi_RegisterHostSetting);
      c.send_data(handle);
      c.send_data(static_cast<uint32_t>(info->type));
      c.send_data(info->flags);
      send(c, info->minValue);
      send(c, info->maxValue);
      send(c, info->step);
      send(c, info->defaultValue);
      send(c, info->currentValue);
      sendHostString(c, info->key);
      sendHostString(c, info->label);
      sendHostString(c, info->section);
      sendHostString(c, info->tooltip);
      c.send_data(enumCount);
      for (uint32_t i = 0; i < enumCount; ++i) {
        sendHostString(c, info->enumLabels[i]);
      }
    }
    *out_handle = handle;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT uint64_t __stdcall remixapi_GetHostSettingsGeneration(void) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_GetHostSettingsGeneration);
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return 0;
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_GetHostSettingsGeneration);
      currentUID = c.get_uid();
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_GetHostSettingsGeneration", 0, currentUID);
    uint64_t generation = 0;
    void* p = nullptr;
    const uint32_t len = DeviceBridge::get_data(&p);
    if (p && len == sizeof(generation)) {
      memcpy(&generation, p, sizeof(generation));
    }
    DeviceBridge::pop_front();
    return generation;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_GetHostSettingValue(
    remixapi_HostSettingHandle handle,
    double*                    out_value) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_GetHostSettingValue);
    if (!out_value) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (handle == 0 || handle >= g_nextHostSettingHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_GetHostSettingValue);
      currentUID = c.get_uid();
      c.send_data(static_cast<uint32_t>(handle));
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_GetHostSettingValue", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const auto result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    void* p = nullptr;
    const uint32_t len = DeviceBridge::get_data(&p);
    if (result == REMIXAPI_ERROR_CODE_SUCCESS && p && len == sizeof(double)) {
      memcpy(out_value, p, sizeof(double));
    }
    DeviceBridge::pop_front();
    return result;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SetHostSettingValue(
    remixapi_HostSettingHandle handle,
    double                     value) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetHostSettingValue);
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    if (handle == 0 || handle >= g_nextHostSettingHandle) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    {
      ClientMessage c(Commands::RemixApi_SetHostSettingValue);
      c.send_data(static_cast<uint32_t>(handle));
      send(c, value);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT void __stdcall remixapi_ClearHostSettings(void) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_ClearHostSettings);
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return;
    }
    // Handles are never reused: a stale one stays unknown to the server.
    g_hostSettingHandles.clear();
    ClientMessage c(Commands::RemixApi_ClearHostSettings);
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_PollHostSettings(
    uint64_t                          last_generation,
    uint64_t*                         out_generation,
    double*                           out_values,
    const remixapi_HostSettingHandle* handles,
    uint32_t                          count) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_PollHostSettings);
    if (!out_generation || (count && (!out_values || !handles))) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    std::lock_guard<std::mutex> lock(g_hostSettingsMutex);
    if (!hostSettingsSupported()) {
      return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    // The round trip waits for the server to work through every command queued before it. The generation the server
    // published at its last Present answers an unchanged one without that wait. ~0 asks for the values in any case.
    if (auto* pBlock = bridge_util::QueryResults::block()) {
      if (last_generation != ~0ull && pBlock->hostSettingsPublished.load(std::memory_order_acquire) &&
          pBlock->hostSettingsGeneration.load(std::memory_order_relaxed) == static_cast<uint32_t>(last_generation)) {
        *out_generation = last_generation;
        return REMIXAPI_ERROR_CODE_SUCCESS;
      }
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_PollHostSettings);
      currentUID = c.get_uid();
      c.send_data(sizeof(last_generation), &last_generation);
      c.send_data(count);
      if (count) {
        c.send_data(count * static_cast<uint32_t>(sizeof(remixapi_HostSettingHandle)), handles);
      }
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_PollHostSettings", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    uint64_t generation = last_generation;
    void* p = nullptr;
    uint32_t len = DeviceBridge::get_data(&p);
    if (p && len == sizeof(generation)) {
      memcpy(&generation, p, sizeof(generation));
    }
    // The server sends the values only when the generation moved.
    const uint32_t sent = DeviceBridge::get_data();
    if (sent) {
      p = nullptr;
      len = DeviceBridge::get_data(&p);
      if (sent == count && p && len == count * sizeof(double)) {
        memcpy(out_values, p, len);
      } else {
        generation = last_generation;   // malformed: report no change
      }
    }
    DeviceBridge::pop_front();
    *out_generation = generation;
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

}


  // ---- Color grading (remix_c.h, "Color grading") ----
  // The grade fields cross the bridge as 16 floats: the matrix, the balance, saturation, contrast, anchor and shoulder.

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SetColorGradePreset(const char* name, float blendSeconds) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetColorGradePreset);
    if (name == nullptr || !(blendSeconds >= 0.0f) || blendSeconds == INFINITY) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_SetColorGradePreset);
      currentUID = c.get_uid();
      send(c, name);
      send(c, blendSeconds);
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_SetColorGradePreset", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    DeviceBridge::pop_front();
    return result;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SetColorGrade(const remixapi_ColorGradeEXT* grade, float blendSeconds) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_SetColorGrade);
    if (grade == nullptr || grade->sType != REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (grade->version != REMIXAPI_COLOR_GRADE_VERSION) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    if (!(blendSeconds >= 0.0f) || blendSeconds == INFINITY) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    float fields[16];
    memcpy(fields, grade->matrix, sizeof(float) * 9);
    memcpy(fields + 9, grade->balance, sizeof(float) * 3);
    fields[12] = grade->saturation;
    fields[13] = grade->contrast;
    fields[14] = grade->anchor;
    fields[15] = grade->shoulder;
    {
      ClientMessage c(Commands::RemixApi_SetColorGrade);
      c.send_data(sizeof(fields), fields);
      send(c, blendSeconds);
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_MixColorGradePresets(const char* nameA, const char* nameB, float weight) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_MixColorGradePresets);
    if (nameA == nullptr || nameB == nullptr || !(weight >= 0.0f && weight <= 1.0f)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_MixColorGradePresets);
      currentUID = c.get_uid();
      send(c, nameA);
      send(c, nameB);
      send(c, weight);
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_MixColorGradePresets", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    DeviceBridge::pop_front();
    return result;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_GetColorGrade(remixapi_ColorGradeEXT* out) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_GetColorGrade);
    if (out == nullptr || out->sType != REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (out->version != REMIXAPI_COLOR_GRADE_VERSION) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_GetColorGrade);
      currentUID = c.get_uid();
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_GetColorGrade", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    if (result == REMIXAPI_ERROR_CODE_SUCCESS) {
      // The server sends the 16 floats only on success.
      void* fields = nullptr;
      const uint32_t size = DeviceBridge::get_data(&fields);
      if (size == sizeof(float) * 16) {
        const float* f = static_cast<const float*>(fields);
        memcpy(out->matrix, f, sizeof(float) * 9);
        memcpy(out->balance, f + 9, sizeof(float) * 3);
        out->saturation = f[12];
        out->contrast = f[13];
        out->anchor = f[14];
        out->shoulder = f[15];
      }
    }
    DeviceBridge::pop_front();
    return result;
  }

  DLLEXPORT remixapi_ErrorCode __stdcall remixapi_SaveColorGradePreset(const char* name, const remixapi_ColorGradeEXT* grade) {
    ASSERT_REMIXAPI_PFN_TYPE(remixapi_SaveColorGradePreset);
    if (name == nullptr || grade == nullptr || grade->sType != REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (grade->version != REMIXAPI_COLOR_GRADE_VERSION) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    float fields[16];
    memcpy(fields, grade->matrix, sizeof(float) * 9);
    memcpy(fields + 9, grade->balance, sizeof(float) * 3);
    fields[12] = grade->saturation;
    fields[13] = grade->contrast;
    fields[14] = grade->anchor;
    fields[15] = grade->shoulder;
    DeviceBridge::ResponseTransaction responseTransaction;
    UID currentUID = 0;
    {
      ClientMessage c(Commands::RemixApi_SaveColorGradePreset);
      currentUID = c.get_uid();
      send(c, name);
      c.send_data(sizeof(fields), fields);
    }
    WAIT_FOR_SERVER_RESPONSE("remixapi_SaveColorGradePreset", REMIXAPI_ERROR_CODE_GENERAL_FAILURE, currentUID);
    const remixapi_ErrorCode result = static_cast<remixapi_ErrorCode>(DeviceBridge::get_data());
    DeviceBridge::pop_front();
    return result;
  }
}
