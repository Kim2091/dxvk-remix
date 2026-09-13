/*
* Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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
#include "rtx_pathtracer_integrate_indirect.h"
#include <algorithm>
#include "dxvk_device.h"
#include "rtx_shader_manager.h"
#include "rtx_context.h"
#include "rtx_options.h"
#include "rtx_sparse_rendering.h"
#include "rtx_neural_radiance_cache.h"
#include "rtx_restir_gi_rayquery.h"
#include "rtx_debug_view.h"

#include "rtx/pass/common_binding_indices.h"
#include "rtx/pass/integrate/integrate_indirect_binding_indices.h"
#include "rtx/pass/integrate/integrate_nee_binding_indices.h"
#include "rtx/pass/integrate/integrate_fused_binding_indices.h"
#include <rtx_shaders/integrate_indirect_sharc_query_fused.h>
#include <rtx_shaders/integrate_indirect_sharc_query_fused_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_fused_ser.h>
#include <rtx_shaders/integrate_indirect_sharc_query_fused_ser_wboit.h>

#include "rtx/concept/surface_material/surface_material_hitgroup.h"

#include <rtx_shaders/integrate_indirect_raygen_neeCache.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen.h>
#include <rtx_shaders/integrate_indirect_raygen.h>
#include <rtx_shaders/integrate_indirect_raygen_ser.h>
#include <rtx_shaders/integrate_indirect_raygen_nrc_neeCache.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_nrc_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_nrc_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_nrc.h>
#include <rtx_shaders/integrate_indirect_raygen_nrc.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_nrc.h>

#include <rtx_shaders/integrate_indirect_rayquery_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery.h>
#include <rtx_shaders/integrate_indirect_rayquery_nrc_neeCache.h>
#include <rtx_shaders/integrate_indirect_rayquery_nrc.h>
#include <rtx_shaders/integrate_indirect_sharc_update.h>
#include <rtx_shaders/integrate_indirect_sharc_update_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_update_raygen.h>
#include <rtx_shaders/integrate_indirect_sharc_update_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred_raygen.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4_raygen.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query.h>
#include <rtx_shaders/integrate_indirect_sharc_query_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_raygen.h>
#include <rtx_shaders/integrate_indirect_sharc_query_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_raygen_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_raygen_stats_wboit.h>

#include <rtx_shaders/integrate_indirect_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_material_rayPortal_closestHit.h>
#include <rtx_shaders/integrate_indirect_pom_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_pom_material_rayPortal_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_material_rayPortal_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_pom_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_pom_material_rayPortal_closestHit.h>
#include <rtx_shaders/integrate_indirect_neeCache_material_rayportal_closestHit.h>
#include <rtx_shaders/integrate_indirect_neeCache_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_neeCache_pom_material_rayportal_closestHit.h>
#include <rtx_shaders/integrate_indirect_neeCache_pom_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_material_rayportal_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_material_opaque_translucent_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_pom_material_rayportal_closestHit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_pom_material_opaque_translucent_closestHit.h>

#include <rtx_shaders/integrate_indirect_miss.h>
#include <rtx_shaders/integrate_indirect_miss_neeCache.h>
#include <rtx_shaders/integrate_indirect_miss_nrc.h>
#include <rtx_shaders/integrate_indirect_miss_nrc_neeCache.h>



#include <rtx_shaders/integrate_indirect_raygen_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_nrc_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_nrc_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_nrc_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_raygen_nrc_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_nrc_wboit.h>
#include <rtx_shaders/integrate_indirect_raygen_ser_nrc_wboit.h>

#include <rtx_shaders/integrate_indirect_rayquery_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_nrc_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_rayquery_nrc_wboit.h>

#include <rtx_shaders/integrate_indirect_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_material_rayPortal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_pom_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_pom_material_rayPortal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_material_rayPortal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_pom_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_pom_material_rayPortal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_neeCache_material_rayportal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_neeCache_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_neeCache_pom_material_rayportal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_neeCache_pom_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_material_rayportal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_material_opaque_translucent_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_pom_material_rayportal_closestHit_wboit.h>
#include <rtx_shaders/integrate_indirect_nrc_neeCache_pom_material_opaque_translucent_closestHit_wboit.h>

#include <rtx_shaders/integrate_indirect_miss_wboit.h>
#include <rtx_shaders/integrate_indirect_miss_neeCache_wboit.h>
#include <rtx_shaders/integrate_indirect_miss_nrc_wboit.h>
#include <rtx_shaders/integrate_indirect_miss_nrc_neeCache_wboit.h>

#include <rtx_shaders/integrate_nee.h>
#include <rtx_shaders/integrate_nee_plain.h>
#include <rtx_shaders/integrate_nee_nrc.h>
#include <rtx_shaders/integrate_nee_restir_gi.h>
#include <rtx_shaders/visualize_nee.h>

#include "dxvk_scoped_annotation.h"
#include "rtx_opacity_micromap_manager.h"
#include "rtx_neural_radiance_cache.h"
#include <rtx_shaders/integrate_indirect_sharc_query_trace.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_stats_wboit.h>

#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_pom.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_pom_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_pom_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_pom_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals_wboit.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals_stats.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals_stats_wboit.h>

#include "rtx_sharc.h"
#include <rtx_shaders/integrate_indirect_sharc_update_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_update_raygen_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred_raygen_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_update_deferred4_raygen_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_stats_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_trace_ser_stats_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats_lean.h>
#include <rtx_shaders/integrate_indirect_sharc_query_miss_no_portals_stats_lean.h>

#include "rtx/pass/sharc/sharc_binding_indices.h"

namespace dxvk {

  // Defined within an unnamed namespace to ensure unique definition across binary
  namespace {
    class IntegrateIndirectRayGenShader : public ManagedShader {
    public:
      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        SAMPLER(INTEGRATE_INDIRECT_BINDING_LINEAR_WRAP_SAMPLER)

        SAMPLERCUBE(INTEGRATE_INDIRECT_BINDING_SKYPROBE)

        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SHARED_TEXTURE_COORD_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SHARED_SUBSURFACE_DATA_INPUT)

        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PRIMARY_CONE_RADIUS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SECONDARY_CONE_RADIUS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PRIMARY_WORLD_POSITION_INPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_PRIMARY_RTXDI_RESERVOIR)

        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_RAY_ORIGIN_DIRECTION_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_THROUGHPUT_CONE_RADIUS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_ACTIVE_LOCAL_PIXEL_COORDS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_FIRST_HIT_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_FIRST_SAMPLED_LOBE_DATA_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_LAST_GBUFFER_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PREV_WORLD_POSITION_INPUT)
        SAMPLER3D(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT)
        SAMPLER3D(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PRIMARY_HIT_DISTANCE_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SECONDARY_HIT_DISTANCE_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_LAST_COMPOSITE_INPUT)

        TEXTURE2DARRAY(INTEGRATE_INDIRECT_BINDING_GRADIENTS_INPUT)

        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_PATH_DATA0_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_UPDATE_PATH_DATA0_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_PATH_DATA1_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_UPDATE_PATH_DATA1_INPUT)

        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_RG_INPUT)
        TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_B_INPUT)

        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PRIMARY_DIRECT_DIFFUSE_LOBE_RADIANCE_INPUT_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_PRIMARY_DIRECT_SPECULAR_LOBE_RADIANCE_INPUT_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SECONDARY_COMBINED_DIFFUSE_LOBE_RADIANCE_INPUT_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_SECONDARY_COMBINED_SPECULAR_LOBE_RADIANCE_INPUT_OUTPUT)

        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NRC_QUERY_PATH_INFO_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_PATH_INFO_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NRC_QUERY_RADIANCE_PARAMS_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NRC_COUNTERS_INPUT_OUTPUT)

        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RESERVOIR_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RADIANCE_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_RESTIR_GI_HIT_GEOMETRY_OUTPUT)

        STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NEE_CACHE)
        STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_SAMPLE)
        STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_PRIMITIVE_ID_PREFIX_SUM)
        RW_STRUCTURED_BUFFER(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_TASK)
        RW_TEXTURE2D(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_THREAD_TASK)


        RW_TEXTURE2D(INTEGRATE_INSTRUMENTATION)

      END_PARAMETER()
    };

    class IntegrateIndirectSharcBaseShader : public IntegrateIndirectRayGenShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateIndirectRayGenShader::getResourceSlots();
        // SHARC stages do not consume NRC or ReSTIR GI state.
        slots.erase(std::remove_if(slots.begin(), slots.end(), [](const auto& slot) {
          switch (slot.slot) {
          case INTEGRATE_INDIRECT_BINDING_NRC_PATH_DATA0_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_UPDATE_PATH_DATA0_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_PATH_DATA1_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_UPDATE_PATH_DATA1_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_RG_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_GBUFFER_SURFACE_RADIANCE_B_INPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_QUERY_PATH_INFO_INPUT_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_PATH_INFO_INPUT_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_QUERY_RADIANCE_PARAMS_INPUT_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_NRC_COUNTERS_INPUT_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RESERVOIR_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RADIANCE_OUTPUT:
          case INTEGRATE_INDIRECT_BINDING_RESTIR_GI_HIT_GEOMETRY_OUTPUT:
            return true;
          default:
            return false;
          }
        }), slots.end());
        return slots;
      }
    };

    class IntegrateIndirectSharcShader : public IntegrateIndirectSharcBaseShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateIndirectSharcBaseShader::getResourceSlots();
        slots.push_back({ SHARC_BINDING_HASH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ SHARC_BINDING_RESOLVED, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT });
        return slots;
      }
    };

    class IntegrateIndirectSharcFusedShader : public IntegrateIndirectSharcShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateIndirectSharcShader::getResourceSlots();
        slots.push_back({ INTEGRATE_FUSED_BINDING_SHARED_MATERIAL_DATA0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_SHARED_MATERIAL_DATA1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_WORLD_SHADING_NORMAL, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_ALBEDO, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_VIEW_DIRECTION, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_POSITION_ERROR, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_BASE_REFLECTIVITY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_INDIRECT_DIFFUSE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT });
        slots.push_back({ INTEGRATE_FUSED_BINDING_PRIMARY_INDIRECT_SPECULAR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_IMAGE_VIEW_TYPE_2D, VK_ACCESS_SHADER_WRITE_BIT });
        return slots;
      }
    };

    class IntegrateIndirectSharcStatsShader : public IntegrateIndirectSharcShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateIndirectSharcShader::getResourceSlots();
        slots.push_back({ SHARC_BINDING_QUERY_STATS, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT });
        return slots;
      }
    };

    class IntegrateIndirectSharcUpdateShader : public IntegrateIndirectSharcBaseShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateIndirectSharcBaseShader::getResourceSlots();
        slots.push_back({ SHARC_BINDING_HASH, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT });
        slots.push_back({ SHARC_BINDING_ACCUMULATION, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_IMAGE_VIEW_TYPE_MAX_ENUM, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT });
        slots.erase(std::remove_if(slots.begin(), slots.end(), [](const auto& slot) {
          return slot.slot == INTEGRATE_INDIRECT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT
              || slot.slot == INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RESERVOIR_OUTPUT
              || slot.slot == INTEGRATE_INDIRECT_BINDING_RESTIR_GI_RADIANCE_OUTPUT
              || slot.slot == INTEGRATE_INDIRECT_BINDING_RESTIR_GI_HIT_GEOMETRY_OUTPUT
              || slot.slot == INTEGRATE_INDIRECT_BINDING_NEE_CACHE_TASK
              || slot.slot == BINDING_DEBUG_VIEW_TEXTURE;
        }), slots.end());
        for (auto& slot : slots) {
          if (slot.slot == INTEGRATE_INDIRECT_BINDING_NEE_CACHE_THREAD_TASK) {
            slot.access = VK_ACCESS_SHADER_READ_BIT;
          }
        }
        return slots;
      }
    };

    class IntegrateIndirectClosestHitShader : public ManagedShader {

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    class IntegrateIndirectMissShader : public ManagedShader {

      BEGIN_PARAMETER()
      END_PARAMETER()
    };

    template<typename BaseShader>
    class IntegrateIndirectLeanShader : public BaseShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = BaseShader::getResourceSlots();
        // Lean stages never reuse RTXDI samples or access previous-frame lights.
        slots.erase(std::remove_if(slots.begin(), slots.end(), [](const auto& slot) {
          return slot.slot == INTEGRATE_INDIRECT_BINDING_PRIMARY_RTXDI_RESERVOIR
              || slot.slot == BINDING_PREVIOUS_LIGHT_DATA_BUFFER;
        }), slots.end());
        return slots;
      }
    };

    DxvkRaytracingPipelineShaders getLeanSharcTracePipelineShaders(bool serEnabled, bool ommEnabled, bool statsEnabled) {
      DxvkRaytracingPipelineShaders shaders;
      if (statsEnabled) {
        shaders.addGeneralShader(serEnabled
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcStatsShader>, integrate_indirect_sharc_query_trace_ser_stats_lean)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcStatsShader>, integrate_indirect_sharc_query_trace_stats_lean));
        shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals_stats_lean));
        shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats_lean), nullptr, nullptr);
      } else {
        shaders.addGeneralShader(serEnabled
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcShader>, integrate_indirect_sharc_query_trace_ser_lean)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcShader>, integrate_indirect_sharc_query_trace_lean));
        shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals_lean));
        shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom_lean), nullptr, nullptr);
      }
      shaders.debugName = "SHARC Lean Query";
      if (ommEnabled) {
        shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
      }
      return shaders;
    }

    Rc<DxvkShader> getLeanSharcUpdateShader(bool rayGeneration, bool deferred, bool shortPath) {
      if (rayGeneration) {
        if (!deferred) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_raygen_lean);
        }
        return shortPath
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_deferred4_raygen_lean)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_deferred_raygen_lean);
      } else {
        if (!deferred) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_lean);
        }
        return shortPath
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_deferred4_lean)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectLeanShader<IntegrateIndirectSharcUpdateShader>, integrate_indirect_sharc_update_deferred_lean);
      }
    }

    DxvkRaytracingPipelineShaders getSharcTracePipelineShaders(
      bool serEnabled, bool ommEnabled, bool wboitEnabled, bool statsEnabled, bool includePortals, bool pomEnabled, bool fusedAssembly = false) {
      DxvkRaytracingPipelineShaders shaders;
      if (statsEnabled) {
        if (serEnabled) {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_trace_ser_stats_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_trace_ser_stats));
        } else {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_trace_stats_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_trace_stats));
        }
        if (includePortals) {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_stats_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_stats));
        } else {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals_stats_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals_stats));
        }
        if (includePortals) {
          if (pomEnabled) {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_stats_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_stats), nullptr, nullptr);
          } else {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_pom_stats_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_pom_stats), nullptr, nullptr);
          }
        } else {
          if (pomEnabled) {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_stats_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_stats), nullptr, nullptr);
          } else {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom_stats), nullptr, nullptr);
          }
        }
      } else {
        if (fusedAssembly) {
          if (serEnabled) {
            shaders.addGeneralShader(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcFusedShader, integrate_indirect_sharc_query_fused_ser_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcFusedShader, integrate_indirect_sharc_query_fused_ser));
          } else {
            shaders.addGeneralShader(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcFusedShader, integrate_indirect_sharc_query_fused_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcFusedShader, integrate_indirect_sharc_query_fused));
          }
        } else if (serEnabled) {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_trace_ser_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_trace_ser));
        } else {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_trace_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_trace));
        }
        if (includePortals) {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss));
        } else {
          shaders.addGeneralShader(wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_sharc_query_miss_no_portals));
        }
        if (includePortals) {
          if (pomEnabled) {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit), nullptr, nullptr);
          } else {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_pom_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_pom), nullptr, nullptr);
          }
        } else {
          if (pomEnabled) {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals), nullptr, nullptr);
          } else {
            shaders.addHitGroup(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_sharc_query_closesthit_no_portals_no_pom), nullptr, nullptr);
          }
        }
      }
      shaders.debugName = serEnabled ? "SHARC Query TraceRay + SER" : "SHARC Query TraceRay";
      if (ommEnabled) {
        shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
      }
      return shaders;
    }

    class IntegrateNEEShader : public ManagedShader {
      SHADER_SOURCE(IntegrateNEEShader, VK_SHADER_STAGE_COMPUTE_BIT, integrate_nee)

      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA0_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA1_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_TEXTURE_COORD_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DATA_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_ACTIVE_LOCAL_PIXEL_COORDS_INPUT)

        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_HIT_DISTANCE_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_ALBEDO_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_VIEW_DIRECTION_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_CONE_RADIUS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_POSITION_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_POSITION_ERROR_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_INPUT)
        STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_PRIMITIVE_ID_PREFIX_SUM_INPUT)

        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_BASE_REFLECTIVITY_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT)

        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_DIFFUSE_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_SPECULAR_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_RESTIR_GI_RESERVOIR_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_BSDF_FACTOR2_OUTPUT)

        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE_TASK)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE_SAMPLE)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_NEE_CACHE_THREAD_TASK)
      END_PARAMETER()
    };

    class IntegrateNEEPlainShader : public IntegrateNEEShader {
    public:
      static std::vector<dxvk::DxvkResourceSlot> getResourceSlots() {
        auto slots = IntegrateNEEShader::getResourceSlots();
        slots.erase(std::remove_if(slots.begin(), slots.end(), [](const auto& slot) {
          return slot.slot == INTEGRATE_NEE_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT
            || slot.slot == INTEGRATE_NEE_BINDING_RESTIR_GI_RESERVOIR_OUTPUT
            || slot.slot == INTEGRATE_NEE_BINDING_BSDF_FACTOR2_OUTPUT;
        }), slots.end());
        return slots;
      }
    };

    class VisualizeNEEShader : public ManagedShader {
      SHADER_SOURCE(VisualizeNEEShader, VK_SHADER_STAGE_COMPUTE_BIT, visualize_nee)

      PUSH_CONSTANTS(VisualizeNeeArgs)
      
      BINDLESS_ENABLED()

      BEGIN_PARAMETER()
        COMMON_RAYTRACING_BINDINGS

        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_FLAGS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA0_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA1_INPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_TEXTURE_COORD_INPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SURFACE_INDEX_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DATA_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT)

        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_HIT_DISTANCE_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_ALBEDO_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_VIEW_DIRECTION_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_CONE_RADIUS_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_POSITION_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_POSITION_ERROR_INPUT)
        TEXTURE2D(INTEGRATE_NEE_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_INPUT)
        STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_PRIMITIVE_ID_PREFIX_SUM_INPUT)

        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_BASE_REFLECTIVITY_INPUT_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT)

        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_DIFFUSE_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_SPECULAR_RADIANCE_HIT_DISTANCE_OUTPUT)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_RESTIR_GI_RESERVOIR_OUTPUT)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_BSDF_FACTOR2_OUTPUT)

        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE_TASK)
        RW_STRUCTURED_BUFFER(INTEGRATE_NEE_BINDING_NEE_CACHE_SAMPLE)
        RW_TEXTURE2D(INTEGRATE_NEE_BINDING_NEE_CACHE_THREAD_TASK)
      END_PARAMETER()
    };

    PREWARM_SHADER_PIPELINE(VisualizeNEEShader);
  }

  DxvkPathtracerIntegrateIndirect::DxvkPathtracerIntegrateIndirect(DxvkDevice* device) 
    : CommonDeviceObject(device)
    , m_integrateIndirectMode(IntegrateIndirectMode::Count) {
  }

  void DxvkPathtracerIntegrateIndirect::prewarmShaders(DxvkPipelineManager& pipelineManager) const {
    ScopedCpuProfileZoneN("Indirect Integrate Shader Prewarming");

    IntegrateNEEShader::getShader();
    GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEPlainShader, integrate_nee_plain);
    GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEShader, integrate_nee_nrc);
    GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEShader, integrate_nee_restir_gi);

    const bool isNrcSupported = NeuralRadianceCache::checkIsSupported(device());
    const bool isOpacityMicromapSupported = OpacityMicromapManager::checkIsOpacityMicromapSupported(*m_device);
    const bool isShaderExecutionReorderingSupported = 
      RtxContext::checkIsShaderExecutionReorderingSupported(*m_device) &&
      RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled();
    // Note: Portal enablement is controlled only via the configuration so unlike other things which can be enabled/disabled via
    // ImGui at runtime this is fine to use as a guide for which permutations need to be generated (much like if OMM or SER are
    // supported on a given platform, as this fact will not change during runtime either).
    const bool portalsEnabled = RtxOptions::rayPortalModelTextureHashes().size() > 0;

    if (RtxOptions::Shader::prewarmAllVariants()) {
      for (int32_t nrcEnabled = isNrcSupported ? 1 : 0; nrcEnabled >= 0; nrcEnabled--) {
        for (int32_t useNeeCache = 1; useNeeCache >= 0; useNeeCache--) {
          for (int32_t wboitEnabled = 1; wboitEnabled >= 0; wboitEnabled--) {
            for (int32_t includesPortals = portalsEnabled; includesPortals >= 0; includesPortals--) {
              for (int32_t useRayQuery = 1; useRayQuery >= 0; useRayQuery--) {
                for (int32_t serEnabled = isShaderExecutionReorderingSupported; serEnabled >= 0; serEnabled--) {
                  for (int32_t ommEnabled = isOpacityMicromapSupported; ommEnabled >= 0; ommEnabled--) {
                    for (int32_t pomEnabled = 1; pomEnabled >= 0; pomEnabled--) {
                      pipelineManager.registerRaytracingShaders(getPipelineShaders(useRayQuery, serEnabled, ommEnabled, useNeeCache, includesPortals, pomEnabled, nrcEnabled, wboitEnabled));
                    }
                  }
                }
              }
            }

            getComputeShader(useNeeCache, nrcEnabled, wboitEnabled);
          }
        }
      }
      // Prewarm the SHARC compute fallbacks separately from the legacy variants.
      if (RtxSharc::isSupported(*m_device)) {
        getComputeShader(NeeCachePass::enable(), false, false, true, false);
        getComputeShader(NeeCachePass::enable(), false, false, false, true);
      }
    } else {
      // Note: The getters for these SER/OMM enabled flags also check if SER/OMMs are supported, so we do not need to check for that manually.
      const bool serEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled();
      const bool ommEnabled = OpacityMicromapManager::checkIsOpacityMicromapSupported(*m_device) && RtxOptions::OpacityMicromap::enable();
      const bool useNeeCache = NeeCachePass::enable();
      const bool nrcEnabled = RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::NeuralRadianceCache;
      const bool wboitEnabled = RtxOptions::wboitEnabled();

      if (RtxSharc::isSupported(*m_device)
          && RtxOptions::integrateIndirectMode() == IntegrateIndirectMode::Sharc) {
        if (RtxSharc::leanSecondary()) {
          for (bool stats : { false, true }) {
            pipelineManager.registerRaytracingShaders(getLeanSharcTracePipelineShaders(serEnabled, ommEnabled, stats));
          }
          getLeanSharcUpdateShader(RtxSharc::updateRayGeneration(), RtxSharc::deferredUpdates(), RtxSharc::updateBounces() <= 4);
        }
        getComputeShader(useNeeCache, false, false, true, false);
        getComputeShader(useNeeCache, false, false, false, true);
        if (RtxSharc::queryTraceRay()) {
          for (int32_t includesPortals = portalsEnabled; includesPortals >= 0; --includesPortals) {
            for (int32_t pomEnabled = 1; pomEnabled >= 0; --pomEnabled) {
              pipelineManager.registerRaytracingShaders(getSharcTracePipelineShaders(serEnabled, ommEnabled, wboitEnabled,
                RtxSharc::measureGpuTime() && RtxSharc::collectQueryStats(), includesPortals != 0, pomEnabled != 0));
            }
          }
        }
      }

      for (int32_t includesPortals = portalsEnabled; includesPortals >= 0; includesPortals--) {
        // Prewarm POM on and off, as that can change based on game content (if nothing in the frame has a height texture, then POM turns off)
        for (int32_t pomEnabled = 1; pomEnabled >= 0; pomEnabled--) {
          DxvkComputePipelineShaders shaders;
          switch (RtxOptions::renderPassIntegrateIndirectRaytraceMode()) {
          case RaytraceMode::RayQuery:
            getComputeShader(useNeeCache, nrcEnabled, wboitEnabled);
            break;
          case RaytraceMode::RayQueryRayGen:
            pipelineManager.registerRaytracingShaders(getPipelineShaders(true, serEnabled, ommEnabled, useNeeCache, includesPortals, pomEnabled, nrcEnabled, wboitEnabled));
            break;
          case RaytraceMode::TraceRay:
            pipelineManager.registerRaytracingShaders(getPipelineShaders(false, serEnabled, ommEnabled, useNeeCache, includesPortals, pomEnabled, nrcEnabled, wboitEnabled));
            break;
          case RaytraceMode::Count:
            assert(false && "Invalid RaytraceMode in DxvkPathtracerIntegrateIndirect::prewarmShaders");
            break;
          }
        }
      }
    }
  }

  void DxvkPathtracerIntegrateIndirect::logIntegrateIndirectMode() {
    if (m_integrateIndirectMode != RtxOptions::integrateIndirectMode()) {
      m_integrateIndirectMode = RtxOptions::integrateIndirectMode();

      switch (m_integrateIndirectMode) {
      default:
        assert(0);
        break;
      case IntegrateIndirectMode::ImportanceSampled:
        Logger::info("[RTX] Integrate Indirect Mode: Importance Sampled - activated");
        break;
      case IntegrateIndirectMode::ReSTIRGI:
        Logger::info("[RTX] Integrate Indirect Mode: ReSTIR GI - activated");
        break;
      case IntegrateIndirectMode::NeuralRadianceCache:
        Logger::info("[RTX] Integrate Indirect Mode: Neural Radiance Cache - activated");
        break;
      case IntegrateIndirectMode::Sharc:
        Logger::info("[RTX] Integrate Indirect Mode: SHARC - activated");
        break;
      }
    }
  }

  void DxvkPathtracerIntegrateIndirect::dispatch(
    RtxContext* ctx, 
    const Resources::RaytracingOutput& rtOutput,
    bool sharcUpdate, bool fusedAssembly) {

    const uint32_t frameIdx = ctx->getDevice()->getCurrentFrameId();

    logIntegrateIndirectMode();

    // Bind resources

    // Note: Clamp to edge used to avoid interpolation to black on the edges of the view.
    Rc<DxvkSampler> linearClampSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    Rc<DxvkSampler> linearWrapSampler = ctx->getResourceManager().getSampler(VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    Rc<DxvkBuffer> primitiveIDPrefixSumBuffer = ctx->getSceneManager().getCurrentFramePrimitiveIDPrefixSumBuffer();

    ctx->bindCommonRayTracingResources(rtOutput);

    ctx->bindResourceSampler(INTEGRATE_INDIRECT_BINDING_LINEAR_WRAP_SAMPLER, linearWrapSampler);

    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SKYPROBE, ctx->getResourceManager().getSkyProbe(ctx).view, nullptr);
    ctx->bindResourceSampler(INTEGRATE_INDIRECT_BINDING_SKYPROBE, linearClampSampler);

    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SHARED_FLAGS_INPUT, rtOutput.m_sharedFlags.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SHARED_MEDIUM_MATERIAL_INDEX_INPUT, rtOutput.m_sharedMediumMaterialIndex.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SHARED_TEXTURE_COORD_INPUT, rtOutput.m_sharedTextureCoord.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SHARED_SURFACE_INDEX_INPUT, rtOutput.m_sharedSurfaceIndex.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SHARED_SUBSURFACE_DATA_INPUT, rtOutput.m_sharedSubsurfaceData.view, nullptr);

    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PRIMARY_CONE_RADIUS_INPUT, rtOutput.m_primaryConeRadius.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SECONDARY_CONE_RADIUS_INPUT, rtOutput.m_secondaryConeRadius.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PRIMARY_WORLD_POSITION_INPUT, rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceBuffer(INTEGRATE_INDIRECT_BINDING_PRIMARY_RTXDI_RESERVOIR, DxvkBufferSlice(rtOutput.m_rtxdiReservoirBuffer, 0, rtOutput.m_rtxdiReservoirBuffer->info().size));
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_RAY_ORIGIN_DIRECTION_INPUT, rtOutput.m_indirectRayOriginDirection.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_ACTIVE_LOCAL_PIXEL_COORDS_INPUT, rtOutput.m_sparseRenderingIndirectActiveLocalPixelCoords.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_FIRST_HIT_PERCEPTUAL_ROUGHNESS_INPUT, rtOutput.m_indirectFirstHitPerceptualRoughness.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_LAST_GBUFFER_INPUT, rtOutput.m_gbufferLast.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PREV_WORLD_POSITION_INPUT, rtOutput.getPreviousPrimaryWorldPositionWorldTriangleNormal().view(Resources::AccessType::Read, rtOutput.getPreviousPrimaryWorldPositionWorldTriangleNormal().matchesWriteFrameIdx(frameIdx - 1)), nullptr);

    const RtxGlobalVolumetrics& globalVolumetrics = ctx->getCommonObjects()->metaGlobalVolumetrics();
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT, globalVolumetrics.getCurrentVolumeAccumulatedRadianceY().view, nullptr);
    ctx->bindResourceSampler(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_Y_INPUT, linearClampSampler);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT, globalVolumetrics.getCurrentVolumeAccumulatedRadianceCoCg().view, nullptr);
    ctx->bindResourceSampler(INTEGRATE_INDIRECT_BINDING_VOLUME_FILTERED_RADIANCE_CO_CG_INPUT, linearClampSampler);

    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PRIMARY_HIT_DISTANCE_INPUT, rtOutput.m_primaryHitDistance.view, nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SECONDARY_HIT_DISTANCE_INPUT, rtOutput.m_secondaryHitDistance.view, nullptr);

    const DxvkReSTIRGIRayQuery& restirGI = ctx->getCommonObjects()->metaReSTIRGIRayQuery();
    if (restirGI.isActive()) {
      const uint32_t isLastCompositeOutputValid = restirGI.getLastCompositeOutput().matchesWriteFrameIdx(frameIdx - 1);
      assert(isLastCompositeOutputValid == rtOutput.m_raytraceArgs.isLastCompositeOutputValid && "Last composite state changed since CB was initialized");
      ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_LAST_COMPOSITE_INPUT, restirGI.getLastCompositeOutput().view(Resources::AccessType::Read, isLastCompositeOutputValid), nullptr);
    } else {
      ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_LAST_COMPOSITE_INPUT, nullptr, nullptr);
    }
    
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_FIRST_SAMPLED_LOBE_DATA_INPUT, rtOutput.m_indirectFirstSampledLobeData.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_GRADIENTS_INPUT, rtOutput.m_rtxdiGradients.view, nullptr);

    // Input / Output resources

    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PRIMARY_DIRECT_DIFFUSE_LOBE_RADIANCE_INPUT_OUTPUT, rtOutput.m_primaryDirectDiffuseRadiance.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_PRIMARY_DIRECT_SPECULAR_LOBE_RADIANCE_INPUT_OUTPUT, rtOutput.m_primaryDirectSpecularRadiance.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SECONDARY_COMBINED_DIFFUSE_LOBE_RADIANCE_INPUT_OUTPUT, rtOutput.m_secondaryCombinedDiffuseRadiance.view(Resources::AccessType::ReadWrite), nullptr);
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_SECONDARY_COMBINED_SPECULAR_LOBE_RADIANCE_INPUT_OUTPUT, rtOutput.m_secondaryCombinedSpecularRadiance.view(Resources::AccessType::ReadWrite), nullptr);

    // Output resources

    ctx->bindResourceBuffer(INTEGRATE_INDIRECT_BINDING_NEE_CACHE, DxvkBufferSlice(rtOutput.m_neeCache, 0, rtOutput.m_neeCache->info().size));
    ctx->bindResourceBuffer(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_SAMPLE, DxvkBufferSlice(rtOutput.m_neeCacheSample, 0, rtOutput.m_neeCacheSample->info().size));
    ctx->bindResourceBuffer(INTEGRATE_INDIRECT_BINDING_PRIMITIVE_ID_PREFIX_SUM, DxvkBufferSlice(primitiveIDPrefixSumBuffer, 0, primitiveIDPrefixSumBuffer->info().size));
    ctx->bindResourceBuffer(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_TASK, DxvkBufferSlice(rtOutput.m_neeCacheTask, 0, rtOutput.m_neeCacheTask->info().size));
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_NEE_CACHE_THREAD_TASK, rtOutput.m_neeCacheThreadTask.view, nullptr);

    // Aliased resources
    // m_indirectRadiance writes the actual output carried forward and therefore it must be bound with write access last
    ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_THROUGHPUT_CONE_RADIUS_INPUT, rtOutput.m_indirectThroughputConeRadius.view(Resources::AccessType::Read), nullptr);

    // Bind necessary resources for Neural Radiance Cache
    NeuralRadianceCache& nrc = ctx->getCommonObjects()->metaNeuralRadianceCache();
    RtxSharc& sharc = ctx->getCommonObjects()->metaSharc();
    if (sharc.isActive()) {
      sharc.bindResources(*ctx);
    } else {
      nrc.bindIntegrateIndirectPathTracingResources(*ctx);
    }

    // Bind necessary resources for ReSTIR GI
    DxvkReSTIRGIRayQuery& reSTIRGI = ctx->getCommonObjects()->metaReSTIRGIRayQuery();
    if (!sharc.isActive()) {
      reSTIRGI.bindIntegrateIndirectPathTracingResources(*ctx);
    }

    if (!sharcUpdate) {
      ctx->bindResourceView(INTEGRATE_INDIRECT_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_OUTPUT, rtOutput.m_indirectRadianceHitDistance.view(Resources::AccessType::Write), nullptr);
    }
    
    if (fusedAssembly) {
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_SHARED_MATERIAL_DATA0, rtOutput.m_sharedMaterialData0.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_SHARED_MATERIAL_DATA1, rtOutput.m_sharedMaterialData1.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA, rtOutput.m_sharedSubsurfaceDiffusionProfileData.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_WORLD_SHADING_NORMAL, rtOutput.m_primaryWorldShadingNormal.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS, rtOutput.m_primaryPerceptualRoughness.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_ALBEDO, rtOutput.m_primaryAlbedo.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_VIEW_DIRECTION, rtOutput.m_primaryViewDirection.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_POSITION_ERROR, rtOutput.m_primaryPositionError.view, nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_BASE_REFLECTIVITY, rtOutput.m_primaryBaseReflectivity.view(Resources::AccessType::ReadWrite), nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_INDIRECT_DIFFUSE, rtOutput.m_primaryIndirectDiffuseRadiance.view(Resources::AccessType::Write), nullptr);
      ctx->bindResourceView(INTEGRATE_FUSED_BINDING_PRIMARY_INDIRECT_SPECULAR, rtOutput.m_primaryIndirectSpecularRadiance.view(Resources::AccessType::Write), nullptr);
    }

    DebugView& debugView = ctx->getDevice()->getCommon()->metaDebugView();
    ctx->bindResourceView(INTEGRATE_INSTRUMENTATION, debugView.getInstrumentation(), nullptr);

    const bool nrcEnabled = nrc.isActive() && !sharc.isActive();

    const VkExtent3D& rayDims = nrcEnabled
      ? nrc.calcRaytracingResolution()
      : rtOutput.m_compositeOutputExtent;

    const bool serEnabled = RtxOptions::isShaderExecutionReorderingInPathtracerIntegrateIndirectEnabled();
    const bool ommEnabled = RtxOptions::getEnableOpacityMicromap();
    const bool includePortals = RtxOptions::rayPortalModelTextureHashes().size() > 0 || rtOutput.m_raytraceArgs.numActiveRayPortals > 0;
    const bool pomEnabled = rtOutput.m_raytraceArgs.pomMode != DisplacementMode::Off && RtxOptions::Displacement::enableIndirectHit();
    const bool wboitEnabled = RtxOptions::wboitEnabled();

    // Trace indirect ray
    {
      ScopedGpuProfileZone(ctx, "Integrate Indirect Raytracing");
      const NeeCachePass& neeCache = ctx->getCommonObjects()->metaNeeCache();
      const bool neeCacheEnabled = neeCache.isActive();
      // Both update backends map one logical invocation per tile to a rotating source pixel.
      VkExtent3D dispatchDims = rayDims;
      if (sharcUpdate) {
        const uint32_t tileSize = std::max(1u, rtOutput.m_raytraceArgs.sharcArgs.updateTileSize);
        dispatchDims.width = (rayDims.width + tileSize - 1) / tileSize;
        dispatchDims.height = (rayDims.height + tileSize - 1) / tileSize;
        dispatchDims.depth = 1;
      }
      const VkExtent3D workgroups = util::computeBlockCount(dispatchDims, VkExtent3D { 16, 8, 1 });
      if (sharcUpdate || sharc.isActive()) {
        if (sharc.isLeanActive()) {
          if (!sharcUpdate) {
            ctx->bindRaytracingPipelineShaders(getLeanSharcTracePipelineShaders(serEnabled, ommEnabled, sharc.queryStatsActive()));
            ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
          } else {
            auto shader = getLeanSharcUpdateShader(RtxSharc::updateRayGeneration(), RtxSharc::deferredUpdates(),
              rtOutput.m_raytraceArgs.sharcArgs.updateBounces <= 4);
            if (RtxSharc::updateRayGeneration()) {
              DxvkRaytracingPipelineShaders shaders;
              shaders.addGeneralShader(shader);
              shaders.debugName = "SHARC Lean Update";
              if (ommEnabled) {
                shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
              }
              ctx->bindRaytracingPipelineShaders(shaders);
              ctx->traceRays(dispatchDims.width, dispatchDims.height, dispatchDims.depth);
            } else {
              ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, shader);
              ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
            }
          }
          return;
        }
        if (!sharcUpdate) {
          if (RtxSharc::queryTraceRay()) {
            ctx->bindRaytracingPipelineShaders(getSharcTracePipelineShaders(serEnabled, ommEnabled, wboitEnabled, sharc.queryStatsActive(), includePortals, pomEnabled, fusedAssembly));
            ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
          } else if (RtxSharc::queryRayGeneration()) {
            DxvkRaytracingPipelineShaders shaders;
            if (sharc.queryStatsActive()) {
              shaders.addGeneralShader(wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_raygen_stats_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_raygen_stats));
            } else {
              shaders.addGeneralShader(wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_raygen_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_raygen));
            }
            shaders.debugName = "SHARC Query RayGen";
            if (ommEnabled) {
              shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
            }
            ctx->bindRaytracingPipelineShaders(shaders);
            ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
          } else {
            ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, sharc.queryStatsActive()
              ? (wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_stats_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcStatsShader, integrate_indirect_sharc_query_stats))
              : (wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcShader, integrate_indirect_sharc_query)));
            const auto queryGroups = util::computeBlockCount(dispatchDims, VkExtent3D { 8, 8, 1 });
            ctx->dispatch(queryGroups.width, queryGroups.height, queryGroups.depth);
          }
          return;
        }
        if (RtxSharc::updateRayGeneration()) {
          DxvkRaytracingPipelineShaders shaders;
          if (RtxSharc::deferredUpdates()) {
            if (rtOutput.m_raytraceArgs.sharcArgs.updateBounces <= 4) {
              shaders.addGeneralShader(wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred4_raygen_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred4_raygen));
            } else {
              shaders.addGeneralShader(wboitEnabled
                ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred_raygen_wboit)
                : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred_raygen));
            }
          } else {
            shaders.addGeneralShader(wboitEnabled
              ? GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_raygen_wboit)
              : GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_raygen));
          }
          shaders.debugName = "SHARC Update RayGen";
          if (ommEnabled) {
            shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;
          }
          ctx->bindRaytracingPipelineShaders(shaders);
          ctx->traceRays(dispatchDims.width, dispatchDims.height, dispatchDims.depth);
        } else {
          ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT,
            getComputeShader(neeCacheEnabled, nrcEnabled, wboitEnabled, true, false));
          ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
        }
        return;
      }
      switch (RtxOptions::renderPassIntegrateIndirectRaytraceMode()) {
      case RaytraceMode::RayQuery:
        ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, getComputeShader(neeCacheEnabled, nrcEnabled, wboitEnabled, sharcUpdate, sharc.isActive() && !sharcUpdate));
        ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
        break;
      case RaytraceMode::RayQueryRayGen:
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(true, serEnabled, ommEnabled, neeCacheEnabled, includePortals, pomEnabled, nrcEnabled, wboitEnabled));
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
        break;
      case RaytraceMode::TraceRay:
        ctx->bindRaytracingPipelineShaders(getPipelineShaders(false, serEnabled, ommEnabled, neeCacheEnabled, includePortals, pomEnabled, nrcEnabled, wboitEnabled));
        ctx->traceRays(rayDims.width, rayDims.height, rayDims.depth);
        break;
      case RaytraceMode::Count:
        assert(false && "Invalid RaytraceMode in DxvkPathtracerIntegrateIndirect::dispatch");
        break;
      }
    }
  }

  struct DxvkPathtracerIntegrateIndirect::AssemblyResources {
    // Keep aliased resources as references: view access must occur in binding order.
    const VkExtent3D& compositeOutputExtent;
    const Resources::Resource& sharedFlags;
    const Resources::Resource& sharedMaterialData0;
    const Resources::Resource& sharedMaterialData1;
    const Resources::Resource& sharedTextureCoord;
    const Resources::AliasedResource& sharedSurfaceIndex;
    const Resources::Resource& sharedSubsurfaceData;
    const Resources::Resource& sharedSubsurfaceDiffusionProfileData;
    const Resources::Resource& sparseRenderingIndirectActiveLocalPixelCoords;
    const Resources::Resource& primaryWorldShadingNormal;
    const Resources::Resource& primaryPerceptualRoughness;
    const Resources::Resource& primaryHitDistance;
    const Resources::Resource& primaryAlbedo;
    const Resources::Resource& primaryViewDirection;
    const Resources::Resource& primaryConeRadius;
    const Resources::Resource& primaryPositionError;
    const Resources::AliasedResource& indirectRadianceHitDistance;
    const Resources::AliasedResource& primaryBaseReflectivity;
    const Resources::AliasedResource& primaryIndirectDiffuseRadiance;
    const Resources::AliasedResource& primaryIndirectSpecularRadiance;
    const Rc<DxvkBuffer>& neeCache;
    const Rc<DxvkBuffer>& neeCacheTask;
    const Rc<DxvkBuffer>& neeCacheSample;
    const Resources::Resource& neeCacheThreadTask;
    const VkExtent3D& finalOutputExtent;
    const Resources::AliasedResource& primaryWorldPosition;

    explicit AssemblyResources(const Resources::RaytracingOutput& rtOutput)
      : compositeOutputExtent(rtOutput.m_compositeOutputExtent)
      , sharedFlags(rtOutput.m_sharedFlags)
      , sharedMaterialData0(rtOutput.m_sharedMaterialData0)
      , sharedMaterialData1(rtOutput.m_sharedMaterialData1)
      , sharedTextureCoord(rtOutput.m_sharedTextureCoord)
      , sharedSurfaceIndex(rtOutput.m_sharedSurfaceIndex)
      , sharedSubsurfaceData(rtOutput.m_sharedSubsurfaceData)
      , sharedSubsurfaceDiffusionProfileData(rtOutput.m_sharedSubsurfaceDiffusionProfileData)
      , sparseRenderingIndirectActiveLocalPixelCoords(rtOutput.m_sparseRenderingIndirectActiveLocalPixelCoords)
      , primaryWorldShadingNormal(rtOutput.m_primaryWorldShadingNormal)
      , primaryPerceptualRoughness(rtOutput.m_primaryPerceptualRoughness)
      , primaryHitDistance(rtOutput.m_primaryHitDistance)
      , primaryAlbedo(rtOutput.m_primaryAlbedo)
      , primaryViewDirection(rtOutput.m_primaryViewDirection)
      , primaryConeRadius(rtOutput.m_primaryConeRadius)
      , primaryPositionError(rtOutput.m_primaryPositionError)
      , indirectRadianceHitDistance(rtOutput.m_indirectRadianceHitDistance)
      , primaryBaseReflectivity(rtOutput.m_primaryBaseReflectivity)
      , primaryIndirectDiffuseRadiance(rtOutput.m_primaryIndirectDiffuseRadiance)
      , primaryIndirectSpecularRadiance(rtOutput.m_primaryIndirectSpecularRadiance)
      , neeCache(rtOutput.m_neeCache)
      , neeCacheTask(rtOutput.m_neeCacheTask)
      , neeCacheSample(rtOutput.m_neeCacheSample)
      , neeCacheThreadTask(rtOutput.m_neeCacheThreadTask)
      , finalOutputExtent(rtOutput.m_finalOutputExtent)
      , primaryWorldPosition(rtOutput.getCurrentPrimaryWorldPositionWorldTriangleNormal()) { }
  };

  void DxvkPathtracerIntegrateIndirect::dispatchLighting(RtxContext* ctx, const Resources::RaytracingOutput& rtOutput) {
    const auto& activeSharc = ctx->getCommonObjects()->metaSharc();
    const bool fusedAssembly = RtxSharc::fuseAssembly() && activeSharc.isActive()
      && RtxSharc::queryTraceRay() && !activeSharc.isLeanActive()
      && !RtxSharc::collectQueryStats() && !SparseRendering::Options::enableSparseRendering()
      && ctx->getCommonObjects()->metaDebugView().debugViewIdx() == 0
      && !ctx->getCommonObjects()->metaNeuralRadianceCache().isActive()
      && !ctx->getCommonObjects()->metaReSTIRGIRayQuery().isActive()
      && !rtOutput.m_indirectThroughputConeRadius.sharesTheSameView(rtOutput.m_primaryIndirectDiffuseRadiance);
    if (m_fusedAssemblyActive != fusedAssembly) {
      Logger::info(fusedAssembly ? "[SHARC] Combined query/assembly active" : "[SHARC] Standalone query/assembly active");
      m_fusedAssemblyActive = fusedAssembly;
    }
    {
      ScopedGpuProfileZone(ctx, "Integrate Indirect Raytracing");
      ctx->setFramePassStage(RtxFramePassStage::IndirectIntegration);

      RtxSharc& sharc = ctx->getCommonObjects()->metaSharc();
      if (sharc.isActive()) {
        // SHARC's sparse update writes the cache; resolve must publish it
        // before the full-resolution query pass reads the resolved entries.
        sharc.recordTimestamp(*ctx, RtxSharc::TimingPoint::Begin);
        {
          ScopedGpuProfileZone(ctx, "SHARC Update");
          dispatch(ctx, rtOutput, true);
        }
        sharc.recordTimestamp(*ctx, RtxSharc::TimingPoint::UpdateEnd);
        sharc.dispatchResolve(*ctx, rtOutput);
        sharc.beginQueryStats(*ctx);
        sharc.recordTimestamp(*ctx, RtxSharc::TimingPoint::ResolveEnd);
        {
          ScopedGpuProfileZone(ctx, "SHARC Query");
          dispatch(ctx, rtOutput, false, fusedAssembly);
        }
        sharc.recordTimestamp(*ctx, RtxSharc::TimingPoint::QueryEnd);
        sharc.endQueryStats(*ctx);
      } else {
        dispatch(ctx, rtOutput);
      }
    }
    ctx->recordGpuStageTiming("IndirectIntegration");

    if (!fusedAssembly) {
      ctx->setFramePassStage(RtxFramePassStage::NEE_Integration);
      ctx->bindCommonRayTracingResources(rtOutput);
      dispatchNEE(ctx, AssemblyResources(rtOutput));
    }
    ctx->recordGpuStageTiming("IndirectAssembly");
  }

  void DxvkPathtracerIntegrateIndirect::dispatchNEE(RtxContext* ctx, const AssemblyResources& resources) {
    // Sample triangles in the NEE cache and perform NEE
    // Construct restir input sample
    const auto rayDims = resources.compositeOutputExtent;
    VkExtent3D workgroups = util::computeBlockCount(rayDims, VkExtent3D { INTEGRATE_NEE_THREADS_DISPATCH_WIDTH, INTEGRATE_NEE_THREADS_DISPATCH_HEIGHT, 1 });
    Rc<DxvkBuffer> primitiveIDPrefixSumBuffer = ctx->getSceneManager().getCurrentFramePrimitiveIDPrefixSumBuffer();
    NeuralRadianceCache& nrc = ctx->getCommonObjects()->metaNeuralRadianceCache();
    DxvkReSTIRGIRayQuery& reSTIRGI = ctx->getCommonObjects()->metaReSTIRGIRayQuery();
    const bool nrcEnabled = nrc.isActive();
    const bool restirGiEnabled = reSTIRGI.isActive();
    const uint32_t debugViewIndex = ctx->getCommonObjects()->metaDebugView().debugViewIdx();
    const bool visualizeNee = debugViewIndex == DEBUG_VIEW_NEE_CACHE_LIGHT_HISTOGRAM || debugViewIndex == DEBUG_VIEW_NEE_CACHE_HISTOGRAM ||
      debugViewIndex == DEBUG_VIEW_NEE_CACHE_ACCUMULATE_MAP || debugViewIndex == DEBUG_VIEW_NEE_CACHE_HASH_MAP || debugViewIndex == DEBUG_VIEW_NEE_CACHE_TRIANGLE_CANDIDATE;

    ScopedGpuProfileZone(ctx, "Integrate NEE");
    ctx->setFramePassStage(RtxFramePassStage::NEE_Integration);

    // Inputs

    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_FLAGS_INPUT, resources.sharedFlags.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA0_INPUT, resources.sharedMaterialData0.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_MATERIAL_DATA1_INPUT, resources.sharedMaterialData1.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_TEXTURE_COORD_INPUT, resources.sharedTextureCoord.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_SURFACE_INDEX_INPUT, resources.sharedSurfaceIndex.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DATA_INPUT, resources.sharedSubsurfaceData.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_SHARED_SUBSURFACE_DIFFUSION_PROFILE_DATA_INPUT, resources.sharedSubsurfaceDiffusionProfileData.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_ACTIVE_LOCAL_PIXEL_COORDS_INPUT, resources.sparseRenderingIndirectActiveLocalPixelCoords.view, nullptr);

    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_SHADING_NORMAL_INPUT, resources.primaryWorldShadingNormal.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_PERCEPTUAL_ROUGHNESS_INPUT, resources.primaryPerceptualRoughness.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_HIT_DISTANCE_INPUT, resources.primaryHitDistance.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_ALBEDO_INPUT, resources.primaryAlbedo.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_VIEW_DIRECTION_INPUT, resources.primaryViewDirection.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_CONE_RADIUS_INPUT, resources.primaryConeRadius.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_WORLD_POSITION_INPUT, resources.primaryWorldPosition.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_POSITION_ERROR_INPUT, resources.primaryPositionError.view, nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_INDIRECT_RADIANCE_HIT_DISTANCE_INPUT, resources.indirectRadianceHitDistance.view(Resources::AccessType::Read), nullptr);
    ctx->bindResourceBuffer(INTEGRATE_NEE_BINDING_PRIMITIVE_ID_PREFIX_SUM_INPUT, DxvkBufferSlice(primitiveIDPrefixSumBuffer, 0, primitiveIDPrefixSumBuffer->info().size));

    // Inputs / Outputs

    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_BASE_REFLECTIVITY_INPUT_OUTPUT, resources.primaryBaseReflectivity.view(Resources::AccessType::ReadWrite), nullptr);
    if (nrcEnabled || restirGiEnabled || visualizeNee) {
      ctx->bindResourceBuffer(INTEGRATE_NEE_BINDING_NRC_TRAINING_PATH_VERTICES_INPUT_OUTPUT, nrc.getBufferSlice(*ctx, NeuralRadianceCache::ResourceType::TrainingPathVertices));
    }

    // Outputs

    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_DIFFUSE_RADIANCE_HIT_DISTANCE_OUTPUT, resources.primaryIndirectDiffuseRadiance.view(Resources::AccessType::Write), nullptr);
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_PRIMARY_INDIRECT_SPECULAR_RADIANCE_HIT_DISTANCE_OUTPUT, resources.primaryIndirectSpecularRadiance.view(Resources::AccessType::Write), nullptr);

    if (nrcEnabled || restirGiEnabled || visualizeNee) {
      ctx->bindResourceView(INTEGRATE_NEE_BINDING_BSDF_FACTOR2_OUTPUT, reSTIRGI.getBsdfFactor2().view, nullptr);
      reSTIRGI.bindIntegrateIndirectNeeResources(*ctx);
    }

    ctx->bindResourceBuffer(INTEGRATE_NEE_BINDING_NEE_CACHE, DxvkBufferSlice(resources.neeCache, 0, resources.neeCache->info().size));
    ctx->bindResourceBuffer(INTEGRATE_NEE_BINDING_NEE_CACHE_TASK, DxvkBufferSlice(resources.neeCacheTask, 0, resources.neeCacheTask->info().size));
    ctx->bindResourceBuffer(INTEGRATE_NEE_BINDING_NEE_CACHE_SAMPLE, DxvkBufferSlice(resources.neeCacheSample, 0, resources.neeCacheSample->info().size));
    ctx->bindResourceView(INTEGRATE_NEE_BINDING_NEE_CACHE_THREAD_TASK, resources.neeCacheThreadTask.view, nullptr);

    Rc<DxvkShader> integrateNeeShader;
    if (nrcEnabled && !restirGiEnabled) {
      integrateNeeShader = GET_SHADER_VARIANT(
        VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEShader, integrate_nee_nrc);
    } else if (restirGiEnabled && !nrcEnabled) {
      integrateNeeShader = GET_SHADER_VARIANT(
        VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEShader, integrate_nee_restir_gi);
    } else if (!nrcEnabled && !restirGiEnabled) {
      integrateNeeShader = GET_SHADER_VARIANT(
        VK_SHADER_STAGE_COMPUTE_BIT, IntegrateNEEPlainShader, integrate_nee_plain);
    } else {
      integrateNeeShader = IntegrateNEEShader::getShader();
    }
    ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, integrateNeeShader);
    ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);

    // Visualize the nee cache when debug view is chosen.
    if (visualizeNee) {
      VisualizeNeeArgs args;
      auto mousePos = ImGui::GetMousePos();
      const VkExtent3D& finalResolution = resources.finalOutputExtent;
      args.mouseUV = vec2(mousePos.x / finalResolution.width, mousePos.y / finalResolution.height);
      args.mouseUV.x = std::clamp(args.mouseUV.x, 0.0f, 1.0f);
      args.mouseUV.y = std::clamp(args.mouseUV.y, 0.0f, 1.0f);
      ctx->pushConstants(0, sizeof(args), &args);
      ctx->bindShader(VK_SHADER_STAGE_COMPUTE_BIT, VisualizeNEEShader::getShader());
      ctx->dispatch(workgroups.width, workgroups.height, workgroups.depth);
    }
  }

  DxvkRaytracingPipelineShaders DxvkPathtracerIntegrateIndirect::getPipelineShaders(
    const bool useRayQuery,
    const bool serEnabled,
    const bool ommEnabled,
    const bool useNeeCache,
    const bool includePortals,
    const bool pomEnabled,
    const bool nrcEnabled,
    const bool wboitEnabled) {

    DxvkRaytracingPipelineShaders shaders;
    if (wboitEnabled) {
      if (useRayQuery) {
        if (nrcEnabled) {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_nrc_neeCache_wboit));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_nrc_wboit));
          }
        } else {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_neeCache_wboit));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_wboit));
          }
        }
        shaders.debugName = "Integrate Indirect RayQuery (RGS)";
      } else {
        if (nrcEnabled) {
          if (serEnabled) {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_nrc_neeCache_wboit));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_nrc_wboit));
            }
          } else {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_nrc_neeCache_wboit));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_nrc_wboit));
            }
          }
        } else {
          if (serEnabled) {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_neeCache_wboit));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_wboit));
            }
          } else {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_neeCache_wboit));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_wboit));
            }
          }
        }

        if (nrcEnabled) {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_nrc_neeCache_wboit));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_nrc_wboit));
          }
        } else {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_neeCache_wboit));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_wboit));
          }
        }

        if (nrcEnabled) {
          if (useNeeCache) {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_pom_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_pom_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            }
          } else {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_pom_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_pom_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            }
          }
        } else {
          if (useNeeCache) {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_pom_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_pom_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            }
          } else {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_pom_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_pom_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_material_rayportal_closestHit_wboit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_material_opaque_translucent_closestHit_wboit), nullptr, nullptr);
              }
            }
          }
        }
      }
      shaders.debugName = "Integrate Indirect TraceRay (RGS)";
    } else {
      if (useRayQuery) {
        if (nrcEnabled) {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_nrc_neeCache));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_nrc));
          }
        } else {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen_neeCache));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_raygen));
          }
        }
        shaders.debugName = "Integrate Indirect RayQuery (RGS)";
      } else {
        if (nrcEnabled) {
          if (serEnabled) {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_nrc_neeCache));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_nrc));
            }
          } else {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_nrc_neeCache));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_nrc));
            }
          }
        } else {
          if (serEnabled) {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser_neeCache));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_ser));
            }
          } else {
            if (useNeeCache) {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen_neeCache));
            } else {
              shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_RAYGEN_BIT_KHR, IntegrateIndirectRayGenShader, integrate_indirect_raygen));
            }
          }
        }

        if (nrcEnabled) {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_nrc_neeCache));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_nrc));
          }
        } else {
          if (useNeeCache) {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss_neeCache));
          } else {
            shaders.addGeneralShader(GET_SHADER_VARIANT(VK_SHADER_STAGE_MISS_BIT_KHR, IntegrateIndirectMissShader, integrate_indirect_miss));
          }
        }

        if (nrcEnabled) {
          if (useNeeCache) {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_pom_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_pom_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_neeCache_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            }
          } else {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_pom_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_pom_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_nrc_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            }
          }
        } else {
          if (useNeeCache) {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_pom_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_pom_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_neeCache_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            }
          } else {
            if (pomEnabled) {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_pom_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_pom_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            } else {
              if (includePortals) {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_material_rayportal_closestHit), nullptr, nullptr);
              } else {
                shaders.addHitGroup(GET_SHADER_VARIANT(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, IntegrateIndirectClosestHitShader, integrate_indirect_material_opaque_translucent_closestHit), nullptr, nullptr);
              }
            }
          }
        }

        shaders.debugName = "Integrate Indirect TraceRay (RGS)";
      }
    }

    if (ommEnabled)
      shaders.pipelineFlags |= VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;

    return shaders;
  }

  Rc<DxvkShader> DxvkPathtracerIntegrateIndirect::getComputeShader(const bool useNeeCache, const bool nrcEnabled, const bool wboitEnabled, const bool sharcUpdate, const bool sharcQuery) const {
    if (sharcUpdate) {
      if (RtxSharc::deferredUpdates()) {
        if (RtxSharc::updateBounces() <= 4) {
          return (wboitEnabled
            ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred4_wboit)
            : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred4));
        }
        return (wboitEnabled
          ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred_wboit)
          : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_deferred));
      }
      return (wboitEnabled
        ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update_wboit)
        : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcUpdateShader, integrate_indirect_sharc_update));
    }
    if (sharcQuery) {
      return (wboitEnabled
        ? GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcShader, integrate_indirect_sharc_query_wboit)
        : GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectSharcShader, integrate_indirect_sharc_query));
    }
    if (wboitEnabled) {
      if (nrcEnabled) {
        if (useNeeCache) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_nrc_neeCache_wboit);
        } else {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_nrc_wboit);
        }
      } else {
        if (useNeeCache) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_neeCache_wboit);
        } else {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_wboit);
        }
      }
    } else {
      if (nrcEnabled) {
        if (useNeeCache) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_nrc_neeCache);
        } else {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_nrc);
        }
      } else {
        if (useNeeCache) {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery_neeCache);
        } else {
          return GET_SHADER_VARIANT(VK_SHADER_STAGE_COMPUTE_BIT, IntegrateIndirectRayGenShader, integrate_indirect_rayquery);
        }
      }
    }
  }

  const char* DxvkPathtracerIntegrateIndirect::raytraceModeToString(RaytraceMode raytraceMode) {
    switch (raytraceMode) {
    case RaytraceMode::RayQuery:
      return "Ray Query [CS]";
    case RaytraceMode::RayQueryRayGen:
      return "Ray Query [RGS]";
    case RaytraceMode::TraceRay:
      return "Trace Ray [RGS]";
    default:
      return "Unknown";
    }
  }
}
