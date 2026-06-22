/*
* Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef SHARED_CONSTANTS_H
#define SHARED_CONSTANTS_H

// contains constants shared between shader and host code

static const uint8_t surfaceMaterialTypeOpaque = uint8_t(0u);
static const uint8_t surfaceMaterialTypeTranslucent = uint8_t(1u);
static const uint8_t surfaceMaterialTypeRayPortal = uint8_t(2u);
// Fork: tag for multi-layer-terrain extension entries in the surface-material
// extension cache. Like surfaceMaterialTypeSubsurface (implicit -- subsurface
// extension entries use flags = 0), this tag is informational rather than used
// for polymorphic dispatch -- the shader knows an entry is multi-layer-terrain
// because the parent opaque material's m_multiLayerTerrainIndex points at it.
// Fits in the existing 2-bit surfaceMaterialTypeMask.
static const uint8_t surfaceMaterialTypeMultiLayerTerrain = uint8_t(3u);
static const uint8_t surfaceMaterialTypeMask = uint8_t(0x3u);

#define COMMON_MATERIAL_FLAG_TYPE_MASK surfaceMaterialTypeMask
#define COMMON_MATERIAL_FLAG_TYPE_OFFSET(X) (2 + X)

// NOTE: Each material memory structure contains a set of flags.  The first 2 bits in that flag identify the material type (opaque, etc).
//       We must ensure all other material flags are written to byte addresses after these first two bits.  Use the COMMON_MATERIAL_FLAG_TYPE_OFFSET(x) 
//       macro, and ensure there is enough storage in the flags to represent desired bits accordingly.

// maximum value for thin film thickness in nanometers
#define OPAQUE_SURFACE_MATERIAL_THIN_FILM_MAX_THICKNESS (1500.0f)
// bits for flags field in OpaqueSurfaceMaterial
#define OPAQUE_SURFACE_MATERIAL_FLAG_USE_THIN_FILM_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALPHA_IS_THIN_FILM_THICKNESS (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(1))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IGNORE_ALPHA_CHANNEL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(2))
#define OPAQUE_SURFACE_MATERIAL_FLAG_IS_RAYTRACED_RENDER_TARGET (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(3))
#define OPAQUE_SURFACE_MATERIAL_FLAG_HAS_DISPLACEMENT (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(4))
// Set when the albedo/emissive source texture uses an sRGB VkFormat, so the sampler hardware already
// linearized it on read. The shader skips its own gammaToLinear() for that channel to avoid double
// linearization (constants remain gamma-encoded and are always converted). See opaque_surface_material_interaction.slangh.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ALBEDO_TEXTURE_IS_SRGB (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(5))
#define OPAQUE_SURFACE_MATERIAL_FLAG_EMISSIVE_TEXTURE_IS_SRGB (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(6))
// Fork (2026-07-26): marks a particle material whose sprites are known to see open sky (weather
// precipitation - the spawn-time shelter probe guarantees it). The resolver's opacity lighting
// approximation adds a sky-ambient term for such particles on top of the froxel radiance sample,
// supplying the skylight that the froxel grid does not contain (its integrator has no sky term).
#define OPAQUE_SURFACE_MATERIAL_FLAG_SKY_LIT_PARTICLE (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(7))
// Fork: set on legacy materials whose normal texture came through the FNV PS-classifier
// protocol path; tells the sample-time decode in opaque_surface_material_interaction.slangh
// to read the texture as RGB tangent-space ([0,1] unsigned snorm in .rgb) instead of
// Remix's native octahedral encoding.
#define OPAQUE_SURFACE_MATERIAL_FLAG_TANGENT_SPACE_NORMAL (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(8))
// Fork: set on opaque materials produced from FNV multi-layer terrain draws (see
// LegacyMaterialData::terrainAlbedoTextures[] / terrainNormalTextures[] and
// kRemixMultiLayerTerrainBit). Tells the shader to read the per-layer texture
// indices via the m_multiLayerTerrainIndex aux slot pointing into the surface-
// material extension cache (RtMultiLayerTerrainMaterial entries).
#define OPAQUE_SURFACE_MATERIAL_FLAG_MULTI_LAYER_TERRAIN (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(9))
// Fork: set on legacy materials following the Bethesda/Gamebryo DXT5n convention --
// specular intensity packed into the alpha channel of the NormalMap. Tells the
// roughness load in opaque_surface_material_interaction.slangh to override the
// roughness texture path with `roughness = 1.0 - normalSample.a`. Implies (and only
// makes sense alongside) OPAQUE_SURFACE_MATERIAL_FLAG_TANGENT_SPACE_NORMAL since both
// fire on the same FNV PS-classifier capture path.
#define OPAQUE_SURFACE_MATERIAL_FLAG_ROUGHNESS_FROM_NORMAL_ALPHA (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(10))
// Offsets 8-10 (bits 10-12 of the uint16_t flags field) were moved off 5-7 when this
// commit was brought onto numos3 -- 5-7 are the sRGB-linearize and sky-lit-particle bits.


#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_HAS_HEIGHT_TEXTURE (1 << 0)
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_USE_THIN_FILM_LAYER (1 << 1)
// flags overlap with type field when in gbuffer, which occupies last 2 bits.
#define OPAQUE_SURFACE_MATERIAL_INTERACTION_FLAG_MASK 0x3F


// Note: Bits for flags field in TranslucentSurfaceMaterial and TranslucentSurfaceMaterialInteraction
// If set, then the texture bound to transmittanceOrDiffuseTextureIndex is an albedo map for the diffuse layer
#define TRANSLUCENT_SURFACE_MATERIAL_FLAG_USE_DIFFUSE_LAYER (1 << COMMON_MATERIAL_FLAG_TYPE_OFFSET(0))

#endif // ifndef SHARED_CONSTANTS_H
