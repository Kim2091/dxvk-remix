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
/*
* The C++ mirror below is a port of the ReSTIR PT reference implementation:
*   Copyright (c) 2022, Daqi Lin.  All rights reserved.
*   ReSTIR_PT/Source/RenderPasses/ReSTIRPTPass/PathReservoir.slang
* Licensed under BSD-3-Clause (see ThirdPartyLicenses.txt, "ReSTIR PT" entry).
*/

// ReSTIR PT (fork) -- CPU unit test for the reservoir's RIS arithmetic.
//
// ==========================  KEEP IN SYNC  ==================================
// The structs and methods in the `mirror` namespace below are a HAND MIRROR of
//   src/dxvk/shaders/rtx/algorithm/fork_restir_pt/restir_pt_reservoir.slangh
// They are deliberately NOT a shared header. The shader version has to stay
// readable as a line-by-line port of the reference, and the test version has to
// stay readable as an independent statement of what the arithmetic should do; a
// shared header would make a transcription error invisible in both at once.
//
// If restir_pt_reservoir.slangh changes, change this file too. The two
// differences that are intentional and permanent:
//   - the accept draw takes a float instead of an inout RAB_RandomSamplerState,
//     so the tests can drive an exact sequence;
//   - vec3 is a plain struct rather than Slang's built-in.
// ============================================================================
//
// What is actually being locked here (the phase 2 estimator identity):
//
//   With M pinned to 1 by PathBuilder::finalize and one streaming candidate per
//   path terminal, F * weight after finalizeRIS is an unbiased single-sample
//   estimator of  SUM over terminals i of ( F_i / p_i ).
//
//   Derivation, which test (c) checks numerically:
//     w_i     = phat(F_i) / p_i                 (add)
//     weight  = SUM_i w_i / phat(F_selected)    (finalizeRIS, M == 1)
//     P(sel j)= w_j / SUM_i w_i                 (streaming RIS)
//     E[F_sel * weight] = SUM_j (w_j / SUM w) * F_j * SUM w / phat(F_j)
//                       = SUM_j F_j / p_j                             QED
//
//   That identity is the whole reason the final shading pass may write F*weight
//   and nothing else. If it breaks, PT mode's brightness stops matching GI's.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <limits>

#include "../../test_utils.h"

namespace dxvk {
  // Note: Logger needed by some shared code used in this Unit Test.
  Logger Logger::s_instance("test_fork_restir_pt_reservoir.log");
}

namespace mirror {

  // ports ReSTIRPTPass/StaticParams.slang kMaximumPathLength.
  static constexpr int kRestirPtMaximumPathLength = 15;

  struct vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    vec3() = default;
    vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    explicit vec3(float v) : x(v), y(v), z(v) { }

    bool operator==(const vec3& o) const { return x == o.x && y == o.y && z == o.z; }
  };

  // ports restir_pt_reservoir.slangh RestirPtPathFlags
  // (ReSTIRPTPass/PathReservoir.slang:21-140).
  struct RestirPtPathFlags {
    int flags = 0;

    void insertIsDeltaEvent(bool isDeltaEvent, bool beforeRcVertex) {
      flags &= (beforeRcVertex ? ~(0x100) : ~(0x200));
      if (isDeltaEvent) { flags |= 1 << (beforeRcVertex ? 8 : 9); }
    }

    void insertIsTransmissionEvent(bool isTransmissionEvent, bool beforeRcVertex) {
      flags &= (beforeRcVertex ? ~(0x400) : ~(0x800));
      if (isTransmissionEvent) { flags |= 1 << (beforeRcVertex ? 10 : 11); }
    }

    void insertIsSpecularBounce(bool isSpecularBounce, bool beforeRcVertex) {
      flags &= (beforeRcVertex ? ~(0x4000000) : ~(0x8000000));
      if (isSpecularBounce) { flags |= 1 << (beforeRcVertex ? 26 : 27); }
    }

    bool decodeIsDeltaEvent(bool beforeRcVertex) const {
      return (beforeRcVertex ? ((flags >> 8) & 1) : ((flags >> 9) & 1)) != 0;
    }

    bool decodeIsTransmissionEvent(bool beforeRcVertex) const {
      return (beforeRcVertex ? ((flags >> 10) & 1) : ((flags >> 11) & 1)) != 0;
    }

    bool decodeIsSpecularBounce(bool beforeRcVertex) const {
      return (beforeRcVertex ? ((flags >> 26) & 1) : ((flags >> 27) & 1)) != 0;
    }

    // Fork-owned bits 20/21. NOT a synonym for the specularBounce pair above --
    // see the long note on RestirPtPathFlags::insertIsSpecularLobeClass.
    void insertIsSpecularLobeClass(bool isSpecularLobeClass, bool beforeRcVertex) {
      flags &= (beforeRcVertex ? ~(0x100000) : ~(0x200000));
      if (isSpecularLobeClass) { flags |= 1 << (beforeRcVertex ? 20 : 21); }
    }

    bool decodeIsSpecularLobeClass(bool beforeRcVertex) const {
      return (beforeRcVertex ? ((flags >> 20) & 1) : ((flags >> 21) & 1)) != 0;
    }

    void insertPathLength(int pathLength) {
      flags &= ~0xF;
      flags |= pathLength & 0xF;
    }

    void insertRcVertexLength(int rcVertexLength) {
      flags &= ~0xF0;
      flags |= (rcVertexLength & 0xF) << 4;
    }

    int pathLength() const { return flags & 0xF; }
    int rcVertexLength() const { return (flags >> 4) & 0xF; }

    void insertLastVertexNEE(bool isNEE) {
      flags &= ~0x10000;
      flags |= (int(isNEE) & 1) << 16;
    }

    bool lastVertexNEE() const { return ((flags >> 16) & 1) != 0; }

    void insertLightType(uint32_t lightType) {
      flags &= ~0xc0000;
      flags |= ((int(lightType) & 3) << 18);
    }

    uint32_t lightType() const { return uint32_t((flags >> 18) & 3); }
  };

  // ports restir_pt_reservoir.slangh RestirPtReservoir
  // (ReSTIRPTPass/PathReservoir.slang:224-451), reduced to the fields the
  // arithmetic touches.
  struct RestirPtReservoir {
    float M = 0.0f;
    float weight = 0.0f;
    RestirPtPathFlags pathFlags;
    uint32_t srcPixel = 0u;
    vec3 F;
    float lightPdf = 0.0f;
    vec3 cachedJacobian;
    uint32_t srcFrameIdx = 0u;
    vec3 rcVertexWi;
    vec3 rcVertexIrradiance;

    void init() {
      M = 0.0f;
      weight = 0.0f;
      pathFlags.flags = 0;
      pathFlags.insertRcVertexLength(kRestirPtMaximumPathLength);
      F = vec3();
      srcPixel = 0u;
      srcFrameIdx = 0u;
      lightPdf = 0.0f;
      cachedJacobian = vec3(1.0f);
      rcVertexWi = vec3();
      rcVertexIrradiance = vec3();
    }

    static float toScalar(const vec3& color) {
      return color.x * 0.299f + color.y * 0.587f + color.z * 0.114f;
    }

    static float computeWeight(const vec3& color, bool binarize) {
      float w = toScalar(color);
      if (binarize && w > 0.0f) { w = 1.0f; }
      return w;
    }

    // Difference from the shader: the accept draw arrives as a plain float.
    bool add(const vec3& in_F, float p, float acceptRnd) {
      M += 1.0f;

      const float w = toScalar(in_F) / p;

      if (std::isnan(w) || w == 0.0f) { return false; }

      weight += w;

      if (acceptRnd * weight <= w) {
        F = in_F;
        return true;
      }

      return false;
    }

    void copySampleFrom(const RestirPtReservoir& in) {
      pathFlags = in.pathFlags;
      srcPixel = in.srcPixel;
      srcFrameIdx = in.srcFrameIdx;
      cachedJacobian = in.cachedJacobian;
      lightPdf = in.lightPdf;
      rcVertexWi = in.rcVertexWi;
      rcVertexIrradiance = in.rcVertexIrradiance;
    }

    bool merge(const vec3& in_F, float in_Jacobian, const RestirPtReservoir& in,
               float acceptRnd, float misWeight, bool forceAdd) {
      const float w = toScalar(in_F) * in_Jacobian * in.M * in.weight * misWeight;

      M += in.M;

      if (std::isnan(w) || w == 0.0f) { return false; }

      weight += w;

      if (forceAdd || acceptRnd * weight <= w) {
        copySampleFrom(in);
        F = in_F;
        return true;
      }

      return false;
    }

    bool mergeWithResamplingMIS(const vec3& in_F, float in_Jacobian, const RestirPtReservoir& in,
                                float acceptRnd, float misWeight, bool forceAdd) {
      const float w = toScalar(in_F) * in_Jacobian * in.weight * misWeight;

      M += in.M;

      if (std::isnan(w) || w == 0.0f) { return false; }

      weight += w;

      if (forceAdd || acceptRnd * weight <= w) {
        copySampleFrom(in);
        F = in_F;
        return true;
      }

      return false;
    }

    bool mergeInSamplePixel(const RestirPtReservoir& in, float acceptRnd) {
      const float w = in.weight;

      M += in.M;

      if (std::isnan(w) || w == 0.0f) { return false; }

      weight += w;

      if (acceptRnd * weight <= w) {
        copySampleFrom(in);
        F = in.F;
        return true;
      }

      return false;
    }

    void prepareMerging() {
      weight *= toScalar(F) * M;
    }

    void finalizeRIS() {
      const float pHat = toScalar(F);
      if (pHat == 0.0f || M == 0.0f) { weight = 0.0f; }
      else { weight = weight / (pHat * M); }
    }

    void finalizeGRIS() {
      const float pHat = toScalar(F);
      if (pHat == 0.0f) { weight = 0.0f; }
      else { weight = weight / pHat; }
    }
  };

  // ports restir_pt_path_builder.slangh RestirPtPathBuilder::finalize.
  inline void pathBuilderFinalize(RestirPtReservoir& r) {
    r.M = 1.0f;
  }

  // =========================================================================
  // Phase 3 mirror -- the shift's Jacobian core and the pairwise MIS arithmetic.
  //
  // KEEP IN SYNC with
  //   src/dxvk/shaders/rtx/algorithm/fork_restir_pt/restir_pt_shift.slangh
  //   src/dxvk/shaders/rtx/pass/fork_restir_pt/fork_restir_pt_spatial_reuse.comp.slang
  // by the same rule as the reservoir mirror above: two independent statements of
  // the same arithmetic, never a shared header.
  // =========================================================================

  inline vec3 sub(const vec3& a, const vec3& b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
  inline float dot3(const vec3& a, const vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

  inline vec3 normalizeOrZ(const vec3& v) {
    const float len = std::sqrt(dot3(v, v));
    if (!(len > 0.0f)) { return vec3(0.0f, 0.0f, 1.0f); }
    return vec3(v.x / len, v.y / len, v.z / len);
  }

  // ports restir_pt_shift.slangh restirPtIsJacobianInvalid
  // (ReSTIRPTPass/Shift.slang:224-227).
  inline bool isJacobianInvalid(float jacobian) {
    return jacobian <= 0.0f || std::isnan(jacobian) || std::isinf(jacobian);
  }

  // =========================================================================
  // The BSDF LOBE-CLASS CONVENTION, which the self-shift identity rests on.
  //
  // KEEP IN SYNC with restir_pt_shift.slangh (restirPtGetAllowedLobeClass,
  // restirPtEvalBsdfCosine, restirPtEvalPdfBsdf) and with
  // restir_pt_trace_core.slangh (restirPtSampleScatterDirection, and the
  // isSpecularLobeClass derivation in restirPtClassifyScatterEvent).
  //
  // dxvk-remix opaque lobes, in the order used below:
  //   0 diffuseReflection   1 specularReflection   2 diffuseTransmission
  // (opacityTransmission is a dirac; the shift rejects deltas outright and the
  // codebase's own all-lobe pdf marginal leaves it out too.)
  // =========================================================================

  static const unsigned kLobeDiffuseReflection   = 1u << 0;
  static const unsigned kLobeSpecularReflection  = 1u << 1;
  static const unsigned kLobeDiffuseTransmission = 1u << 2;
  static const unsigned kLobeClassAll =
    kLobeDiffuseReflection | kLobeSpecularReflection | kLobeDiffuseTransmission;

  static constexpr int kLobeIndexDiffuseReflection = 0;
  static constexpr int kLobeIndexSpecularReflection = 1;
  static constexpr int kLobeIndexDiffuseTransmission = 2;

  // ports restir_pt_shift.slangh restirPtGetAllowedLobeClass
  // (ReSTIRPTPass/PathTracer.slang:37-41).
  inline unsigned allowedLobeClass(bool isSpecularLobeClass) {
    return isSpecularLobeClass
      ? kLobeSpecularReflection
      : (kLobeDiffuseReflection | kLobeDiffuseTransmission);
  }

  // One opaque material evaluated at one (view, scatter) direction pair. All
  // three arrays are indexed by the lobe indices above.
  struct LobeState {
    float selectionProbability[3] = { 0.0f, 0.0f, 0.0f };  // normalized P_lobe
    float solidAnglePdf[3] = { 0.0f, 0.0f, 0.0f };         // per-lobe solid angle pdf
    vec3 projectedWeight[3];                               // f * cos, per lobe
  };

  // ports restir_pt_trace_core.slangh restirPtSampleScatterDirection: the trace
  // kernel folds the SAMPLED lobe's projected weight over that lobe's own pdf and
  // over the probability of having selected it. This is the quantity the shift has
  // to reproduce at the destination pixel.
  inline vec3 traceThroughputForSampledLobe(const LobeState& m, int lobe) {
    const float denom = m.solidAnglePdf[lobe] * m.selectionProbability[lobe];
    const vec3& f = m.projectedWeight[lobe];
    return vec3(f.x / denom, f.y / denom, f.z / denom);
  }

  // ports restir_pt_shift.slangh restirPtEvalBsdfCosine.
  inline vec3 shiftEvalBsdfCosine(const LobeState& m, unsigned allowed) {
    vec3 sum;
    for (int i = 0; i < 3; ++i) {
      const unsigned bit = 1u << i;
      if ((allowed & bit) == 0u) { continue; }
      sum.x += m.projectedWeight[i].x;
      sum.y += m.projectedWeight[i].y;
      sum.z += m.projectedWeight[i].z;
    }
    return sum;
  }

  // ports restir_pt_shift.slangh restirPtEvalPdfBsdf -- `pdfSingle` is the
  // class-restricted sum of selection-weighted per-lobe pdfs (BxDF.slang:1022-1051),
  // NOT a renormalisation over the class.
  inline float shiftEvalPdfBsdf(const LobeState& m, unsigned allowed, float& pdfAll) {
    pdfAll = 0.0f;
    float pdfSingle = 0.0f;
    for (int i = 0; i < 3; ++i) {
      if (m.selectionProbability[i] <= 0.0f) { continue; }
      const float pdf = m.selectionProbability[i] * m.solidAnglePdf[i];
      if ((allowed & (1u << i)) != 0u) { pdfSingle += pdf; }
      pdfAll += pdf;
    }
    return pdfSingle;
  }

  // The shift's primary-vertex factor, dstF1 / dstPDF1.
  inline vec3 shiftPrimaryFactor(const LobeState& m, unsigned allowed) {
    float pdfAll = 0.0f;
    const float pdfSingle = shiftEvalPdfBsdf(m, allowed, pdfAll);
    const vec3 f = shiftEvalBsdfCosine(m, allowed);
    return vec3(f.x / pdfSingle, f.y / pdfSingle, f.z / pdfSingle);
  }

  // =========================================================================
  // TEXTURE FOOTPRINT: the cone the trace resolves the reconnection vertex with,
  // and the cone the shift's rebuild must reproduce.
  //
  // KEEP IN SYNC with restir_pt_shift.slangh (restirPtShiftSpreadAngleFromSolidAnglePdf
  // and the rcSpreadAngle it hands restirPtReconstructRcVertex),
  // restir_pt_trace_core.slangh (calculateSpreadAngleFromSolidAnglePdfLocal) and
  // concept/surface/surface_interaction.slangh:620-622.
  // =========================================================================

  static constexpr float kPi = 3.14159265358979323846f;

  // ports calculateSpreadAngleFromSolidAnglePdfLocal (integrator.slangh:90-121).
  inline float spreadAngleFromSolidAnglePdf(
    float incomingSpreadAngle, float solidAnglePdf, float indirectRaySpreadAngleFactor) {

    if (solidAnglePdf == 1.0f) { return incomingSpreadAngle; }

    const float clamped = std::min(1.0f, std::max(0.0f, 1.0f / (4.0f * kPi * solidAnglePdf)));
    const float newSpreadAngle = 2.0f * std::asin(std::sqrt(clamped)) * indirectRaySpreadAngleFactor;

    return std::max(incomingSpreadAngle, newSpreadAngle);
  }

  // What the TRACE ends up with: rayInteractionCreate (ray_interaction.slangh:49)
  // gives coneRadius + spreadAngle * hitDistance, and surfaceInteractionCreate then
  // uses that verbatim under kFootprintFromRayDirection.
  inline float traceConeRadiusAtRcVertex(
    float primaryConeRadius, float scatterSpreadAngle, float segmentLength) {
    return primaryConeRadius + scatterSpreadAngle * segmentLength;
  }

  // What the REBUILD ends up with: kFootprintFromRayOrigin computes
  // rayInteraction.coneRadius + length(positionOffset) * ray.spreadAngle
  // (surface_interaction.slangh:622), where positionOffset is measured from the
  // destination primary vertex -- i.e. it is the segment length. So the rebuild
  // matches the trace exactly IF, and only if, it is handed the same spread angle.
  inline float rebuildConeRadiusAtRcVertex(
    float primaryConeRadius, float rebuildSpreadAngle, float segmentLength) {
    return primaryConeRadius + rebuildSpreadAngle * segmentLength;
  }

  // =========================================================================
  // PREFIX / SUFFIX THROUGHPUT, and where the connection segment's attenuation
  // is allowed to live.
  //
  // KEEP IN SYNC with restir_pt_path_state.slangh (recordPrefixThp,
  // getCurrentThp) and the two fold sites in restir_pt_trace_core.slangh -- one
  // on ARRIVAL at the reconnection vertex, one after the scatter FROM it.
  // =========================================================================

  struct PathThroughput {
    vec3 prefix = vec3(1.0f);
    vec3 suffix = vec3(1.0f);

    // ports RestirPtPathState::getCurrentThp.
    vec3 current() const {
      return vec3(suffix.x * prefix.x, suffix.y * prefix.y, suffix.z * prefix.z);
    }

    // ports RestirPtPathState::recordPrefixThp.
    void recordPrefix() {
      prefix = vec3(prefix.x * suffix.x, prefix.y * suffix.y, prefix.z * suffix.z);
      suffix = vec3(1.0f);
    }

    void multiplySuffix(const vec3& v) {
      suffix = vec3(suffix.x * v.x, suffix.y * v.y, suffix.z * v.z);
    }
  };

  inline vec3 mul3(const vec3& a, const vec3& b) {
    return vec3(a.x * b.x, a.y * b.y, a.z * b.z);
  }

  // =========================================================================
  // The debug view 887 metric.
  // KEEP IN SYNC with fork_restir_pt_spatial_reuse.comp.slang
  // (restirPtSelfShiftRelativeError).
  // =========================================================================

  static constexpr float kSelfShiftMinScale = 1e-3f;

  inline float maxComponent(const vec3& v) { return std::max(v.x, std::max(v.y, v.z)); }

  inline float selfShiftRelativeError(const vec3& a, const vec3& b, bool& judged) {
    const float scale = std::max(maxComponent(a), maxComponent(b));

    judged = scale > kSelfShiftMinScale;
    if (!judged) { return 0.0f; }

    const vec3 difference(std::fabs(a.x - b.x), std::fabs(a.y - b.y), std::fabs(a.z - b.z));
    return maxComponent(difference) / scale;
  }

  // The metric this replaced, kept only so the test can state what was wrong with
  // it. maxChannel(|a-b|) / max(1e-4, luma(b)).
  inline float retiredSelfShiftError(const vec3& a, const vec3& b) {
    const vec3 difference(std::fabs(a.x - b.x), std::fabs(a.y - b.y), std::fabs(a.z - b.z));
    return maxComponent(difference) / std::max(1e-4f, RestirPtReservoir::toScalar(b));
  }

  inline float maxRelativeDifference(const vec3& a, const vec3& b) {
    float worst = 0.0f;
    const float ax[3] = { a.x, a.y, a.z };
    const float bx[3] = { b.x, b.y, b.z };
    for (int i = 0; i < 3; ++i) {
      const float scale = std::max(1e-8f, std::fabs(bx[i]));
      worst = std::max(worst, std::fabs(ax[i] - bx[i]) / scale);
    }
    return worst;
  }

  // ports restir_pt_shift.slangh restirPtReconnectionGeometryJacobian
  // (ReSTIRPTPass/Shift.slang:450-469).
  inline float reconnectionGeometryJacobian(
    const vec3& dstPrimaryPosition, const vec3& srcPrimaryPosition,
    const vec3& rcPosition, const vec3& rcFaceNormal) {

    const vec3 shiftedDisplacement = sub(rcPosition, dstPrimaryPosition);
    const float shiftedDistanceSquared = dot3(shiftedDisplacement, shiftedDisplacement);
    const float shiftedCosine = std::fabs(dot3(rcFaceNormal, normalizeOrZ(shiftedDisplacement)));

    const vec3 originalDisplacement = sub(rcPosition, srcPrimaryPosition);
    const float originalDistanceSquared = dot3(originalDisplacement, originalDisplacement);
    const float originalCosine = std::fabs(dot3(rcFaceNormal, normalizeOrZ(originalDisplacement)));

    if (shiftedDistanceSquared <= 0.0f || originalCosine <= 0.0f) { return 0.0f; }

    return (shiftedCosine / shiftedDistanceSquared) * originalDistanceSquared / originalCosine;
  }
}

namespace dxvk {

  class TestApp {
  public:

    void check(bool condition, const char* what) {
      ++m_checks;
      if (!condition) {
        throw DxvkError(str::format("FAILED: ", what));
      }
    }

    void checkClose(float actual, float expected, const char* what, float tolerance = 1e-5f) {
      ++m_checks;
      const float diff = std::fabs(actual - expected);
      const float scale = std::max(1.0f, std::fabs(expected));
      if (diff / scale > tolerance) {
        throw DxvkError(str::format("FAILED: ", what, " -- expected ", expected, ", got ", actual));
      }
    }

    // (a) add(): M and w_sum bookkeeping, and the acceptance rule, against a
    // hand-computed sequence.
    void testAddBookkeeping() {
      mirror::RestirPtReservoir r;
      r.init();

      check(r.M == 0.0f, "a: fresh reservoir has M == 0");
      check(r.weight == 0.0f, "a: fresh reservoir has w_sum == 0");
      check(r.pathFlags.rcVertexLength() == mirror::kRestirPtMaximumPathLength,
            "a: init pins rcVertexLength to kMaximumPathLength");

      // Grey candidates make phat == the component value (0.299+0.587+0.114 == 1).
      const mirror::vec3 a(1.0f, 1.0f, 1.0f);   // phat 1
      const mirror::vec3 b(3.0f, 3.0f, 3.0f);   // phat 3

      // First add: w = 1/0.5 = 2. w_sum = 2. Acceptance is unconditional for the
      // first non-zero candidate (rnd * 2 <= 2 for every rnd in [0,1]).
      check(r.add(a, 0.5f, 0.999f), "a: first non-zero candidate is always selected");
      checkClose(r.weight, 2.0f, "a: w_sum after first add");
      checkClose(r.M, 1.0f, "a: M after first add");
      check(r.F == a, "a: F is the first candidate");

      // Second add: w = 3/1 = 3, w_sum = 5. Accept iff rnd * 5 <= 3, i.e.
      // rnd <= 0.6. Drive it just above the boundary: reject.
      mirror::RestirPtReservoir reject = r;
      check(!reject.add(b, 1.0f, 0.61f), "a: rnd above w/w_sum rejects");
      checkClose(reject.weight, 5.0f, "a: w_sum still accumulates on rejection");
      checkClose(reject.M, 2.0f, "a: M still increments on rejection");
      check(reject.F == a, "a: rejected candidate does not replace F");

      // ... and just below: accept.
      mirror::RestirPtReservoir accept = r;
      check(accept.add(b, 1.0f, 0.59f), "a: rnd below w/w_sum accepts");
      check(accept.F == b, "a: accepted candidate replaces F");
      checkClose(accept.weight, 5.0f, "a: w_sum identical whether or not the candidate is selected");

      // A zero-contribution candidate is rejected but still counts toward M --
      // the reference increments M before the early out (PathReservoir.slang:293-297).
      mirror::RestirPtReservoir zeroCase = r;
      check(!zeroCase.add(mirror::vec3(0.0f), 1.0f, 0.0f), "a: zero-phat candidate is rejected");
      checkClose(zeroCase.M, 2.0f, "a: M increments even for a rejected zero candidate");
      checkClose(zeroCase.weight, 2.0f, "a: w_sum unchanged by a zero candidate");
    }

    // (b) finalizeRIS() == w_sum / (phat * M), including both guards.
    void testFinalizeRIS() {
      {
        mirror::RestirPtReservoir r;
        r.init();
        r.add(mirror::vec3(2.0f), 0.5f, 0.0f);   // w = 4, phat = 2
        r.add(mirror::vec3(2.0f), 0.25f, 1.0f);  // w = 8, w_sum = 12
        // M is 2 here; the streaming builder would pin it to 1, but finalizeRIS
        // has to be correct for either.
        r.finalizeRIS();
        checkClose(r.weight, 12.0f / (2.0f * 2.0f), "b: weight == w_sum / (phat * M)");
      }

      {
        // Zero phat guard: F == 0 must produce weight 0, not a division by zero.
        mirror::RestirPtReservoir r;
        r.init();
        r.M = 4.0f;
        r.weight = 17.0f;
        r.F = mirror::vec3(0.0f);
        r.finalizeRIS();
        check(r.weight == 0.0f, "b: zero phat guard sets weight to 0");
      }

      {
        // Zero M guard: no candidate was ever offered.
        mirror::RestirPtReservoir r;
        r.init();
        r.F = mirror::vec3(1.0f);
        r.weight = 5.0f;
        r.M = 0.0f;
        r.finalizeRIS();
        check(r.weight == 0.0f, "b: zero M guard sets weight to 0");
      }

      {
        // finalizeGRIS divides by phat only -- no M -- which is the whole
        // difference between the two (PathReservoir.slang:443-450).
        mirror::RestirPtReservoir r;
        r.init();
        r.M = 7.0f;
        r.weight = 6.0f;
        r.F = mirror::vec3(3.0f);
        r.finalizeGRIS();
        checkClose(r.weight, 2.0f, "b: finalizeGRIS ignores M");
      }
    }

    // (c) THE PHASE 2 IDENTITY. Exhaustively enumerate every selection outcome of
    // a multi-candidate streaming build and check that the estimator averages to
    // SUM_i F_i / p_i, which is what path.L accumulates.
    void testStreamingEstimatorIsUnbiased() {
      struct Candidate { mirror::vec3 F; float p; };

      // Deliberately mixed: different magnitudes and different source pdfs, the
      // way real terminals (NEE at bounce 1, emissive at bounce 2, sky at 3) are.
      const std::vector<Candidate> candidates = {
        { mirror::vec3(0.5f, 0.25f, 0.75f), 1.0f },
        { mirror::vec3(2.0f, 1.0f, 0.5f),   0.8f },
        { mirror::vec3(0.1f, 0.1f, 4.0f),   0.64f },
      };

      float referenceX = 0.0f, referenceY = 0.0f, referenceZ = 0.0f;
      for (const Candidate& c : candidates) {
        referenceX += c.F.x / c.p;
        referenceY += c.F.y / c.p;
        referenceZ += c.F.z / c.p;
      }

      // Enumerate the selection tree. At step i the candidate is accepted with
      // probability w_i / w_sum_i; walk both branches with their exact weights.
      struct State { mirror::RestirPtReservoir r; double prob; };
      std::vector<State> states;
      {
        State s;
        s.r.init();
        s.prob = 1.0;
        states.push_back(s);
      }

      for (const Candidate& c : candidates) {
        std::vector<State> next;
        for (const State& s : states) {
          // Recompute the acceptance probability the same way add() does.
          const float w = mirror::RestirPtReservoir::toScalar(c.F) / c.p;
          const float wSumAfter = s.r.weight + w;
          const double pAccept = (w <= 0.0f) ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

          if (pAccept > 0.0) {
            State accepted = s;
            // acceptRnd 0 always accepts.
            accepted.r.add(c.F, c.p, 0.0f);
            accepted.prob = s.prob * pAccept;
            next.push_back(accepted);
          }

          if (pAccept < 1.0) {
            State rejected = s;
            // acceptRnd 1 rejects whenever w < w_sum.
            rejected.r.add(c.F, c.p, 1.0f);
            rejected.prob = s.prob * (1.0 - pAccept);
            next.push_back(rejected);
          }
        }
        states = next;
      }

      double totalProb = 0.0;
      double eX = 0.0, eY = 0.0, eZ = 0.0;

      for (State& s : states) {
        // What the trace kernel does at loop exit.
        mirror::pathBuilderFinalize(s.r);
        s.r.finalizeRIS();

        totalProb += s.prob;
        eX += s.prob * double(s.r.F.x) * double(s.r.weight);
        eY += s.prob * double(s.r.F.y) * double(s.r.weight);
        eZ += s.prob * double(s.r.F.z) * double(s.r.weight);
      }

      checkClose(float(totalProb), 1.0f, "c: selection probabilities sum to 1");
      checkClose(float(eX), referenceX, "c: E[F.x * weight] == sum F_i.x / p_i", 1e-4f);
      checkClose(float(eY), referenceY, "c: E[F.y * weight] == sum F_i.y / p_i", 1e-4f);
      checkClose(float(eZ), referenceZ, "c: E[F.z * weight] == sum F_i.z / p_i", 1e-4f);
    }

    // (c2) The merge family round-trips the estimator in the single-candidate
    // case: merging a finalized reservoir into an empty one, then finalizing,
    // must reproduce the same contribution.
    void testMergeRoundTrip() {
      mirror::RestirPtReservoir src;
      src.init();
      src.srcPixel = 0x00230011u;
      src.srcFrameIdx = 4242u;
      src.lightPdf = 3.5f;
      src.pathFlags.insertPathLength(3);
      src.pathFlags.insertLastVertexNEE(true);

      const mirror::vec3 F(1.5f, 0.5f, 0.25f);
      const float p = 0.4f;

      src.add(F, p, 0.0f);
      mirror::pathBuilderFinalize(src);
      src.finalizeRIS();

      const float expectedX = F.x / p;
      checkClose(src.F.x * src.weight, expectedX, "c2: single-candidate reservoir reproduces F/p");

      // WHICH FORM OF `weight` EACH ENTRY POINT EXPECTS -- easy to get backwards,
      // and the reason this test exists. `weight` carries two meanings
      // (PathReservoir.slang:227):
      //
      //   merge / mergeWithResamplingMIS take a FINALIZED incoming reservoir
      //     (weight == contribution weight). They multiply in phat(in_F)
      //     themselves, so a pre-scaled input would square it.
      //
      //   mergeInSamplePixel takes a PREPAREMERGING'd reservoir (weight == a
      //     resampling weight), because it copies F verbatim and does no phat
      //     multiply of its own.
      //
      //   prepareMerging is applied to the DESTINATION, never the source -- the
      //     reference calls it on dstReservoir before streaming neighbours in
      //     (SpatialReuse.cs.slang:175, TemporalReuse.cs.slang:102).

      // Identity shift: same integrand, unit Jacobian, unit MIS weight.
      mirror::RestirPtReservoir dst;
      dst.init();
      check(dst.mergeWithResamplingMIS(src.F, 1.0f, src, 0.0f, 1.0f, false),
            "c2: the only candidate is selected");
      dst.finalizeGRIS();

      checkClose(dst.F.x * dst.weight, expectedX, "c2: mergeWithResamplingMIS + finalizeGRIS conserves the estimate");
      check(dst.pathFlags.pathLength() == 3, "c2: merge carries pathFlags");
      check(dst.pathFlags.lastVertexNEE(), "c2: merge carries the NEE terminal flag");
      check(dst.srcPixel == 0x00230011u, "c2: merge carries the replay identity (pixel)");
      check(dst.srcFrameIdx == 4242u, "c2: merge carries the replay identity (frame)");
      checkClose(dst.lightPdf, 3.5f, "c2: merge carries lightPdf");

      // merge() additionally folds the incoming M into the resampling weight, so
      // with M == 1 the two must agree exactly.
      mirror::RestirPtReservoir dst2;
      dst2.init();
      check(dst2.merge(src.F, 1.0f, src, 0.0f, 1.0f, false), "c2: merge selects its only candidate");
      dst2.finalizeGRIS();
      checkClose(dst2.F.x * dst2.weight, expectedX, "c2: merge agrees with mergeWithResamplingMIS at M == 1");

      // mergeInSamplePixel copies F verbatim, so its input has to be scaled.
      mirror::RestirPtReservoir prepared = src;
      prepared.prepareMerging();
      checkClose(prepared.weight,
                 src.weight * mirror::RestirPtReservoir::toScalar(src.F) * src.M,
                 "c2: prepareMerging scales w by phat * M");

      mirror::RestirPtReservoir dst3;
      dst3.init();
      check(dst3.mergeInSamplePixel(prepared, 0.0f), "c2: mergeInSamplePixel selects its only candidate");
      check(dst3.F == src.F, "c2: mergeInSamplePixel copies F from the source");
      dst3.finalizeGRIS();
      checkClose(dst3.F.x * dst3.weight, expectedX, "c2: mergeInSamplePixel + finalizeGRIS conserves the estimate");

      // prepareMerging is the exact inverse of finalizeGRIS at M == 1, which is
      // what lets a reuse pass load a stored reservoir, merge more into it, and
      // finalize again without the estimate drifting.
      mirror::RestirPtReservoir roundTrip = src;
      roundTrip.prepareMerging();
      roundTrip.finalizeGRIS();
      checkClose(roundTrip.weight, src.weight, "c2: prepareMerging then finalizeGRIS is the identity at M == 1");
    }

    // (d) NaN rejection. PathReservoir.slang:297 and :327 both bail on a NaN
    // resampling weight -- without that, one bad vertex poisons the reservoir and
    // then the denoiser history.
    void testNanRejection() {
      const float nan = std::numeric_limits<float>::quiet_NaN();

      {
        mirror::RestirPtReservoir r;
        r.init();
        r.add(mirror::vec3(1.0f), 1.0f, 0.0f);
        const float wSumBefore = r.weight;
        const mirror::vec3 good = r.F;

        check(!r.add(mirror::vec3(nan, nan, nan), 1.0f, 0.0f), "d: NaN candidate is rejected by add");
        check(r.weight == wSumBefore, "d: NaN candidate does not corrupt w_sum");
        check(r.F == good, "d: NaN candidate does not replace F");
        check(!std::isnan(r.weight), "d: w_sum stays finite");
      }

      {
        // A zero source pdf produces inf, not NaN, for a non-zero numerator --
        // which the reference does NOT reject. Locking the actual behaviour so a
        // future change to it is a deliberate one. The trace kernel never emits
        // a zero russianRoulettePdf (it is a product of strictly positive
        // survival probabilities), and the final shading pass has its own
        // isinf guard on the resolved weight, which is where this is caught.
        //
        // Note the draw is 0.5, not 0: the acceptance test is
        // `acceptRnd * weight <= w`, and 0 * inf is NaN, which compares false --
        // so an infinite candidate offered with a zero draw is REJECTED while the
        // same candidate with any positive draw is accepted. That asymmetry is
        // inherited from the reference and is harmless only because the
        // downstream guard exists.
        mirror::RestirPtReservoir r;
        r.init();
        const bool selected = r.add(mirror::vec3(1.0f), 0.0f, 0.5f);
        check(selected, "d: an infinite weight is still selected (guarded downstream, not here)");
        check(std::isinf(r.weight), "d: an infinite weight propagates to w_sum");

        mirror::RestirPtReservoir rZeroDraw;
        rZeroDraw.init();
        check(!rZeroDraw.add(mirror::vec3(1.0f), 0.0f, 0.0f), "d: 0 * inf is NaN, so a zero draw rejects an infinite candidate");
      }

      {
        // 0/0 IS a NaN and must be rejected.
        mirror::RestirPtReservoir r;
        r.init();
        check(!r.add(mirror::vec3(0.0f), 0.0f, 0.0f), "d: 0/0 is rejected");
        check(r.weight == 0.0f, "d: 0/0 leaves w_sum clean");
      }

      {
        mirror::RestirPtReservoir dst;
        dst.init();
        mirror::RestirPtReservoir in;
        in.init();
        in.M = 1.0f;
        in.weight = nan;
        in.F = mirror::vec3(1.0f);

        check(!dst.merge(mirror::vec3(1.0f), 1.0f, in, 0.0f, 1.0f, false), "d: NaN merge is rejected");
        check(!std::isnan(dst.weight), "d: rejected NaN merge leaves w_sum clean");
        checkClose(dst.M, 1.0f, "d: rejected merge still folds in M");
      }
    }

    // (e) pathFlags encode/decode round-trips every field at its extremes. The
    // bit positions ARE the reference's, and a phase 3 shift reading a
    // misplaced bit would silently mis-route or mis-shift paths.
    void testPathFlagsRoundTrip() {
      for (int len = 0; len <= 15; ++len) {
        for (int rcLen = 0; rcLen <= 15; ++rcLen) {
          mirror::RestirPtPathFlags f;
          f.insertPathLength(len);
          f.insertRcVertexLength(rcLen);
          check(f.pathLength() == len, "e: pathLength round-trips");
          check(f.rcVertexLength() == rcLen, "e: rcVertexLength round-trips");
        }
      }

      for (uint32_t lightType = 0; lightType <= 3; ++lightType) {
        mirror::RestirPtPathFlags f;
        f.insertLightType(lightType);
        check(f.lightType() == lightType, "e: lightType round-trips");
      }

      // Both bits of each pair, independently, and without disturbing anything
      // else -- this is what the shared-word packing makes easy to get wrong.
      for (int mask = 0; mask < 64; ++mask) {
        const bool deltaBefore = (mask & 1) != 0;
        const bool deltaAt = (mask & 2) != 0;
        const bool transBefore = (mask & 4) != 0;
        const bool transAt = (mask & 8) != 0;
        const bool specBefore = (mask & 16) != 0;
        const bool specAt = (mask & 32) != 0;

        mirror::RestirPtPathFlags f;
        f.insertPathLength(15);
        f.insertRcVertexLength(15);
        f.insertLastVertexNEE(true);
        f.insertLightType(3);
        f.insertIsDeltaEvent(deltaBefore, true);
        f.insertIsDeltaEvent(deltaAt, false);
        f.insertIsTransmissionEvent(transBefore, true);
        f.insertIsTransmissionEvent(transAt, false);
        f.insertIsSpecularBounce(specBefore, true);
        f.insertIsSpecularBounce(specAt, false);

        check(f.decodeIsDeltaEvent(true) == deltaBefore, "e: delta-before round-trips");
        check(f.decodeIsDeltaEvent(false) == deltaAt, "e: delta-at round-trips");
        check(f.decodeIsTransmissionEvent(true) == transBefore, "e: transmission-before round-trips");
        check(f.decodeIsTransmissionEvent(false) == transAt, "e: transmission-at round-trips");
        check(f.decodeIsSpecularBounce(true) == specBefore, "e: specular-before round-trips");
        check(f.decodeIsSpecularBounce(false) == specAt, "e: specular-at round-trips");

        // Neighbours undisturbed.
        check(f.pathLength() == 15, "e: event bits do not disturb pathLength");
        check(f.rcVertexLength() == 15, "e: event bits do not disturb rcVertexLength");
        check(f.lastVertexNEE(), "e: event bits do not disturb lastVertexNEE");
        check(f.lightType() == 3u, "e: event bits do not disturb lightType");
      }

      // Insert-false must CLEAR, not just fail to set (the reference's masks are
      // written that way and the transfer* helpers rely on it).
      mirror::RestirPtPathFlags f;
      f.insertIsSpecularBounce(true, true);
      check(f.decodeIsSpecularBounce(true), "e: specular-before set");
      f.insertIsSpecularBounce(false, true);
      check(!f.decodeIsSpecularBounce(true), "e: specular-before cleared");
    }

    // toScalar is the RIS target function; it must stay the reference's Rec.601
    // luma and not drift to the codebase's calcBt709Luminance.
    void testTargetFunction() {
      checkClose(mirror::RestirPtReservoir::toScalar(mirror::vec3(1.0f, 0.0f, 0.0f)), 0.299f, "f: toScalar red weight");
      checkClose(mirror::RestirPtReservoir::toScalar(mirror::vec3(0.0f, 1.0f, 0.0f)), 0.587f, "f: toScalar green weight");
      checkClose(mirror::RestirPtReservoir::toScalar(mirror::vec3(0.0f, 0.0f, 1.0f)), 0.114f, "f: toScalar blue weight");
      checkClose(mirror::RestirPtReservoir::toScalar(mirror::vec3(1.0f)), 1.0f, "f: toScalar of white is 1");

      checkClose(mirror::RestirPtReservoir::computeWeight(mirror::vec3(0.25f), true), 1.0f, "f: computeWeight binarizes");
      checkClose(mirror::RestirPtReservoir::computeWeight(mirror::vec3(0.25f), false), 0.25f, "f: computeWeight passthrough");
      checkClose(mirror::RestirPtReservoir::computeWeight(mirror::vec3(0.0f), true), 0.0f, "f: computeWeight leaves zero alone");
    }

    // -----------------------------------------------------------------------
    // (g) PHASE 3: the reconnection Jacobian's geometry core.
    //
    // Three properties, all of which a transcription slip breaks: the shift onto
    // your own surface is the identity, shifting there and back is the identity,
    // and degenerate placements are rejected rather than returning garbage.
    // -----------------------------------------------------------------------
    void testReconnectionGeometryJacobian() {
      const mirror::vec3 rc(0.0f, 0.0f, 0.0f);
      const mirror::vec3 faceN(0.0f, 0.0f, 1.0f);

      const mirror::vec3 a(1.0f, 0.0f, 2.0f);
      const mirror::vec3 b(-3.0f, 1.5f, 5.0f);

      // J(a -> a) is exactly 1: identical numerator and denominator, bit for bit.
      check(mirror::reconnectionGeometryJacobian(a, a, rc, faceN) == 1.0f,
            "g: J(a -> a) is exactly 1");
      check(mirror::reconnectionGeometryJacobian(b, b, rc, faceN) == 1.0f,
            "g: J(b -> b) is exactly 1");

      // Round trip. If the src/dst roles were swapped anywhere, this lands on
      // (cos_a d_b^2 / cos_b d_a^2)^2 instead of 1 -- the "energy warp correlated
      // with geometry" symptom.
      const float forward = mirror::reconnectionGeometryJacobian(a, b, rc, faceN);
      const float backward = mirror::reconnectionGeometryJacobian(b, a, rc, faceN);

      check(!mirror::isJacobianInvalid(forward), "g: forward Jacobian is valid");
      check(!mirror::isJacobianInvalid(backward), "g: backward Jacobian is valid");
      checkClose(forward * backward, 1.0f, "g: J(a -> b) * J(b -> a) == 1");

      // Hand-computed value, so a sign or an inversion cannot hide behind the
      // round trip alone. cos = |z| / d, so J = (|z_a|/d_a^3) / (|z_b|/d_b^3).
      {
        const mirror::vec3 p(0.0f, 0.0f, 1.0f);   // straight on, distance 1
        const mirror::vec3 q(0.0f, 3.0f, 4.0f);   // distance 5, cos = 4/5
        const float expected = (1.0f / 1.0f) / ((4.0f / 5.0f) / 25.0f);
        checkClose(mirror::reconnectionGeometryJacobian(p, q, rc, faceN), expected,
                   "g: hand-computed geometry Jacobian");
      }

      // Degenerate placements. A source vertex coplanar with the reconnection
      // vertex's triangle has zero cosine on the source side -- an infinite
      // Jacobian if the guard is missing.
      {
        const mirror::vec3 coplanar(4.0f, 0.0f, 0.0f);
        const float j = mirror::reconnectionGeometryJacobian(a, coplanar, rc, faceN);
        check(mirror::isJacobianInvalid(j), "g: coplanar source is rejected");
      }

      {
        // Destination sitting exactly on the reconnection vertex: zero distance.
        const float j = mirror::reconnectionGeometryJacobian(rc, a, rc, faceN);
        check(mirror::isJacobianInvalid(j), "g: zero-length connection is rejected");
      }

      // The guard itself, on the values the reference names.
      const float nan = std::numeric_limits<float>::quiet_NaN();
      const float inf = std::numeric_limits<float>::infinity();

      check(mirror::isJacobianInvalid(nan), "g: NaN Jacobian is invalid");
      check(mirror::isJacobianInvalid(inf), "g: infinite Jacobian is invalid");
      check(mirror::isJacobianInvalid(0.0f), "g: zero Jacobian is invalid");
      check(mirror::isJacobianInvalid(-1.0f), "g: negative Jacobian is invalid");
      check(!mirror::isJacobianInvalid(1e-8f), "g: a tiny positive Jacobian is valid");

      // The full shift Jacobian is the geometry term times two pdf ratios; each
      // ratio inverts under a src/dst swap, so the whole product round-trips.
      {
        const float dstPdf1 = 0.37f, srcPdf1 = 1.91f;
        const float dstPdf2 = 4.25f, srcPdf2 = 0.62f;

        const float full = forward * (dstPdf1 / srcPdf1) * (dstPdf2 / srcPdf2);
        const float fullBack = backward * (srcPdf1 / dstPdf1) * (srcPdf2 / dstPdf2);

        checkClose(full * fullBack, 1.0f, "g: full shift Jacobian round-trips");
      }
    }

    // -----------------------------------------------------------------------
    // (h) PHASE 3: PAIRWISE RESAMPLING MIS IS UNBIASED.
    //
    // The strongest assertion available for this phase, and the direct extension
    // of (c). Build K+1 synthetic pixels over ONE shared discrete path domain, so
    // that shifting is the identity with Jacobian 1 and "pixel px's integrand at
    // path j" is simply F[px][j]. Every pixel builds a phase-2 reservoir over that
    // domain; the central pixel then runs the pairwise combine exactly as
    // fork_restir_pt_spatial_reuse.comp.slang does, with the merge draws
    // enumerated rather than sampled.
    //
    //   assert  E[F_c(Y) * W_out]  ==  SUM_j F_c(j) / p_j
    //
    // which is what the central pixel's own estimator was already unbiased for.
    // Reuse is allowed to change the variance and nothing else.
    //
    // What this catches, from plan-phase3's symptom table: a missing
    // /(validNeighborCount + 1) (~x(k+1) too bright), a forgotten dstReservoir
    // .init() or a canonical-weight slip, and `merge` used where
    // `mergeWithResamplingMIS` belongs -- the last of which is nearly invisible
    // in-game at M == 1 and unmissable here.
    // -----------------------------------------------------------------------
    void testPairwiseResamplingMISIsUnbiased() {
      // One reservoir outcome of a phase-2 streaming build: which path index was
      // selected, with what probability and what contribution weight.
      struct Outcome {
        int index = 0;
        double probability = 0.0;
        mirror::RestirPtReservoir reservoir;
      };

      // Enumerate a pixel's streaming RIS build exhaustively. Each candidate is
      // offered in order; the selected index is the only thing that survives, and
      // its probability is w_j / w_sum.
      auto buildPixel = [](const std::vector<mirror::vec3>& F, const std::vector<float>& p) {
        std::vector<Outcome> outcomes;

        mirror::RestirPtReservoir reference;
        reference.init();
        for (size_t j = 0; j < F.size(); ++j) {
          reference.add(F[j], p[j], 1.0f);
        }

        const float wSum = reference.weight;

        for (size_t j = 0; j < F.size(); ++j) {
          const float w = mirror::RestirPtReservoir::toScalar(F[j]) / p[j];
          if (!(w > 0.0f)) { continue; }

          Outcome o;
          o.index = int(j);
          o.probability = double(w) / double(wSum);

          o.reservoir.init();
          o.reservoir.F = F[j];
          o.reservoir.weight = wSum;
          o.reservoir.M = 1.0f;
          // Stash the path index so the shift can be modelled as the identity.
          o.reservoir.srcPixel = uint32_t(j);
          mirror::pathBuilderFinalize(o.reservoir);
          o.reservoir.finalizeRIS();

          outcomes.push_back(o);
        }

        return outcomes;
      };

      // One (K + 1) x J integrand table plus one shared source-pdf vector is a
      // whole test case. Deliberately asymmetric: differing scales per pixel, and
      // at least one path a neighbour cannot see at all (F == 0), which is the
      // shift-to-zero case the canonical weight has to compensate for.
      struct Case {
        const char* name;
        std::vector<std::vector<mirror::vec3>> F;  // [pixel][path], pixel 0 is central
        std::vector<float> p;
      };

      std::vector<Case> cases;

      cases.push_back({
        "k=1, mild asymmetry",
        {
          { mirror::vec3(1.0f, 0.5f, 0.25f), mirror::vec3(0.2f, 2.0f, 0.1f), mirror::vec3(0.7f, 0.7f, 3.0f) },
          { mirror::vec3(0.8f, 0.6f, 0.30f), mirror::vec3(0.3f, 1.4f, 0.2f), mirror::vec3(0.5f, 0.9f, 2.1f) },
        },
        { 1.0f, 0.8f, 0.5f }
      });

      cases.push_back({
        "k=2, one neighbour blind to a path",
        {
          { mirror::vec3(2.0f, 1.0f, 0.5f),  mirror::vec3(0.4f, 0.4f, 4.0f), mirror::vec3(1.1f, 0.2f, 0.9f) },
          { mirror::vec3(0.1f, 0.1f, 0.1f),  mirror::vec3(0.0f, 0.0f, 0.0f), mirror::vec3(3.0f, 1.0f, 0.5f) },
          { mirror::vec3(5.0f, 0.25f, 1.0f), mirror::vec3(0.9f, 0.9f, 0.9f), mirror::vec3(0.0f, 0.0f, 0.0f) },
        },
        { 1.0f, 0.64f, 0.32f }
      });

      cases.push_back({
        "k=3, wide dynamic range",
        {
          { mirror::vec3(0.05f, 0.05f, 0.05f), mirror::vec3(9.0f, 0.1f, 0.1f) },
          { mirror::vec3(0.02f, 0.90f, 0.03f), mirror::vec3(0.3f, 0.3f, 6.0f) },
          { mirror::vec3(1.50f, 0.02f, 0.20f), mirror::vec3(0.0f, 0.0f, 0.0f) },
          { mirror::vec3(0.40f, 0.40f, 0.40f), mirror::vec3(2.2f, 0.5f, 0.1f) },
        },
        { 0.9f, 0.45f }
      });

      for (const Case& c : cases) {
        const int pixelCount = int(c.F.size());
        const int neighborCount = pixelCount - 1;
        const int pathCount = int(c.F[0].size());

        // The target: what the central pixel's own estimator estimates.
        double refX = 0.0, refY = 0.0, refZ = 0.0;
        for (int j = 0; j < pathCount; ++j) {
          refX += double(c.F[0][j].x) / double(c.p[j]);
          refY += double(c.F[0][j].y) / double(c.p[j]);
          refZ += double(c.F[0][j].z) / double(c.p[j]);
        }

        std::vector<std::vector<Outcome>> perPixel;
        for (int px = 0; px < pixelCount; ++px) {
          perPixel.push_back(buildPixel(c.F[px], c.p));
        }

        // Enumerate the joint selection of every pixel's reservoir.
        std::vector<int> pick(pixelCount, 0);
        double totalProbability = 0.0;
        double eX = 0.0, eY = 0.0, eZ = 0.0;

        // Odometer over perPixel[px].size().
        bool done = false;
        while (!done) {
          double jointProbability = 1.0;
          for (int px = 0; px < pixelCount; ++px) {
            jointProbability *= perPixel[px][pick[px]].probability;
          }

          const mirror::RestirPtReservoir centralReservoir = perPixel[0][pick[0]].reservoir;
          const int centralPath = perPixel[0][pick[0]].index;

          // --- the pairwise combine, mirroring the shader ---------------------
          //
          // Shifts are the identity on this domain, so the shifted integrand of a
          // path j onto pixel P is simply F[P][j], and every Jacobian is 1.
          int validNeighborCount = 0;
          float canonicalWeight = 1.0f;

          struct MergeState { mirror::RestirPtReservoir dst; double probability; };

          std::vector<MergeState> states;
          {
            MergeState s;
            s.dst.init();
            s.probability = 1.0;
            states.push_back(s);
          }

          for (int i = 1; i < pixelCount; ++i) {
            const mirror::RestirPtReservoir neighborReservoir = perPixel[i][pick[i]].reservoir;
            const int neighborPath = perPixel[i][pick[i]].index;

            ++validNeighborCount;

            // (a) central -> neighbour, for the canonical weight. ports
            // SpatialReuse.cs.slang:372-379.
            const mirror::vec3 prefixIntegrand = c.F[i][centralPath];
            const float prefixApproxPdf =
              mirror::RestirPtReservoir::computeWeight(prefixIntegrand, false) * 1.0f;

            canonicalWeight += 1.0f;

            if (prefixApproxPdf > 0.0f) {
              canonicalWeight -= prefixApproxPdf * neighborReservoir.M /
                (prefixApproxPdf * neighborReservoir.M +
                 centralReservoir.M * mirror::RestirPtReservoir::computeWeight(centralReservoir.F, false) /
                   float(neighborCount));
            }

            // (b) neighbour -> central. ports :381-400.
            const mirror::vec3 shiftedIntegrand = c.F[0][neighborPath];
            const float dstJacobian = 1.0f;

            std::vector<MergeState> next;

            for (const MergeState& s : states) {
              // shiftAndMergeReservoir with forceMerge: the temp reservoir ends up
              // holding the shifted integrand plus the SOURCE's M and weight.
              mirror::RestirPtReservoir temp = s.dst;
              const bool selected = temp.merge(shiftedIntegrand, dstJacobian, neighborReservoir,
                                               /*acceptRnd=*/ 0.0f, /*misWeight=*/ 1.0f, /*forceAdd=*/ true);
              if (!selected) { temp.F = mirror::vec3(); }
              temp.M = neighborReservoir.M;
              temp.weight = neighborReservoir.weight;

              float neighborWeight = 0.0f;

              if (selected) {
                const float neighborPHat =
                  mirror::RestirPtReservoir::computeWeight(neighborReservoir.F, false) / dstJacobian;

                neighborWeight = neighborPHat * neighborReservoir.M /
                  (neighborPHat * neighborReservoir.M +
                   mirror::RestirPtReservoir::computeWeight(temp.F, false) * centralReservoir.M /
                     float(neighborCount));

                if (std::isnan(neighborWeight) || std::isinf(neighborWeight)) { neighborWeight = 0.0f; }
              }

              // mergeWithResamplingMIS, both branches of the accept draw.
              const float w = mirror::RestirPtReservoir::toScalar(temp.F) * dstJacobian *
                              temp.weight * neighborWeight;
              const float wSumAfter = s.dst.weight + w;
              const double pAccept = (!(w > 0.0f) || std::isnan(w))
                ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

              if (pAccept > 0.0) {
                MergeState accepted = s;
                accepted.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 0.0f, neighborWeight, false);
                accepted.probability = s.probability * pAccept;
                next.push_back(accepted);
              }

              if (pAccept < 1.0) {
                MergeState rejected = s;
                rejected.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 1.0f, neighborWeight, false);
                rejected.probability = s.probability * (1.0 - pAccept);
                next.push_back(rejected);
              }
            }

            states = next;
          }

          // The canonical merge. ports :403 -- shifted onto itself, Jacobian 1.
          {
            std::vector<MergeState> next;

            for (const MergeState& s : states) {
              const float w = mirror::RestirPtReservoir::toScalar(centralReservoir.F) * 1.0f *
                              centralReservoir.weight * canonicalWeight;
              const float wSumAfter = s.dst.weight + w;
              const double pAccept = (!(w > 0.0f) || std::isnan(w))
                ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

              if (pAccept > 0.0) {
                MergeState accepted = s;
                accepted.dst.mergeWithResamplingMIS(centralReservoir.F, 1.0f, centralReservoir, 0.0f, canonicalWeight, false);
                accepted.probability = s.probability * pAccept;
                next.push_back(accepted);
              }

              if (pAccept < 1.0) {
                MergeState rejected = s;
                rejected.dst.mergeWithResamplingMIS(centralReservoir.F, 1.0f, centralReservoir, 1.0f, canonicalWeight, false);
                rejected.probability = s.probability * (1.0 - pAccept);
                next.push_back(rejected);
              }
            }

            states = next;
          }

          // ports :405-409 plus the output guards at :534-535.
          for (MergeState& s : states) {
            if (s.dst.weight > 0.0f) {
              s.dst.finalizeGRIS();
              s.dst.weight /= float(validNeighborCount + 1);
            }

            if (s.dst.weight < 0.0f) { s.dst.weight = 0.0f; }
            if (std::isnan(s.dst.weight) || std::isinf(s.dst.weight)) { s.dst.weight = 0.0f; }

            const double probability = jointProbability * s.probability;

            totalProbability += probability;
            eX += probability * double(s.dst.F.x) * double(s.dst.weight);
            eY += probability * double(s.dst.F.y) * double(s.dst.weight);
            eZ += probability * double(s.dst.F.z) * double(s.dst.weight);
          }

          // Advance the odometer.
          int px = 0;
          for (; px < pixelCount; ++px) {
            if (++pick[px] < int(perPixel[px].size())) { break; }
            pick[px] = 0;
          }
          done = (px == pixelCount);
        }

        checkClose(float(totalProbability), 1.0f,
                   str::format("h: [", c.name, "] outcome probabilities sum to 1").c_str(), 1e-4f);
        checkClose(float(eX), float(refX),
                   str::format("h: [", c.name, "] E[F.x * W] == sum F_c.x / p").c_str(), 1e-3f);
        checkClose(float(eY), float(refY),
                   str::format("h: [", c.name, "] E[F.y * W] == sum F_c.y / p").c_str(), 1e-3f);
        checkClose(float(eZ), float(refZ),
                   str::format("h: [", c.name, "] E[F.z * W] == sum F_c.z / p").c_str(), 1e-3f);
      }
    }

    // -----------------------------------------------------------------------
    // (i) PHASE 3: the pairwise MIS weights are a partition of unity, and the
    // guards on the way out of the reuse pass.
    //
    // The partition check is the algebraic half of (h): it holds pointwise, for
    // any number of VALID neighbours, whatever is inside the individual ratios --
    // which is exactly why the deferred /(k' + 1) is the whole normalisation and
    // why dropping it brightens the image by that factor.
    // -----------------------------------------------------------------------
    void testPairwiseWeightsPartitionAndGuards() {
      const int configuredCount = 4;   // the divisor the reference uses inside the ratios

      // Arbitrary per-neighbour target-function values at the canonical path.
      const std::vector<float> neighborPHatAtCanonical = { 0.7f, 0.0f, 3.25f, 0.02f };
      const float centralPHat = 1.3f;

      for (int validCount = 1; validCount <= 4; ++validCount) {
        float canonicalWeight = 1.0f;
        float neighborWeightSum = 0.0f;

        for (int i = 0; i < validCount; ++i) {
          const float prefixApproxPdf = neighborPHatAtCanonical[i];

          canonicalWeight += 1.0f;

          if (prefixApproxPdf > 0.0f) {
            const float a = prefixApproxPdf /
              (prefixApproxPdf + centralPHat / float(configuredCount));
            canonicalWeight -= a;
            neighborWeightSum += a;
          }
        }

        // Every m_i is a_i, m_c is canonicalWeight; the deferred division by
        // (validCount + 1) is what turns the sum into 1.
        checkClose((canonicalWeight + neighborWeightSum) / float(validCount + 1), 1.0f,
                   "i: pairwise MIS weights sum to 1 after the deferred division");
      }

      // A zero-integrand shift (the neighbour cannot see the canonical path at
      // all) must leave the canonical weight at its full +1 for that neighbour --
      // that is how the canonical sample compensates for a lost shift.
      {
        float canonicalWeight = 1.0f;
        const float prefixApproxPdf = 0.0f;
        canonicalWeight += 1.0f;
        if (prefixApproxPdf > 0.0f) { canonicalWeight -= 0.5f; }
        checkClose(canonicalWeight, 2.0f, "i: a shift-to-zero neighbour leaves the canonical weight whole");
      }

      // ports SpatialReuse.cs.slang:397 -- the neighbour weight guard. A zero
      // Jacobian that slipped past isJacobianInvalid would make this infinite.
      {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();

        float neighborWeight = inf;
        if (std::isnan(neighborWeight) || std::isinf(neighborWeight)) { neighborWeight = 0.0f; }
        check(neighborWeight == 0.0f, "i: infinite neighbour weight is zeroed");

        neighborWeight = nan;
        if (std::isnan(neighborWeight) || std::isinf(neighborWeight)) { neighborWeight = 0.0f; }
        check(neighborWeight == 0.0f, "i: NaN neighbour weight is zeroed");
      }

      // ports :534-535 -- the output guards, mirrored so the reuse pass's last
      // line of defence is locked the same way final shading's already is.
      {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();

        mirror::RestirPtReservoir r;
        r.init();
        r.F = mirror::vec3(1.0f);

        r.weight = -3.0f;
        if (r.weight < 0.0f) { r.weight = 0.0f; }
        if (std::isnan(r.weight) || std::isinf(r.weight)) { r.weight = 0.0f; }
        check(r.weight == 0.0f, "i: negative output weight is clamped to 0");

        r.weight = inf;
        if (r.weight < 0.0f) { r.weight = 0.0f; }
        if (std::isnan(r.weight) || std::isinf(r.weight)) { r.weight = 0.0f; }
        check(r.weight == 0.0f, "i: infinite output weight is zeroed");

        r.weight = nan;
        if (r.weight < 0.0f) { r.weight = 0.0f; }
        if (std::isnan(r.weight) || std::isinf(r.weight)) { r.weight = 0.0f; }
        check(r.weight == 0.0f, "i: NaN output weight is zeroed");
      }

      // The inherited `0 * inf` quirk documented in (d) reaches the reuse pass
      // through mergeWithResamplingMIS too: an infinite resampling weight offered
      // with a zero draw is REJECTED, with any positive draw ACCEPTED. Locked here
      // so a later change to it is deliberate; it stays harmless only because the
      // output guards above and final shading's own guards both exist.
      {
        mirror::RestirPtReservoir in;
        in.init();
        in.M = 1.0f;
        in.weight = std::numeric_limits<float>::infinity();
        in.F = mirror::vec3(1.0f);

        mirror::RestirPtReservoir zeroDraw;
        zeroDraw.init();
        check(!zeroDraw.mergeWithResamplingMIS(mirror::vec3(1.0f), 1.0f, in, 0.0f, 1.0f, false),
              "i: 0 * inf is NaN, so a zero draw rejects an infinite resampling weight");

        mirror::RestirPtReservoir positiveDraw;
        positiveDraw.init();
        check(positiveDraw.mergeWithResamplingMIS(mirror::vec3(1.0f), 1.0f, in, 0.5f, 1.0f, false),
              "i: the same infinite weight is accepted with a positive draw");
        check(std::isinf(positiveDraw.weight), "i: the infinity propagates to w_sum, for the guards to catch");
      }
    }

    // -----------------------------------------------------------------------
    // (j) PHASE 3: THE SELF-SHIFT IDENTITY, at the level the GPU got wrong.
    //
    // For dst == src the reconnection shift must return the path's own integrand.
    // Its primary-vertex factor is dstF1 / dstPDF1, evaluated over a LOBE CLASS;
    // the trace kernel's corresponding factor is the SAMPLED lobe's
    // f*cos / (pdf_lobe * P_lobe). Those agree only when
    //
    //   (1) the class contains exactly one lobe with non-zero selection
    //       probability, and
    //   (2) that lobe is the one the path actually sampled.
    //
    // (2) is what broke in-game: the shift was reading the fork's
    // `specularBounce` flag, which folds in a roughness threshold because this
    // fork repurposed it for denoiser routing. On any surface rougher than the
    // threshold a specular sample was labelled "not specular", so the shift
    // evaluated the diffuse lobe instead. `minOpaqueSpecularLobeSamplingProbability`
    // is 0.25 (rtx_options.h:831), which forces the specular lobe to be picked on
    // a large minority of pixels everywhere -- hence a dithered failure over an
    // entire exterior rather than a localised one. Debug view 887 showed it as
    // red (integrand) with green (Jacobian) clean, because the same wrong class is
    // used on both sides of every pdf ratio and cancels there.
    //
    // This test locks (1) and (2) directly, and re-enacts the bug so the
    // diagnosis itself cannot be quietly undone.
    // -----------------------------------------------------------------------
    void testLobeClassConventionSelfShift() {
      // The fork's specularRoughnessThreshold default (rtx_fork_restir_pt_rayquery.h).
      const float specularRoughnessThreshold = 0.2f;

      struct Material {
        const char* name;
        float perceptualRoughness;
        mirror::LobeState lobes;
        bool diffuseClassHasTwoActiveLobes;
      };

      std::vector<Material> materials;

      {
        // Rough dielectric terrain -- an FNV cliff or ground. Specular probability
        // is floored at 0.25 by minOpaqueSpecularLobeSamplingProbability even
        // though the material barely reflects, so roughly half of all pixels here
        // sample the specular lobe. This is the case that failed in-game.
        Material m;
        m.name = "rough dielectric terrain";
        m.perceptualRoughness = 0.70f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseReflection] = 0.55f;
        m.lobes.selectionProbability[mirror::kLobeIndexSpecularReflection] = 0.45f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseTransmission] = 0.0f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseReflection] = 0.28f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexSpecularReflection] = 1.90f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseTransmission] = 0.0f;
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseReflection] = mirror::vec3(0.090f, 0.080f, 0.070f);
        m.lobes.projectedWeight[mirror::kLobeIndexSpecularReflection] = mirror::vec3(0.012f, 0.012f, 0.012f);
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseTransmission] = mirror::vec3();
        m.diffuseClassHasTwoActiveLobes = false;
        materials.push_back(m);
      }

      {
        // Smooth metal -- roughness BELOW the threshold, so the roughness-folded
        // flag happens to agree with the lobe class. This is why the bug hid on
        // shiny surfaces and showed up on terrain.
        Material m;
        m.name = "smooth metal";
        m.perceptualRoughness = 0.05f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseReflection] = 0.25f;
        m.lobes.selectionProbability[mirror::kLobeIndexSpecularReflection] = 0.75f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseTransmission] = 0.0f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseReflection] = 0.31f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexSpecularReflection] = 44.0f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseTransmission] = 0.0f;
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseReflection] = mirror::vec3(0.004f, 0.004f, 0.004f);
        m.lobes.projectedWeight[mirror::kLobeIndexSpecularReflection] = mirror::vec3(0.900f, 0.750f, 0.400f);
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseTransmission] = mirror::vec3();
        m.diffuseClassHasTwoActiveLobes = false;
        materials.push_back(m);
      }

      {
        // Thin-opaque subsurface (foliage). diffuseTransmission is ACTIVE, so the
        // diffuse class genuinely holds two lobes and condition (1) fails. This is
        // the residual the plan predicted; it is asserted as a bounded
        // approximation rather than as an identity, so it stays documented.
        Material m;
        m.name = "thin-opaque foliage";
        m.perceptualRoughness = 0.60f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseReflection] = 0.40f;
        m.lobes.selectionProbability[mirror::kLobeIndexSpecularReflection] = 0.20f;
        m.lobes.selectionProbability[mirror::kLobeIndexDiffuseTransmission] = 0.40f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseReflection] = 0.30f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexSpecularReflection] = 2.00f;
        m.lobes.solidAnglePdf[mirror::kLobeIndexDiffuseTransmission] = 0.25f;
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseReflection] = mirror::vec3(0.100f, 0.120f, 0.060f);
        m.lobes.projectedWeight[mirror::kLobeIndexSpecularReflection] = mirror::vec3(0.020f, 0.020f, 0.020f);
        m.lobes.projectedWeight[mirror::kLobeIndexDiffuseTransmission] = mirror::vec3(0.050f, 0.070f, 0.030f);
        m.diffuseClassHasTwoActiveLobes = true;
        materials.push_back(m);
      }

      for (const Material& m : materials) {
        for (int lobe = 0; lobe < 3; ++lobe) {
          if (m.lobes.selectionProbability[lobe] <= 0.0f) { continue; }

          // The fork's honest lobe class: a pure lobe-type test, no roughness.
          // ports restirPtClassifyScatterEvent's isSpecularLobeClass.
          const bool isSpecularLobeClass = (lobe == mirror::kLobeIndexSpecularReflection);

          const mirror::vec3 traced = mirror::traceThroughputForSampledLobe(m.lobes, lobe);
          const mirror::vec3 shifted =
            mirror::shiftPrimaryFactor(m.lobes, mirror::allowedLobeClass(isSpecularLobeClass));

          const float relative = mirror::maxRelativeDifference(shifted, traced);

          const bool classIsSingleLobe =
            !(m.diffuseClassHasTwoActiveLobes && lobe != mirror::kLobeIndexSpecularReflection);

          if (classIsSingleLobe) {
            // THE SELF-SHIFT IDENTITY. Debug view 887 paints this quantity x100,
            // so anything above 1e-2 here is a visibly non-black pixel in-game.
            check(relative < 1e-5f,
                  str::format("j: [", m.name, ", lobe ", lobe,
                              "] self-shift reproduces the traced throughput").c_str());
          } else {
            // Known, bounded approximation: two active lobes in one class. Asserted
            // in BOTH directions so it can neither silently grow nor be silently
            // fixed without updating this comment.
            check(relative > 1e-3f && relative < 0.5f,
                  str::format("j: [", m.name, ", lobe ", lobe,
                              "] two-active-lobe class is a bounded approximation").c_str());
          }
        }
      }

      // --- Re-enactment of the shipped bug ---------------------------------
      // The class the shift WOULD get if it read the roughness-folded
      // `specularBounce` flag instead of the lobe class.
      auto buggyRoughnessFoldedClass = [&](const Material& m, int lobe) {
        const bool isSpecularBounce =
          (lobe != mirror::kLobeIndexDiffuseReflection) &&
          (m.perceptualRoughness <= specularRoughnessThreshold);
        return mirror::allowedLobeClass(isSpecularBounce);
      };

      {
        const Material& terrain = materials[0];
        const int lobe = mirror::kLobeIndexSpecularReflection;

        const mirror::vec3 traced = mirror::traceThroughputForSampledLobe(terrain.lobes, lobe);
        const mirror::vec3 buggy =
          mirror::shiftPrimaryFactor(terrain.lobes, buggyRoughnessFoldedClass(terrain, lobe));

        // Not a subtle drift: on this material the wrong class is off by a factor
        // of ~40, which is why the failure read as saturated red rather than as a
        // faint tint.
        const float relative = mirror::maxRelativeDifference(buggy, traced);
        check(relative > 10.0f,
              "j: the roughness-folded class is grossly wrong for a specular sample on a rough surface");

        // ...and it is CORRECT for a diffuse sample on the same material, which is
        // exactly what made the in-game failure stippled instead of uniform: only
        // the pixels that happened to draw the specular lobe were wrong.
        const int diffuseLobe = mirror::kLobeIndexDiffuseReflection;
        const mirror::vec3 tracedDiffuse = mirror::traceThroughputForSampledLobe(terrain.lobes, diffuseLobe);
        const mirror::vec3 buggyDiffuse =
          mirror::shiftPrimaryFactor(terrain.lobes, buggyRoughnessFoldedClass(terrain, diffuseLobe));
        check(mirror::maxRelativeDifference(buggyDiffuse, tracedDiffuse) < 1e-5f,
              "j: the roughness-folded class is correct for a diffuse sample -- hence a STIPPLED failure");
      }

      {
        // On a surface smoother than the threshold the two agree for every lobe,
        // which is why the bug never showed on shiny geometry.
        const Material& metal = materials[1];
        for (int lobe = 0; lobe < 2; ++lobe) {
          const mirror::vec3 traced = mirror::traceThroughputForSampledLobe(metal.lobes, lobe);
          const mirror::vec3 buggy =
            mirror::shiftPrimaryFactor(metal.lobes, buggyRoughnessFoldedClass(metal, lobe));
          check(mirror::maxRelativeDifference(buggy, traced) < 1e-5f,
                "j: below the roughness threshold the two conventions coincide");
        }
      }

      // --- The two flag pairs must stay independent ------------------------
      // If a later change aliases the lobe class back onto the specularBounce bits
      // (or reuses 20/21 for something else), this fails.
      for (int mask = 0; mask < 16; ++mask) {
        const bool routingBefore = (mask & 1) != 0;
        const bool routingAt = (mask & 2) != 0;
        const bool classBefore = (mask & 4) != 0;
        const bool classAt = (mask & 8) != 0;

        mirror::RestirPtPathFlags f;
        f.insertPathLength(9);
        f.insertRcVertexLength(1);
        f.insertLightType(2);
        f.insertIsDeltaEvent(true, true);
        f.insertIsSpecularBounce(routingBefore, true);
        f.insertIsSpecularBounce(routingAt, false);
        f.insertIsSpecularLobeClass(classBefore, true);
        f.insertIsSpecularLobeClass(classAt, false);

        check(f.decodeIsSpecularBounce(true) == routingBefore, "j: routing-before survives the lobe class");
        check(f.decodeIsSpecularBounce(false) == routingAt, "j: routing-at survives the lobe class");
        check(f.decodeIsSpecularLobeClass(true) == classBefore, "j: lobe-class-before round-trips");
        check(f.decodeIsSpecularLobeClass(false) == classAt, "j: lobe-class-at round-trips");
        check(f.pathLength() == 9, "j: lobe class does not disturb pathLength");
        check(f.rcVertexLength() == 1, "j: lobe class does not disturb rcVertexLength");
        check(f.lightType() == 2u, "j: lobe class does not disturb lightType");
        check(f.decodeIsDeltaEvent(true), "j: lobe class does not disturb the delta bit");
      }
    }

    // -----------------------------------------------------------------------
    // (k) PHASE 3: THE ESCAPE / SKY-RECONNECTION SELF-SHIFT IDENTITY.
    //
    // The env-map branch of the shift (Shift.slang:403-432) is the one piece with
    // the least in-game exposure, and the residual hunt after Gate 1 needed it
    // ruled in or out. Structurally it is the SIMPLE branch: it never reconstructs
    // a reconnection vertex -- there is no hit to reconstruct -- so none of the
    // rebuild fidelity gaps can reach it. What it does have is a chain of
    // conventions that must line up exactly:
    //
    //   trace:  F                 = prefixThp * sky * segmentAttenuation
    //           rcVertexIrradiance = sky * segmentAttenuation   (markEscapeVertexAsRcVertex)
    //           lightPdf           = 0                          (this kernel never NEE-samples the sky)
    //   shift:  integrand          = dstF1/dstPDF1 * evalMIS(1, dstPDF1All, 1, lightPdf) * rcVertexIrradiance
    //
    // so the identity holds iff (a) dstF1/dstPDF1 reproduces prefixThp -- the lobe
    // class again, already locked by (j) -- (b) the MIS weight degenerates to
    // exactly 1 at lightPdf 0, and (c) the segment attenuation is applied EXACTLY
    // ONCE. (c) is the live hazard: the finite-segment branch multiplies the
    // destination's visibility attenuation into the integrand, and doing the same
    // here would double-count, because for an escape the source's attenuation is
    // already baked into the stored irradiance.
    // -----------------------------------------------------------------------
    void testEscapeShiftIdentity() {
      // A rough dielectric primary that scattered diffusely into the sky.
      mirror::LobeState primary;
      primary.selectionProbability[mirror::kLobeIndexDiffuseReflection] = 0.6f;
      primary.selectionProbability[mirror::kLobeIndexSpecularReflection] = 0.4f;
      primary.solidAnglePdf[mirror::kLobeIndexDiffuseReflection] = 0.26f;
      primary.solidAnglePdf[mirror::kLobeIndexSpecularReflection] = 1.40f;
      primary.projectedWeight[mirror::kLobeIndexDiffuseReflection] = mirror::vec3(0.11f, 0.10f, 0.08f);
      primary.projectedWeight[mirror::kLobeIndexSpecularReflection] = mirror::vec3(0.02f, 0.02f, 0.02f);

      const mirror::vec3 skyRadiance(3.0f, 3.6f, 5.0f);
      const mirror::vec3 segmentAttenuation(0.82f, 0.86f, 0.93f);  // fog over the escape segment

      // ports restir_pt_shift.slangh restirPtEvalMIS (PathTracer.slang:550-560,
      // balance heuristic), including the fork's zero-sum guard.
      auto evalMIS = [](float n0, float p0, float n1, float p1) {
        const float q0 = n0 * p0;
        const float q1 = n1 * p1;
        const float sum = q0 + q1;
        return (sum > 0.0f) ? (q0 / sum) : 0.0f;
      };

      for (int lobe = 0; lobe < 2; ++lobe) {
        const bool isSpecularLobeClass = (lobe == mirror::kLobeIndexSpecularReflection);
        const unsigned allowed = mirror::allowedLobeClass(isSpecularLobeClass);

        // --- trace side ---------------------------------------------------
        const mirror::vec3 prefixThp = mirror::traceThroughputForSampledLobe(primary, lobe);
        const mirror::vec3 tracedF(
          prefixThp.x * skyRadiance.x * segmentAttenuation.x,
          prefixThp.y * skyRadiance.y * segmentAttenuation.y,
          prefixThp.z * skyRadiance.z * segmentAttenuation.z);

        // What markEscapeVertexAsRcVertex stores: sky times the SOURCE segment's
        // attenuation, and lightPdf 0.
        const mirror::vec3 rcVertexIrradiance(
          skyRadiance.x * segmentAttenuation.x,
          skyRadiance.y * segmentAttenuation.y,
          skyRadiance.z * segmentAttenuation.z);
        const float lightPdf = 0.0f;

        // --- shift side ---------------------------------------------------
        float pdfAll = 0.0f;
        const float dstPDF1 = mirror::shiftEvalPdfBsdf(primary, allowed, pdfAll);
        const mirror::vec3 dstF1 = mirror::shiftEvalBsdfCosine(primary, allowed);

        const float misWeight = evalMIS(1.0f, pdfAll, 1.0f, lightPdf);
        checkClose(misWeight, 1.0f, "k: the escape MIS weight degenerates to exactly 1 at lightPdf 0");

        const mirror::vec3 shifted(
          dstF1.x / dstPDF1 * misWeight * rcVertexIrradiance.x,
          dstF1.y / dstPDF1 * misWeight * rcVertexIrradiance.y,
          dstF1.z / dstPDF1 * misWeight * rcVertexIrradiance.z);

        check(mirror::maxRelativeDifference(shifted, tracedF) < 1e-5f,
              str::format("k: [lobe ", lobe, "] the escape self-shift reproduces F exactly").c_str());
      }

      // --- The double-count hazard, stated as a test -----------------------
      // Folding the destination's visibility attenuation in here -- which is what
      // the FINITE-segment branch correctly does -- squares the fog term, because
      // the stored irradiance already carries the source's. Asserted to be wrong by
      // a margin the 887 view would show, so nobody "unifies" the two branches.
      {
        const int lobe = mirror::kLobeIndexDiffuseReflection;
        const unsigned allowed = mirror::allowedLobeClass(false);

        const mirror::vec3 prefixThp = mirror::traceThroughputForSampledLobe(primary, lobe);
        const mirror::vec3 tracedF(
          prefixThp.x * skyRadiance.x * segmentAttenuation.x,
          prefixThp.y * skyRadiance.y * segmentAttenuation.y,
          prefixThp.z * skyRadiance.z * segmentAttenuation.z);

        float pdfAll = 0.0f;
        const float dstPDF1 = mirror::shiftEvalPdfBsdf(primary, allowed, pdfAll);
        const mirror::vec3 dstF1 = mirror::shiftEvalBsdfCosine(primary, allowed);

        // The bug: attenuation applied a second time.
        const mirror::vec3 doubleCounted(
          dstF1.x / dstPDF1 * skyRadiance.x * segmentAttenuation.x * segmentAttenuation.x,
          dstF1.y / dstPDF1 * skyRadiance.y * segmentAttenuation.y * segmentAttenuation.y,
          dstF1.z / dstPDF1 * skyRadiance.z * segmentAttenuation.z * segmentAttenuation.z);

        const float relative = mirror::maxRelativeDifference(doubleCounted, tracedF);
        check(relative > 0.05f,
              "k: double-counting the escape segment attenuation is a visible error, not a rounding one");
      }

      // The env branch is only entered for a path whose FIRST scatter escaped, i.e.
      // stored pathLength 0 against the hardcoded rcVertexLength of 1. The phase 2
      // off-by-one made this 1 and the branch never fired; lock the arithmetic so
      // the length convention cannot drift back.
      {
        mirror::RestirPtPathFlags f;
        f.insertPathLength(0);
        f.insertLightType(0u);  // kRestirPtLightTypeEnvMap
        f.insertLastVertexNEE(false);

        const int shiftRcVertexLength = 1;  // kRestirPtShiftRcVertexLength
        check(f.pathLength() + 1 == shiftRcVertexLength,
              "k: a bounce-1 escape satisfies the env branch's escaped-vertex test");

        mirror::RestirPtPathFlags offByOne;
        offByOne.insertPathLength(1);
        check(offByOne.pathLength() + 1 != shiftRcVertexLength,
              "k: the phase 2 off-by-one would have missed the env branch entirely");
      }
    }

    // -----------------------------------------------------------------------
    // (l) PHASE 3: THE RECONNECTION VERTEX'S TEXTURE FOOTPRINT.
    //
    // The shift REBUILDS the reconnection vertex instead of re-tracing it, so it
    // has to reproduce the cone the trace resolved that vertex with -- otherwise
    // it reads albedo and roughness at a different mip and computes a different
    // BSDF. The trace's cone is widened by the primary scatter's own pdf; the
    // screen-space pixel cone is roughly two orders of magnitude narrower.
    //
    // The first cut of the rebuild passed the screen-space pixel spread, which
    // measured as a two-to-four mip error and was the dominant term in the debug
    // view 887 residual. This locks the derivation, and quantifies the miss so the
    // magnitude claim is on record rather than in a commit message.
    // -----------------------------------------------------------------------
    void testReconnectionVertexFootprint() {
      // rtx_options.h:844.
      const float indirectRaySpreadAngleFactor = 0.05f;
      // rtx_context.cpp:1319, fov / 2 / resolution.y. ~70 degrees at 1080p.
      const float screenSpacePixelSpreadHalfAngle = 1.22f / 2.0f / 1080.0f;

      struct Case {
        const char* name;
        float primaryScatterSolidAnglePdf;  // what cachedJacobian.x stores
        float primaryConeRadius;            // grows with distance from the camera
        float segmentLength;                // primary vertex -> reconnection vertex
        float minimumMipLevelsMissed;       // what the WRONG spread angle costs
      };

      const std::vector<Case> cases = {
        // A cosine-hemisphere diffuse bounce is the overwhelmingly common case.
        { "distant terrain, diffuse bounce",  0.22f, 3.30f, 300.0f, 2.0f },
        { "near geometry, diffuse bounce",    0.22f, 0.25f, 100.0f, 3.0f },
        { "grazing diffuse bounce",           0.05f, 1.00f, 200.0f, 3.0f },
        // A rough specular lobe is more peaked, so its cone is narrower -- but
        // still far wider than a pixel.
        { "rough specular bounce",            4.00f, 1.00f, 150.0f, 1.0f },
      };

      for (const Case& c : cases) {
        const float scatterSpreadAngle = mirror::spreadAngleFromSolidAnglePdf(
          screenSpacePixelSpreadHalfAngle, c.primaryScatterSolidAnglePdf, indirectRaySpreadAngleFactor);

        // The scatter always widens the cone; if this ever stopped being true the
        // whole concern would evaporate, so state it.
        check(scatterSpreadAngle > screenSpacePixelSpreadHalfAngle,
              str::format("l: [", c.name, "] the scatter widens the ray cone").c_str());

        const float traceCone = mirror::traceConeRadiusAtRcVertex(
          c.primaryConeRadius, scatterSpreadAngle, c.segmentLength);

        // THE FIX: rebuild with the same spread angle, derived from the stored pdf.
        const float rebuiltCone = mirror::rebuildConeRadiusAtRcVertex(
          c.primaryConeRadius, scatterSpreadAngle, c.segmentLength);

        check(rebuiltCone == traceCone,
              str::format("l: [", c.name, "] the rebuilt footprint matches the trace exactly").c_str());

        // THE BUG: rebuild with the screen-space pixel spread instead.
        const float buggyCone = mirror::rebuildConeRadiusAtRcVertex(
          c.primaryConeRadius, screenSpacePixelSpreadHalfAngle, c.segmentLength);

        // Quantified as MIP LEVELS, because that is what a cone radius ratio means
        // for a texture read and it is the number that makes the magnitude claim
        // checkable: one mip is a 2x footprint.
        const float mipLevelsMissed = std::log2(traceCone / buggyCone);

        check(mipLevelsMissed >= c.minimumMipLevelsMissed,
              str::format("l: [", c.name, "] the screen-space spread misses by at least ",
                          c.minimumMipLevelsMissed, " mip levels").c_str());
      }

      // BOTH destination-side texture consumers must use that cone, not just the
      // rebuild. The visibility ray re-reads the alpha of every cutout surface it
      // crosses; reading it at a different mip returns a different attenuation than
      // the trace applied, and on a cutout edge that is the difference between
      // alpha ~0.4 and alpha ~0 or ~1. Fixing only the rebuild leaves half the bug.
      {
        const Case& c = cases[0];
        const float scatterSpreadAngle = mirror::spreadAngleFromSolidAnglePdf(
          screenSpacePixelSpreadHalfAngle, c.primaryScatterSolidAnglePdf, indirectRaySpreadAngleFactor);

        // One value feeds the rebuild and the visibility ray alike.
        const float rebuildSpread = scatterSpreadAngle;
        const float visibilitySpread = scatterSpreadAngle;

        check(visibilitySpread == rebuildSpread,
              "l: the visibility ray uses the same cone spread as the vertex rebuild");
        check(visibilitySpread != screenSpacePixelSpreadHalfAngle,
              "l: and it is not the screen-space pixel spread");
      }

      // A dirac scatter has pdf exactly 1 by convention and must NOT be widened --
      // the cone math breaks down there, and the reference guards it the same way.
      {
        const float dirac = mirror::spreadAngleFromSolidAnglePdf(
          screenSpacePixelSpreadHalfAngle, 1.0f, indirectRaySpreadAngleFactor);
        check(dirac == screenSpacePixelSpreadHalfAngle, "l: a dirac scatter leaves the cone alone");
      }
    }

    // -----------------------------------------------------------------------
    // (m) PHASE 3: THE SELF-SHIFT PARITY METRIC ITSELF.
    //
    // Debug view 887 is a measuring instrument, and the first one shipped was not
    // trustworthy: it divided a per-CHANNEL difference by the LUMA of the stored
    // integrand, floored that denominator at 1e-4, and then saturated. On the
    // blue-dominant sky-lit paths that fill an FNV exterior it inflated readings by
    // up to 1/0.114 = 8.8x, and it turned dim reservoirs into enormous relative
    // errors out of nothing. A full round of investigation was spent on numbers it
    // had distorted, so the replacement's properties are asserted here.
    // -----------------------------------------------------------------------
    void testSelfShiftMetric() {
      bool judged = false;

      // Identical inputs read exactly zero.
      {
        const mirror::vec3 a(0.4f, 0.7f, 1.2f);
        check(mirror::selfShiftRelativeError(a, a, judged) == 0.0f, "m: identical inputs read 0");
        check(judged, "m: a bright reservoir is judged");
      }

      // Symmetric and scale-invariant: only the RATIO matters, and swapping the
      // arguments cannot change the answer.
      {
        const mirror::vec3 a(2.0f, 2.0f, 2.0f);
        const mirror::vec3 b(1.0f, 1.0f, 1.0f);
        checkClose(mirror::selfShiftRelativeError(a, b, judged), 0.5f, "m: a 2x difference reads 0.5");
        checkClose(mirror::selfShiftRelativeError(b, a, judged), 0.5f, "m: the metric is symmetric");

        const mirror::vec3 aScaled(200.0f, 200.0f, 200.0f);
        const mirror::vec3 bScaled(100.0f, 100.0f, 100.0f);
        checkClose(mirror::selfShiftRelativeError(aScaled, bScaled, judged), 0.5f,
                   "m: the metric is scale invariant");

        const mirror::vec3 fivefold(5.0f, 5.0f, 5.0f);
        checkClose(mirror::selfShiftRelativeError(fivefold, b, judged), 0.8f, "m: a 5x difference reads 0.8");
      }

      // Bounded in [0,1] for non-negative radiance, so it cannot manufacture a
      // reading no matter how extreme the inputs.
      {
        // Note: not named `small` -- rpcndr.h, reached through windows.h, #defines
        // that to `char`.
        const mirror::vec3 enormous(1e6f, 1e6f, 1e6f);
        const mirror::vec3 faint(1e-2f, 1e-2f, 1e-2f);
        const float e = mirror::selfShiftRelativeError(enormous, faint, judged);
        check(e > 0.99f && e <= 1.0f, "m: an extreme mismatch saturates at 1, it does not run away");

        const mirror::vec3 zero(0.0f, 0.0f, 0.0f);
        checkClose(mirror::selfShiftRelativeError(faint, zero, judged), 1.0f, "m: against zero reads exactly 1");
      }

      // NO CHROMA BIAS. This is the fault that cost a round of investigation: a
      // 10% error confined to the blue channel of a blue-dominant path.
      {
        const mirror::vec3 storedF(0.05f, 0.10f, 1.00f);   // sky-lit indirect, blue dominant
        const mirror::vec3 shifted(0.05f, 0.10f, 1.10f);   // 10% high in blue only

        const float honest = mirror::selfShiftRelativeError(shifted, storedF, judged);
        checkClose(honest, 0.10f / 1.10f, "m: a 10% blue-channel error reads as ~10%", 1e-3f);

        // What the retired metric said about the same pair.
        const float retired = mirror::retiredSelfShiftError(shifted, storedF);
        check(retired > 4.0f * honest,
              "m: the retired metric inflated a blue-dominant error more than fourfold");
        check(retired > 0.5f,
              "m: the retired metric read a 10% error as over 50%, which is how a small residual looked large");
      }

      // NO MANUFACTURED MAGNITUDE on dim reservoirs. The retired metric's 1e-4
      // denominator floor turned a negligible absolute difference into a huge
      // relative one; the replacement excludes the pixel instead.
      {
        const mirror::vec3 dimA(2e-4f, 2e-4f, 2e-4f);
        const mirror::vec3 dimB(0.0f, 0.0f, 0.0f);

        const float honest = mirror::selfShiftRelativeError(dimA, dimB, judged);
        check(!judged, "m: a reservoir below the minimum scale is excluded, not judged");
        check(honest == 0.0f, "m: an excluded reservoir contributes no error");

        const float retired = mirror::retiredSelfShiftError(dimA, dimB);
        check(retired > 1.0f, "m: the retired metric read that same negligible pixel as over 100%");
      }

      // A dim STORED integrand against a bright shift is still judged -- exclusion
      // must not become a way to hide a blow-up.
      {
        const mirror::vec3 bright(1.0f, 1.0f, 1.0f);
        const mirror::vec3 dim(1e-5f, 1e-5f, 1e-5f);
        const float e = mirror::selfShiftRelativeError(bright, dim, judged);
        check(judged, "m: a bright shift against a dim reservoir is still judged");
        check(e > 0.99f, "m: and it reads as a near-total mismatch");
      }
    }

    // -----------------------------------------------------------------------
    // (n) PHASE 3: THE CONNECTION SEGMENT'S ATTENUATION BELONGS TO THE PREFIX.
    //
    // This one was a real energy error in spatial reuse, not just a failing
    // harness, and it is the subtlest thing in the phase.
    //
    // The trace folds segment 1's attenuation -- alpha cutout plus volumetric --
    // into `thp`, the SUFFIX accumulator, on arrival at the reconnection vertex.
    // Any terminal offered AT that vertex (NEE there, emissive there) then builds
    // its postfix, and so the stored rcVertexIrradiance, from that suffix -- so
    // the attenuation is baked into the irradiance. The shift afterwards multiplies
    // the DESTINATION's own connection-segment attenuation into the integrand,
    // because for a shifted path that segment is new. Result: the same term twice.
    //
    // The fix is to close the prefix on ARRIVAL at the reconnection vertex, so the
    // attenuation sits in the prefix -- exactly the part the shift replaces.
    //
    // Two things are asserted: that getCurrentThp is INVARIANT under the extra
    // fold (so F, path.L and Russian roulette cannot move), and that the stored
    // irradiance is then free of the attenuation.
    // -----------------------------------------------------------------------
    void testConnectionAttenuationBelongsToPrefix() {
      // Primary scatter weight, segment-1 attenuation (a cutout leaf edge at
      // alpha 0.5 plus a little fog), and the terminal's own quantities.
      const mirror::vec3 primaryScatterWeight(0.42f, 0.38f, 0.31f);
      const mirror::vec3 segmentAttenuation(0.50f, 0.52f, 0.55f);
      const mirror::vec3 incidentRadiance(3.0f, 3.6f, 5.0f);
      const mirror::vec3 rcVertexBsdfWeight(0.20f, 0.18f, 0.15f);

      // --- the trace, up to the terminal AT the reconnection vertex ---------
      auto walkToReconnectionVertex = [&](bool foldOnArrival) {
        mirror::PathThroughput t;

        // Primary scatter: fold its weight, then close the prefix (this fold has
        // always been there).
        t.multiplySuffix(primaryScatterWeight);
        t.recordPrefix();

        // Segment 1 travelled: its attenuation lands in the SUFFIX.
        t.multiplySuffix(segmentAttenuation);

        // Arrival at the reconnection vertex. THE FIX.
        if (foldOnArrival) { t.recordPrefix(); }

        return t;
      };

      const mirror::PathThroughput fixed = walkToReconnectionVertex(true);
      const mirror::PathThroughput unfixed = walkToReconnectionVertex(false);

      // (1) INVARIANCE. getCurrentThp is what every radiance accumulation site and
      // the Russian roulette test read, so if the fold moved it, the shaded image
      // and the replay parity view would both move. Asserted BIT-EXACT, not close:
      // the fold only moves a factor from one operand of a product to the other,
      // and IEEE multiplication is commutative.
      const mirror::vec3 currentFixed = fixed.current();
      const mirror::vec3 currentUnfixed = unfixed.current();

      check(currentFixed.x == currentUnfixed.x &&
            currentFixed.y == currentUnfixed.y &&
            currentFixed.z == currentUnfixed.z,
            "n: getCurrentThp is BIT-IDENTICAL under the arrival fold");

      const mirror::vec3 expectedCurrent = mirror::mul3(primaryScatterWeight, segmentAttenuation);
      check(mirror::maxRelativeDifference(currentFixed, expectedCurrent) < 1e-6f,
            "n: and it is still the full prefix-times-suffix product");

      // (2) The suffix -- which is what a terminal at this vertex hands the builder
      // as its postfix, and hence what lands in rcVertexIrradiance -- must be free
      // of the segment attenuation after the fix, and carry it before.
      check(fixed.suffix.x == 1.0f && fixed.suffix.y == 1.0f && fixed.suffix.z == 1.0f,
            "n: after the arrival fold the suffix is exactly 1 at the reconnection vertex");
      check(mirror::maxRelativeDifference(unfixed.suffix, segmentAttenuation) < 1e-6f,
            "n: without it the suffix still carries the connection segment's attenuation");

      // (3) THE DOUBLE COUNT, end to end. Build the stored irradiance the way
      // addNeeVertex does for a terminal AT the reconnection vertex (the reference's
      // `weight = 1` exclusion, then x lightPdf / misWeight), shift it back onto the
      // same pixel, and compare against the traced F.
      const float lightPdf = 2.5f;
      const float misWeight = 1.0f;

      auto storedIrradiance = [&](const mirror::PathThroughput& t) {
        // postfix = suffix * (radiance / lightPdf) * mis, with the rc vertex's own
        // BSDF excluded; the builder then multiplies by lightPdf / mis.
        const mirror::vec3 postfix(
          t.suffix.x * incidentRadiance.x / lightPdf * misWeight,
          t.suffix.y * incidentRadiance.y / lightPdf * misWeight,
          t.suffix.z * incidentRadiance.z / lightPdf * misWeight);
        const float toIrradiance = lightPdf / misWeight;
        return mirror::vec3(postfix.x * toIrradiance, postfix.y * toIrradiance, postfix.z * toIrradiance);
      };

      // The traced F for this candidate: full throughput x the rc BSDF x radiance.
      const mirror::vec3 tracedF(
        currentFixed.x * rcVertexBsdfWeight.x * incidentRadiance.x / lightPdf,
        currentFixed.y * rcVertexBsdfWeight.y * incidentRadiance.y / lightPdf,
        currentFixed.z * rcVertexBsdfWeight.z * incidentRadiance.z / lightPdf);

      // The self-shift: the destination's primary factor (which for a self-shift is
      // the primary scatter weight), the rc BSDF over the light pdf, the stored
      // irradiance, and the connection segment's attenuation supplied by the
      // visibility ray.
      auto selfShift = [&](const mirror::vec3& irradiance) {
        return mirror::vec3(
          primaryScatterWeight.x * rcVertexBsdfWeight.x / lightPdf * irradiance.x * segmentAttenuation.x,
          primaryScatterWeight.y * rcVertexBsdfWeight.y / lightPdf * irradiance.y * segmentAttenuation.y,
          primaryScatterWeight.z * rcVertexBsdfWeight.z / lightPdf * irradiance.z * segmentAttenuation.z);
      };

      const mirror::vec3 shiftedFixed = selfShift(storedIrradiance(fixed));
      check(mirror::maxRelativeDifference(shiftedFixed, tracedF) < 1e-5f,
            "n: with the arrival fold, the self-shift of a terminal-at-rc reproduces F");

      const mirror::vec3 shiftedUnfixed = selfShift(storedIrradiance(unfixed));
      const float relative = mirror::maxRelativeDifference(shiftedUnfixed, tracedF);

      // Error is exactly (1 - A) on each channel; for a 0.5 cutout texel that is 50%.
      checkClose(relative, 1.0f - segmentAttenuation.x,
                 "n: without it the error is exactly one minus the segment attenuation", 1e-3f);
      check(relative > 0.4f,
            "n: which for a half-transparent cutout texel is a ~50% error, not a rounding one");

      // (4) The fold must not disturb candidates BEYOND the reconnection vertex:
      // the scatter fold further down still composes, and the total prefix is the
      // same product either way.
      {
        const mirror::vec3 rcScatterWeight(0.31f, 0.29f, 0.27f);

        mirror::PathThroughput afterFixed = fixed;
        afterFixed.multiplySuffix(rcScatterWeight);
        afterFixed.recordPrefix();

        mirror::PathThroughput afterUnfixed = unfixed;
        afterUnfixed.multiplySuffix(rcScatterWeight);
        afterUnfixed.recordPrefix();

        check(mirror::maxRelativeDifference(afterFixed.prefix, afterUnfixed.prefix) < 1e-5f,
              "n: past the reconnection vertex the prefix is the same product either way");
        check(afterFixed.suffix.x == 1.0f, "n: and the suffix restarts at 1 for the bounce-2 terminals");
      }
    }

    // -----------------------------------------------------------------------
    // (o) PHASE 3: a zero view direction must not become a NaN.
    //
    // The reconnection-vertex rebuild has no view direction to give until the
    // position is known, so it passes zero. `normalize(0)` is NaN, and every
    // comparison against a NaN is false -- which in the baked-terrain cascade
    // selection (surface_interaction.slangh:581) silently pins the cascade to
    // level 0 for a point that may be hundreds of metres out. Latent rather than
    // live in the scenes measured so far, but real wherever baked terrain hosts a
    // reconnection vertex.
    // -----------------------------------------------------------------------
    void testZeroViewDirectionIsSafe() {
      const mirror::vec3 zero(0.0f, 0.0f, 0.0f);
      const mirror::vec3 fallback(0.0f, 0.0f, 1.0f);

      // What plain normalize does.
      {
        const float length = std::sqrt(mirror::dot3(zero, zero));
        check(length == 0.0f, "o: a zero view direction has zero length");
        const float naive = zero.x / length;
        check(std::isnan(naive), "o: plain normalize of it is NaN");
        // The specific consequence: the threshold test silently takes the false
        // branch, so cascade 0 is chosen without anything looking wrong.
        check(!(naive >= 0.5f), "o: and every comparison against that NaN is false");
      }

      // What safeNormalize does.
      {
        const mirror::vec3 safe = mirror::normalizeOrZ(zero);
        check(safe.x == fallback.x && safe.y == fallback.y && safe.z == fallback.z,
              "o: safeNormalize returns the fallback instead");
        check(!std::isnan(safe.x) && !std::isnan(safe.y) && !std::isnan(safe.z),
              "o: and the result is finite, so the comparison below it is meaningful");
      }

      // A real direction is untouched by the guard.
      {
        const mirror::vec3 real(0.0f, 3.0f, 4.0f);
        const mirror::vec3 safe = mirror::normalizeOrZ(real);
        checkClose(safe.y, 0.6f, "o: a real direction normalizes normally");
        checkClose(safe.z, 0.8f, "o: a real direction normalizes normally (z)");
      }
    }

    // -----------------------------------------------------------------------
    // (p) PHASE 4: TALBOT RESAMPLING MIS IS UNBIASED, and unbiased INDEPENDENTLY
    // OF THE CONFIDENCE WEIGHTS.
    //
    // The temporal analogue of (h), on the same enumerate-every-draw harness.
    // Two candidates -- the current frame's reservoir and the previous frame's --
    // over ONE shared discrete path domain, so shifting is the identity with
    // Jacobian 1 and "the temporal path shifted onto the central pixel" is simply
    // F_c[j]. The central pixel then runs the Talbot combine exactly as
    // fork_restir_pt_temporal_reuse.comp.slang does.
    //
    //   assert  E[F_c(Y) * W_out]  ==  SUM_j F_c(j) / p_j
    //
    // The SECOND assertion is the one that matters most for this phase, and it is
    // why every case is run at several history lengths: the M-cap changes M_T, M_T
    // appears in both Talbot denominators, and the result must not move at all.
    // A drift here would show in-game as "the image gets brighter (or dimmer) as
    // rtx.restirPT.temporalHistoryLength is raised" -- which is exactly the
    // symptom the phase's A/B watches for, and which is otherwise indistinguishable
    // from a tuning preference.
    //
    // What this catches: the two Talbot denominators swapped (they are NOT
    // symmetric -- one divides by the Jacobian, the other multiplies), p_self taken
    // from the wrong reservoir, a missing confidence weight on either term, and
    // finalizeRIS used where finalizeGRIS belongs (which reintroduces a division by
    // M and darkens everything by the accumulated history length).
    // -----------------------------------------------------------------------
    void testTalbotResamplingMISIsUnbiased() {
      struct Outcome {
        int index = 0;
        double probability = 0.0;
        mirror::RestirPtReservoir reservoir;
      };

      // Same exhaustive streaming build (p) shares with (h): each candidate is
      // offered in order, the selected index survives with probability w_j / w_sum.
      auto buildPixel = [](const std::vector<mirror::vec3>& F, const std::vector<float>& p) {
        std::vector<Outcome> outcomes;

        mirror::RestirPtReservoir reference;
        reference.init();
        for (size_t j = 0; j < F.size(); ++j) {
          reference.add(F[j], p[j], 1.0f);
        }

        const float wSum = reference.weight;

        for (size_t j = 0; j < F.size(); ++j) {
          const float w = mirror::RestirPtReservoir::toScalar(F[j]) / p[j];
          if (!(w > 0.0f)) { continue; }

          Outcome o;
          o.index = int(j);
          o.probability = double(w) / double(wSum);

          o.reservoir.init();
          o.reservoir.F = F[j];
          o.reservoir.weight = wSum;
          o.reservoir.M = 1.0f;
          o.reservoir.srcPixel = uint32_t(j);
          mirror::pathBuilderFinalize(o.reservoir);
          o.reservoir.finalizeRIS();

          outcomes.push_back(o);
        }

        return outcomes;
      };

      struct Case {
        const char* name;
        std::vector<mirror::vec3> centralF;   // the current frame's integrands
        std::vector<mirror::vec3> temporalF;  // the previous frame's, same domain
        std::vector<float> p;
      };

      std::vector<Case> cases;

      cases.push_back({
        "ordinary",
        { mirror::vec3(1.0f, 0.5f, 0.25f), mirror::vec3(0.2f, 2.0f, 0.1f), mirror::vec3(0.7f, 0.7f, 3.0f) },
        { mirror::vec3(0.8f, 0.6f, 0.30f), mirror::vec3(0.3f, 1.4f, 0.2f), mirror::vec3(0.5f, 0.9f, 2.1f) },
        { 1.0f, 0.8f, 0.5f }
      });

      // A path the previous frame could not see at all. Its Talbot weight there is
      // 0 and the canonical weight must absorb the whole of it -- the temporal twin
      // of (h)'s shift-to-zero neighbour, and the case a "both weights are 1/2"
      // simplification silently gets wrong.
      cases.push_back({
        "history blind to a path",
        { mirror::vec3(2.0f, 1.0f, 0.5f), mirror::vec3(0.4f, 0.4f, 4.0f), mirror::vec3(1.1f, 0.2f, 0.9f) },
        { mirror::vec3(0.1f, 0.1f, 0.1f), mirror::vec3(0.0f, 0.0f, 0.0f), mirror::vec3(3.0f, 1.0f, 0.5f) },
        { 1.0f, 0.64f, 0.32f }
      });

      cases.push_back({
        "wide dynamic range",
        { mirror::vec3(0.05f, 0.05f, 0.05f), mirror::vec3(9.0f, 0.1f, 0.1f) },
        { mirror::vec3(1.50f, 0.02f, 0.20f), mirror::vec3(0.3f, 0.3f, 6.0f) },
        { 0.9f, 0.45f }
      });

      // The M-cap's output. 1 is "no history yet", 20 is the shipped default, 200
      // is the far end of the slider.
      const std::vector<float> historyMs = { 1.0f, 3.0f, 20.0f, 200.0f };

      for (const Case& c : cases) {
        const int pathCount = int(c.centralF.size());

        // The target: what the central pixel's own estimator already estimates.
        double refX = 0.0, refY = 0.0, refZ = 0.0;
        for (int j = 0; j < pathCount; ++j) {
          refX += double(c.centralF[j].x) / double(c.p[j]);
          refY += double(c.centralF[j].y) / double(c.p[j]);
          refZ += double(c.centralF[j].z) / double(c.p[j]);
        }

        const std::vector<Outcome> centralOutcomes = buildPixel(c.centralF, c.p);
        const std::vector<Outcome> temporalOutcomes = buildPixel(c.temporalF, c.p);

        for (const float historyM : historyMs) {
          double totalProbability = 0.0;
          double eX = 0.0, eY = 0.0, eZ = 0.0;

          for (const Outcome& central : centralOutcomes) {
            for (const Outcome& temporalRaw : temporalOutcomes) {
              const double jointProbability = central.probability * temporalRaw.probability;

              const mirror::RestirPtReservoir centralReservoir = central.reservoir;
              const int centralPath = central.index;

              mirror::RestirPtReservoir temporalReservoir = temporalRaw.reservoir;
              const int temporalPath = temporalRaw.index;

              // The confidence weight the M-cap produced after some number of
              // accumulated frames. What (p) asserts is that the estimator does not
              // depend on it AT ALL; the cap arithmetic itself is locked in (q).
              // currentM is 1 here, as it is in-game before any reuse has run.
              const float currentM = centralReservoir.M;
              temporalReservoir.M = historyM;

              struct MergeState { mirror::RestirPtReservoir dst; double probability; };

              std::vector<MergeState> states;
              {
                MergeState s;
                s.dst.init();
                s.probability = 1.0;
                states.push_back(s);
              }

              // --- i = curSampleId ------------------------------------------
              {
                const mirror::RestirPtReservoir tempDst = centralReservoir;
                const bool possible = tempDst.weight > 0.0f;

                float pSum = 0.0f;
                float pSelf = 0.0f;

                if (possible) {
                  pSelf = mirror::RestirPtReservoir::computeWeight(tempDst.F, false) * currentM;
                  pSum = pSelf;

                  // The SECOND shift: the current sample projected into the
                  // temporal domain. Identity here, so it is F_T at the central
                  // pixel's path.
                  pSum += mirror::RestirPtReservoir::computeWeight(c.temporalF[centralPath], false) *
                          1.0f * temporalReservoir.M;
                }

                const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

                std::vector<MergeState> next;

                for (const MergeState& s : states) {
                  const float w = mirror::RestirPtReservoir::toScalar(tempDst.F) * 1.0f *
                                  tempDst.weight * misWeight;
                  const float wSumAfter = s.dst.weight + w;
                  const double pAccept = (!(w > 0.0f) || std::isnan(w))
                    ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

                  if (pAccept > 0.0) {
                    MergeState accepted = s;
                    accepted.dst.mergeWithResamplingMIS(tempDst.F, 1.0f, tempDst, 0.0f, misWeight, false);
                    accepted.probability = s.probability * pAccept;
                    next.push_back(accepted);
                  }

                  if (pAccept < 1.0) {
                    MergeState rejected = s;
                    rejected.dst.mergeWithResamplingMIS(tempDst.F, 1.0f, tempDst, 1.0f, misWeight, false);
                    rejected.probability = s.probability * (1.0 - pAccept);
                    next.push_back(rejected);
                  }
                }

                states = next;
              }

              // --- i = prevSampleId -----------------------------------------
              {
                std::vector<MergeState> next;

                for (const MergeState& s : states) {
                  // shiftAndMergeReservoir with forceMerge. THE FIRST SHIFT:
                  // temporal -> central, identity here, so F_c at the temporal
                  // pixel's path.
                  const mirror::vec3 shiftedIntegrand = c.centralF[temporalPath];
                  const float dstJacobian = 1.0f;

                  mirror::RestirPtReservoir temp = s.dst;
                  const bool selected = temp.merge(shiftedIntegrand, dstJacobian, temporalReservoir,
                                                   /*acceptRnd=*/ 0.0f, /*misWeight=*/ 1.0f, /*forceAdd=*/ true);
                  if (!selected) { temp.F = mirror::vec3(); }
                  temp.M = temporalReservoir.M;
                  temp.weight = temporalReservoir.weight;

                  float pSum = 0.0f;
                  float pSelf = 0.0f;

                  if (selected) {
                    pSum = mirror::RestirPtReservoir::computeWeight(temp.F, false) * currentM;
                    // Mirrors THE THIRD SHIFT (the consistent-target rule): the shader
                    // re-evaluates the history path's target on the reconstructed
                    // temporal surface instead of reading temporalReservoir.F. In this
                    // test's identity world the re-evaluation IS the stored value
                    // (c.temporalF[temporalPath] == temporalReservoir.F by
                    // construction), which is exactly the blind spot group (r) covers
                    // with an adversarial re-evaluation table.
                    pSelf = mirror::RestirPtReservoir::computeWeight(c.temporalF[temporalPath], false) /
                            dstJacobian * temporalReservoir.M;
                    pSum += pSelf;
                  }

                  const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

                  const float w = mirror::RestirPtReservoir::toScalar(temp.F) * dstJacobian *
                                  temp.weight * misWeight;
                  const float wSumAfter = s.dst.weight + w;
                  const double pAccept = (!(w > 0.0f) || std::isnan(w))
                    ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

                  if (pAccept > 0.0) {
                    MergeState accepted = s;
                    accepted.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 0.0f, misWeight, false);
                    accepted.probability = s.probability * pAccept;
                    next.push_back(accepted);
                  }

                  if (pAccept < 1.0) {
                    MergeState rejected = s;
                    rejected.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 1.0f, misWeight, false);
                    rejected.probability = s.probability * (1.0 - pAccept);
                    next.push_back(rejected);
                  }
                }

                states = next;
              }

              // ports :222-225 and :298 -- finalizeGRIS, NOT finalizeRIS, and no
              // division by a candidate count: Talbot's weights already form a
              // partition of unity, which is the whole difference from pairwise.
              for (MergeState& s : states) {
                if (s.dst.weight > 0.0f) { s.dst.finalizeGRIS(); }

                if (s.dst.weight < 0.0f) { s.dst.weight = 0.0f; }
                if (std::isnan(s.dst.weight) || std::isinf(s.dst.weight)) { s.dst.weight = 0.0f; }

                const double probability = jointProbability * s.probability;

                totalProbability += probability;
                eX += probability * double(s.dst.F.x) * double(s.dst.weight);
                eY += probability * double(s.dst.F.y) * double(s.dst.weight);
                eZ += probability * double(s.dst.F.z) * double(s.dst.weight);
              }
            }
          }

          const std::string label = str::format("p: [", c.name, ", M_T ", int(historyM), "] ");

          checkClose(float(totalProbability), 1.0f,
                     (label + "outcome probabilities sum to 1").c_str(), 1e-4f);
          checkClose(float(eX), float(refX),
                     (label + "E[F.x * W] == sum F_c.x / p").c_str(), 1e-3f);
          checkClose(float(eY), float(refY),
                     (label + "E[F.y * W] == sum F_c.y / p").c_str(), 1e-3f);
          checkClose(float(eZ), float(refZ),
                     (label + "E[F.z * W] == sum F_c.z / p").c_str(), 1e-3f);
        }
      }
    }

    // -----------------------------------------------------------------------
    // (q) PHASE 4: the Talbot weights are a partition of unity, and the M-cap
    // does not disturb it.
    //
    // The algebraic half of (p): it holds pointwise, for any target-function
    // values and any pair of confidence weights, which is what makes Talbot need
    // NO deferred normalisation of the kind pairwise MIS needs. The one way to
    // break it is to make the two denominators disagree -- e.g. by dividing by the
    // Jacobian on one side and forgetting to multiply on the other.
    // -----------------------------------------------------------------------
    void testTalbotWeightsPartitionAndMCap() {
      struct Sample { float centralPHat; float temporalPHat; float jacobian; };

      const std::vector<Sample> samples = {
        { 1.30f, 0.70f, 1.00f },
        { 0.02f, 5.00f, 0.25f },
        { 4.00f, 0.01f, 3.75f },
        { 1.00f, 0.00f, 1.00f },  // history blind to this path
      };

      const std::vector<std::pair<float, float>> confidences = {
        { 1.0f, 1.0f }, { 1.0f, 20.0f }, { 1.0f, 200.0f }, { 3.0f, 7.0f },
      };

      for (const Sample& s : samples) {
        for (const std::pair<float, float>& mc : confidences) {
          const float currentM = mc.first;
          const float temporalM = mc.second;

          // Both denominators are the same sum, expressed in the central pixel's
          // measure: the central term direct, the temporal term converted.
          const float centralTerm = s.centralPHat * currentM;
          const float temporalTerm = (s.temporalPHat / s.jacobian) * temporalM;
          const float denominator = centralTerm + temporalTerm;

          if (denominator <= 0.0f) { continue; }

          const float misCentral = centralTerm / denominator;
          const float misTemporal = temporalTerm / denominator;

          checkClose(misCentral + misTemporal, 1.0f,
                     "q: Talbot MIS weights sum to 1 with no deferred normalisation");

          if (s.temporalPHat == 0.0f) {
            checkClose(misCentral, 1.0f, "q: a history blind to the path leaves the canonical weight whole");
            checkClose(misTemporal, 0.0f, "q: and takes none for itself");
          }
        }
      }

      // The M-cap itself: history confidence is clamped to a multiple of the
      // current reservoir's, never raised by it. ports TemporalReuse.cs.slang:139.
      {
        const float historyLength = 20.0f;

        auto cap = [historyLength](float currentM, float temporalM) {
          return std::min(historyLength * currentM, temporalM);
        };

        checkClose(cap(1.0f, 5.0f), 5.0f, "q: an under-cap history is left alone");
        checkClose(cap(1.0f, 50.0f), 20.0f, "q: an over-cap history is clamped");
        checkClose(cap(1.0f, 20.0f), 20.0f, "q: exactly at the cap is a fixed point");
        checkClose(cap(0.0f, 50.0f), 0.0f,
                   "q: a pixel whose current reservoir found nothing keeps no history either");
      }

      // Both Talbot denominators must be THE SAME SUM. Locking this directly,
      // because the two are written out separately in the shader (once per
      // candidate) and a transcription slip in either is invisible in the partition
      // check above, which only ever sees one of them.
      {
        const float centralPHat = 1.3f, temporalPHat = 0.7f, jacobian = 2.5f;
        const float currentM = 1.0f, temporalM = 20.0f;

        // Candidate "current": p_self direct, other term is the current sample
        // shifted INTO the temporal domain, hence x jacobian.
        const float sumFromCurrentSide =
          centralPHat * currentM + (temporalPHat * jacobian) * temporalM;

        // Candidate "temporal": p_self is the temporal target converted OUT of the
        // temporal domain, hence / jacobian... which is only the same sum when the
        // two Jacobians are reciprocal, as they are for a shift and its inverse.
        const float inverseJacobian = 1.0f / jacobian;
        const float sumFromTemporalSide =
          centralPHat * currentM + (temporalPHat / inverseJacobian) * temporalM;

        checkClose(sumFromCurrentSide, sumFromTemporalSide,
                   "q: both Talbot denominators are the same sum under a reciprocal Jacobian");
      }
    }

    // -----------------------------------------------------------------------
    // (r) PHASE 4 FIX: TALBOT STAYS UNBIASED UNDER AN ASYMMETRIC SHIFT FAILURE.
    //
    // The blind spot of (p), closed. (p) verified the Talbot arithmetic with
    // IDENTITY shifts, where the stored history integrand and its re-evaluation
    // on the temporal surface are the same number by construction -- so it could
    // not see the in-game energy runaway, whose mechanism is exactly their
    // disagreement. Here the two are separate tables:
    //
    //   temporalTrueF[j]  what the REAL previous surface computed -- the stored
    //                     history integrand, and the history pixel's true target.
    //   reevalF[j]        what the RECONSTRUCTED temporal surface computes when
    //                     the shift machinery re-evaluates path j against it
    //                     (visibility ray from the rebuilt origin, borrowed
    //                     material). q_T in the shader's terms.
    //
    // The PRE-FIX shader mixed them: the canonical candidate's MIS denominator
    // used reevalF while the temporal candidate's numerator used temporalTrueF.
    // Where reevalF spuriously reports zero (the visibility ray from the rebuilt
    // origin fails) the canonical weight becomes 1 while the temporal weight
    // stays ~M_T/(M_T+M_C): the weights sum to nearly 2 and the estimator
    // inflates. The FIXED shader (THE CONSISTENT-TARGET RULE) computes both from
    // reevalF, so the weights partition unity for ANY disagreement.
    //
    // Both semantics are enumerated here, each against its own closed-form
    // prediction, so the test permanently locks BOTH that the fix is exact and
    // that the pre-fix arithmetic inflates by the predicted amount -- the
    // mechanism stays pinned even after the code that had it is gone.
    //
    //   assert  E[F_c(Y) * W_out]  ==  SUM_j (m_C(j) + m_T(j)) F_c(j) / p_j
    //
    // with, per path j (currentM == 1, J_j == |d central / d temporal|):
    //   cM        = phat_c(j)                          [central target]
    //   qTerm     = phat_reeval(j) / J_j * M_T         [re-evaluated, both slots, FIXED]
    //   sTerm     = phat_stored(j) / J_j * M_T         [stored, numerator slot, PRE-FIX]
    //   m_C(j)    = cM / (cM + qTerm)
    //   m_T(j)    = num / (cM + num),  num = qTerm (fixed) or sTerm (pre-fix),
    //               and 0 unless the history can hold j and the t->c shift lands.
    //
    // FIXED semantics therefore satisfies m_C + m_T == 1 wherever the temporal
    // candidate is live, i.e. E == SUM_j F_c(j)/p_j exactly -- EXCEPT on
    // "overclaim" paths (reevalF > 0 where temporalTrueF == 0): there the
    // temporal candidate cannot exist, m_C < 1 discounts the canonical with no
    // compensation, and the estimator shows the DECLARED residual deficit --
    // asserted numerically below, never a surplus.
    //
    // The temporal candidates are built over p'_j = p_j * J_j, which is what "the
    // same underlying paths, parametrised in the temporal domain's measure" means
    // in a discrete mirror; it is exactly the condition under which the shifted
    // UCW bookkeeping closes, and it makes the /dstJacobian in pSelf and the
    // *1/J in the canonical's denominator load-bearing rather than decorative.
    // -----------------------------------------------------------------------
    void testTalbotAsymmetricShiftIsUnbiased() {
      struct Outcome {
        int index = 0;
        double probability = 0.0;
        mirror::RestirPtReservoir reservoir;
      };

      // The same exhaustive streaming build (p) uses.
      auto buildPixel = [](const std::vector<mirror::vec3>& F, const std::vector<float>& p) {
        std::vector<Outcome> outcomes;

        mirror::RestirPtReservoir reference;
        reference.init();
        for (size_t j = 0; j < F.size(); ++j) {
          reference.add(F[j], p[j], 1.0f);
        }

        const float wSum = reference.weight;

        for (size_t j = 0; j < F.size(); ++j) {
          const float w = mirror::RestirPtReservoir::toScalar(F[j]) / p[j];
          if (!(w > 0.0f)) { continue; }

          Outcome o;
          o.index = int(j);
          o.probability = double(w) / double(wSum);

          o.reservoir.init();
          o.reservoir.F = F[j];
          o.reservoir.weight = wSum;
          o.reservoir.M = 1.0f;
          o.reservoir.srcPixel = uint32_t(j);
          mirror::pathBuilderFinalize(o.reservoir);
          o.reservoir.finalizeRIS();

          outcomes.push_back(o);
        }

        return outcomes;
      };

      const std::vector<mirror::vec3> centralF = {
        mirror::vec3(1.0f, 0.5f, 0.25f), mirror::vec3(0.2f, 2.0f, 0.1f), mirror::vec3(0.7f, 0.7f, 3.0f)
      };
      const std::vector<float> p = { 1.0f, 0.8f, 0.5f };

      const std::vector<mirror::vec3> faithful = {
        mirror::vec3(0.8f, 0.6f, 0.30f), mirror::vec3(0.3f, 1.4f, 0.2f), mirror::vec3(0.5f, 0.9f, 2.1f)
      };

      struct Case {
        const char* name;
        std::vector<mirror::vec3> temporalTrueF;
        std::vector<mirror::vec3> reevalF;
        std::vector<float> jacobian;
      };

      std::vector<Case> cases;

      // The in-game mechanism: the re-evaluation spuriously reports one path
      // impossible (a visibility ray from the rebuilt origin self-intersects).
      cases.push_back({
        "spurious re-evaluation failure",
        faithful,
        { mirror::vec3(0.0f), faithful[1], faithful[2] },
        { 1.0f, 1.0f, 1.0f }
      });

      // The whole surface is broken: every re-evaluation returns zero. The fixed
      // pass must degenerate to the canonical sample at weight 1 -- unbiased.
      cases.push_back({
        "surface-wide re-evaluation failure",
        faithful,
        { mirror::vec3(0.0f), mirror::vec3(0.0f), mirror::vec3(0.0f) },
        { 1.0f, 1.0f, 1.0f }
      });

      // A continuous 2x underestimate (material drift, wrong mip): not a support
      // change, still an asymmetry the pre-fix mixing turns into energy.
      cases.push_back({
        "re-evaluation underestimates 2x",
        faithful,
        { mirror::vec3(0.4f, 0.3f, 0.15f), mirror::vec3(0.15f, 0.7f, 0.1f), mirror::vec3(0.25f, 0.45f, 1.05f) },
        { 1.0f, 1.0f, 1.0f }
      });

      // The converse disagreement: the re-evaluation CLAIMS a path the real
      // history could never hold. This is the fix's declared residual -- a
      // deficit, asserted exactly, never a surplus.
      cases.push_back({
        "re-evaluation overclaims a dead path",
        { mirror::vec3(0.1f, 0.1f, 0.1f), mirror::vec3(0.0f), mirror::vec3(3.0f, 1.0f, 0.5f) },
        { mirror::vec3(0.1f, 0.1f, 0.1f), mirror::vec3(0.3f, 1.4f, 0.2f), mirror::vec3(3.0f, 1.0f, 0.5f) },
        { 1.0f, 1.0f, 1.0f }
      });

      // Non-unit Jacobians plus a spurious failure: catches a dropped
      // /dstJacobian in pSelf or a dropped *1/J in the canonical's denominator.
      cases.push_back({
        "spurious failure under non-unit Jacobians",
        faithful,
        { faithful[0], faithful[1], mirror::vec3(0.0f) },
        { 1.0f, 2.0f, 0.25f }
      });

      const std::vector<float> historyMs = { 1.0f, 3.0f, 20.0f, 200.0f };

      // Closed-form prediction of the enumerated estimator, either semantics.
      auto predict = [&](const Case& c, float historyM, bool useStoredHistoryTarget,
                         double& outX, double& outY, double& outZ) {
        outX = outY = outZ = 0.0;
        for (size_t j = 0; j < centralF.size(); ++j) {
          const double cM = double(mirror::RestirPtReservoir::toScalar(centralF[j])) * 1.0;
          const double qTerm = double(mirror::RestirPtReservoir::toScalar(c.reevalF[j])) /
                               double(c.jacobian[j]) * double(historyM);
          const double sTerm = double(mirror::RestirPtReservoir::toScalar(c.temporalTrueF[j])) /
                               double(c.jacobian[j]) * double(historyM);

          const double mC = (cM > 0.0) ? cM / (cM + qTerm) : 0.0;

          const bool historyCanHold = mirror::RestirPtReservoir::toScalar(c.temporalTrueF[j]) > 0.0f;
          const bool shiftLands = mirror::RestirPtReservoir::toScalar(centralF[j]) > 0.0f && c.jacobian[j] > 0.0f;

          double mT = 0.0;
          if (historyCanHold && shiftLands) {
            const double num = useStoredHistoryTarget ? sTerm : qTerm;
            mT = (num > 0.0) ? num / (cM + num) : 0.0;
          }

          const double share = mC + mT;
          outX += share * double(centralF[j].x) / double(p[j]);
          outY += share * double(centralF[j].y) / double(p[j]);
          outZ += share * double(centralF[j].z) / double(p[j]);
        }
      };

      // The enumerated Talbot combine, (p)'s harness with the two-table split.
      auto enumerate = [&](const Case& c, float historyM, bool useStoredHistoryTarget,
                           double& totalProbability, double& eX, double& eY, double& eZ) {
        std::vector<float> temporalP(p.size());
        for (size_t j = 0; j < p.size(); ++j) { temporalP[j] = p[j] * c.jacobian[j]; }

        const std::vector<Outcome> centralOutcomes = buildPixel(centralF, p);
        const std::vector<Outcome> temporalOutcomes = buildPixel(c.temporalTrueF, temporalP);

        totalProbability = 0.0;
        eX = eY = eZ = 0.0;

        for (const Outcome& central : centralOutcomes) {
          for (const Outcome& temporalRaw : temporalOutcomes) {
            const double jointProbability = central.probability * temporalRaw.probability;

            const mirror::RestirPtReservoir centralReservoir = central.reservoir;
            const int centralPath = central.index;

            mirror::RestirPtReservoir temporalReservoir = temporalRaw.reservoir;
            const int temporalPath = temporalRaw.index;

            const float currentM = centralReservoir.M;
            temporalReservoir.M = historyM;

            struct MergeState { mirror::RestirPtReservoir dst; double probability; };

            std::vector<MergeState> states;
            {
              MergeState s;
              s.dst.init();
              s.probability = 1.0;
              states.push_back(s);
            }

            // --- i = curSampleId --------------------------------------------
            {
              const mirror::RestirPtReservoir tempDst = centralReservoir;
              const bool possible = tempDst.weight > 0.0f;

              float pSum = 0.0f;
              float pSelf = 0.0f;

              if (possible) {
                pSelf = mirror::RestirPtReservoir::computeWeight(tempDst.F, false) * currentM;
                pSum = pSelf;

                // The central -> temporal re-evaluation: reevalF, times the
                // c->t Jacobian 1/J_j. Both semantics use this term -- the
                // pre-fix bug was never here.
                pSum += mirror::RestirPtReservoir::computeWeight(c.reevalF[centralPath], false) *
                        (1.0f / c.jacobian[centralPath]) * temporalReservoir.M;
              }

              const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

              std::vector<MergeState> next;

              for (const MergeState& s : states) {
                const float w = mirror::RestirPtReservoir::toScalar(tempDst.F) * 1.0f *
                                tempDst.weight * misWeight;
                const float wSumAfter = s.dst.weight + w;
                const double pAccept = (!(w > 0.0f) || std::isnan(w))
                  ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

                if (pAccept > 0.0) {
                  MergeState accepted = s;
                  accepted.dst.mergeWithResamplingMIS(tempDst.F, 1.0f, tempDst, 0.0f, misWeight, false);
                  accepted.probability = s.probability * pAccept;
                  next.push_back(accepted);
                }

                if (pAccept < 1.0) {
                  MergeState rejected = s;
                  rejected.dst.mergeWithResamplingMIS(tempDst.F, 1.0f, tempDst, 1.0f, misWeight, false);
                  rejected.probability = s.probability * (1.0 - pAccept);
                  next.push_back(rejected);
                }
              }

              states = next;
            }

            // --- i = prevSampleId -------------------------------------------
            {
              std::vector<MergeState> next;

              for (const MergeState& s : states) {
                // THE FIRST SHIFT, temporal -> central: lands on the central
                // pixel's integrand for that path, with the t->c Jacobian J_j.
                // Evaluated at the REAL central surface, so it is faithful.
                const mirror::vec3 shiftedIntegrand = centralF[temporalPath];
                const float dstJacobian = c.jacobian[temporalPath];

                mirror::RestirPtReservoir temp = s.dst;
                const bool selected = temp.merge(shiftedIntegrand, dstJacobian, temporalReservoir,
                                                 /*acceptRnd=*/ 0.0f, /*misWeight=*/ 1.0f, /*forceAdd=*/ true);
                if (!selected) { temp.F = mirror::vec3(); }
                temp.M = temporalReservoir.M;
                temp.weight = temporalReservoir.weight;

                float pSum = 0.0f;
                float pSelf = 0.0f;

                if (selected) {
                  pSum = mirror::RestirPtReservoir::computeWeight(temp.F, false) * currentM;

                  // THE CONTESTED SLOT. Fixed semantics: THE THIRD SHIFT's
                  // re-evaluation (reevalF, self-Jacobian 1). Pre-fix semantics:
                  // the stored history integrand. The entire energy-runaway bug
                  // is this one line's choice of table.
                  const float selfTarget = useStoredHistoryTarget
                    ? mirror::RestirPtReservoir::computeWeight(temporalReservoir.F, false)
                    : mirror::RestirPtReservoir::computeWeight(c.reevalF[temporalPath], false);

                  pSelf = selfTarget / dstJacobian * temporalReservoir.M;
                  pSum += pSelf;
                }

                const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);

                const float w = mirror::RestirPtReservoir::toScalar(temp.F) * dstJacobian *
                                temp.weight * misWeight;
                const float wSumAfter = s.dst.weight + w;
                const double pAccept = (!(w > 0.0f) || std::isnan(w))
                  ? 0.0 : std::min(1.0, double(w) / double(wSumAfter));

                if (pAccept > 0.0) {
                  MergeState accepted = s;
                  accepted.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 0.0f, misWeight, false);
                  accepted.probability = s.probability * pAccept;
                  next.push_back(accepted);
                }

                if (pAccept < 1.0) {
                  MergeState rejected = s;
                  rejected.dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, 1.0f, misWeight, false);
                  rejected.probability = s.probability * (1.0 - pAccept);
                  next.push_back(rejected);
                }
              }

              states = next;
            }

            for (MergeState& s : states) {
              if (s.dst.weight > 0.0f) { s.dst.finalizeGRIS(); }

              if (s.dst.weight < 0.0f) { s.dst.weight = 0.0f; }
              if (std::isnan(s.dst.weight) || std::isinf(s.dst.weight)) { s.dst.weight = 0.0f; }

              const double probability = jointProbability * s.probability;

              totalProbability += probability;
              eX += probability * double(s.dst.F.x) * double(s.dst.weight);
              eY += probability * double(s.dst.F.y) * double(s.dst.weight);
              eZ += probability * double(s.dst.F.z) * double(s.dst.weight);
            }
          }
        }
      };

      // The unbiased target.
      double truthX = 0.0, truthY = 0.0, truthZ = 0.0;
      for (size_t j = 0; j < centralF.size(); ++j) {
        truthX += double(centralF[j].x) / double(p[j]);
        truthY += double(centralF[j].y) / double(p[j]);
        truthZ += double(centralF[j].z) / double(p[j]);
      }

      for (const Case& c : cases) {
        bool hasOverclaim = false;
        bool hasUnderclaim = false;
        for (size_t j = 0; j < centralF.size(); ++j) {
          const float stored = mirror::RestirPtReservoir::toScalar(c.temporalTrueF[j]);
          const float reeval = mirror::RestirPtReservoir::toScalar(c.reevalF[j]);
          if (reeval > 0.0f && stored == 0.0f) { hasOverclaim = true; }
          if (reeval < stored) { hasUnderclaim = true; }
        }

        for (const float historyM : historyMs) {
          for (const bool useStored : { false, true }) {
            double predX = 0.0, predY = 0.0, predZ = 0.0;
            predict(c, historyM, useStored, predX, predY, predZ);

            double totalProbability = 0.0, eX = 0.0, eY = 0.0, eZ = 0.0;
            enumerate(c, historyM, useStored, totalProbability, eX, eY, eZ);

            const std::string label = str::format(
              "r: [", c.name, ", M_T ", int(historyM), ", ", useStored ? "stored" : "fixed", "] ");

            checkClose(float(totalProbability), 1.0f,
                       (label + "outcome probabilities sum to 1").c_str(), 1e-4f);
            checkClose(float(eX), float(predX),
                       (label + "E[F.x * W] matches the closed-form prediction").c_str(), 1e-3f);
            checkClose(float(eY), float(predY),
                       (label + "E[F.y * W] matches the closed-form prediction").c_str(), 1e-3f);
            checkClose(float(eZ), float(predZ),
                       (label + "E[F.z * W] matches the closed-form prediction").c_str(), 1e-3f);

            if (!useStored) {
              if (!hasOverclaim) {
                // The fix's guarantee: exact unbiasedness under ANY underclaiming
                // disagreement, including total failure.
                checkClose(float(eX), float(truthX), (label + "E[F.x * W] == sum F_c.x / p").c_str(), 1e-3f);
                checkClose(float(eY), float(truthY), (label + "E[F.y * W] == sum F_c.y / p").c_str(), 1e-3f);
                checkClose(float(eZ), float(truthZ), (label + "E[F.z * W] == sum F_c.z / p").c_str(), 1e-3f);
              } else {
                // The declared residual: a deficit, exactly (1 - m_C) of each
                // dead-but-claimed path's canonical share, and NEVER a surplus.
                double deficitX = 0.0, deficitY = 0.0, deficitZ = 0.0;
                for (size_t j = 0; j < centralF.size(); ++j) {
                  const float stored = mirror::RestirPtReservoir::toScalar(c.temporalTrueF[j]);
                  const float reeval = mirror::RestirPtReservoir::toScalar(c.reevalF[j]);
                  if (!(reeval > 0.0f && stored == 0.0f)) { continue; }

                  const double cM = double(mirror::RestirPtReservoir::toScalar(centralF[j]));
                  const double qTerm = double(reeval) / double(c.jacobian[j]) * double(historyM);
                  const double mC = cM / (cM + qTerm);

                  deficitX += (1.0 - mC) * double(centralF[j].x) / double(p[j]);
                  deficitY += (1.0 - mC) * double(centralF[j].y) / double(p[j]);
                  deficitZ += (1.0 - mC) * double(centralF[j].z) / double(p[j]);
                }

                checkClose(float(eX), float(truthX - deficitX),
                           (label + "the declared residual is exactly the dead-path deficit (x)").c_str(), 1e-3f);
                checkClose(float(eY), float(truthY - deficitY),
                           (label + "the declared residual is exactly the dead-path deficit (y)").c_str(), 1e-3f);
                checkClose(float(eZ), float(truthZ - deficitZ),
                           (label + "the declared residual is exactly the dead-path deficit (z)").c_str(), 1e-3f);
                check(eX <= truthX + 1e-6 && eY <= truthY + 1e-6 && eZ <= truthZ + 1e-6,
                      (label + "the residual is a deficit, never a surplus").c_str());
              }
            } else {
              if (hasUnderclaim) {
                // The pre-fix mechanism, permanently pinned: mixing the stored
                // numerator with the re-evaluated denominator INFLATES.
                check(eX > truthX + 1e-4 * std::max(1.0, truthX),
                      (label + "stored-target semantics inflates E[F.x * W] above the truth").c_str());
              }
            }
          }
        }
      }

      // One hand-derived anchor, so the magnitude is pinned by arithmetic done
      // OUTSIDE the code under test (the project's magnitude-discipline rule).
      // Case "spurious re-evaluation failure", M_T = 20, stored semantics,
      // channel x: only path 0 is poisoned, so
      //   excess = m_T_old(0) * F_c(0).x / p_0
      //   phat_c(0)      = 0.299*1.0 + 0.587*0.5 + 0.114*0.25 = 0.62100
      //   phat_stored(0) = 0.299*0.8 + 0.587*0.6 + 0.114*0.30 = 0.62560
      //   m_T_old(0)     = 0.6256*20 / (0.621 + 0.6256*20)    = 0.952726
      //   E_old.x        = 2.65 + 0.952726 * 1.0/1.0          = 3.602726
      {
        const Case& c = cases[0];
        double predX = 0.0, predY = 0.0, predZ = 0.0;
        predict(c, 20.0f, true, predX, predY, predZ);
        checkClose(float(predX), 3.602726f,
                   "r: hand-derived pre-fix inflation anchor (spurious failure, M_T 20, x)", 1e-4f);
      }
    }

    // -----------------------------------------------------------------------
    // (s) PHASE 4 FIX: THE TEMPORAL FEEDBACK LOOP DOES NOT GROW.
    //
    // The test that would have caught the in-game runaway, and the one that
    // proves it is gone. The output reservoir is fed back as the next frame's
    // temporal candidate for 60 frames -- M-cap dynamics included -- under a
    // re-evaluation table that spuriously reports paths impossible, and the
    // estimator's per-frame mean is measured against the truth.
    //
    // FIXED semantics: flat at the truth, every frame, for any failure table --
    // the partition of unity holds pointwise, so no per-frame factor exceeds 1.
    //
    // STORED (pre-fix) semantics, full failure table: the mean follows the
    // deterministic ramp
    //     r_f = 1.5 + 0.5 f                     while M_T ramps (f <= 19)
    //     r_f = 21 - 10 * (20/21)^(f-19)        once M_T saturates at 20
    // i.e. ~1.5x on the first reused frame, ~11x by frame 19, ~19.6x by frame
    // 59, converging to 1 + historyLength = 21x. At historyLength 3 the fixed
    // point is 4x -- the numeric shape of the in-game bisection result "history
    // length 3 mitigates the symptom but does not remove it".
    // -----------------------------------------------------------------------
    struct FeedbackRng {
      uint64_t state;
      explicit FeedbackRng(uint64_t seed) : state(seed) { }

      // splitmix64, folded to a 24-bit mantissa in [0, 1). Deterministic across
      // platforms, unlike std::uniform_real_distribution.
      float next() {
        state += 0x9E3779B97F4A7C15ull;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= (z >> 31);
        return float(uint32_t((z >> 40) & 0xFFFFFFu)) * (1.0f / 16777216.0f);
      }
    };

    void testTemporalFeedbackLoopDoesNotGrow() {
      const std::vector<mirror::vec3> centralF = {
        mirror::vec3(1.0f, 0.5f, 0.25f), mirror::vec3(0.2f, 2.0f, 0.1f), mirror::vec3(0.7f, 0.7f, 3.0f)
      };
      const std::vector<float> p = { 1.0f, 0.8f, 0.5f };

      double truthX = 0.0;
      for (size_t j = 0; j < centralF.size(); ++j) {
        truthX += double(centralF[j].x) / double(p[j]);
      }

      // Identity domain at steady state: a faithful re-evaluation returns the
      // central integrand. The failure tables corrupt it the way the broken
      // temporal surface did in-game.
      const std::vector<mirror::vec3> reevalAllFail = {
        mirror::vec3(0.0f), mirror::vec3(0.0f), mirror::vec3(0.0f)
      };
      const std::vector<mirror::vec3> reevalPartialFail = {
        mirror::vec3(0.0f), centralF[1], centralF[2]
      };

      // The CENTRAL target's two evaluators. `centralExact` is the shift agreeing
      // with the trace -- everywhere debug view 887 is black. `centralUnderReports`
      // is the 887 residual population: the shift returns 82% of what the trace
      // stored on one path, a fraction rather than the ~2x the temporal split had.
      // That single number is what turns a runaway you cannot miss into one that
      // takes ~10 seconds to become visible.
      const std::vector<mirror::vec3> centralExact = centralF;
      const std::vector<mirror::vec3> centralUnderReports = {
        centralF[0], mirror::vec3(0.164f, 1.64f, 0.082f), centralF[2]
      };

      // centralReevalF is what the SHIFT evaluator returns for the central target;
      // centralF is what the TRACE stored. They coincide wherever debug view 887 is
      // black, and the residual population is where they do not -- see (t).
      auto runLoop = [&](const std::vector<mirror::vec3>& reevalF,
                         const std::vector<mirror::vec3>& centralReevalF, float historyLength,
                         bool useStoredHistoryTarget, bool useStoredCentralTarget,
                         int frames, int chains, uint64_t seed,
                         std::vector<double>& frameMean) {
        frameMean.assign(size_t(frames), 0.0);
        FeedbackRng rng(seed);

        for (int chain = 0; chain < chains; ++chain) {
          // Frame 0 has no history: the temporal dispatch is skipped in-game
          // (m_historyValid false) and the trace output becomes the history.
          mirror::RestirPtReservoir history;
          int historyPath = -1;
          {
            history.init();
            for (size_t j = 0; j < centralF.size(); ++j) {
              if (history.add(centralF[j], p[j], rng.next())) { historyPath = int(j); }
            }
            history.srcPixel = uint32_t(historyPath);
            mirror::pathBuilderFinalize(history);
            history.finalizeRIS();
          }

          for (int frame = 0; frame < frames; ++frame) {
            mirror::RestirPtReservoir canonical;
            canonical.init();
            int canonicalPath = -1;
            for (size_t j = 0; j < centralF.size(); ++j) {
              if (canonical.add(centralF[j], p[j], rng.next())) { canonicalPath = int(j); }
            }
            canonical.srcPixel = uint32_t(canonicalPath);
            mirror::pathBuilderFinalize(canonical);
            canonical.finalizeRIS();

            const float currentM = canonical.M;

            mirror::RestirPtReservoir temporal = history;
            const int temporalPath = historyPath;
            temporal.M = std::min(historyLength * currentM, temporal.M);

            mirror::RestirPtReservoir dst;
            dst.init();

            // --- i = curSampleId ------------------------------------------
            {
              const bool possible = canonical.weight > 0.0f;

              float pSum = 0.0f;
              float pSelf = 0.0f;

              // THE CENTRAL-SIDE SLOT. Pre-fix this read the stored trace integrand
              // while the temporal candidate's denominator below read the shift's
              // re-evaluation of the same target -- two evaluators, one target, and
              // the surplus compounds exactly as the temporal-side split did, only
              // slower because the disagreement is a fraction rather than ~2x.
              const mirror::vec3 canonicalTarget = useStoredCentralTarget
                ? canonical.F
                : centralReevalF[canonicalPath];

              if (possible) {
                pSelf = mirror::RestirPtReservoir::computeWeight(canonicalTarget, false) * currentM;
                pSum = pSelf;
                pSum += mirror::RestirPtReservoir::computeWeight(reevalF[canonicalPath], false) *
                        temporal.M;
              }

              const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);
              // F serves double duty as radiance and target, and finalizeGRIS
              // normalises by toScalar(F), so the merged integrand must be the same
              // evaluator the weight used.
              dst.mergeWithResamplingMIS(canonicalTarget, 1.0f, canonical, rng.next(), misWeight, false);
            }

            // --- i = prevSampleId -----------------------------------------
            {
              mirror::RestirPtReservoir temp = dst;
              const mirror::vec3 shiftedIntegrand = centralReevalF[temporalPath];
              const float dstJacobian = 1.0f;

              const bool selected = temp.merge(shiftedIntegrand, dstJacobian, temporal,
                                               0.0f, 1.0f, /*forceAdd=*/ true);
              if (!selected) { temp.F = mirror::vec3(); }
              temp.M = temporal.M;
              temp.weight = temporal.weight;

              float pSum = 0.0f;
              float pSelf = 0.0f;

              if (selected) {
                pSum = mirror::RestirPtReservoir::computeWeight(temp.F, false) * currentM;

                // THE ONE LINE THE WHOLE FIX IS ABOUT. Pre-fix, this slot read the
                // STORED history integrand while the canonical candidate's
                // denominator above read the RE-EVALUATION; where the two disagree
                // the weights stop summing to 1 and the surplus compounds through
                // `history = dst` below. Post-fix, both read the re-evaluation.
                const float selfTarget = useStoredHistoryTarget
                  ? mirror::RestirPtReservoir::computeWeight(temporal.F, false)
                  : mirror::RestirPtReservoir::computeWeight(reevalF[temporalPath], false);

                pSelf = selfTarget / dstJacobian * temporal.M;
                pSum += pSelf;
              }

              const float misWeight = (pSum == 0.0f) ? 0.0f : (pSelf / pSum);
              dst.mergeWithResamplingMIS(temp.F, dstJacobian, temp, rng.next(), misWeight, false);
            }

            if (dst.weight > 0.0f) { dst.finalizeGRIS(); }
            if (dst.weight < 0.0f || std::isnan(dst.weight) || std::isinf(dst.weight)) {
              dst.weight = 0.0f;
            }

            frameMean[size_t(frame)] += double(dst.F.x) * double(dst.weight);

            history = dst;
            historyPath = int(dst.srcPixel);
          }
        }

        for (double& v : frameMean) { v /= double(chains); }
      };

      auto windowMean = [](const std::vector<double>& v, int lo, int hi) {
        double sum = 0.0;
        for (int i = lo; i <= hi; ++i) { sum += v[size_t(i)]; }
        return sum / double(hi - lo + 1);
      };

      const int frames = 60;

      // FIXED semantics, total failure: degenerates to the canonical sample,
      // exactly unbiased, no growth.
      {
        std::vector<double> mean;
        runLoop(reevalAllFail, centralExact, 20.0f, false, false, frames, 120000, 0x51CA9E1Dull, mean);

        const double early = windowMean(mean, 0, 4);
        const double late = windowMean(mean, 40, 59);

        checkClose(float(early), float(truthX),
                   "s: [fixed, all-fail, H 20] early frames sit at the truth", 0.02f);
        checkClose(float(late), float(truthX),
                   "s: [fixed, all-fail, H 20] late frames sit at the truth", 0.02f);
        checkClose(float(late / early), 1.0f,
                   "s: [fixed, all-fail, H 20] growth factor over 60 frames is 1", 0.02f);
      }

      // FIXED semantics, partial failure: the temporal candidate stays live on
      // the healthy paths, the partition still holds, still no growth.
      {
        std::vector<double> mean;
        runLoop(reevalPartialFail, centralExact, 20.0f, false, false, frames, 120000, 0xB0BAFE77ull, mean);

        const double early = windowMean(mean, 0, 4);
        const double late = windowMean(mean, 40, 59);

        checkClose(float(early), float(truthX),
                   "s: [fixed, partial-fail, H 20] early frames sit at the truth", 0.02f);
        checkClose(float(late), float(truthX),
                   "s: [fixed, partial-fail, H 20] late frames sit at the truth", 0.02f);
        checkClose(float(late / early), 1.0f,
                   "s: [fixed, partial-fail, H 20] growth factor over 60 frames is 1", 0.02f);
      }

      // STORED (pre-fix) semantics, total failure, H 20: the runaway, with the
      // deterministic ramp asserted at three depths. This is the in-game bug in
      // four numbers: ~1.5x on the first reused frame, ~11x at frame 19, ~19.6x
      // at frame 59, limit 21x.
      {
        std::vector<double> mean;
        runLoop(reevalAllFail, centralExact, 20.0f, true, false, frames, 40000, 0xDEAD10CCull, mean);

        const double first = mean[0];
        const double ramp = windowMean(mean, 17, 21);   // predicted ~10.5-11.5
        const double late = windowMean(mean, 55, 59);   // predicted ~19.5

        check(first > truthX * 1.35 && first < truthX * 1.65,
              "s: [stored, all-fail, H 20] first reused frame inflates by ~1.5x");
        check(ramp > truthX * 9.0 && ramp < truthX * 13.0,
              "s: [stored, all-fail, H 20] frame ~19 inflates by ~11x (the ramp)");
        check(late > truthX * 15.0 && late < truthX * 21.5,
              "s: [stored, all-fail, H 20] frame ~59 inflates toward the 21x fixed point");
      }

      // STORED semantics, H 3: the fixed point drops to 1 + 3 = 4x -- the
      // bisection's "history length 3 mitigates but does not remove".
      {
        std::vector<double> mean;
        runLoop(reevalAllFail, centralExact, 3.0f, true, false, frames, 40000, 0xF00DFACEull, mean);

        const double late = windowMean(mean, 40, 59);

        check(late > truthX * 3.4 && late < truthX * 4.4,
              "s: [stored, all-fail, H 3] the runaway saturates at ~4x, not ~21x");
      }

      // THE CENTRAL-SIDE SPLIT -- a REAL second inconsistency, MEASURED AND
      // BOUNDED, and deliberately NOT fixed in the shader. This block exists so
      // nobody spends another round chasing it.
      //
      // The central target has the identical two-evaluator structure the temporal
      // one had: the canonical candidate's numerator reads the STORED trace
      // integrand while the temporal candidate's MIS denominator reads the shift's
      // re-evaluation of the same target. Debug view 887 is the measurement of how
      // far those two are apart, and it is black on ordinary opaque surfaces, so
      // the disagreement is confined to its documented residual -- foliage and
      // thin-translucent, alpha-tested geometry, back-facing rebuilt vertices.
      //
      // The reason to model it and then walk away is the MAGNITUDE, which is the
      // whole point. With an 18% disagreement on one path of three the loop grows
      // about 1.5% per 60 frames. Extrapolated at 44 fps that is roughly 12% over
      // ten seconds -- about twenty times too weak to be an in-game white-out, and
      // the "fix" (routing the canonical numerator AND the merged contribution
      // through a fourth self-shift, since F is both radiance and target here and
      // finalizeGRIS normalises by it) costs a fourth visibility ray per pixel
      // while trading that surplus for a deficit of comparable size.
      //
      // So: real, bounded, not worth a ray. If a future change makes the 887
      // residual much larger, re-read this -- the bound below is what would move.
      {
        std::vector<double> storedMean;
        runLoop(reevalPartialFail, centralUnderReports, 20.0f, false, true, frames, 40000,
                0xC0FFEE11ull, storedMean);

        std::vector<double> fixedMean;
        runLoop(reevalPartialFail, centralUnderReports, 20.0f, false, false, frames, 40000,
                0xC0FFEE11ull, fixedMean);

        const double storedEarly = windowMean(storedMean, 0, 4);
        const double storedLate = windowMean(storedMean, 50, 59);
        const double fixedEarly = windowMean(fixedMean, 0, 4);
        const double fixedLate = windowMean(fixedMean, 50, 59);

        const double storedGrowth = storedLate / storedEarly;
        const double fixedGrowth = fixedLate / fixedEarly;

        // It IS an inflation, and it IS caused by the split: routing both slots
        // through one evaluator grows strictly less.
        check(storedGrowth > 1.0, "s: [central split] the stored-target loop does inflate");
        check(storedGrowth > fixedGrowth,
              "s: [central split] and inflates strictly more than the consistent-target form");

        // AND IT IS SMALL. This is the assertion that retires the hypothesis: an
        // 18% evaluator disagreement cannot compound fast enough to white out a
        // frame in ten seconds. If this ever fires, the bound has moved and the
        // trade-off above deserves re-costing.
        check(storedGrowth < 1.05,
              "s: [central split] growth over 60 frames stays under 5% - far too slow to be the in-game symptom");

        // The discriminator against the temporal-side split, so the two mechanisms
        // can never be confused: the temporal split is already ~1.5x on the FIRST
        // reused frame; this one is indistinguishable from the truth there.
        checkClose(float(storedEarly), float(truthX),
                   "s: [central split] early frames are still at the truth - a SLOW ramp, not a jump", 0.03f);
      }
    }

    void run() {
      testAddBookkeeping();
      testFinalizeRIS();
      testStreamingEstimatorIsUnbiased();
      testMergeRoundTrip();
      testNanRejection();
      testPathFlagsRoundTrip();
      testTargetFunction();
      testReconnectionGeometryJacobian();
      testLobeClassConventionSelfShift();
      testEscapeShiftIdentity();
      testReconnectionVertexFootprint();
      testSelfShiftMetric();
      testConnectionAttenuationBelongsToPrefix();
      testZeroViewDirectionIsSafe();
      testPairwiseResamplingMISIsUnbiased();
      testPairwiseWeightsPartitionAndGuards();
      testTalbotResamplingMISIsUnbiased();
      testTalbotWeightsPartitionAndMCap();
      testTalbotAsymmetricShiftIsUnbiased();
      testTemporalFeedbackLoopDoesNotGrow();

      std::cout << "All passed (" << m_checks << " checks)\n";
    }

  private:
    int m_checks = 0;
  };
}

int main() {
  try {
    dxvk::TestApp testApp;
    testApp.run();
  }
  catch (const dxvk::DxvkError& error) {
    std::cerr << error.message() << std::endl;
    return 1;
  }

  return 0;
}
