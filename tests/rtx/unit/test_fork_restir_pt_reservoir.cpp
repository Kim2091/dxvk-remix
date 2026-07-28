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

#include <cmath>
#include <cstdint>
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

    void run() {
      testAddBookkeeping();
      testFinalizeRIS();
      testStreamingEstimatorIsUnbiased();
      testMergeRoundTrip();
      testNanRejection();
      testPathFlagsRoundTrip();
      testTargetFunction();
      testReconnectionGeometryJacobian();
      testPairwiseResamplingMISIsUnbiased();
      testPairwiseWeightsPartitionAndGuards();

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
