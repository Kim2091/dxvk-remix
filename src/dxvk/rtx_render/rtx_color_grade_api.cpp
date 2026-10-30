// The Remix API's colour grading exports (remix_c.h, "Colour grading"). They translate the C structs and
// hand the requests to the runtime's grade controller (rtx_color_grade.h), which is safe to call from any thread.

#include <cmath>
#include <string>

#include <remix/remix_c.h>

#include "rtx_color_grade.h"

namespace {
  using dxvk::ColorGrade;
  using dxvk::color_grade::Params;

  bool validBlendSeconds(float blendSeconds) {
    return std::isfinite(blendSeconds) && blendSeconds >= 0.0f;
  }

  remixapi_ErrorCode checkStruct(const remixapi_ColorGradeEXT* grade) {
    if (grade == nullptr || grade->sType != REMIXAPI_STRUCT_TYPE_COLOR_GRADE_EXT) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    if (grade->version != REMIXAPI_COLOR_GRADE_VERSION) {
      return REMIXAPI_ERROR_CODE_INCOMPATIBLE_VERSION;
    }
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  Params toParams(const remixapi_ColorGradeEXT& g) {
    Params p;
    for (int i = 0; i < 9; ++i) {
      p.matrix[i / 3][i % 3] = g.matrix[i];
    }
    p.balance = { g.balance[0], g.balance[1], g.balance[2] };
    p.saturation = g.saturation;
    p.contrast = g.contrast;
    // A zero-initialized struct gets the standard curve: 0 is outside both ranges and takes the default.
    if (g.anchor != 0.0f) {
      p.anchor = g.anchor;
    }
    if (g.shoulder != 0.0f) {
      p.shoulder = g.shoulder;
    }
    return p;
  }

  void fromParams(const Params& p, remixapi_ColorGradeEXT& g) {
    for (int i = 0; i < 9; ++i) {
      g.matrix[i] = p.matrix[i / 3][i % 3];
    }
    for (int i = 0; i < 3; ++i) {
      g.balance[i] = p.balance[i];
    }
    g.saturation = p.saturation;
    g.contrast = p.contrast;
    g.anchor = p.anchor;
    g.shoulder = p.shoulder;
  }
}

extern "C" {

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_SetColorGradePreset(const char* name, float blendSeconds) {
    if (name == nullptr || !validBlendSeconds(blendSeconds)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return ColorGrade::setPreset(name, blendSeconds) ? REMIXAPI_ERROR_CODE_SUCCESS : REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_SetColorGrade(
      const remixapi_ColorGradeEXT* grade, float blendSeconds) {
    const remixapi_ErrorCode check = checkStruct(grade);
    if (check != REMIXAPI_ERROR_CODE_SUCCESS) {
      return check;
    }
    if (!validBlendSeconds(blendSeconds)) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    ColorGrade::setGrade(toParams(*grade), blendSeconds);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_MixColorGradePresets(
      const char* nameA, const char* nameB, float weight) {
    if (nameA == nullptr || nameB == nullptr || !std::isfinite(weight) || weight < 0.0f || weight > 1.0f) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    return ColorGrade::mixPresets(nameA, nameB, weight) ? REMIXAPI_ERROR_CODE_SUCCESS : REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_GetColorGrade(remixapi_ColorGradeEXT* out) {
    const remixapi_ErrorCode check = checkStruct(out);
    if (check != REMIXAPI_ERROR_CODE_SUCCESS) {
      return check;
    }
    fromParams(ColorGrade::current(), *out);
    return REMIXAPI_ERROR_CODE_SUCCESS;
  }

  REMIXAPI remixapi_ErrorCode REMIXAPI_CALL remixapi_SaveColorGradePreset(
      const char* name, const remixapi_ColorGradeEXT* grade) {
    const remixapi_ErrorCode check = checkStruct(grade);
    if (check != REMIXAPI_ERROR_CODE_SUCCESS) {
      return check;
    }
    if (name == nullptr) {
      return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    }
    switch (ColorGrade::savePreset(name, toParams(*grade))) {
    case ColorGrade::SaveResult::Saved:       return REMIXAPI_ERROR_CODE_SUCCESS;
    case ColorGrade::SaveResult::InvalidName: return REMIXAPI_ERROR_CODE_INVALID_ARGUMENTS;
    case ColorGrade::SaveResult::WriteFailed: return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
    }
    return REMIXAPI_ERROR_CODE_GENERAL_FAILURE;
  }

}
