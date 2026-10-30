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

// Checks the cone generator (pom_cone_generate.comp.slang, copied below on the CPU) against a brute-force
// relaxed cone on synthetic height fields. Soundness: the stored (quantised) cone never exceeds the brute-force one at
// any texel (the hierarchical search skips bounds that cannot change the stored value, so its unquantised result may).
// Tightness (stored / brute force, averaged) is reported, not asserted.
// Also the conservative variant (no ray walk) against a brute-force conservative cone,
// and the blended-terrain cone of pom_sccr_terrain.slangh (per-layer conservative cones combined with the layer
// weights held fixed) against a brute-force conservative cone of the blended field.
// Also the march's crossing refinement (bisections, a false-position step, a final secant): it keeps a
// crossing bracketed and its bracket within the bisection bound.
// Also the curvature correction's base surface (surface_interaction_sccr.slangh, copied below): its UV
// expansion around a hit reproduces the half-strength Phong patch, is 0 at the vertices, stays within its bound on the
// triangle, and matches a circular arc on a sphere.
// Also blends reduced to their heaviest layers (sccrTerrainReduceLayers): the kept set and the
// soundness of the reduced blend's cone.
// Also the self-shadow march's early returns: the same visibility as the full refinement wherever a
// crossing lies before the texel.
// The generator's hierarchical search (square power-of-two maps): its conservative variant against the
// brute-force conservative cone (sound, and exact up to one step of the stored quantisation; sound with the visit cap
// too), and its mean cone and reads per texel against the old windowed generator (kept below as windowedRatio) and
// coarse cones.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kPi = 3.14159265358979f;

// Depths (1 - height) of a height map, quantised like an 8-bit texture. Lookups wrap.
struct DepthField {
  int width = 0;
  int height = 0;
  std::vector<float> depth;

  float at(int x, int y) const {
    const int wx = ((x % width) + width) % width;
    const int wy = ((y % height) + height) % height;
    return depth[size_t(wy) * width + wx];
  }
};

DepthField makeField(int width, int height, const std::function<float(int, int)>& heightAt) {
  DepthField field;
  field.width = width;
  field.height = height;
  field.depth.resize(size_t(width) * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float h = std::clamp(heightAt(x, y), 0.0f, 1.0f);
      field.depth[size_t(y) * width + x] = 1.0f - std::round(h * 255.0f) / 255.0f;
    }
  }
  return field;
}

// The max-height pyramid as least depths; level 0 is the field (square power-of-two fields only).
std::vector<DepthField> buildPyramid(const DepthField& field, uint32_t farLevels) {
  std::vector<DepthField> levels { field };
  for (uint32_t level = 1; level <= farLevels; ++level) {
    const DepthField& source = levels.back();
    DepthField next;
    next.width = std::max(source.width / 2, 1);
    next.height = std::max(source.height / 2, 1);
    next.depth.resize(size_t(next.width) * next.height);
    for (int y = 0; y < next.height; ++y) {
      for (int x = 0; x < next.width; ++x) {
        float least = 1.0f;
        for (int i = 0; i < 4; ++i) {
          const int sx = std::min(2 * x + (i & 1), source.width - 1);
          const int sy = std::min(2 * y + (i >> 1), source.height - 1);
          least = std::min(least, source.depth[size_t(sy) * source.width + sx]);
        }
        next.depth[size_t(y) * next.width + x] = least;
      }
    }
    levels.push_back(next);
  }
  return levels;
}

// PomConeMapManager's generationParams: square power-of-two maps get the whole pyramid (to 1x1) and the hierarchical
// search; other maps none (the window bound).
uint32_t farLevelsFor(int width, int height) {
  const uint32_t w = uint32_t(width);
  uint32_t farLevels = 0;
  if (width == height && w > 1 && (w & (w - 1)) == 0) {
    while ((1u << (farLevels + 1)) <= w) {
      ++farLevels;
    }
  }
  return farLevels;
}

// The old generator's pyramid levels (before the hierarchical search): up to nearRadius * 2^levels >= the width.
uint32_t windowedFarLevelsFor(int width, int height, uint32_t nearRadius) {
  const uint32_t w = uint32_t(width);
  uint32_t farLevels = 0;
  if (width == height && (w & (w - 1)) == 0) {
    uint32_t maxLevel = 0;
    while ((1u << (maxLevel + 1)) <= w) {
      ++maxLevel;
    }
    while (farLevels < maxLevel && (nearRadius << farLevels) < w) {
      ++farLevels;
    }
  }
  return farLevels;
}

// pom_cone_args.h: POM_CONE_MAX_VISITS.
constexpr int kMaxVisits = 2048;

// Floor division by 2^shift, also for negative values (cells left of or above the texture).
int floorShift(int value, int shift) {
  return value >= 0 ? (value >> shift) : -((-value + (1 << shift) - 1) >> shift);
}

// pruneThreshold in pom_cone_generate.comp.slang (8-bit cones).
float pruneThreshold(float ratio) {
  const float stored = std::floor(std::sqrt(std::clamp(ratio, 0.0f, 1.0f)) * 255.0f) / 255.0f;
  return stored * stored * 1.0001f;
}

// nearField in pom_cone_generate.comp.slang.
float nearField(const DepthField& field, int px, int py, float depthP, int radius, uint32_t raySteps, float ratio,
                uint64_t* reads) {
  const float texelU = 1.0f / float(field.width);
  const float texelV = 1.0f / float(field.height);
  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const float depthQ = field.at(px + dx, py + dy);
      if (reads != nullptr) {
        ++*reads;
      }
      const float span = float(std::max(std::abs(dx), std::abs(dy)));
      float t = 1.0f;
      for (uint32_t i = 1; i <= raySteps; ++i) {
        const float tNext = 1.0f + float(i) / span;
        const float depthNext = tNext * depthQ;
        if (depthNext > 1.0f) {
          break;
        }
        const int sx = px + int(std::floor(float(dx) * tNext + 0.5f));
        const int sy = py + int(std::floor(float(dy) * tNext + 0.5f));
        if (reads != nullptr) {
          ++*reads;
        }
        if (field.at(sx, sy) > depthNext) {
          break;
        }
        t = tNext;
      }
      const float exitDepth = t * depthQ;
      if (exitDepth < depthP) {
        const float distance = std::hypot(float(dx) * t * texelU, float(dy) * t * texelV);
        ratio = std::min(ratio, distance / (depthP - exitDepth));
      }
    }
  }
  return ratio;
}

// pyramidSearch in pom_cone_generate.comp.slang: the conservative bounds of every texel outside the near field, by
// branch and bound over the max-height pyramid (square power-of-two fields; pyramid[topLevel] is 1x1).
float pyramidSearch(const std::vector<DepthField>& pyramid, int px, int py, float depthP, int nearRadius, int topLevel,
                    float ratio, int maxVisits, uint64_t* reads) {
  const int width = pyramid[0].width;
  const float texelUv = 1.0f / float(width);
  int windowFirstX;
  int windowFirstY;
  int windowLastX;
  int windowLastY;
  if (nearRadius > 0) {
    windowFirstX = px - (width - 1);
    windowFirstY = py - (width - 1);
    windowLastX = px + (width - 1);
    windowLastY = py + (width - 1);
  } else {
    windowFirstX = px - width / 2;
    windowFirstY = py - width / 2;
    windowLastX = windowFirstX + width - 1;
    windowLastY = windowFirstY + width - 1;
  }
  const int rootFirstX = floorShift(windowFirstX, topLevel);
  const int rootFirstY = floorShift(windowFirstY, topLevel);
  const int rootLastX = floorShift(windowLastX, topLevel);
  const int rootLastY = floorShift(windowLastY, topLevel);
  float threshold = pruneThreshold(ratio);
  int visits = 0;
  for (int iy = 0; iy < 3; ++iy) {
    const int rootY = iy == 0 ? 0 : (iy == 1 ? -1 : 1);
    if (rootY < rootFirstY || rootY > rootLastY) {
      continue;
    }
    for (int ix = 0; ix < 3; ++ix) {
      const int rootX = ix == 0 ? 0 : (ix == 1 ? -1 : 1);
      if (rootX < rootFirstX || rootX > rootLastX) {
        continue;
      }
      int cellX = rootX;
      int cellY = rootY;
      int level = topLevel;
      while (true) {
        const int size = 1 << level;
        const int firstX = cellX * size;
        const int firstY = cellY * size;
        const int lastX = firstX + size - 1;
        const int lastY = firstY + size - 1;
        bool descend = false;
        const bool nearTexel = level == 0 && std::abs(cellX - px) <= nearRadius && std::abs(cellY - py) <= nearRadius;
        if (lastX >= windowFirstX && lastY >= windowFirstY && firstX <= windowLastX && firstY <= windowLastY &&
            !nearTexel) {
          ++visits;
          if (reads != nullptr) {
            ++*reads;
          }
          const float leastDepth = pyramid[level].at(cellX, cellY);
          if (leastDepth < depthP) {
            const int nearestX = std::clamp(px, firstX, lastX);
            const int nearestY = std::clamp(py, firstY, lastY);
            const float bound = std::hypot(float(nearestX - px) * texelUv, float(nearestY - py) * texelUv) /
                                (depthP - leastDepth);
            if (bound < threshold) {
              if (level > 0 && visits < maxVisits) {
                descend = true;
              } else {
                ratio = std::min(ratio, bound);
                threshold = pruneThreshold(ratio);
              }
            }
          }
        }
        if (descend) {
          --level;
          const int halfSize = 1 << level;
          cellX = 2 * cellX + (px >= firstX + halfSize ? 1 : 0);
          cellY = 2 * cellY + (py >= firstY + halfSize ? 1 : 0);
          continue;
        }
        while (level < topLevel) {
          const int halfSize = 1 << level;
          const int parentX = floorShift(cellX, 1);
          const int parentY = floorShift(cellY, 1);
          const int nearSideX = px >= parentX * 2 * halfSize + halfSize ? 1 : 0;
          const int nearSideY = py >= parentY * 2 * halfSize + halfSize ? 1 : 0;
          const int order = ((cellX - 2 * parentX) ^ nearSideX) | (((cellY - 2 * parentY) ^ nearSideY) << 1);
          if (order < 3) {
            const int next = order + 1;
            cellX = 2 * parentX + ((next & 1) ^ nearSideX);
            cellY = 2 * parentY + ((next >> 1) ^ nearSideY);
            break;
          }
          cellX = parentX;
          cellY = parentY;
          ++level;
        }
        if (level == topLevel) {
          break;
        }
      }
    }
  }
  return ratio;
}

// CPU copy of pom_cone_generate.comp.slang, before quantisation. reads (optional) counts the texels and pyramid cells
// read; maxVisits is POM_CONE_MAX_VISITS unless a check lowers it.
float generateRatio(const std::vector<DepthField>& pyramid, int px, int py, int radius, uint32_t raySteps,
                    uint32_t farLevels, uint64_t* reads = nullptr, int maxVisits = kMaxVisits) {
  const DepthField& field = pyramid[0];
  const float texelU = 1.0f / float(field.width);
  const float texelV = 1.0f / float(field.height);
  const float depthP = field.at(px, py);
  if (reads != nullptr) {
    ++*reads;
  }
  float ratio = 1.0f;
  if (depthP <= 0.0f) {
    return ratio;
  }
  const bool hasPyramid = farLevels > 0;
  const int nearRadius = (hasPyramid && raySteps == 0) ? 0 : radius;
  ratio = nearField(field, px, py, depthP, nearRadius, raySteps, ratio, reads);
  if (hasPyramid) {
    ratio = pyramidSearch(pyramid, px, py, depthP, nearRadius, int(farLevels), ratio, maxVisits, reads);
  } else {
    ratio = std::min(ratio, float(radius + 1) * std::min(texelU, texelV) / depthP);
  }
  return ratio;
}

// The old generator (pom_cone_generate.comp.slang before the hierarchical search: a window walked in full, fixed rings
// of pyramid cells beyond it), before quantisation, to compare cone widths and reads. Its pyramid levels come from
// windowedFarLevelsFor; a deeper pyramid serves as well (it reads no further).
float windowedRatio(const std::vector<DepthField>& pyramid, int px, int py, int radius, uint32_t raySteps,
                    uint32_t farLevels, uint64_t* reads = nullptr) {
  const DepthField& field = pyramid[0];
  const float texelU = 1.0f / float(field.width);
  const float texelV = 1.0f / float(field.height);
  const float depthP = field.at(px, py);
  if (reads != nullptr) {
    ++*reads;
  }
  float ratio = 1.0f;
  if (depthP <= 0.0f) {
    return ratio;
  }

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const float depthQ = field.at(px + dx, py + dy);
      if (reads != nullptr) {
        ++*reads;
      }
      const float span = float(std::max(std::abs(dx), std::abs(dy)));
      float t = 1.0f;
      for (uint32_t i = 1; i <= raySteps; ++i) {
        const float tNext = 1.0f + float(i) / span;
        const float depthNext = tNext * depthQ;
        if (depthNext > 1.0f) {
          break;
        }
        const int sx = px + int(std::floor(float(dx) * tNext + 0.5f));
        const int sy = py + int(std::floor(float(dy) * tNext + 0.5f));
        if (reads != nullptr) {
          ++*reads;
        }
        if (field.at(sx, sy) > depthNext) {
          break;
        }
        t = tNext;
      }
      const float exitDepth = t * depthQ;
      if (exitDepth < depthP) {
        const float distance = std::hypot(float(dx) * t * texelU, float(dy) * t * texelV);
        ratio = std::min(ratio, distance / (depthP - exitDepth));
      }
    }
  }

  if (farLevels > 0) {
    const int ringInner = radius / 2;
    for (uint32_t level = 1; level <= farLevels; ++level) {
      const int shift = int(level);
      const int cellSize = 1 << shift;
      const DepthField& cells = pyramid[level];
      const int pCellX = px >> shift;
      const int pCellY = py >> shift;
      for (int cy = -radius; cy <= radius; ++cy) {
        for (int cx = -radius; cx <= radius; ++cx) {
          if (std::max(std::abs(cx), std::abs(cy)) < ringInner) {
            continue;
          }
          const int cellX = pCellX + cx;
          const int cellY = pCellY + cy;
          const float leastDepth = cells.at(cellX, cellY);
          if (reads != nullptr) {
            ++*reads;
          }
          if (leastDepth >= depthP) {
            continue;
          }
          const int nearestX = std::clamp(px, cellX * cellSize, cellX * cellSize + cellSize - 1);
          const int nearestY = std::clamp(py, cellY * cellSize, cellY * cellSize + cellSize - 1);
          const float distance = std::hypot(float(nearestX - px) * texelU, float(nearestY - py) * texelV);
          ratio = std::min(ratio, distance / (depthP - leastDepth));
        }
      }
    }
  } else {
    ratio = std::min(ratio, float(radius + 1) * std::min(texelU, texelV) / depthP);
  }
  return ratio;
}

// The relaxed cone by definition: every candidate within one texture width (farther ones are past the cap of 1 UV),
// each ray walked until it leaves the field or passes the bottom.
float bruteForceRatio(const DepthField& field, int px, int py) {
  const float texelU = 1.0f / float(field.width);
  const float texelV = 1.0f / float(field.height);
  const float depthP = field.at(px, py);
  float ratio = 1.0f;
  if (depthP <= 0.0f) {
    return ratio;
  }
  // A ray along the top of a plateau that wraps all the way round never leaves it; past two widths it cannot matter.
  const int maxWalk = 2 * std::max(field.width, field.height);
  for (int dy = -(field.height - 1); dy <= field.height - 1; ++dy) {
    for (int dx = -(field.width - 1); dx <= field.width - 1; ++dx) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const float depthQ = field.at(px + dx, py + dy);
      const float span = float(std::max(std::abs(dx), std::abs(dy)));
      float t = 1.0f;
      for (int i = 1; i <= maxWalk; ++i) {
        const float tNext = 1.0f + float(i) / span;
        const float depthNext = tNext * depthQ;
        if (depthNext > 1.0f) {
          break;
        }
        const int sx = px + int(std::floor(float(dx) * tNext + 0.5f));
        const int sy = py + int(std::floor(float(dy) * tNext + 0.5f));
        if (field.at(sx, sy) > depthNext) {
          break;
        }
        t = tNext;
      }
      const float exitDepth = t * depthQ;
      if (exitDepth < depthP) {
        const float distance = std::hypot(float(dx) * t * texelU, float(dy) * t * texelV);
        ratio = std::min(ratio, distance / (depthP - exitDepth));
      }
    }
  }
  return ratio;
}

// The conservative cone by definition: no point of the field inside it. Every offset within one texture width covers
// the nearest copy of every texel of a wrapping field, so without a cap this is the exact ratio (infinite, here
// `cap`, when nothing lies above p).
float bruteForceConservativeRatio(const DepthField& field, int px, int py, float cap) {
  const float texelU = 1.0f / float(field.width);
  const float texelV = 1.0f / float(field.height);
  const float depthP = field.at(px, py);
  float ratio = cap;
  for (int dy = -(field.height - 1); dy <= field.height - 1; ++dy) {
    for (int dx = -(field.width - 1); dx <= field.width - 1; ++dx) {
      const float depthQ = field.at(px + dx, py + dy);
      if ((dx == 0 && dy == 0) || depthQ >= depthP) {
        continue;
      }
      ratio = std::min(ratio, std::hypot(float(dx) * texelU, float(dy) * texelV) / (depthP - depthQ));
    }
  }
  return ratio;
}

// The value the cone texture stores, read back as the shader reads it (G squared).
float storedCone(float ratio) {
  const float stored = std::floor(std::sqrt(std::clamp(ratio, 0.0f, 1.0f)) * 255.0f) / 255.0f;
  return stored * stored;
}

// The conservative variant (rayStepsPerCandidate = 0) never exceeds the conservative cone.
void checkConservative(const std::string& name, const DepthField& field, int radius) {
  const uint32_t farLevels = farLevelsFor(field.width, field.height);
  const std::vector<DepthField> pyramid = buildPyramid(field, farLevels);
  double tightnessSum = 0.0;
  int tightnessCount = 0;
  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      const float generated = generateRatio(pyramid, x, y, radius, 0, farLevels);
      const float reference = bruteForceConservativeRatio(field, x, y, 1.0f);
      if (storedCone(generated) > reference * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") conservative generated " << generated
                  << " > brute force " << reference << std::endl;
        throw std::runtime_error("conservative cone generator is not conservative");
      }
      if (reference > 0.0f) {
        tightnessSum += storedCone(generated) / reference;
        ++tightnessCount;
      }
    }
  }
  std::printf("  %-28s %3dx%-3d radius %2d, conservative, far levels %u: sound, tightness %.3f\n", name.c_str(),
              field.width, field.height, radius, farLevels, tightnessCount > 0 ? tightnessSum / tightnessCount : 1.0);
}

// Pom_sccr_terrain.slangh's cone for a blend of layers with fixed weights w_i (W = their sum, plus
// the weight of layers without a height map, which are flat): depth D = sum_i w_i D_i / W, and from each layer's
// conservative cone D_i(q) >= D_i(p) - |q - p| / c_i, so D(q) >= D(p) - |q - p| * sum_i (w_i / W) / c_i. The cone is
// 1 / sum_i (w_i / W) / c_i (at least the smallest c_i); a layer with a stored cone of 0 makes it 0 (a linear step).
// Checked against the exact conservative cone of the blended field. Also counts the texels where the same combination
// of relaxed cones would be too wide (reported only: why terrain layers take the conservative variant).
void checkBlend(const std::string& name, const std::vector<DepthField>& layers, const std::vector<float>& weights,
                float flatWeight, int radius) {
  const DepthField& first = layers[0];
  const uint32_t farLevels = farLevelsFor(first.width, first.height);
  float weightSum = flatWeight;
  for (const float w : weights) {
    weightSum += w;
  }
  std::vector<std::vector<DepthField>> pyramids;
  for (const DepthField& layer : layers) {
    pyramids.push_back(buildPyramid(layer, farLevels));
  }
  DepthField blend;
  blend.width = first.width;
  blend.height = first.height;
  blend.depth.assign(first.depth.size(), 0.0f);
  for (size_t i = 0; i < blend.depth.size(); ++i) {
    for (size_t layer = 0; layer < layers.size(); ++layer) {
      blend.depth[i] += weights[layer] * layers[layer].depth[i] / weightSum;
    }
  }

  constexpr float kNoBound = 1e30f;
  int relaxedTooWide = 0;
  double tightnessSum = 0.0;
  int tightnessCount = 0;
  for (int y = 0; y < blend.height; ++y) {
    for (int x = 0; x < blend.width; ++x) {
      float slope = 0.0f;
      float relaxedSlope = 0.0f;
      bool linear = false;
      bool relaxedLinear = false;
      for (size_t layer = 0; layer < layers.size(); ++layer) {
        const float cone = storedCone(generateRatio(pyramids[layer], x, y, radius, 0, farLevels));
        const float relaxedCone = storedCone(generateRatio(pyramids[layer], x, y, radius, 8, farLevels));
        if (cone > 0.0f) {
          slope += weights[layer] / weightSum / cone;
        } else {
          linear = true;
        }
        if (relaxedCone > 0.0f) {
          relaxedSlope += weights[layer] / weightSum / relaxedCone;
        } else {
          relaxedLinear = true;
        }
      }
      const float combined = linear ? 0.0f : (slope > 0.0f ? 1.0f / slope : kNoBound);
      const float relaxedCombined = relaxedLinear ? 0.0f : (relaxedSlope > 0.0f ? 1.0f / relaxedSlope : kNoBound);
      const float reference = bruteForceConservativeRatio(blend, x, y, kNoBound);
      if (combined > reference * (1.0f + 1e-4f) + 1e-6f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") blended cone " << combined << " > brute force "
                  << reference << std::endl;
        throw std::runtime_error("blended terrain cone is not conservative");
      }
      if (relaxedCombined > reference * (1.0f + 1e-4f) + 1e-6f) {
        ++relaxedTooWide;
      }
      if (reference > 0.0f && reference < kNoBound) {
        tightnessSum += std::min(combined, reference) / reference;
        ++tightnessCount;
      }
    }
  }
  std::printf("  %-28s %3dx%-3d radius %2d, %zu layers + flat %.2f: sound, tightness %.3f, relaxed too wide at %d texels\n",
              name.c_str(), blend.width, blend.height, radius, layers.size(), flatWeight / weightSum,
              tightnessCount > 0 ? tightnessSum / tightnessCount : 1.0, relaxedTooWide);
}

// CPU copy of sccrTerrainReduceLayers (pom_sccr_terrain.slangh): the maxLayers layers with the
// largest weights (ties: the lower index) are kept, the others leave the blend with their weight. Returns the kept mask.
std::vector<bool> reduceLayers(const std::vector<float>& weights, uint32_t maxLayers) {
  std::vector<bool> keep(weights.size(), true);
  if (maxLayers == 0 || weights.size() <= maxLayers) {
    return keep;
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    uint32_t rank = 0;
    for (size_t j = 0; j < weights.size(); ++j) {
      if (j != i && (weights[j] > weights[i] || (weights[j] == weights[i] && j < i))) {
        ++rank;
      }
    }
    keep[i] = rank < maxLayers;
  }
  return keep;
}

// The reduced blend is a blend of the kept layers with their own weights and the flat weight unchanged (W minus the
// dropped weights), so checkBlend on it checks the reduced field's cone exactly as the shader computes it. Also checks
// that the kept set is the top maxLayers, and prints how far the reduced field lies from the full one (in depth units).
void checkReducedBlend(const std::string& name, const std::vector<DepthField>& layers, const std::vector<float>& weights,
                       float flatWeight, int radius, uint32_t maxLayers) {
  const std::vector<bool> keep = reduceLayers(weights, maxLayers);
  std::vector<DepthField> keptLayers;
  std::vector<float> keptWeights;
  float smallestKept = 1e30f, largestDropped = -1.0f;
  for (size_t i = 0; i < layers.size(); ++i) {
    if (keep[i]) {
      keptLayers.push_back(layers[i]);
      keptWeights.push_back(weights[i]);
      smallestKept = std::min(smallestKept, weights[i]);
    } else {
      largestDropped = std::max(largestDropped, weights[i]);
    }
  }
  if (keptLayers.size() != std::min<size_t>(layers.size(), maxLayers) || largestDropped > smallestKept) {
    throw std::runtime_error(name + ": reduced layers are not the heaviest");
  }
  float fullSum = flatWeight, keptSum = flatWeight;
  for (size_t i = 0; i < layers.size(); ++i) {
    fullSum += weights[i];
    keptSum += keep[i] ? weights[i] : 0.0f;
  }
  float maxDifference = 0.0f;
  for (size_t t = 0; t < layers[0].depth.size(); ++t) {
    float full = 0.0f, reduced = 0.0f;
    for (size_t i = 0; i < layers.size(); ++i) {
      full += weights[i] * layers[i].depth[t] / fullSum;
      reduced += keep[i] ? weights[i] * layers[i].depth[t] / keptSum : 0.0f;
    }
    maxDifference = std::max(maxDifference, std::abs(full - reduced));
  }
  std::printf("  %-28s top %u of %zu layers: field moves by at most %.3f of the depth\n", name.c_str(), maxLayers,
              layers.size(), maxDifference);
  checkBlend(name + " top " + std::to_string(maxLayers), keptLayers, keptWeights, flatWeight, radius);
}

// pom_cone_coarse.comp.slang's ring (POM_CONE_COARSE_RING) and its levels (PomConeMapManager::recordCoarse): the
// coarsest is the first where the ring reaches a texture width.
constexpr int kCoarseRing = 3;

uint32_t coarseTopLevel(int width, uint32_t shift) {
  uint32_t widthLog2 = 0;
  while ((1 << (widthLog2 + 1)) <= width) {
    ++widthLog2;
  }
  uint32_t topLevel = shift;
  while (topLevel < widthLog2 && (uint32_t(kCoarseRing) << topLevel) < uint32_t(width)) {
    ++topLevel;
  }
  return topLevel;
}

// CPU copy of pom_cone_coarse.comp.slang, before quantisation: one conservative cone per cell of 2^shift texels, for
// every texel of the cell (square power-of-two fields).
std::vector<float> coarseRatios(const DepthField& field, uint32_t shift) {
  const uint32_t topLevel = coarseTopLevel(field.width, shift);
  const std::vector<DepthField> pyramid = buildPyramid(field, topLevel);
  const int cellSize = 1 << shift;
  const int cells = field.width >> shift;
  const float texelUv = 1.0f / float(field.width);

  // POM_CONE_COARSE_CELLS: the largest 8-neighbour depth difference over the pairs whose first texel is in the cell,
  // and the cell's largest depth.
  std::vector<float> slopes(size_t(cells) * cells, 0.0f);
  std::vector<float> deepests(size_t(cells) * cells, 0.0f);
  for (int cy = 0; cy < cells; ++cy) {
    for (int cx = 0; cx < cells; ++cx) {
      float slope = 0.0f;
      float deepest = 0.0f;
      for (int y = cy * cellSize; y < (cy + 1) * cellSize; ++y) {
        for (int x = cx * cellSize; x < (cx + 1) * cellSize; ++x) {
          const float depth = field.at(x, y);
          deepest = std::max(deepest, depth);
          slope = std::max(slope, std::abs(depth - field.at(x + 1, y)));
          slope = std::max(slope, std::abs(depth - field.at(x, y + 1)));
          slope = std::max(slope, std::abs(depth - field.at(x + 1, y + 1)));
          slope = std::max(slope, std::abs(depth - field.at(x - 1, y + 1)));
        }
      }
      slopes[size_t(cy) * cells + cx] = slope;
      deepests[size_t(cy) * cells + cx] = deepest;
    }
  }
  const auto wrapCell = [cells](int v) { return ((v % cells) + cells) % cells; };

  // POM_CONE_COARSE_CONES.
  std::vector<float> cellRatios(size_t(cells) * cells, 1.0f);
  for (int cy = 0; cy < cells; ++cy) {
    for (int cx = 0; cx < cells; ++cx) {
      const float deepest = deepests[size_t(cy) * cells + cx];
      float ratio = 1.0f;
      if (deepest > 0.0f) {
        float slope = 0.0f;
        float least = 1.0f;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            slope = std::max(slope, slopes[size_t(wrapCell(cy + dy)) * cells + wrapCell(cx + dx)]);
            least = std::min(least, pyramid[shift].at(cx + dx, cy + dy));
          }
        }
        if (least < deepest && slope > 0.0f) {
          ratio = std::min(ratio, texelUv * std::max(1.0f / slope, 1.0f / (deepest - least)));
        }

        const int firstX = cx * cellSize;
        const int firstY = cy * cellSize;
        const int lastX = firstX + cellSize - 1;
        const int lastY = firstY + cellSize - 1;
        const int inner = std::max(2, (kCoarseRing - 1) / 2 + 1);
        for (uint32_t level = shift; level <= topLevel; ++level) {
          const int up = int(level - shift);
          const int size = 1 << level;
          const int ancestorX = cx >> up;
          const int ancestorY = cy >> up;
          for (int oy = -kCoarseRing; oy <= kCoarseRing; ++oy) {
            for (int ox = -kCoarseRing; ox <= kCoarseRing; ++ox) {
              if (std::max(std::abs(ox), std::abs(oy)) < inner) {
                continue;
              }
              const int otherX = ancestorX + ox;
              const int otherY = ancestorY + oy;
              const float leastDepth = pyramid[level].at(otherX, otherY);
              if (leastDepth >= deepest) {
                continue;
              }
              const int otherFirstX = otherX * size;
              const int otherFirstY = otherY * size;
              const int gapX = std::max({ otherFirstX - lastX, firstX - (otherFirstX + size - 1), 0 });
              const int gapY = std::max({ otherFirstY - lastY, firstY - (otherFirstY + size - 1), 0 });
              ratio = std::min(ratio, std::hypot(float(gapX), float(gapY)) * texelUv / (deepest - leastDepth));
            }
          }
        }
      }
      cellRatios[size_t(cy) * cells + cx] = ratio;
    }
  }

  std::vector<float> ratios(field.depth.size());
  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      ratios[size_t(y) * field.width + x] = cellRatios[size_t(y >> shift) * cells + (x >> shift)];
    }
  }
  return ratios;
}

// Coarse cones never exceed the conservative cone, nor (with relaxed set; the brute-force relaxed cone is slow) the
// relaxed one: a cone holding no point of the field is a valid relaxed cone. Also the state while a map generates,
// where G holds generated cones in the finished tiles and coarse ones elsewhere: each texel's stored value is checked
// against its variant's reference (relaxed tiles against the relaxed cone, conservative ones against the
// conservative cone).
void checkCoarse(const std::string& name, const DepthField& field, uint32_t shift, bool relaxed) {
  const std::vector<float> coarse = coarseRatios(field, shift);
  constexpr int kRadius = 8;
  constexpr int kTile = 8;
  const uint32_t farLevels = farLevelsFor(field.width, field.height);
  const std::vector<DepthField> pyramid = buildPyramid(field, farLevels);
  double tightnessSum = 0.0;
  int tightnessCount = 0;
  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      const float ratio = coarse[size_t(y) * field.width + x];
      const float stored = storedCone(ratio);
      if (stored > ratio) {
        throw std::runtime_error("quantised coarse cone exceeds the computed one");
      }
      const float conservative = bruteForceConservativeRatio(field, x, y, 1.0f);
      if (ratio > conservative * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") coarse " << ratio << " > brute-force conservative "
                  << conservative << std::endl;
        throw std::runtime_error("coarse cone is not conservative");
      }
      // Tiles generated so far: a fixed scatter of about a third of them.
      const bool generated = ((x / kTile) * 7 + (y / kTile) * 3) % 3 == 0;
      const float mixedConservative = generated ? storedCone(generateRatio(pyramid, x, y, kRadius, 0, farLevels)) : stored;
      if (mixedConservative > conservative * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") partly generated conservative map " << mixedConservative
                  << " > brute force " << conservative << std::endl;
        throw std::runtime_error("partly generated conservative cone map is not conservative");
      }
      if (relaxed) {
        const float relaxedReference = bruteForceRatio(field, x, y);
        if (conservative > relaxedReference * (1.0f + 1e-5f) + 1e-7f) {
          std::cerr << name << ": texel (" << x << ", " << y << ") conservative " << conservative << " > relaxed "
                    << relaxedReference << std::endl;
          throw std::runtime_error("the conservative cone is wider than the relaxed one");
        }
        if (ratio > relaxedReference * (1.0f + 1e-5f) + 1e-7f) {
          throw std::runtime_error("coarse cone exceeds the relaxed cone");
        }
        const float mixedRelaxed = generated ? storedCone(generateRatio(pyramid, x, y, kRadius, 8, farLevels)) : stored;
        if (mixedRelaxed > relaxedReference * (1.0f + 1e-5f) + 1e-7f) {
          std::cerr << name << ": texel (" << x << ", " << y << ") partly generated relaxed map " << mixedRelaxed
                    << " > brute force " << relaxedReference << std::endl;
          throw std::runtime_error("partly generated relaxed cone map is not sound");
        }
      }
      if (conservative > 0.0f) {
        tightnessSum += ratio / conservative;
        ++tightnessCount;
      }
    }
  }
  std::printf("  %-28s %3dx%-3d cells of %2d, %s: sound, tightness against conservative %.3f\n", name.c_str(),
              field.width, field.height, 1 << shift, relaxed ? "both variants" : "conservative",
              tightnessCount > 0 ? tightnessSum / tightnessCount : 1.0);
}

// The stored 8-bit G (sqrt of the ratio, rounded down) as an integer.
int storedLevel(float ratio) {
  return int(std::floor(std::sqrt(std::clamp(ratio, 0.0f, 1.0f)) * 255.0f));
}

// The hierarchical search on a square power-of-two field. The conservative variant never exceeds the
// brute-force conservative cone (sound) and stores the same G up to one step of rounding (the search is exact; the
// step allows for float rounding between two equal formulas); with the visit cap lowered to 16 it is still sound. The
// default relaxed near field (2 texels, 4 steps) is checked against the relaxed brute force in checkField; here its
// cone is never narrower than the conservative one (it takes the conservative bounds beyond the near field and relaxed
// ones, never smaller, in it). Reported: the mean stored cone (G squared, UV per unit of depth) of coarse cones, the old
// windowed generator (relaxed 8 texels / 8 steps, and conservative), the new conservative variant and the new default,
// and the reads per texel of each generator.
void checkHierarchical(const std::string& name, const DepthField& field) {
  const uint32_t farLevels = farLevelsFor(field.width, field.height);
  if (farLevels == 0) {
    throw std::runtime_error("checkHierarchical needs a square power-of-two field");
  }
  const std::vector<DepthField> pyramid = buildPyramid(field, farLevels);
  const uint32_t windowedLevels = windowedFarLevelsFor(field.width, field.height, 8);
  const std::vector<float> coarse = coarseRatios(field, 3);
  double coarseSum = 0.0;
  double oldRelaxedSum = 0.0;
  double oldConservativeSum = 0.0;
  double conservativeSum = 0.0;
  double relaxedSum = 0.0;
  uint64_t oldRelaxedReads = 0;
  uint64_t oldConservativeReads = 0;
  uint64_t conservativeReads = 0;
  uint64_t relaxedReads = 0;
  int exact = 0;
  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      const float reference = bruteForceConservativeRatio(field, x, y, 1.0f);
      const float conservative = generateRatio(pyramid, x, y, 2, 0, farLevels, &conservativeReads);
      if (storedCone(conservative) > reference * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") hierarchical conservative " << conservative
                  << " > brute force " << reference << std::endl;
        throw std::runtime_error("hierarchical conservative cone is not conservative");
      }
      const int storedReference = storedLevel(reference);
      const int storedConservative = storedLevel(conservative);
      if (storedConservative < storedReference - 1) {
        std::cerr << name << ": texel (" << x << ", " << y << ") hierarchical conservative stores " << storedConservative
                  << ", brute force " << storedReference << std::endl;
        throw std::runtime_error("hierarchical conservative cone is not exact");
      }
      exact += storedConservative == storedReference ? 1 : 0;
      const float capped = generateRatio(pyramid, x, y, 2, 0, farLevels, nullptr, 16);
      if (storedCone(capped) > reference * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") capped search " << capped << " > brute force "
                  << reference << std::endl;
        throw std::runtime_error("hierarchical search with a visit cap is not conservative");
      }
      const float relaxed = generateRatio(pyramid, x, y, 2, 4, farLevels, &relaxedReads);
      if (storedLevel(relaxed) < storedConservative - 1) {
        std::cerr << name << ": texel (" << x << ", " << y << ") relaxed " << relaxed << " < conservative "
                  << conservative << std::endl;
        throw std::runtime_error("hierarchical relaxed cone is narrower than the conservative one");
      }
      coarseSum += storedCone(coarse[size_t(y) * field.width + x]);
      oldRelaxedSum += storedCone(windowedRatio(pyramid, x, y, 8, 8, windowedLevels, &oldRelaxedReads));
      oldConservativeSum += storedCone(windowedRatio(pyramid, x, y, 8, 0, windowedLevels, &oldConservativeReads));
      conservativeSum += storedCone(conservative);
      relaxedSum += storedCone(relaxed);
    }
  }
  const double texels = double(field.width) * field.height;
  std::printf("  %-28s %3dx%-3d hierarchical: sound, exact at %.1f%% of texels; mean cone: coarse %.4f, old relaxed "
              "%.4f, old conservative %.4f, new conservative %.4f, new relaxed (2/4) %.4f; reads per texel: old "
              "%.0f / %.0f, new %.0f / %.0f\n",
              name.c_str(), field.width, field.height, 100.0 * exact / texels, coarseSum / texels,
              oldRelaxedSum / texels, oldConservativeSum / texels, conservativeSum / texels, relaxedSum / texels,
              oldRelaxedReads / texels, oldConservativeReads / texels, conservativeReads / texels,
              relaxedReads / texels);
}

void checkField(const std::string& name, const DepthField& field, int radius, uint32_t raySteps) {
  const uint32_t farLevels = farLevelsFor(field.width, field.height);
  const std::vector<DepthField> pyramid = buildPyramid(field, farLevels);
  double tightnessSum = 0.0;
  int tightnessCount = 0;
  for (int y = 0; y < field.height; ++y) {
    for (int x = 0; x < field.width; ++x) {
      const float generated = generateRatio(pyramid, x, y, radius, raySteps, farLevels);
      const float reference = bruteForceRatio(field, x, y);
      if (storedCone(generated) > reference * (1.0f + 1e-5f) + 1e-7f) {
        std::cerr << name << ": texel (" << x << ", " << y << ") generated " << generated << " > brute force "
                  << reference << std::endl;
        throw std::runtime_error("cone generator is not conservative");
      }
      // The quantised value the cone texture stores must stay below the computed ratio.
      const float stored = std::floor(std::sqrt(std::clamp(generated, 0.0f, 1.0f)) * 255.0f) / 255.0f;
      if (stored * stored > generated) {
        throw std::runtime_error("quantised cone exceeds the computed one");
      }
      if (reference > 0.0f) {
        tightnessSum += storedCone(generated) / reference;
        ++tightnessCount;
      }
    }
  }
  std::printf("  %-28s %3dx%-3d radius %2d, ray steps %2u, far levels %u: sound, tightness %.3f\n", name.c_str(),
              field.width, field.height, radius, raySteps, farLevels,
              tightnessCount > 0 ? tightnessSum / tightnessCount : 1.0);
}

// The march's crossing refinement (pomConeStepMarchRefined in pom_sccr.slangh, copied below): bisections, the last of
// binarySteps a false-position step unless refinePrecision is reached first, then a final secant. Positions are
// (u, v, depth); the field is sampled bilinearly with wrap, as the shader's SampleLevel does.
struct MarchPoint {
  float u, v, z;
};

float sampleBilinear(const DepthField& field, float u, float v) {
  const float fx = u * field.width - 0.5f;
  const float fy = v * field.height - 0.5f;
  const int x0 = int(std::floor(fx));
  const int y0 = int(std::floor(fy));
  const float tx = fx - x0;
  const float ty = fy - y0;
  const float top = field.at(x0, y0) * (1.0f - tx) + field.at(x0 + 1, y0) * tx;
  const float bottom = field.at(x0, y0 + 1) * (1.0f - tx) + field.at(x0 + 1, y0 + 1) * tx;
  return top * (1.0f - ty) + bottom * ty;
}

MarchPoint mix(const MarchPoint& a, const MarchPoint& b, float t) {
  return { a.u + (b.u - a.u) * t, a.v + (b.v - a.v) * t, a.z + (b.z - a.z) * t };
}

float gapAt(const DepthField& field, const MarchPoint& p) {
  return sampleBilinear(field, p.u, p.v) - p.z;
}

struct Refined {
  MarchPoint result;
  MarchPoint above;
  MarchPoint below;
  uint32_t bisections;
  bool falsePosition;
};

Refined refine(const DepthField& field, MarchPoint above, float aboveGap, MarchPoint below, float belowGap,
               uint32_t binarySteps, float refinePrecision) {
  Refined out;
  out.bisections = binarySteps > 0 ? binarySteps - 1 : 0;
  out.falsePosition = binarySteps > 0;
  if (refinePrecision > 0.0f) {
    const float bracket = below.z - above.z;
    const uint32_t needed = bracket > refinePrecision ? uint32_t(std::ceil(std::log2(bracket / refinePrecision))) : 0u;
    if (needed <= out.bisections) {
      out.bisections = needed;
      out.falsePosition = false;
    }
  }
  for (uint32_t j = 0; j < out.bisections; ++j) {
    const MarchPoint middle = mix(above, below, 0.5f);
    const float middleGap = gapAt(field, middle);
    if (middleGap > 0.0f) {
      above = middle;
      aboveGap = middleGap;
    } else {
      below = middle;
      belowGap = middleGap;
    }
  }
  if (out.falsePosition) {
    const MarchPoint secant = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
    const float secantGap = gapAt(field, secant);
    if (secantGap > 0.0f) {
      above = secant;
      aboveGap = secantGap;
    } else {
      below = secant;
      belowGap = secantGap;
    }
  }
  out.result = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
  out.above = above;
  out.below = below;
  return out;
}

// Soundness: the refined bracket still holds a crossing (above the field at one end, at or below it at the other), the
// result lies on it, and it is no longer than bisection guarantees (or refinePrecision, when that ended it). So the
// reported hit is never farther than that bound from a point at or under the surface. Precision against a dense
// reference crossing is reported, for the old rule (6 bisections) and the default (4 bisections and a false-position
// step), not asserted: a bracket over a thin feature can hold several crossings, and any of them is a valid answer.
void checkRefinement(const std::string& name, const DepthField& field) {
  std::mt19937 rng(99);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  const float totalHeight = 0.5f;
  double errorOld = 0.0, errorNew = 0.0, errorLod = 0.0;
  int count = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const float angle = uniform(rng) * 2.0f * kPi;
    const float viewZ = 0.2f + 0.77f * uniform(rng);
    const float slope = std::sqrt(1.0f - viewZ * viewZ) / viewZ * totalHeight;
    const float du = std::cos(angle) * slope;
    const float dv = std::sin(angle) * slope;
    const float minStep = 1.0f / 16.0f;
    MarchPoint p { uniform(rng), uniform(rng), 0.0f };
    MarchPoint prev = p;
    float prevGap = 0.0f;
    float gap = gapAt(field, p);
    bool hasPrev = false;
    while (gap > 0.0f && p.z < 1.0f) {
      prev = p;
      prevGap = gap;
      hasPrev = true;
      const float advance = std::min(minStep, 1.0f - p.z);
      p = { p.u + du * advance, p.v + dv * advance, p.z + advance };
      gap = gapAt(field, p);
    }
    if (gap > 0.0f || !hasPrev) {
      continue;
    }
    const float bracket = p.z - prev.z;
    // A dense reference: the first sign change on the step.
    MarchPoint reference = p;
    float last = prevGap;
    for (int k = 1; k <= 4096; ++k) {
      const MarchPoint q = mix(prev, p, float(k) / 4096.0f);
      const float g = gapAt(field, q);
      if (g <= 0.0f) {
        reference = mix(mix(prev, p, float(k - 1) / 4096.0f), q, last / std::max(last - g, 1e-9f));
        break;
      }
      last = g;
    }
    const float lodPrecision = bracket / 8.0f;
    const struct { uint32_t steps; float refinePrecision; double* error; } rules[] = {
      { 6, 0.0f, nullptr },  // old rule's bound only; its precision is measured below with plain bisections
      { 5, 0.0f, &errorNew },
      { 5, lodPrecision, &errorLod },
    };
    for (const auto& rule : rules) {
      const Refined r = refine(field, prev, prevGap, p, gap, rule.steps, rule.refinePrecision);
      const float finalBracket = r.below.z - r.above.z;
      const float bound = r.falsePosition || rule.refinePrecision <= 0.0f ?
        bracket / float(1u << r.bisections) : std::max(rule.refinePrecision, bracket / float(1u << r.bisections));
      if (gapAt(field, r.above) <= 0.0f || gapAt(field, r.below) > 0.0f) {
        throw std::runtime_error(name + ": refinement lost its bracket");
      }
      if (finalBracket > bound * (1.0f + 1e-4f) + 1e-7f) {
        throw std::runtime_error(name + ": refined bracket longer than bisection guarantees");
      }
      if (r.result.z < r.above.z - 1e-6f || r.result.z > r.below.z + 1e-6f) {
        throw std::runtime_error(name + ": refined hit outside its bracket");
      }
      if (rule.refinePrecision > 0.0f && !r.falsePosition && finalBracket > rule.refinePrecision * (1.0f + 1e-4f)) {
        throw std::runtime_error(name + ": refinement stopped short of its precision");
      }
      if (rule.error != nullptr) {
        *rule.error += std::hypot(r.result.u - reference.u, r.result.v - reference.v) * field.width;
      }
    }
    // The old rule: 6 bisections and the final secant (no false-position step).
    MarchPoint above = prev, below = p;
    float aboveGap = prevGap, belowGap = gap;
    for (int j = 0; j < 6; ++j) {
      const MarchPoint middle = mix(above, below, 0.5f);
      const float middleGap = gapAt(field, middle);
      if (middleGap > 0.0f) {
        above = middle;
        aboveGap = middleGap;
      } else {
        below = middle;
        belowGap = middleGap;
      }
    }
    const MarchPoint old = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
    errorOld += std::hypot(old.u - reference.u, old.v - reference.v) * field.width;
    ++count;
  }
  std::printf("  %-28s %3dx%-3d refinement: sound; mean error in texels: 6 bisections %.4f, 4+false position %.4f, "
              "to 1/8 of the step %.4f (%d rays)\n", name.c_str(), field.width, field.height,
              count > 0 ? errorOld / count : 0.0, count > 0 ? errorNew / count : 0.0, count > 0 ? errorLod / count : 0.0,
              count);
}

// The self-shadow march (pom_sccr.slangh), linear steps of minStep from depth 0 down to the texel. Old rule
// (pomConeStepMarchRefined, then the visibility of its hit): the march samples the texel too, and a gap <= 0 there
// counts as a crossing refined over the last step. New rule (pomConeStepVisibility): reaching the texel is lit, and a
// bracket wholly before the visibility ramp (0) or wholly inside its lit end (1) skips the rest of the refinement.
struct VisibilityResult {
  float visibility;
  int fetches;
  bool crossedBeforeTexel; // old rule: the crossing it refined lies before the texel
};

float visibilityOf(float texelDepth, float hitDepth, float minStep) {
  return std::clamp(1.5f - (texelDepth - hitDepth) / minStep, 0.0f, 1.0f);
}

VisibilityResult oldVisibility(const DepthField& field, MarchPoint start, float du, float dv, float texelDepth,
                               float minStep, uint32_t binarySteps, float refinePrecision) {
  VisibilityResult out { 1.0f, 0, false };
  MarchPoint p = start;
  MarchPoint prev = p;
  float prevGap = 0.0f;
  float gap = 0.0f;
  bool hasPrev = false;
  bool crossed = false;
  const uint32_t maxSteps = uint32_t(std::ceil(texelDepth / minStep)) + 1u;
  for (uint32_t i = 0; i < maxSteps; ++i) {
    gap = gapAt(field, p);
    ++out.fetches;
    if (gap <= 0.0f) {
      crossed = true;
      break;
    }
    if (p.z >= texelDepth) {
      break;
    }
    const float advance = std::min(minStep, texelDepth - p.z);
    prev = p;
    prevGap = gap;
    hasPrev = true;
    p = { p.u + du * advance, p.v + dv * advance, p.z + advance };
  }
  if (!crossed) {
    return out;
  }
  // At the texel up to the rounding of the last step's sum.
  out.crossedBeforeTexel = p.z < texelDepth - 1e-5f;
  if (!hasPrev) {
    out.visibility = visibilityOf(texelDepth, p.z, minStep);
    return out;
  }
  const Refined r = refine(field, prev, prevGap, p, gap, binarySteps, refinePrecision);
  out.fetches += int(r.bisections) + (r.falsePosition ? 1 : 0);
  out.visibility = visibilityOf(texelDepth, r.result.z, minStep);
  return out;
}

VisibilityResult newVisibility(const DepthField& field, MarchPoint start, float du, float dv, float texelDepth,
                               float minStep, uint32_t binarySteps, float refinePrecision) {
  VisibilityResult out { 1.0f, 0, false };
  const float occludedDepth = texelDepth - 1.5f * minStep;
  const float litDepth = texelDepth - 0.5f * minStep;
  MarchPoint p = start;
  MarchPoint above = start;
  float aboveGap = 0.0f;
  float gap = 1.0f;
  bool hasPrev = false;
  const uint32_t maxSteps = uint32_t(std::ceil(texelDepth / minStep)) + 1u;
  for (uint32_t i = 0; i < maxSteps; ++i) {
    if (p.z >= texelDepth) {
      return out;
    }
    gap = gapAt(field, p);
    ++out.fetches;
    if (gap <= 0.0f) {
      break;
    }
    const float remaining = texelDepth - p.z;
    above = p;
    aboveGap = gap;
    hasPrev = true;
    const float advance = std::min(minStep, remaining);
    p = { p.u + du * advance, p.v + dv * advance, p.z + advance };
    if (minStep >= remaining) {
      p.z = texelDepth;
    }
  }
  if (gap > 0.0f) {
    return out;
  }
  if (!hasPrev) {
    out.visibility = visibilityOf(texelDepth, p.z, minStep);
    return out;
  }
  MarchPoint below = p;
  float belowGap = gap;
  if (below.z <= occludedDepth) {
    out.visibility = 0.0f;
    return out;
  }
  if (above.z >= litDepth) {
    return out;
  }
  uint32_t bisections = binarySteps > 0 ? binarySteps - 1 : 0;
  bool falsePosition = binarySteps > 0;
  if (refinePrecision > 0.0f) {
    const float bracket = below.z - above.z;
    const uint32_t needed = bracket > refinePrecision ? uint32_t(std::ceil(std::log2(bracket / refinePrecision))) : 0u;
    if (needed <= bisections) {
      bisections = needed;
      falsePosition = false;
    }
  }
  for (uint32_t j = 0; j < bisections; ++j) {
    const MarchPoint middle = mix(above, below, 0.5f);
    const float middleGap = gapAt(field, middle);
    ++out.fetches;
    if (middleGap > 0.0f) {
      above = middle;
      aboveGap = middleGap;
      if (above.z >= litDepth) {
        return out;
      }
    } else {
      below = middle;
      belowGap = middleGap;
      if (below.z <= occludedDepth) {
        out.visibility = 0.0f;
        return out;
      }
    }
  }
  if (falsePosition) {
    const MarchPoint secant = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
    const float secantGap = gapAt(field, secant);
    ++out.fetches;
    if (secantGap > 0.0f) {
      above = secant;
      aboveGap = secantGap;
    } else {
      below = secant;
      belowGap = secantGap;
    }
  }
  const MarchPoint hit = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
  out.visibility = visibilityOf(texelDepth, hit.z, minStep);
  return out;
}

// Wherever the old rule found a crossing before the texel, both rules give the same visibility (the early returns only
// skip work whose result is already decided). Where it "crossed" at the texel itself (a gap <= 0 there by rounding),
// the new rule says lit: those rays are counted and reported. Fetches per ray are reported.
void checkVisibility(const std::string& name, const DepthField& field) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  const float totalHeight = 0.3f;
  const uint32_t binarySteps = 5;
  int rays = 0, atTexel = 0;
  long oldFetches = 0, newFetches = 0;
  for (int trial = 0; trial < 2000; ++trial) {
    const float angle = uniform(rng) * 2.0f * kPi;
    const float lightZ = 0.15f + 0.8f * uniform(rng);
    const float slope = std::sqrt(1.0f - lightZ * lightZ) / lightZ * totalHeight;
    const float du = -std::cos(angle) * slope;
    const float dv = -std::sin(angle) * slope;
    const MarchPoint texel { uniform(rng), uniform(rng), 0.0f };
    const float texelDepth = sampleBilinear(field, texel.u, texel.v);
    if (texelDepth <= 0.0f) {
      continue;
    }
    const MarchPoint start { texel.u - du * texelDepth, texel.v - dv * texelDepth, 0.0f };
    const float minStep = (trial & 1) ? 1.0f / 16.0f : 1.0f / 40.0f;
    const float refinePrecision = (trial & 2) ? 0.125f * minStep : 0.0f;
    const VisibilityResult a = oldVisibility(field, start, du, dv, texelDepth, minStep, binarySteps, refinePrecision);
    const VisibilityResult b = newVisibility(field, start, du, dv, texelDepth, minStep, binarySteps, refinePrecision);
    ++rays;
    oldFetches += a.fetches;
    newFetches += b.fetches;
    if (a.crossedBeforeTexel || a.visibility >= 1.0f) {
      if (std::abs(a.visibility - b.visibility) > 1e-4f) {
        std::cerr << name << ": ray " << trial << " visibility " << b.visibility << " != old " << a.visibility << std::endl;
        throw std::runtime_error("visibility march early returns changed a result");
      }
    } else {
      // The old value was below 1 here: a refinement over the last step that the new rule does not run.
      ++atTexel;
      if (b.visibility < 1.0f) {
        throw std::runtime_error(name + ": a crossing at the texel is not lit");
      }
    }
  }
  std::printf("  %-28s %3dx%-3d visibility: same as before except %d of %d rays (a crossing at the texel by rounding, "
              "was < 1, now lit); fetches per ray %.2f -> %.2f\n", name.c_str(), field.width, field.height, atTexel,
              rays, rays > 0 ? double(oldFetches) / rays : 0.0, rays > 0 ? double(newFetches) / rays : 0.0);
}

// CPU copy of sccrPhongEdgeCurvature and sccrSurfaceBaseCreate (surface_interaction_sccr.slangh), in double.
struct Vec3 {
  double x, y, z;
};
Vec3 operator-(const Vec3& a, const Vec3& b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec3 operator+(const Vec3& a, const Vec3& b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Vec3 operator*(const Vec3& a, double s) { return { a.x * s, a.y * s, a.z * s }; }
double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
Vec3 normalized(const Vec3& a) { return a * (1.0 / std::sqrt(dot(a, a))); }

struct EdgeCurvature {
  double c01, c02, c12, bound;
};

EdgeCurvature phongEdgeCurvature(const Vec3 p[3], const Vec3 n[3], const Vec3& triangleNormal) {
  const double tilt0 = dot(n[0], triangleNormal);
  const double tilt1 = dot(n[1], triangleNormal);
  const double tilt2 = dot(n[2], triangleNormal);
  const Vec3 edge01 = p[1] - p[0];
  const Vec3 edge02 = p[2] - p[0];
  const Vec3 edge12 = p[2] - p[1];
  EdgeCurvature c;
  c.c01 = 0.5 * (dot(edge01, n[0]) * tilt0 - dot(edge01, n[1]) * tilt1);
  c.c02 = 0.5 * (dot(edge02, n[0]) * tilt0 - dot(edge02, n[2]) * tilt2);
  c.c12 = 0.5 * (dot(edge12, n[1]) * tilt1 - dot(edge12, n[2]) * tilt2);
  c.bound = std::max({ std::abs(c.c01), std::abs(c.c02), std::abs(c.c12) }) / 3.0;
  return c;
}

// The patch itself, from barycentrics.
double phongHeight(const EdgeCurvature& c, double b0, double b1, double b2) {
  return -(b0 * b1 * c.c01 + b0 * b2 * c.c02 + b1 * b2 * c.c12);
}

struct BaseExpansion {
  double h0, gu, gv, huu, huv, hvv;
};

BaseExpansion baseExpansion(const double uv[3][2], const double bary[3], const EdgeCurvature& c) {
  const double duv1[2] = { uv[1][0] - uv[0][0], uv[1][1] - uv[0][1] };
  const double duv2[2] = { uv[2][0] - uv[0][0], uv[2][1] - uv[0][1] };
  const double invDeterminant = 1.0 / (duv1[0] * duv2[1] - duv2[0] * duv1[1]);
  const double m1[2] = { duv2[1] * invDeterminant, -duv2[0] * invDeterminant };
  const double m2[2] = { -duv1[1] * invDeterminant, duv1[0] * invDeterminant };
  const double m0[2] = { -(m1[0] + m2[0]), -(m1[1] + m2[1]) };
  BaseExpansion e;
  e.h0 = -(bary[0] * bary[1] * c.c01 + bary[0] * bary[2] * c.c02 + bary[1] * bary[2] * c.c12);
  const double d0 = c.c01 * bary[1] + c.c02 * bary[2];
  const double d1 = c.c01 * bary[0] + c.c12 * bary[2];
  const double d2 = c.c02 * bary[0] + c.c12 * bary[1];
  e.gu = -(m0[0] * d0 + m1[0] * d1 + m2[0] * d2);
  e.gv = -(m0[1] * d0 + m1[1] * d1 + m2[1] * d2);
  e.huu = -2.0 * (c.c01 * m0[0] * m1[0] + c.c02 * m0[0] * m2[0] + c.c12 * m1[0] * m2[0]);
  e.hvv = -2.0 * (c.c01 * m0[1] * m1[1] + c.c02 * m0[1] * m2[1] + c.c12 * m1[1] * m2[1]);
  e.huv = -(c.c01 * (m0[0] * m1[1] + m1[0] * m0[1]) + c.c02 * (m0[0] * m2[1] + m2[0] * m0[1]) +
            c.c12 * (m1[0] * m2[1] + m2[0] * m1[1]));
  return e;
}

double evalExpansion(const BaseExpansion& e, double du, double dv) {
  return e.h0 + e.gu * du + e.gv * dv + 0.5 * (e.huu * du * du + 2.0 * e.huv * du * dv + e.hvv * dv * dv);
}

void checkBaseSurface() {
  std::mt19937 rng(4321);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  double worstExpansion = 0.0;
  for (int trial = 0; trial < 2000; ++trial) {
    Vec3 p[3];
    Vec3 n[3];
    double uv[3][2];
    for (int i = 0; i < 3; ++i) {
      p[i] = { uniform(rng) * 100.0, uniform(rng) * 100.0, uniform(rng) * 10.0 };
      uv[i][0] = uniform(rng) * 4.0;
      uv[i][1] = uniform(rng) * 4.0;
    }
    const Vec3 triangleNormal = normalized(cross(p[1] - p[0], p[2] - p[0]));
    for (int i = 0; i < 3; ++i) {
      n[i] = normalized(triangleNormal + Vec3 { uniform(rng), uniform(rng), uniform(rng) } * 0.5);
    }
    const double determinant = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1]) - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
    if (std::abs(determinant) < 1e-3) {
      continue;
    }
    const EdgeCurvature c = phongEdgeCurvature(p, n, triangleNormal);
    double b1 = unit(rng), b2 = unit(rng);
    if (b1 + b2 > 1.0) {
      b1 = 1.0 - b1;
      b2 = 1.0 - b2;
    }
    const double bary[3] = { 1.0 - b1 - b2, b1, b2 };
    const BaseExpansion e = baseExpansion(uv, bary, c);
    const double hitUv[2] = { bary[0] * uv[0][0] + bary[1] * uv[1][0] + bary[2] * uv[2][0],
                              bary[0] * uv[0][1] + bary[1] * uv[1][1] + bary[2] * uv[2][1] };
    const double scale = std::max(c.bound, 1e-9);
    // 0 at the vertices.
    for (int i = 0; i < 3; ++i) {
      const double h = evalExpansion(e, uv[i][0] - hitUv[0], uv[i][1] - hitUv[1]);
      worstExpansion = std::max(worstExpansion, std::abs(h) / scale);
      if (std::abs(h) > 1e-6 * scale + 1e-9) {
        throw std::runtime_error("base surface expansion is not 0 at a vertex");
      }
    }
    // The patch everywhere (inside and outside the triangle), and its bound inside.
    for (int k = 0; k < 16; ++k) {
      const double q1 = uniform(rng) * 1.5, q2 = uniform(rng) * 1.5;
      const double q[3] = { 1.0 - q1 - q2, q1, q2 };
      const double du = q[0] * uv[0][0] + q[1] * uv[1][0] + q[2] * uv[2][0] - hitUv[0];
      const double dv = q[0] * uv[0][1] + q[1] * uv[1][1] + q[2] * uv[2][1] - hitUv[1];
      const double direct = phongHeight(c, q[0], q[1], q[2]);
      const double expanded = evalExpansion(e, du, dv);
      worstExpansion = std::max(worstExpansion, std::abs(direct - expanded) / scale);
      if (std::abs(direct - expanded) > 1e-6 * std::max(scale, std::abs(direct)) + 1e-9) {
        throw std::runtime_error("base surface expansion differs from the Phong patch");
      }
      if (q[0] >= 0.0 && q[1] >= 0.0 && q[2] >= 0.0 && std::abs(direct) > c.bound * (1.0 + 1e-9)) {
        throw std::runtime_error("base surface exceeds its bound on the triangle");
      }
    }
  }

  // A small triangle on a sphere with radial normals: the patch at an edge midpoint is the arc's sagitta.
  const double radius = 100.0;
  double worstSagitta = 0.0;
  for (double angle : { 0.05, 0.1, 0.2 }) {
    const Vec3 n[3] = { Vec3 { 0.0, 0.0, 1.0 }, Vec3 { std::sin(angle), 0.0, std::cos(angle) },
                        Vec3 { 0.0, std::sin(angle), std::cos(angle) } };
    const Vec3 p[3] = { n[0] * radius, n[1] * radius, n[2] * radius };
    Vec3 triangleNormal = normalized(cross(p[1] - p[0], p[2] - p[0]));
    if (dot(triangleNormal, n[0]) < 0.0) {
      triangleNormal = triangleNormal * -1.0;
    }
    const EdgeCurvature c = phongEdgeCurvature(p, n, triangleNormal);
    const double chord = std::sqrt(dot(p[1] - p[0], p[1] - p[0]));
    const double sagitta = radius - std::sqrt(radius * radius - 0.25 * chord * chord);
    const double h = phongHeight(c, 0.5, 0.5, 0.0);
    const double error = std::abs(h - sagitta) / sagitta;
    worstSagitta = std::max(worstSagitta, error);
    if (!(h > 0.0) || error > 0.05) {
      throw std::runtime_error("base surface does not match a circular arc on a sphere");
    }
  }
  std::printf("  base surface: expansion error %.2e of the bound, sagitta error %.2f%%\n", worstExpansion, worstSagitta * 100.0);
}

// Live terrain weights (pom_sccr_terrain.slangh, PomLiveLayeredConeField). A terrain-like
// blend: height layers that tile every UV unit, per-vertex layer weights on a grid (bilinear between vertices, as the
// FNV weight maps reproduce the game's per-vertex blend), and a flat layer. Rays are aimed at relief points q* of the
// true field D(q) = sum_i w_i(q) D_i(q) / W(q) from many directions; a march whose weights are held at the ray's mesh
// point (the old field, PomLayeredConeField) and the live march are compared:
// - rigidity: where nothing occludes q* (a dense march of the true field finds it first), how far each march's hit
//   lies from q*. The relief is rigid when the hit does not depend on the direction the texel is seen from; the frozen
//   march's error is the sliding. Asserted: the live march finds q* within the refinement's precision on all but a
//   handful of rays, and its mean error is far below the frozen one.
// - crossings: on random rays, how often the live march's hit lies beyond the dense reference's first crossing (a
//   step that passed a crossing), with the weight-slope term G and without it. Asserted: rare with G.
struct TerrainModel {
  std::vector<DepthField> layers;               // height layers
  std::vector<std::vector<float>> storedCones;  // per layer and texel: the stored conservative cone (G squared)
  int gridSize = 0;                             // vertices per side; vertex (i, j) at uv (i, j) * spacing
  float spacing = 0.0f;
  int layerCount = 0;                           // height layers + 1 flat layer (the last)
  std::vector<float> vertexWeights;             // gridSize^2 * layerCount, normalised per vertex

  std::vector<float> weightsAt(float u, float v) const {
    const float fx = std::clamp(u / spacing, 0.0f, float(gridSize - 1));
    const float fy = std::clamp(v / spacing, 0.0f, float(gridSize - 1));
    const int x0 = std::min(int(fx), gridSize - 2);
    const int y0 = std::min(int(fy), gridSize - 2);
    const float tx = fx - x0, ty = fy - y0;
    std::vector<float> w(layerCount);
    for (int l = 0; l < layerCount; ++l) {
      auto at = [&](int x, int y) { return vertexWeights[(size_t(y) * gridSize + x) * layerCount + l]; };
      w[l] = (at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx) * (1 - ty) + (at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx) * ty;
    }
    return w;
  }

  // The cone a layer sample reads: the smallest stored cone of the four texels the bilinear sample blends.
  float coneAt(size_t layer, float u, float v) const {
    const DepthField& field = layers[layer];
    const int x0 = int(std::floor(u * field.width - 0.5f));
    const int y0 = int(std::floor(v * field.height - 0.5f));
    float cone = 1e30f;
    for (int dy = 0; dy <= 1; ++dy) {
      for (int dx = 0; dx <= 1; ++dx) {
        const int wx = (((x0 + dx) % field.width) + field.width) % field.width;
        const int wy = (((y0 + dy) % field.height) + field.height) % field.height;
        cone = std::min(cone, storedCones[layer][size_t(wy) * field.width + wx]);
      }
    }
    return cone;
  }
};

// A field as PomLiveLayeredConeField (live) or PomLayeredConeField (frozen weights) computes it. keep: the height layers
// kept; dropped height layers leave W.
struct TerrainFieldModel {
  const TerrainModel* model = nullptr;
  bool live = true;
  std::vector<float> frozenWeights;
  std::vector<bool> keep;
  float weightSlope = 0.0f;

  float depthAt(float u, float v, float* coneOut) const {
    const std::vector<float> w = live ? model->weightsAt(u, v) : frozenWeights;
    const size_t heightLayers = model->layers.size();
    float weightSum = 0.0f, depth = 0.0f, slope = 0.0f;
    bool linear = false;
    for (int l = 0; l < model->layerCount; ++l) {
      const bool isHeight = size_t(l) < heightLayers;
      if ((isHeight && !keep[l]) || w[l] <= 0.0f) {
        continue;
      }
      weightSum += w[l];
      if (!isHeight) {
        continue;
      }
      depth += w[l] * sampleBilinear(model->layers[l], u, v);
      const float cone = model->coneAt(l, u, v);
      if (cone > 0.0f) {
        slope += w[l] / cone;
      } else {
        linear = true;
      }
    }
    const float inv = weightSum > 1e-5f ? 1.0f / weightSum : 0.0f;
    slope = slope * inv + weightSlope;
    if (coneOut != nullptr) {
      *coneOut = linear ? 0.0f : (slope * 16.0f > 1.0f ? 1.0f / slope : 16.0f);
    }
    return depth * inv;
  }
};

// CPU copy of pomConeStepMarchRefined (pom_sccr.slangh), over a TerrainFieldModel.
MarchPoint terrainMarch(const TerrainFieldModel& field, MarchPoint start, float du, float dv, float minStep,
                        uint32_t coneSteps, uint32_t binarySteps, bool& crossed, int* samples = nullptr) {
  const float rayRatio = std::hypot(du, dv);
  const uint32_t maxSteps = coneSteps + uint32_t(std::ceil(1.0f / minStep)) + 1u;
  MarchPoint position = start, prev = start;
  float gap = 0.0f, prevGap = 0.0f;
  bool hasPrev = false;
  crossed = false;
  for (uint32_t i = 0; i < maxSteps; ++i) {
    float cone = 0.0f;
    gap = field.depthAt(position.u, position.v, &cone) - position.z;
    if (samples != nullptr) {
      ++*samples;
    }
    if (gap <= 0.0f) {
      crossed = true;
      break;
    }
    if (position.z >= 1.0f) {
      break;
    }
    float advance = minStep;
    if (i < coneSteps && cone > 0.0f) {
      advance = std::max(cone * gap / (rayRatio + cone), minStep);
    }
    advance = std::min(advance, 1.0f - position.z);
    prev = position;
    prevGap = gap;
    hasPrev = true;
    position = { position.u + du * advance, position.v + dv * advance, position.z + advance };
  }
  if (!crossed || !hasPrev) {
    return position;
  }
  MarchPoint above = prev, below = position;
  float aboveGap = prevGap, belowGap = gap;
  auto gapOf = [&](const MarchPoint& p) { return field.depthAt(p.u, p.v, nullptr) - p.z; };
  const uint32_t bisections = binarySteps > 0 ? binarySteps - 1 : 0;
  for (uint32_t j = 0; j < bisections; ++j) {
    const MarchPoint middle = mix(above, below, 0.5f);
    const float middleGap = gapOf(middle);
    if (middleGap > 0.0f) { above = middle; aboveGap = middleGap; } else { below = middle; belowGap = middleGap; }
  }
  if (binarySteps > 0) {
    const MarchPoint secant = mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
    const float secantGap = gapOf(secant);
    if (secantGap > 0.0f) { above = secant; aboveGap = secantGap; } else { below = secant; belowGap = secantGap; }
  }
  return mix(above, below, std::clamp(aboveGap / std::max(aboveGap - belowGap, 1e-6f), 0.0f, 1.0f));
}

// The first crossing of the true (live, every layer) field by dense steps from start, refined by bisection.
bool denseFirstCrossing(const TerrainFieldModel& field, MarchPoint start, float du, float dv, MarchPoint& hit) {
  constexpr int kSteps = 4096;
  MarchPoint prev = start;
  float prevGap = field.depthAt(start.u, start.v, nullptr) - start.z;
  if (prevGap <= 0.0f) {
    hit = start;
    return true;
  }
  for (int k = 1; k <= kSteps; ++k) {
    const float z = float(k) / kSteps;
    const MarchPoint p { start.u + du * z, start.v + dv * z, start.z + z };
    const float gap = field.depthAt(p.u, p.v, nullptr) - p.z;
    if (gap <= 0.0f) {
      MarchPoint a = prev, b = p;
      for (int j = 0; j < 20; ++j) {
        const MarchPoint m = mix(a, b, 0.5f);
        if (field.depthAt(m.u, m.v, nullptr) - m.z > 0.0f) { a = m; } else { b = m; }
      }
      hit = b;
      return true;
    }
    prev = p;
    prevGap = gap;
  }
  return false;
}

// Total variation of the blend's shares between two weight vectors (sccrTerrainShareVariation).
float shareVariation(const std::vector<float>& a, const std::vector<float>& b, const std::vector<bool>& inBlend) {
  float sumA = 0.0f, sumB = 0.0f;
  for (size_t l = 0; l < a.size(); ++l) {
    if (inBlend[l]) { sumA += a[l]; sumB += b[l]; }
  }
  float tv = 0.0f;
  for (size_t l = 0; l < a.size(); ++l) {
    if (inBlend[l]) {
      tv += std::abs((sumA > 1e-5f ? a[l] / sumA : 0.0f) - (sumB > 1e-5f ? b[l] / sumB : 0.0f));
    }
  }
  return tv;
}

// The live field of a march from start to end (sccrTerrainLiveFieldCreate with probes start, start, end): every height
// layer kept (terrainMaxLayers 0), G from the share variation over the segment times the safety factor (1).
TerrainFieldModel liveFieldFor(const TerrainModel& model, MarchPoint start, float du, float dv, float safety) {
  TerrainFieldModel field;
  field.model = &model;
  field.live = true;
  field.keep.assign(model.layers.size(), true);
  const std::vector<bool> inBlend(model.layerCount, true);
  const float length = std::hypot(du, dv);
  if (length > 1e-6f) {
    const float tv = shareVariation(model.weightsAt(start.u, start.v), model.weightsAt(start.u + du, start.v + dv), inBlend);
    field.weightSlope = safety * 0.5f * tv / length;
  }
  return field;
}

void checkLiveWeights(const std::string& name, const TerrainModel& model, float totalHeight) {
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  TerrainFieldModel truth;
  truth.model = &model;
  truth.live = true;
  truth.keep.assign(model.layers.size(), true);
  const float extent = model.spacing * float(model.gridSize - 1);
  const float minStep = 1.0f / 64.0f;
  const float texel = 1.0f / float(model.layers[0].width);

  // Rigidity.
  double frozenError = 0.0, liveError = 0.0, frozenMax = 0.0, liveMax = 0.0;
  int rigidRays = 0, liveOff = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const MarchPoint target0 { extent * (0.2f + 0.6f * uniform(rng)), extent * (0.2f + 0.6f * uniform(rng)), 0.0f };
    const float targetDepth = truth.depthAt(target0.u, target0.v, nullptr);
    const MarchPoint target { target0.u, target0.v, targetDepth };
    const float angle = uniform(rng) * 2.0f * kPi;
    const float viewZ = 0.1f + 0.8f * uniform(rng); // 26 to 84 degrees from the normal
    const float slope = std::sqrt(1.0f - viewZ * viewZ) / viewZ * totalHeight;
    const float du = std::cos(angle) * slope, dv = std::sin(angle) * slope;
    // The ray's mesh point (the mesh is the top of the field, depth 0).
    const MarchPoint start { target.u - du * targetDepth, target.v - dv * targetDepth, 0.0f };
    MarchPoint reference;
    if (!denseFirstCrossing(truth, start, du, dv, reference) ||
        std::hypot(reference.u - target.u, reference.v - target.v) > 0.01f * texel) {
      continue; // occluded (or grazing the field): q* is not what this ray sees
    }
    TerrainFieldModel frozen;
    frozen.model = &model;
    frozen.live = false;
    frozen.frozenWeights = model.weightsAt(start.u, start.v);
    frozen.keep.assign(model.layers.size(), true);
    bool crossed = false;
    const MarchPoint frozenHit = terrainMarch(frozen, start, du, dv, minStep, 16, 5, crossed);
    const MarchPoint liveHit = terrainMarch(liveFieldFor(model, start, du, dv, 1.0f), start, du, dv, minStep, 16, 5, crossed);
    const double fe = std::hypot(frozenHit.u - target.u, frozenHit.v - target.v) / texel;
    const double le = std::hypot(liveHit.u - target.u, liveHit.v - target.v) / texel;
    frozenError += fe;
    liveError += le;
    frozenMax = std::max(frozenMax, fe);
    liveMax = std::max(liveMax, le);
    if (le > 0.5) {
      ++liveOff;
    }
    ++rigidRays;
  }
  if (rigidRays < 500) {
    throw std::runtime_error(name + ": too few unoccluded rays for the rigidity check");
  }
  const double frozenMean = frozenError / rigidRays, liveMean = liveError / rigidRays;
  // Also in texels of a 1024-wide layer map (texel * 1024 / width), the size of a game's layer textures.
  const double toGameTexels = 1024.0 / double(model.layers[0].width);
  std::printf("  %-28s live weights, rigidity over %d rays (hit to the aimed relief point, texels of a 1024 map): "
              "frozen mean %.2f max %.1f, live mean %.3f max %.1f (%d over half a test texel)\n",
              name.c_str(), rigidRays, frozenMean * toGameTexels, frozenMax * toGameTexels, liveMean * toGameTexels,
              liveMax * toGameTexels, liveOff);
  if (liveOff > rigidRays / 200 || liveMean > 0.05 || liveMean * 10.0 > frozenMean) {
    throw std::runtime_error(name + ": live terrain weights do not give a rigid relief");
  }

  // Crossings passed by a step, with and without the weight-slope term, and the march samples (before refinement) of
  // each and of the frozen march.
  int passedWithSlope = 0, passedWithout = 0, rays = 0;
  int samplesWithSlope = 0, samplesWithout = 0, samplesFrozen = 0, passedFrozen = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    const MarchPoint start { extent * (0.2f + 0.6f * uniform(rng)), extent * (0.2f + 0.6f * uniform(rng)), 0.0f };
    const float angle = uniform(rng) * 2.0f * kPi;
    const float viewZ = 0.1f + 0.8f * uniform(rng);
    const float slope = std::sqrt(1.0f - viewZ * viewZ) / viewZ * totalHeight;
    const float du = std::cos(angle) * slope, dv = std::sin(angle) * slope;
    MarchPoint reference;
    if (!denseFirstCrossing(truth, start, du, dv, reference)) {
      continue;
    }
    ++rays;
    {
      TerrainFieldModel frozen;
      frozen.model = &model;
      frozen.live = false;
      frozen.frozenWeights = model.weightsAt(start.u, start.v);
      frozen.keep.assign(model.layers.size(), true);
      bool crossed = false;
      const MarchPoint hit = terrainMarch(frozen, start, du, dv, minStep, 16, 5, crossed, &samplesFrozen);
      // The same count for the frozen field against its own reference: what the layer cones alone let pass.
      MarchPoint frozenReference;
      if (denseFirstCrossing(frozen, start, du, dv, frozenReference) && hit.z > frozenReference.z + minStep / 16.0f + 1e-4f) {
        ++passedFrozen;
      }
    }
    for (const float safety : { 1.0f, 0.0f }) {
      bool crossed = false;
      const MarchPoint hit = terrainMarch(liveFieldFor(model, start, du, dv, safety), start, du, dv, minStep, 16, 5, crossed,
                                          safety > 0.0f ? &samplesWithSlope : &samplesWithout);
      // Beyond the first crossing by more than the refinement's precision (a step of minStep bisected 4 times).
      if (hit.z > reference.z + minStep / 16.0f + 1e-4f) {
        ++(safety > 0.0f ? passedWithSlope : passedWithout);
      }
    }
  }
  std::printf("  %-28s live weights, %d rays: a step passed the first crossing on %d with the weight slope, %d without "
              "(frozen field, its own crossing: %d); samples per march: frozen %.2f, live %.2f, live without the weight slope %.2f\n",
              name.c_str(), rays, passedWithSlope, passedWithout, passedFrozen, double(samplesFrozen) / std::max(rays, 1),
              double(samplesWithSlope) / std::max(rays, 1), double(samplesWithout) / std::max(rays, 1));
  if (passedWithSlope > rays / 200) {
    throw std::runtime_error(name + ": the live terrain march passes crossings");
  }
}

TerrainModel makeTerrainModel(const std::vector<DepthField>& layers, int gridSize, float spacing, bool sharp, uint32_t seed) {
  TerrainModel model;
  model.layers = layers;
  model.gridSize = gridSize;
  model.spacing = spacing;
  model.layerCount = int(layers.size()) + 1;
  for (const DepthField& layer : layers) {
    std::vector<float> cones(layer.depth.size());
    for (int y = 0; y < layer.height; ++y) {
      for (int x = 0; x < layer.width; ++x) {
        cones[size_t(y) * layer.width + x] = storedCone(bruteForceConservativeRatio(layer, x, y, 1.0f));
      }
    }
    model.storedCones.push_back(std::move(cones));
  }
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  model.vertexWeights.assign(size_t(gridSize) * gridSize * model.layerCount, 0.0f);
  for (int v = 0; v < gridSize * gridSize; ++v) {
    float* w = &model.vertexWeights[size_t(v) * model.layerCount];
    if (sharp) {
      // Painted land: one layer per vertex (sometimes the flat one), so weights swap fully from vertex to vertex.
      w[std::min(int(uniform(rng) * model.layerCount), model.layerCount - 1)] = 1.0f;
    } else {
      float sum = 0.0f;
      for (int l = 0; l < model.layerCount; ++l) {
        w[l] = uniform(rng) * uniform(rng);
        sum += w[l];
      }
      for (int l = 0; l < model.layerCount; ++l) {
        w[l] /= sum;
      }
    }
  }
  return model;
}

void run() {
  std::cout << std::endl << "Begin cone generator tests" << std::endl;

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  struct Shape {
    std::string name;
    std::function<float(int, int, int, int)> heightAt;
  };
  std::vector<float> noise(64 * 64);
  for (float& v : noise) {
    v = uniform(rng);
  }
  std::vector<float> spikes(64 * 64);
  for (float& v : spikes) {
    v = uniform(rng) < 0.03f ? uniform(rng) : 0.0f;
  }
  const std::vector<Shape> shapes = {
    { "plateau", [](int x, int y, int w, int h) {
        return (x >= w / 4 && x < 3 * w / 4 && y >= h / 4 && y < 3 * h / 4) ? 1.0f : 0.0f; } },
    { "steps", [](int x, int, int w, int) { return float((x * 4) / w) / 3.0f; } },
    { "sine", [](int x, int y, int w, int h) {
        return 0.5f + 0.5f * std::sin(2.0f * kPi * 2.0f * x / w) * std::sin(2.0f * kPi * 3.0f * y / h); } },
    { "spikes", [&spikes](int x, int y, int, int) { return spikes[size_t(y) * 64 + x]; } },
    { "noise", [&noise](int x, int y, int, int) { return noise[size_t(y) * 64 + x]; } },
  };

  struct Config {
    int width;
    int height;
    int radius;
    uint32_t raySteps;
  };
  const Config configs[] = {
    { 32, 32, 8, 8 },  // square power of two: far field
    { 32, 32, 2, 4 },  // far field, the default near field
    { 32, 32, 2, 8 },  // far field, small window
    { 32, 32, 8, 0 },  // no ray walk
    { 24, 24, 4, 8 },  // not a power of two: window bound
    { 32, 16, 4, 8 },  // not square: window bound
  };
  for (const Config& config : configs) {
    for (const Shape& shape : shapes) {
      const DepthField field = makeField(config.width, config.height, [&](int x, int y) {
        return shape.heightAt(x, y, config.width, config.height);
      });
      checkField(shape.name, field, config.radius, config.raySteps);
    }
  }

  // The conservative variant, on the same fields and windows.
  const Config conservativeConfigs[] = {
    { 32, 32, 8, 0 },  // far field
    { 32, 32, 2, 0 },  // far field, small window
    { 24, 24, 4, 0 },  // window bound
    { 32, 16, 4, 0 },  // not square: window bound
  };
  for (const Config& config : conservativeConfigs) {
    for (const Shape& shape : shapes) {
      const DepthField field = makeField(config.width, config.height, [&](int x, int y) {
        return shape.heightAt(x, y, config.width, config.height);
      });
      checkConservative(shape.name, field, config.radius);
    }
  }

  // Blends of conservative layer cones, as pom_sccr_terrain.slangh combines them. Shapes by index:
  // 0 plateau, 1 steps, 2 sine, 3 spikes, 4 noise.
  struct Blend {
    std::string name;
    std::vector<size_t> shapes;
    std::vector<float> weights;
    float flatWeight;
  };
  const Blend blends[] = {
    { "sine+steps", { 2, 1 }, { 0.7f, 0.3f }, 0.0f },
    { "noise+plateau", { 4, 0 }, { 0.5f, 0.5f }, 0.0f },
    { "spikes+sine+flat", { 3, 2 }, { 0.5f, 0.25f }, 0.25f },
    { "plateau+steps+noise", { 0, 1, 4 }, { 0.2f, 0.3f, 0.5f }, 0.0f },
  };
  const Config blendConfigs[] = {
    { 32, 32, 8, 0 },  // far field
    { 24, 24, 4, 0 },  // window bound
  };
  for (const Config& config : blendConfigs) {
    for (const Blend& blend : blends) {
      std::vector<DepthField> layers;
      for (const size_t shape : blend.shapes) {
        layers.push_back(makeField(config.width, config.height, [&](int x, int y) {
          return shapes[shape].heightAt(x, y, config.width, config.height);
        }));
      }
      checkBlend(blend.name, layers, blend.weights, blend.flatWeight, config.radius);
    }
  }

  // Blends reduced to their heaviest layers (sccrTerrainReduceLayers), with a tie at the cut.
  const Blend wideBlends[] = {
    { "4 layers", { 0, 1, 2, 4 }, { 0.1f, 0.2f, 0.3f, 0.4f }, 0.0f },
    { "5 layers+flat, tie", { 3, 2, 1, 0, 4 }, { 0.2f, 0.2f, 0.2f, 0.15f, 0.05f }, 0.2f },
  };
  for (const Blend& blend : wideBlends) {
    std::vector<DepthField> layers;
    for (const size_t shape : blend.shapes) {
      layers.push_back(makeField(32, 32, [&](int x, int y) { return shapes[shape].heightAt(x, y, 32, 32); }));
    }
    for (const uint32_t maxLayers : { 3u, 2u }) {
      checkReducedBlend(blend.name, layers, blend.weights, blend.flatWeight, 8, maxLayers);
    }
  }

  // Live terrain weights (PomLiveLayeredConeField): sine, noise and steps layers and a flat one,
  // tiling every UV unit, per-vertex weights every 0.25 UV (painted: one layer per vertex; mixed: random blends).
  // Depths 0.026 and 0.08 UV: a march at 80 degrees travels about 0.6 and 1.8 vertex spacings.
  {
    std::vector<DepthField> terrainLayers;
    for (const size_t shape : { size_t(2), size_t(4), size_t(1) }) {
      terrainLayers.push_back(makeField(32, 32, [&](int x, int y) { return shapes[shape].heightAt(x, y, 32, 32); }));
    }
    const TerrainModel painted = makeTerrainModel(terrainLayers, 9, 0.25f, true, 7);
    const TerrainModel mixed = makeTerrainModel(terrainLayers, 9, 0.25f, false, 8);
    checkLiveWeights("painted, depth 0.026", painted, 0.026f);
    checkLiveWeights("painted, depth 0.08", painted, 0.08f);
    checkLiveWeights("mixed, depth 0.026", mixed, 0.026f);
  }

  // Coarse cones (pom_cone_coarse.comp.slang), alone and mixed with generated tiles. The manager uses cells of 8 up to
  // 2048-wide maps; other cell sizes exercise more and fewer pyramid levels.
  struct CoarseConfig {
    int width;
    uint32_t shift;
    bool relaxed;
  };
  const CoarseConfig coarseConfigs[] = {
    { 32, 3, true },
    { 32, 2, true },
    { 64, 3, false },
    { 64, 4, false },
  };
  for (const CoarseConfig& config : coarseConfigs) {
    for (const Shape& shape : shapes) {
      const DepthField field = makeField(config.width, config.width, [&](int x, int y) {
        return shape.heightAt(x, y, config.width, config.width);
      });
      checkCoarse(shape.name, field, config.shift, config.relaxed);
    }
  }

  // The hierarchical search (pom_cone_generate.comp.slang), against the brute-force conservative cone, the old windowed
  // generator and coarse cones.
  for (const Shape& shape : shapes) {
    const DepthField field = makeField(64, 64, [&](int x, int y) { return shape.heightAt(x, y, 64, 64); });
    checkHierarchical(shape.name, field);
  }

  // The march's crossing refinement (pom_sccr.slangh).
  for (const Shape& shape : shapes) {
    const DepthField field = makeField(64, 64, [&](int x, int y) { return shape.heightAt(x, y, 64, 64); });
    checkRefinement(shape.name, field);
  }

  // The self-shadow march's early returns (pom_sccr.slangh).
  for (const Shape& shape : shapes) {
    const DepthField field = makeField(64, 64, [&](int x, int y) { return shape.heightAt(x, y, 64, 64); });
    checkVisibility(shape.name, field);
  }

  // The curvature correction's base surface (surface_interaction_sccr.slangh).
  checkBaseSurface();

  std::cout << "All cone generator tests passed" << std::endl;
}

} // anonymous namespace

int main() {
  try {
    run();
  }
  catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
