// Standalone probe using Remix's actual reference-counted pointer implementation.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../src/util/rc/util_rc.h"
#include "../src/util/rc/util_rc_ptr.h"

namespace {
struct ProbeBlas : dxvk::RcObject {
  explicit ProbeBlas(uint32_t* count) : destroyed(count) { }
  ~ProbeBlas() { ++*destroyed; }
  uint32_t* destroyed;
};
using BlasRef = dxvk::Rc<ProbeBlas>;

void require(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

NOINLINE void scanCapture(const std::vector<BlasRef>& pool,
    const std::vector<ProbeBlas*>& assigned, std::vector<BlasRef>& cached) {
  for (size_t i = 0; i < assigned.size(); ++i) {
    if (assigned[i]) {
      for (const auto& blas : pool) {
        if (blas.ptr() == assigned[i]) {
          cached[i] = blas;
          break;
        }
      }
    }
  }
}

NOINLINE void directCapture(const std::vector<BlasRef>& pool,
    const std::vector<ProbeBlas*>& assigned, std::vector<BlasRef>& cached) {
  for (size_t i = 0; i < assigned.size(); ++i) {
    cached[i] = BlasRef(assigned[i]);
  }
}

void checkLifetime() {
  uint32_t destroyed = 0;
  std::vector<BlasRef> pool;
  for (uint32_t i = 0; i < 257; ++i) { pool.emplace_back(new ProbeBlas(&destroyed)); }
  // Null, first/middle/last entries and repeated references.
  std::vector<ProbeBlas*> assigned { nullptr, pool.front().ptr(), pool[128].ptr(), pool.back().ptr(), pool.front().ptr() };
  std::vector<BlasRef> before(assigned.size()), after(assigned.size());
  scanCapture(pool, assigned, before);
  directCapture(pool, assigned, after);
  for (size_t i = 0; i < assigned.size(); ++i) {
    require(before[i].ptr() == after[i].ptr() && after[i].ptr() == assigned[i], "captured identity");
  }
  pool.clear();
  require(destroyed == 254, "cache keeps all three referenced objects alive");
  before.clear();
  require(destroyed == 254, "direct references retain ownership after old references release");
  after.clear();
  require(destroyed == 257, "final release deletes every object exactly once");
  std::printf("PASS: identity, nulls, duplicate references, and lifetime\n");
}

void benchmark(uint32_t poolSize) {
  uint32_t destroyed = 0;
  std::vector<BlasRef> pool;
  for (uint32_t i = 0; i < poolSize; ++i) { pool.emplace_back(new ProbeBlas(&destroyed)); }
  constexpr uint32_t kBuckets = 64;
  constexpr uint32_t kRepeats = 500;
  std::vector<ProbeBlas*> assigned;
  for (uint32_t i = 0; i < kBuckets; ++i) { assigned.push_back(pool[(i * 2654435761u + 12345) % poolSize].ptr()); }
  std::vector<BlasRef> cached(kBuckets);
  std::vector<double> times[2];
  for (uint32_t batch = 0; batch < 24; ++batch) {
    for (uint32_t order = 0; order < 2; ++order) {
      uint32_t mode = (batch + order) % 2;
      auto capture = mode ? directCapture : scanCapture;
      const auto start = std::chrono::steady_clock::now();
      for (uint32_t repeat = 0; repeat < kRepeats; ++repeat) {
        capture(pool, assigned, cached);
        // Cache records are newly constructed in the renderer; reproduce their release too.
        for (auto& ref : cached) { ref = nullptr; }
      }
      double microseconds = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / kRepeats;
      if (batch >= 4) { times[mode].push_back(microseconds); }
    }
  }
  for (auto& values : times) { std::sort(values.begin(), values.end()); }
  std::printf("pool=%u buckets=%u median CPU us/capture: scan=%.3f direct=%.3f\n", poolSize, kBuckets, times[0][10], times[1][10]);
}
}

int main() {
  checkLifetime();
  for (uint32_t size : { 16u, 256u, 4096u, 16384u }) { benchmark(size); }
  return 0;
}
