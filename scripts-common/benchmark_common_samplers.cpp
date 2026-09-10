// Standalone Vulkan driver-cost probe. See docs/CoreRenderingOptimizations.md.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    std::fprintf(stderr, "%s failed: %d\n", operation, result);
    std::exit(1);
  }
}

VkSamplerCreateInfo samplerInfo(bool sky) {
  VkSamplerCreateInfo info { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  info.magFilter = VK_FILTER_LINEAR;
  info.minFilter = VK_FILTER_LINEAR;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  info.addressModeV = sky ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
  info.addressModeW = info.addressModeV;
  info.maxLod = sky ? VK_LOD_CLAMP_NONE : 0.0f;
  // All other fields match the zero-initialized DxvkSamplerCreateInfo conversion.
  return info;
}
}

int main() {
  VkApplicationInfo application { VK_STRUCTURE_TYPE_APPLICATION_INFO };
  application.pApplicationName = "Common binding sampler driver probe";
  application.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo instanceInfo { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
  instanceInfo.pApplicationInfo = &application;
  VkInstance instance;
  check(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

  uint32_t deviceCount = 0;
  check(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr), "enumerate device count");
  if (!deviceCount) {
    return 1;
  }
  std::vector<VkPhysicalDevice> devices(deviceCount);
  check(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()), "enumerate devices");
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(devices[0], &properties);
  std::printf("Device: %s; raw driver version: %u\n", properties.deviceName, properties.driverVersion);

  uint32_t familyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(devices[0], &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(devices[0], &familyCount, families.data());
  uint32_t family = 0;
  while (family < familyCount && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
    ++family;
  }
  if (family == familyCount) {
    return 1;
  }
  float priority = 1.0f;
  VkDeviceQueueCreateInfo queueInfo { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
  queueInfo.queueFamilyIndex = family;
  queueInfo.queueCount = 1;
  queueInfo.pQueuePriorities = &priority;
  VkDeviceCreateInfo deviceInfo { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
  deviceInfo.queueCreateInfoCount = 1;
  deviceInfo.pQueueCreateInfos = &queueInfo;
  VkDevice device;
  check(vkCreateDevice(devices[0], &deviceInfo, nullptr, &device), "vkCreateDevice");

  const std::array<VkSamplerCreateInfo, 2> infos { samplerInfo(false), samplerInfo(true) };
  constexpr uint32_t kPairs = 64;
  std::array<VkSampler, kPairs * 2> samplers;
  std::vector<double> times;
  for (uint32_t batch = 0; batch < 24; ++batch) {
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < samplers.size(); ++i) {
      check(vkCreateSampler(device, &infos[i % 2], nullptr, &samplers[i]), "vkCreateSampler");
    }
    // Retain each batch before destroying, rather than repeatedly recycling one handle.
    for (VkSampler sampler : samplers) {
      vkDestroySampler(device, sampler, nullptr);
    }
    const double microseconds = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count() / kPairs;
    if (batch >= 4) {
      times.push_back(microseconds);
    }
  }
  std::sort(times.begin(), times.end());
  std::printf("CPU us per created/destroyed sampler pair: median=%.3f min=%.3f max=%.3f\n",
    (times[9] + times[10]) * 0.5, times.front(), times.back());
  std::printf("PASS: both exact configurations accepted; 3072 samplers created and destroyed.\n");
  std::printf("Measures driver calls only; excludes Remix wrapper, bindings and GPU work.\n");
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return 0;
}
