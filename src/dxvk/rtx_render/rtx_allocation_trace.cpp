#include "rtx_allocation_trace.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "dxvk_device.h"
#include "dxvk_image.h"
#include "rtx_bindless_resource_manager.h"
#include "rtx_context.h"
#include "rtx_fork_hooks.h"
#include "rtx_retained_scene.h"
#include "rtx_scene_manager.h"
#include "rtx_texture_manager.h"

namespace dxvk {
  namespace {
    constexpr uint64_t kLargeBytes = 8ull << 20;
    constexpr uint32_t kSmallLinesPerSecond = 20;
    // An image freed within this many frames of the application's release went through ordinary deferred destruction.
    constexpr uint32_t kHeldFrames = 16;
    constexpr size_t kMaxGroupLines = 24;
    constexpr size_t kMaxHeldLines = 32;
    constexpr size_t kMaxScreenLines = 32;
    // The D3D9 frontend gives its primary images this debug name and logs them itself, with their D3D9 description.
    constexpr const char* kD3D9PrimaryName = "D3D9 texture primary";

    struct Entry {
      uint64_t id = 0;
      const char* name = "";
      uint32_t width = 0;
      uint32_t height = 0;
      uint32_t depth = 1;
      uint32_t mipLevels = 1;
      uint32_t layers = 1;
      VkFormat vkFormat = VK_FORMAT_UNDEFINED;
      uint64_t bytes = 0;

      bool d3d9 = false;
      AllocationTrace::D3D9Info info;
      const std::atomic<uint32_t>* publicRefs = nullptr;
      const std::atomic<uint32_t>* privateRefs = nullptr;

      // True when the creation line was written. The release and free lines follow it.
      bool logged = false;
      bool released = false;
      uint32_t releasedFrame = 0;
    };

    struct Counters {
      uint32_t created = 0;
      uint64_t createdBytes = 0;
      uint32_t released = 0;
      uint32_t freed = 0;
      uint64_t freedBytes = 0;
      uint32_t freedAfterHold = 0;
    };

    struct ScreenSet {
      uint32_t owned = 0;
      uint64_t ownedBytes = 0;
      uint32_t held = 0;
      uint64_t heldBytes = 0;

      bool operator==(const ScreenSet& other) const {
        return owned == other.owned && ownedBytes == other.ownedBytes && held == other.held && heldBytes == other.heldBytes;
      }
    };

    struct State {
      std::mutex mutex;
      std::unordered_map<const DxvkImage*, Entry> entries;
      std::atomic<uint32_t> frame { 0 };
      uint64_t nextId = 1;
      Counters sinceSummary;
      uint64_t droppedLines = 0;
      uint64_t lineSecondMs = 0;
      uint32_t smallLinesInSecond = 0;
      uint32_t backBufferWidth = 0;
      uint32_t backBufferHeight = 0;
      ScreenSet screen;
      uint64_t lastSummaryMs = 0;
    };

    // Never freed, because images can be destroyed during static destruction.
    State& state() {
      static State* s = new State();
      return *s;
    }

    uint64_t steadyMs() {
      return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    std::string mib(uint64_t bytes) {
      const uint64_t tenths = (bytes * 10 + (1ull << 19)) >> 20;
      return str::format(tenths / 10, ".", tenths % 10, " MiB");
    }

    std::string signedMib(int64_t bytes) {
      return bytes < 0 ? "-" + mib(uint64_t(-bytes)) : mib(uint64_t(bytes));
    }

    std::string extent(uint32_t width, uint32_t height, uint32_t depth) {
      return depth > 1 ? str::format(width, "x", height, "x", depth) : str::format(width, "x", height);
    }

    // Images of one shape form one group in the summary.
    std::string shape(const Entry& e) {
      if (e.d3d9) {
        return str::format(e.info.kind, " ", extent(e.width, e.height, e.depth), " ", e.info.format, " mips ", e.mipLevels,
                           " pool ", e.info.pool, " usage ", e.info.usage);
      }
      return str::format(e.name, " ", extent(e.width, e.height, e.depth), " ", e.vkFormat, " mips ", e.mipLevels,
                         " layers ", e.layers);
    }

    bool isScreenSized(const State& s, const Entry& e) {
      return e.d3d9 && (e.info.renderTarget || e.info.depthStencil) && !e.info.backBuffer &&
             s.backBufferWidth != 0 && e.width == s.backBufferWidth && e.height == s.backBufferHeight;
    }

    // Lines for images of at least kLargeBytes are always written. The others share kSmallLinesPerSecond.
    bool admitLine(State& s, uint64_t bytes) {
      if (bytes >= kLargeBytes) {
        return true;
      }
      const uint64_t now = steadyMs();
      if (now - s.lineSecondMs >= 1000) {
        s.lineSecondMs = now;
        s.smallLinesInSecond = 0;
      }
      if (s.smallLinesInSecond < kSmallLinesPerSecond) {
        s.smallLinesInSecond++;
        return true;
      }
      s.droppedLines++;
      return false;
    }

    // Logs the screen-sized render targets and depth stencils when their set changed. Called with the mutex held.
    void updateScreenSet(State& s, bool force) {
      ScreenSet now;
      for (const auto& [image, e] : s.entries) {
        if (!isScreenSized(s, e)) {
          continue;
        }
        if (e.released) {
          now.held++;
          now.heldBytes += e.bytes;
        } else {
          now.owned++;
          now.ownedBytes += e.bytes;
        }
      }
      if (!force && now == s.screen) {
        return;
      }
      s.screen = now;
      Logger::status(str::format("[AllocTrace] screen frame ", s.frame.load(), ": render targets and depth stencils at ",
                                 s.backBufferWidth, "x", s.backBufferHeight, ": ", now.owned, " app-owned (", mib(now.ownedBytes),
                                 "), ", now.held, " still alive after the app released them (", mib(now.heldBytes), ")"));
    }

    // What in Remix references each image, gathered on the CS thread before the summary takes the mutex.
    struct Holders {
      std::unordered_map<const DxvkImage*, uint32_t> textureCache;
      std::unordered_map<const DxvkImage*, uint32_t> drawCallCache;
      std::unordered_map<const DxvkImage*, uint32_t> drawCallCachedMaterials;
      std::unordered_map<const DxvkImage*, uint32_t> retainedScene;
      std::unordered_map<const DxvkImage*, uint32_t> contextBindings;
      std::unordered_map<const DxvkImage*, uint32_t> bindlessTables;
      std::unordered_map<const DxvkImage*, uint32_t> pendingRelease;
      std::unordered_map<const DxvkImage*, uint32_t> hostImports;
      std::unordered_map<const DxvkImage*, uint32_t> apiTextures;
    };

    void countRef(const TextureRef& ref, std::unordered_map<const DxvkImage*, uint32_t>& counts) {
      if (const DxvkImageView* view = ref.getImageView()) {
        counts[view->image().ptr()]++;
      }
    }

    // The color textures and the legacy material protocol's textures of a D3D9 draw's material.
    void countMaterial(const LegacyMaterialData& material, std::unordered_map<const DxvkImage*, uint32_t>& counts) {
      countRef(material.getColorTexture(), counts);
      countRef(material.getColorTexture2(), counts);
      countRef(material.fnvProtocol.diffuseTexture, counts);
      countRef(material.fnvProtocol.normalTexture, counts);
      countRef(material.fnvProtocol.glowTexture, counts);
      countRef(material.fnvProtocol.heightTexture, counts);
    }

    void countBlas(const BlasEntry& blas, std::unordered_map<const DxvkImage*, uint32_t>& input,
                   std::unordered_map<const DxvkImage*, uint32_t>& cachedMaterials) {
      countMaterial(blas.input.getMaterialData(), input);
      for (const auto& [hash, material] : blas.getCachedMaterials()) {
        countMaterial(material, cachedMaterials);
      }
    }

    uint32_t countOf(const std::unordered_map<const DxvkImage*, uint32_t>& counts, const DxvkImage* image) {
      const auto it = counts.find(image);
      return it == counts.end() ? 0 : it->second;
    }

    // Returns the image's Rc reference count, and a reference in keep that the caller drops after unlocking.
    // The entry is erased under the mutex at the start of ~DxvkImage, so the object is still valid here.
    // A count of 0 means the image is being destroyed and is waiting for the mutex.
    uint32_t probeRefs(const DxvkImage* image, std::vector<Rc<DxvkImage>>& keep) {
      DxvkImage* mutableImage = const_cast<DxvkImage*>(image);
      const uint32_t refs = mutableImage->incRef() - 1;
      if (refs > 0) {
        keep.emplace_back(mutableImage);
      }
      mutableImage->decRef();
      return refs;
    }

    std::string holderFields(const Holders& holders, const DxvkImage* image, std::vector<Rc<DxvkImage>>& keep) {
      const uint32_t refs = probeRefs(image, keep);
      return str::format("Rc refs ", refs, refs == 0 ? " (being freed)" : "",
                         ", GPU use ", image->isInUse(DxvkAccess::Read) ? "yes" : "no",
                         ", texture cache ", countOf(holders.textureCache, image),
                         ", draw-call cache ", countOf(holders.drawCallCache, image),
                         ", draw-call cached materials ", countOf(holders.drawCallCachedMaterials, image),
                         ", retained scene ", countOf(holders.retainedScene, image),
                         ", context bindings ", countOf(holders.contextBindings, image),
                         ", bindless tables ", countOf(holders.bindlessTables, image),
                         ", pending release ", countOf(holders.pendingRelease, image),
                         ", host import ", countOf(holders.hostImports, image),
                         ", API texture ", countOf(holders.apiTextures, image));
    }
  }

  void AllocationTrace::onImageCreated(DxvkImage& image, const char* name) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    if (!active()) {
      return;
    }
    const DxvkImageCreateInfo& createInfo = image.info();
    Entry e;
    e.id = s.nextId++;
    e.name = name != nullptr ? name : "unnamed";
    e.width = createInfo.extent.width;
    e.height = createInfo.extent.height;
    e.depth = createInfo.extent.depth;
    e.mipLevels = createInfo.mipLevels;
    e.layers = createInfo.numLayers;
    e.vkFormat = createInfo.format;
    e.bytes = image.memSize();
    image.markAllocationTraced();
    s.sinceSummary.created++;
    s.sinceSummary.createdBytes += e.bytes;
    if (e.bytes >= kLargeBytes && std::strcmp(e.name, kD3D9PrimaryName) != 0) {
      e.logged = true;
      Logger::status(str::format("[AllocTrace] + #", e.id, " frame ", s.frame.load(), " ", shape(e), " ", mib(e.bytes)));
    }
    s.entries[&image] = std::move(e);
  }

  void AllocationTrace::onImageDestroyed(const DxvkImage& image) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    const auto it = s.entries.find(&image);
    if (it == s.entries.end()) {
      return;
    }
    const Entry& e = it->second;
    const uint32_t frame = s.frame.load();
    const uint32_t framesAfterRelease = e.released ? frame - e.releasedFrame : 0;
    s.sinceSummary.freed++;
    s.sinceSummary.freedBytes += e.bytes;
    if (e.released && framesAfterRelease > kHeldFrames) {
      s.sinceSummary.freedAfterHold++;
    }
    if (e.logged) {
      Logger::status(str::format("[AllocTrace] x #", e.id, " frame ", frame, " image freed ", shape(e), " ", mib(e.bytes),
                                 e.released ? str::format(", ", framesAfterRelease, " frames after the app released it") : std::string()));
    }
    const bool screenSized = isScreenSized(s, e);
    s.entries.erase(it);
    if (screenSized) {
      updateScreenSet(s, false);
    }
  }

  void AllocationTrace::onD3D9Created(const DxvkImage& image, const D3D9Info& info) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    const auto it = s.entries.find(&image);
    if (it == s.entries.end()) {
      return;
    }
    Entry& e = it->second;
    e.d3d9 = true;
    e.info = info;
    e.width = info.width;
    e.height = info.height;
    e.depth = info.depth;
    e.mipLevels = info.mipLevels;
    if ((info.renderTarget || info.depthStencil || e.bytes >= kLargeBytes) && admitLine(s, e.bytes)) {
      e.logged = true;
      Logger::status(str::format("[AllocTrace] + #", e.id, " frame ", s.frame.load(), " ", shape(e), " ", mib(e.bytes),
                                 isScreenSized(s, e) ? " (screen-sized)" : ""));
    }
    if (isScreenSized(s, e)) {
      updateScreenSet(s, false);
    }
  }

  void AllocationTrace::onD3D9OwnerRefs(const DxvkImage& image, const std::atomic<uint32_t>* publicRefs,
                                        const std::atomic<uint32_t>* privateRefs) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    const auto it = s.entries.find(&image);
    if (it != s.entries.end()) {
      it->second.publicRefs = publicRefs;
      it->second.privateRefs = privateRefs;
    }
  }

  void AllocationTrace::onD3D9Released(const DxvkImage& image) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    const auto it = s.entries.find(&image);
    if (it == s.entries.end()) {
      return;
    }
    Entry& e = it->second;
    e.released = true;
    e.releasedFrame = s.frame.load();
    e.publicRefs = nullptr;
    e.privateRefs = nullptr;
    s.sinceSummary.released++;
    if (e.logged) {
      Logger::status(str::format("[AllocTrace] - #", e.id, " frame ", e.releasedFrame, " app released ", shape(e), " ", mib(e.bytes)));
    }
    if (isScreenSized(s, e)) {
      updateScreenSet(s, false);
    }
  }

  void AllocationTrace::onBackBuffers(uint32_t width, uint32_t height, const std::string& format, uint32_t count) {
    State& s = state();
    std::lock_guard lock { s.mutex };
    const bool changed = width != s.backBufferWidth || height != s.backBufferHeight;
    s.backBufferWidth = width;
    s.backBufferHeight = height;
    if (!active()) {
      return;
    }
    Logger::status(str::format("[AllocTrace] backbuffers frame ", s.frame.load(), ": ", count, " x ", width, "x", height, " ", format,
                               changed ? " (size changed)" : ""));
    if (changed) {
      updateScreenSet(s, true);
    }
  }

  void AllocationTrace::setHolderCounters(HolderCounter hostImports, HolderCounter apiTextures) {
    s_countHostImports.store(hostImports);
    s_countApiTextures.store(apiTextures);
  }

  void AllocationTrace::onPresent(RtxContext& ctx) {
    State& s = state();
    s.frame.store(ctx.getDevice()->getCurrentFrameId(), std::memory_order_relaxed);
    const bool wanted = allocationTrace();
    if (!wanted && !active()) {
      return;
    }

    if (wanted && !active()) {
      std::lock_guard lock { s.mutex };
      s.entries.clear();
      s.sinceSummary = Counters();
      s.droppedLines = 0;
      s.screen = ScreenSet();
      s.lastSummaryMs = steadyMs();
      s_active.store(true, std::memory_order_relaxed);
      Logger::status(str::format("[AllocTrace] on at frame ", s.frame.load(), ", back buffer ", s.backBufferWidth, "x",
                                 s.backBufferHeight, ". Images allocated before this line are not tracked."));
      return;
    }

    const uint64_t now = steadyMs();
    const uint32_t intervalSeconds = allocationTraceSummarySeconds();
    const bool summaryDue = !wanted || (intervalSeconds > 0 && now - s.lastSummaryMs >= uint64_t(intervalSeconds) * 1000);
    if (!summaryDue) {
      return;
    }
    s.lastSummaryMs = now;

    // Gathered before the mutex is taken: releasing a reference while holding another subsystem's lock
    // can free an image, and its destructor takes the mutex.
    Holders holders;
    RtxTextureManager& textureManager = ctx.getCommonObjects()->getTextureManager();
    for (const TextureRef& ref : textureManager.getTextureTable()) {
      countRef(ref, holders.textureCache);
    }
    textureManager.countPendingAppReleases(holders.pendingRelease);
    SceneManager& sceneManager = ctx.getSceneManager();
    for (const auto& [hash, blas] : sceneManager.m_drawCallCache.getEntries()) {
      countBlas(blas, holders.drawCallCache, holders.drawCallCachedMaterials);
    }
    {
      const RetainedScene& retained = sceneManager.retainedScene();
      for (const auto& [key, geometry] : retained.m_geometry) {
        countBlas(geometry->blas, holders.retainedScene, holders.retainedScene);
      }
      for (const auto* list : { &retained.m_retiredGeometry, &retained.m_clearedGeometry }) {
        for (const auto& geometry : *list) {
          countBlas(geometry->blas, holders.retainedScene, holders.retainedScene);
        }
      }
      for (const auto& [handle, entry] : retained.m_entries) {
        countMaterial(entry.desc.drawCall.getMaterialData(), holders.retainedScene);
        countMaterial(entry.replacementInput.getMaterialData(), holders.retainedScene);
      }
    }
    ctx.countBoundImages(holders.contextBindings);
    for (const auto& table : sceneManager.m_bindlessResourceManager.m_tables[BindlessResourceManager::Textures]) {
      if (table == nullptr) {
        continue;
      }
      for (const Rc<DxvkResource>& resource : table->descriptorResources) {
        if (const DxvkImageView* view = dynamic_cast<const DxvkImageView*>(resource.ptr())) {
          holders.bindlessTables[view->image().ptr()]++;
        }
      }
    }
    if (const HolderCounter countHostImports = s_countHostImports.load()) {
      countHostImports(holders.hostImports);
    }
    if (const HolderCounter countApiTextures = s_countApiTextures.load()) {
      countApiTextures(holders.apiTextures);
    }

    uint64_t appTextureBytes = 0;
    {
      DxvkMemoryAllocator& allocator = ctx.getCommonObjects()->memoryManager();
      const VkPhysicalDeviceMemoryProperties& props = allocator.getMemoryProperties();
      for (uint32_t heapId = 0; heapId < props.memoryHeapCount; heapId++) {
        if (props.memoryHeaps[heapId].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
          appTextureBytes += allocator.getMemoryHeaps()[heapId].stats.usedByCategory(DxvkMemoryStats::Category::AppTexture);
        }
      }
    }

    // References taken by probeRefs, dropped after the mutex is released.
    std::vector<Rc<DxvkImage>> keep;
    {
      std::lock_guard lock { s.mutex };
      const uint32_t frame = s.frame.load();

      struct Group {
        uint32_t count = 0;
        uint64_t bytes = 0;
        uint32_t owned = 0;
        uint32_t held = 0;
      };
      std::unordered_map<std::string, Group> groups;
      std::vector<std::pair<const DxvkImage*, const Entry*>> held;
      std::vector<std::pair<const DxvkImage*, const Entry*>> screen;
      uint32_t total = 0, owned = 0, released = 0, heldLong = 0, noOwner = 0;
      uint64_t totalBytes = 0, ownedBytes = 0, releasedBytes = 0, heldLongBytes = 0, noOwnerBytes = 0;
      for (const auto& [image, e] : s.entries) {
        total++;
        totalBytes += e.bytes;
        Group& group = groups[shape(e)];
        group.count++;
        group.bytes += e.bytes;
        if (!e.d3d9) {
          noOwner++;
          noOwnerBytes += e.bytes;
        } else if (!e.released) {
          owned++;
          ownedBytes += e.bytes;
          group.owned++;
        } else {
          released++;
          releasedBytes += e.bytes;
          group.held++;
          if (frame - e.releasedFrame > kHeldFrames) {
            heldLong++;
            heldLongBytes += e.bytes;
            held.emplace_back(image, &e);
          }
        }
        if (isScreenSized(s, e)) {
          screen.emplace_back(image, &e);
        }
      }

      const Counters& c = s.sinceSummary;
      Logger::status(str::format(
        "[AllocTrace] summary frame ", frame, wanted ? "" : " (trace off)", ": tracked ", total, " images / ", mib(totalBytes),
        ": app-owned ", owned, " / ", mib(ownedBytes), ", alive after the app released them ", released, " / ", mib(releasedBytes),
        " (of those over ", kHeldFrames, " frames: ", heldLong, " / ", mib(heldLongBytes), "), no D3D9 owner ", noOwner, " / ", mib(noOwnerBytes),
        " | AppTexture memory ", mib(appTextureBytes), ", not tracked ", signedMib(int64_t(appTextureBytes) - int64_t(totalBytes)),
        " | since the last summary: created ", c.created, " / ", mib(c.createdBytes), ", app released ", c.released,
        ", freed ", c.freed, " / ", mib(c.freedBytes), " (", c.freedAfterHold, " of them over ", kHeldFrames, " frames after the app released them)",
        " | lines dropped ", s.droppedLines, " | back buffer ", s.backBufferWidth, "x", s.backBufferHeight));
      s.sinceSummary = Counters();

      std::vector<std::pair<const std::string*, const Group*>> sortedGroups;
      sortedGroups.reserve(groups.size());
      for (const auto& [key, group] : groups) {
        sortedGroups.emplace_back(&key, &group);
      }
      std::sort(sortedGroups.begin(), sortedGroups.end(), [](const auto& a, const auto& b) { return a.second->bytes > b.second->bytes; });
      for (size_t i = 0; i < std::min(sortedGroups.size(), kMaxGroupLines); i++) {
        const Group& group = *sortedGroups[i].second;
        Logger::status(str::format("[AllocTrace] group ", *sortedGroups[i].first, ": ", group.count, " / ", mib(group.bytes),
                                   ", app-owned ", group.owned, ", alive after app release ", group.held));
      }
      if (sortedGroups.size() > kMaxGroupLines) {
        uint32_t restCount = 0;
        uint64_t restBytes = 0;
        for (size_t i = kMaxGroupLines; i < sortedGroups.size(); i++) {
          restCount += sortedGroups[i].second->count;
          restBytes += sortedGroups[i].second->bytes;
        }
        Logger::status(str::format("[AllocTrace] group (", sortedGroups.size() - kMaxGroupLines, " smaller groups): ", restCount, " / ", mib(restBytes)));
      }

      const auto byBytes = [](const auto& a, const auto& b) { return a.second->bytes > b.second->bytes; };
      std::sort(held.begin(), held.end(), byBytes);
      for (size_t i = 0; i < std::min(held.size(), kMaxHeldLines); i++) {
        const Entry& e = *held[i].second;
        Logger::status(str::format("[AllocTrace] held #", e.id, " ", shape(e), " ", mib(e.bytes), ", released at frame ", e.releasedFrame,
                                   " (", frame - e.releasedFrame, " frames ago): ", holderFields(holders, held[i].first, keep)));
      }
      if (held.size() > kMaxHeldLines) {
        Logger::status(str::format("[AllocTrace] held: ", held.size() - kMaxHeldLines, " smaller images not listed"));
      }

      std::sort(screen.begin(), screen.end(), byBytes);
      for (size_t i = 0; i < std::min(screen.size(), kMaxScreenLines); i++) {
        const Entry& e = *screen[i].second;
        // The owner's reference counts are read under the mutex, which its destruction takes before they go away.
        const std::string owner = e.released
          ? str::format("app released it at frame ", e.releasedFrame)
          : e.publicRefs != nullptr
            ? str::format("app-owned, COM refs public ", e.publicRefs->load(), " private ", e.privateRefs->load() & 0x7fffffffu)
            : std::string("app-owned");
        Logger::status(str::format("[AllocTrace] screen-sized #", e.id, " ", shape(e), " ", mib(e.bytes), ": ", owner, ", ",
                                   holderFields(holders, screen[i].first, keep)));
      }

      if (!wanted) {
        s_active.store(false, std::memory_order_relaxed);
        s.entries.clear();
        Logger::status(str::format("[AllocTrace] off at frame ", frame));
      }
    }
  }
}
