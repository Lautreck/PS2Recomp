#pragma once

#include "runtime/gs/gs_backend.h"
#include "runtime/gs/gs_texture_page_cache.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct GSPageShadows;

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    // Page indexes (modulo 512) MakePageRect marks for one texel. Matches PixelStorageTraits::PageId.
    static uint32_t PagesCoveringTexel(uint32_t blockBase, uint32_t bufferWidth, uint32_t psm, int x, int y,
                                       uint16_t *pages, uint32_t cap);
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

private:
    void ResetUnlocked();
    void LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut);
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    uint32_t ReadTextureFast(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    uint32_t ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y);
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    struct PixelPipeline; // what WritePixel needs from the draw state, worked out once per primitive
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    void WritePixel(const PixelPipeline &pipe, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog,
                    bool zAlreadyPassed = false);
    // True when WritePixel would return without storing, based on the depth test alone. Same comparison as WritePixel.
    bool DepthRejects(const PixelPipeline &pipe, int x, int y, int z) const;
    // Fetched texels from the previous pixel of this primitive. A later pixel with the same integer UV,
    // or the next bilinear cell to the right, reuses them. The stored words are SamplePoint's results.
    struct TexelReuse
    {
        int keyU = 0x80000000;
        int keyV = 0;
        uint32_t c00 = 0;
        uint32_t c10 = 0;
        uint32_t c01 = 0;
        uint32_t c11 = 0;
        uint8_t valid = 0; // bit 0..3 set only when that corner was fetched, not copied from a neighbour
    };
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v,
                           TexelReuse *reuse = nullptr);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);
    uint32_t SamplePoint(const GSDrawState &state, int sampleU, int sampleV, int texW, int texH, uint8_t wrapU,
                         uint8_t wrapV, uint16_t minU, uint16_t maxU, uint16_t minV, uint16_t maxV);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};
    // CLUT slots: m_clut is the current one. A load that would change the table queued primitives still read
    // copies into the next slot (no heap traffic). Slots are reused once the workers have drawn the queue.
    static constexpr uint32_t kClutSlots = 512u;
    mutable std::array<std::array<uint16_t, 512>, kClutSlots> m_clutPool{};
    mutable uint16_t *m_clut = nullptr;
    mutable uint32_t m_clutSlot = 0u;
    // ~0 = the slot is current. Any other value is the published queue count at which it stopped being current;
    // the slot is free once every worker has drawn that far. Waiting for a full drain only when none are free.
    mutable std::array<uint64_t, kClutSlots> m_clutRetire{};
    // Bumped when GS memory (or a finished draw that might have written it) changes. A repeated CLUT load of
    // the same registers is skipped while this stamp still matches the load.
    mutable uint32_t m_memoryStamp = 1u;
    struct ClutLoadKey
    {
        uint32_t cbp = 0xffffffffu;
        uint32_t packed = 0u;
        uint32_t cbw = 0u;
        uint32_t cou = 0u;
        uint32_t cov = 0u;
        uint32_t stamp = 0u;
    };
    ClutLoadKey m_clutLoad{};
    std::array<uint32_t, 2> m_clutCbp{};
    mutable GSMem::TexturePageCache m_texturePageCache; // mutable: the drawing threads' barrier invalidates it

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    // Drawing threads (PS2X_GS_THREADS=<n> or "auto"; off by default). Every primitive is queued for all workers
    // and each one draws only the rows of its own 16-row bands, so no two threads write the same byte. A primitive
    // whose texture pages overlap the frame or Z pages it writes is drawn on the submitting thread instead: the
    // bands would otherwise sample rows another band has not finished. Reads and
    // direct writes of GS memory, local transfers and presentation first wait until everything queued is drawn.
    // CLUT loads wait only when they read pages a queued primitive draws into. An image upload into pages a
    // queued primitive draws into waits; an upload into pages a queued primitive only samples does not: the old
    // page bytes are kept for jobs already queued and the upload lands in live memory for jobs queued after it.
    // The drawing threads drop their texture page caches when they reach a primitive queued after one.
    // The GS memory pages under a rectangle of a buffer: page rows and columns from the format's page size
    // (64x32 for 32-bit and the 8H/4HL/4HH formats, 64x64 for 16-bit, 128x64 for 8-bit, 128x128 for 4-bit).
    struct PageRect
    {
        uint32_t base = 0u, stride = 1u, extra = 0u; // first page, pages per row of the buffer, +1 if it starts mid-page
        uint32_t row0 = 0u, row1 = 0u, col0 = 0u, col1 = 0u;
        bool empty = true;
    };
    struct Job
    {
        GSPrimitiveBatch batch;
        const uint16_t *clut = nullptr; // CLUT table current when the primitive was queued
        uint32_t epoch = 0u;            // texture cache epoch (see m_epoch)
        // Pages this job contributed to the queued masks, so they can be dropped once every worker has drawn it.
        PageRect frame{};
        PageRect depth{};
        PageRect sampled[4]{};
        uint8_t sampledCount = 0u;
        int boxX0 = 0, boxY0 = 0, boxX1 = -1, boxY1 = -1;
        uint32_t frameBase = 0u; // block address (FBP is in pages)
        uint32_t depthBase = 0u;
        uint32_t bufferWidth = 1u;
        uint8_t framePsm = 0u;
        uint8_t depthPsm = 0u;
        bool drew = false;
        bool depthTouched = false;
        // Pages MarkPages added for this job, so an upload can test the whole in-flight queue without walking
        // each rectangle. Bit index is the page modulo 512, the same index the count masks use.
        uint64_t frameBits[8]{};
        uint64_t depthBits[8]{};
    };
    static constexpr size_t kJobChunkSize = 4096u;
    static constexpr size_t kMaxJobChunks = 1024u;
    static constexpr uint32_t kPageCount = 512u; // 4 MiB of GS memory in 8 KiB pages
    // How many in-flight jobs cover each page. A finished job decrements. Zero means no queued draw uses it.
    using PageMask = std::array<uint16_t, kPageCount>;
    enum class DrainReason : uint8_t
    {
        Other,          // reads, direct writes, local transfers, presentation, reset
        SampleConflict, // a primitive samples pages a queued one draws into
        WriteConflict,  // a primitive draws into pages a queued one samples
        Upload,         // an image upload into pages a queued primitive uses
        ClutSource,     // a CLUT load from pages a queued primitive draws into
        QueueFull,
        Count
    };
    void StartWorkersIfConfigured();
    void StopWorkers();
    void WaitForWorkersUnlocked(DrainReason reason = DrainReason::Other) const;
    void WakeWorkers() const;
    void WorkerLoop(unsigned index);
    static PageRect MakePageRect(uint32_t blockBase, uint32_t bufferWidth, uint32_t psm, int x0, int y0, int x1, int y1);
    bool PagesQueued(const PageRect &rect) const;
    // Distinct page indexes the rectangle covers, in the order the mask walks them.
    uint32_t CollectPages(const PageRect &rect, uint16_t *pages, uint32_t cap) const;
    // Copies queued read pages the rectangle covers. Jobs still on m_epoch sample the copy. Returns how many.
    uint32_t PublishReadShadows(const PageRect &rect, uint16_t *pages, uint32_t cap);
    void BumpPageSeq(const uint16_t *pages, uint32_t count) const;
    void ReleasePageShadows() const;
    static void MarkPages(PageMask &mask, const PageRect &rect);
    static void UnmarkPages(PageMask &mask, const PageRect &rect);
    static bool AnyPages(const PageMask &mask, const PageRect &rect);
    static void SetPageBits(uint64_t bits[8], const PageRect &rect);
    // Drop pages of jobs every worker has already drawn. An upload that only shares a page with those is not a conflict.
    void RetireFinishedJobs() const;
    // writeHit was at page granularity. True when no in-flight job writes a pixel of this upload (same buffer, disjoint
    // box) and none writes the same page through a different buffer. Queues longer than kBoxScanMax keep the page wait.
    bool UploadMissesInFlightPixels(int ux0, int uy0, int ux1, int uy1, uint32_t dbp, uint32_t dbw, uint8_t dpsm) const;
    static constexpr uint32_t kBoxScanMax = 8192u;
    // Pixel rectangle the rasterizer visits (empty when it returns before touching GS memory).
    static bool DrawBox(const GSPrimitiveBatch &batch, int &x0, int &y0, int &x1, int &y1);
    // Pages a textured sprite or triangle can sample. False when this primitive does not sample.
    // *full is set when the UV range is not a safe subset and the whole texture is marked.
    bool AppendSampledPages(const GSPrimitiveBatch &batch, PageRect *rects, int &count, bool &full) const;
    bool m_workersConfigured = false;
    unsigned m_bandCount = 1;
    std::vector<std::thread> m_workers;
    std::vector<std::unique_ptr<GSMem::TexturePageCache>> m_workerCaches;
    // Queue: fixed table of chunks (allocated on first use and kept), written only by the submitting thread.
    // m_published = generation << 32 | jobs queued; a worker's progress uses the same encoding. Waiting for the
    // workers starts a new generation, so the next primitive goes to slot 0 again.
    std::unique_ptr<std::unique_ptr<Job[]>[]> m_jobChunks;
    mutable std::atomic<uint64_t> m_published{0u};
    struct alignas(64) WorkerProgress
    {
        std::atomic<uint64_t> done{0u};
    };
    std::unique_ptr<WorkerProgress[]> m_progress;
    mutable std::atomic<unsigned> m_sleepers{0u};
    mutable std::atomic<bool> m_drainWaiting{false};
    mutable uint32_t m_lastWakeCount = 0u;
    mutable uint32_t m_epoch = 0u; // bumped after anything that changes texture memory behind the workers' caches
    mutable std::mutex m_jobMutex;
    mutable std::condition_variable m_jobCv;
    mutable std::condition_variable m_idleCv;
    // GS memory pages the queued primitives draw into (frame and Z pages the rasterizer visits) and sample
    // (texels inside the primitive's UV range, wrapped): a primitive that samples pages a queued one draws
    // into, or draws into pages a queued one samples, waits for the workers first.
    mutable PageMask m_queuedWritePages{};
    mutable PageMask m_queuedReadPages{};
    // PS2X_GS_DRAIN_STATS=1: once a second, primitives queued and why the queue was drained.
    // =2 also keeps the old whole-texture mask and prints the first conflicts of each second (tbp/tbw/psm/UV
    // against the draw box, or an upload's dbp) plus how many drains the tighter test removed.
    mutable PageMask m_queuedReadPagesWide{};
    bool m_drainStats = false;
    int m_drainDetail = 0;
    mutable std::array<uint32_t, static_cast<size_t>(DrainReason::Count)> m_drainCounts{};
    mutable uint32_t m_statPrimitives = 0u;
    mutable uint32_t m_savedSample = 0u;
    mutable uint32_t m_savedWrite = 0u;
    mutable uint32_t m_savedUpload = 0u;
    mutable uint32_t m_texFull = 0u;
    mutable uint32_t m_texTight = 0u;
    mutable uint32_t m_clutLoads = 0u;
    mutable uint32_t m_clutSkipped = 0u;
    mutable uint32_t m_detailPrints = 0u;
    mutable uint32_t m_shadowUploads = 0u; // read-only uploads that kept a page snapshot instead of waiting
    mutable uint32_t m_boxSkip = 0u;       // page write-hits whose in-flight pixels do not meet the upload
    mutable uint32_t m_boxFailQueue = 0u;  // write-hit left as a wait because the in-flight queue exceeded kBoxScanMax
    mutable uint32_t m_boxFailAlias = 0u;  // X outside the buffer width (page-row alias); the box test does not apply
    mutable uint32_t m_boxFailBuf = 0u;    // a different buffer writes the same page
    mutable uint32_t m_boxFailOverlap = 0u; // same buffer and the pixel boxes meet
    mutable uint32_t m_uploadNoWrite = 0u;  // UploadImage calls that store no pixels (skipped the drain)
    mutable uint32_t m_retired = 0u;       // jobs [0, m_retired) have been removed from the page counts
    mutable int64_t m_statSecond = -1;
    mutable std::unique_ptr<GSPageShadows> m_pageShadows;
    std::atomic<bool> m_stopWorkers{false};
};

// How many primitives were drawn on the submitting thread since the last call because their texture
// pages overlapped the frame or Z pages of that same primitive. Zero when drawing threads are off.
uint32_t GS_TakeSelfOverlapSerialDraws();
