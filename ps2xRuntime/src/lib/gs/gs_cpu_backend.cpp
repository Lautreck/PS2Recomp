#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

using namespace GSInternal;

namespace
{
std::atomic<uint32_t> g_gsSelfOverlapSerial{0u};
}

// The helpers below run for every pixel or texel; left to the compiler they stay calls.
#if defined(_MSC_VER)
#define GS_FORCE_INLINE __forceinline
#else
#define GS_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Busy-wait step for the drawing threads' short spins.
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#define GS_CPU_RELAX() _mm_pause()
#else
#define GS_CPU_RELAX() std::this_thread::yield()
#endif

// Page snapshots for uploads that overlap queued texture reads. Nodes stay live until the next drain
// (the workers are idle then). head is newest-first; epochs along older links do not increase.
struct GSPageShadows
{
    struct Node
    {
        uint32_t epoch = 0u;
        Node *older = nullptr;
        alignas(64) uint8_t bytes[8192]{};
    };
    std::atomic<Node *> head[512]{};
    std::atomic<uint32_t> seq[512]{};
    std::vector<std::unique_ptr<Node>> owned;
    std::vector<Node *> freeNodes;
};

namespace
{
    thread_local const GSPageShadows *t_pageShadows = nullptr;
    thread_local uint32_t t_jobEpoch = 0xffffffffu;
}

namespace GSMem
{
void CopyStablePage(uint8_t *dst, const uint8_t *vram, uint32_t pageBase) noexcept
{
    constexpr uint32_t kPage = TexturePageCache::kPageSize;
    if (!dst)
        return;
    if (!t_pageShadows || !vram)
    {
        if (vram)
            std::memcpy(dst, vram + pageBase, kPage);
        return;
    }
    const uint32_t index = (pageBase / kPage) & 511u;
    for (;;)
    {
        const uint32_t seq1 = t_pageShadows->seq[index].load(std::memory_order_acquire);
        const GSPageShadows::Node *best = nullptr;
        for (const GSPageShadows::Node *node = t_pageShadows->head[index].load(std::memory_order_acquire); node;
             node = node->older)
        {
            if (node->epoch < t_jobEpoch)
                break;
            best = node;
        }
        std::memcpy(dst, best ? best->bytes : vram + pageBase, kPage);
        const uint32_t seq2 = t_pageShadows->seq[index].load(std::memory_order_acquire);
        if (seq1 == seq2 && (seq1 & 1u) == 0u)
            return;
        GS_CPU_RELAX();
    }
}
}

namespace
{
    // Which rows this thread draws (see GSCpuBackend's drawing threads); outside them a thread draws every row.
    thread_local unsigned t_bandIndex = 0u;
    thread_local unsigned t_bandCount = 1u;
    thread_local GSMem::TexturePageCache *t_textureCache = nullptr;
    thread_local const uint16_t *t_clut = nullptr; // the CLUT table of the primitive a drawing thread is on
    constexpr unsigned kBandRowShift = 4u; // 16-row bands: the tallest block of any pixel format

    GS_FORCE_INLINE bool ownsRow(int y)
    {
        return t_bandCount <= 1u || ((static_cast<unsigned>(y) >> kBandRowShift) % t_bandCount) == t_bandIndex;
    }

    GS_FORCE_INLINE float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    GS_FORCE_INLINE u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    GS_FORCE_INLINE u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    GS_FORCE_INLINE u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    GS_FORCE_INLINE uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};

    GS_FORCE_INLINE int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    GS_FORCE_INLINE bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    // True when no pixel of a primitive can be written: the alpha test always fails (ATST NEVER, or a flat colour
    // whose alpha fails it) and AFAIL is KEEP. Games draw such sprites over the whole screen.
    bool alphaTestDiscardsPrimitive(uint64_t testReg, bool textured, uint8_t vertexAlpha)
    {
        if ((testReg & 0x1u) == 0u || ((testReg >> 12) & 0x3u) != 0u)
            return false;
        if (((testReg >> 1) & 0x7u) == 0u)
            return true;
        return !textured && !passesAlphaTest(testReg, vertexAlpha);
    }

    GS_FORCE_INLINE PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    GS_FORCE_INLINE bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    GS_FORCE_INLINE TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    GS_FORCE_INLINE bool isFourBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    GS_FORCE_INLINE bool isEightBitIndexedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H;
    }

    GS_FORCE_INLINE uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        // The value is never negative, so + 0.5 and truncation round like lround, without its CRT call (MSVC
        // only inlines rounding with /arch:AVX; it was a fifth of the drawing time).
        return clampU8(static_cast<int>(top + (bottom - top) * fy + 0.5f));
    }

    // floor() for texture coordinates, without the CRT call.
    GS_FORCE_INLINE int floorToInt(float value)
    {
        const int truncated = static_cast<int>(value);
        return truncated - (value < static_cast<float>(truncated) ? 1 : 0);
    }

    // One axis of the texels a primitive can read, in texture space. Two intervals when REPEAT wraps.
    struct AxisCover
    {
        bool full = false;
        int a0 = 0;
        int a1 = -1;
        int b0 = 0;
        int b1 = -1;
    };

    void coverRepeat(int i0, int i1, int size, AxisCover &axis)
    {
        if (size <= 1)
        {
            axis.a0 = 0;
            axis.a1 = 0;
            return;
        }
        if (i1 < i0)
            std::swap(i1, i0);
        if (i1 - i0 >= size)
        {
            axis.full = true;
            return;
        }
        const int start = static_cast<int>(static_cast<uint32_t>(i0) & static_cast<uint32_t>(size - 1));
        const int span = i1 - i0;
        if (start + span < size)
        {
            axis.a0 = start;
            axis.a1 = start + span;
            return;
        }
        axis.a0 = start;
        axis.a1 = size - 1;
        axis.b0 = 0;
        axis.b1 = start + span - size;
    }

    void coverAxis(int i0, int i1, int size, uint8_t mode, int regionMin, int regionMax, AxisCover &axis)
    {
        axis = {};
        if (size <= 0)
        {
            axis.full = true;
            return;
        }
        if (i1 < i0)
            std::swap(i1, i0);
        switch (mode & 3u)
        {
        case 0:
            coverRepeat(i0, i1, size, axis);
            return;
        case 1:
            axis.a0 = clampInt(i0, 0, size - 1);
            axis.a1 = clampInt(i1, 0, size - 1);
            if (axis.a0 > axis.a1)
                std::swap(axis.a0, axis.a1);
            return;
        case 2:
            axis.a0 = std::min(std::max(i0, regionMin), regionMax);
            axis.a1 = std::min(std::max(i1, regionMin), regionMax);
            if (axis.a0 > axis.a1)
                std::swap(axis.a0, axis.a1);
            return;
        default:
            if (i1 - i0 > 4096)
            {
                axis.full = true;
                return;
            }
            int mn = 0x7fffffff;
            int mx = -0x7fffffff;
            for (int k = i0; k <= i1; ++k)
            {
                const int wrapped = static_cast<int>((static_cast<uint32_t>(k) & static_cast<uint32_t>(regionMin)) |
                                                     static_cast<uint32_t>(regionMax));
                mn = std::min(mn, wrapped);
                mx = std::max(mx, wrapped);
            }
            axis.a0 = mn;
            axis.a1 = mx;
            return;
        }
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    // Display lines shown per frame-buffer line: MAGV + 1, doubled in interlaced FRAME mode (FFMD = 1), where each
    // field reads every buffer line while DH still counts the lines of both fields.
    uint32_t decodeDisplayLineRepeat(uint64_t display64, const GSSmode2State &smode2)
    {
        const uint32_t magv = static_cast<uint32_t>((display64 >> 27) & 0x3u);
        return (magv + 1u) * ((smode2.interlaced && smode2.frameMode) ? 2u : 1u);
    }

    // Spreads the first ceil(height / repeat) rows over `height` rows, each repeated `repeat` times, starting
    // `offset` rows lower (the odd field of an interlaced picture sits one line below the even one).
    void repeatPresentationLines(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, uint32_t repeat,
                                 uint32_t offset = 0u)
    {
        if (pixels.empty() || (repeat <= 1u && offset == 0u))
            return;
        for (uint32_t y = height; y-- > 0u;)
        {
            const uint32_t sourceY = (y < offset ? 0u : y - offset) / repeat;
            if (sourceY != y)
                std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                            pixels.data() + sourceY * kHostFrameWidth * 4u,
                            width * 4u);
        }
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }

    // PS2X_GS_FRAME_DUMP=<dir> writes the presented image at 40s and 95s (BGR BMP). PrintWindow captures of
    // this window are white, so the present pixels are the screenshot.
    PresentationFrame dumpPresented(PresentationFrame frame)
    {
        const char *dir = std::getenv("PS2X_GS_FRAME_DUMP");
        if (!dir || dir[0] == '\0' || frame.pixels.empty() || frame.width == 0u || frame.height == 0u)
            return frame;
        static const auto t0 = std::chrono::steady_clock::now();
        static int wrote40 = 0;
        static int wrote95 = 0;
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const char *name = nullptr;
        if (wrote40 == 0 && sec >= 40.0 && sec < 41.5)
        {
            name = "frame_40.bmp";
            wrote40 = 1;
        }
        else if (wrote95 == 0 && sec >= 95.0 && sec < 96.5)
        {
            name = "frame_95.bmp";
            wrote95 = 1;
        }
        if (!name)
            return frame;
        char path[512];
        std::snprintf(path, sizeof(path), "%s\\%s", dir, name);
        std::FILE *fp = std::fopen(path, "wb");
        if (!fp)
            return frame;
        const uint32_t w = frame.width;
        const uint32_t h = frame.height;
        const uint32_t rowBytes = (w * 3u + 3u) & ~3u;
        const uint32_t imageBytes = rowBytes * h;
        uint8_t header[54] = {};
        header[0] = 'B';
        header[1] = 'M';
        auto put32 = [&](int offset, uint32_t value) {
            header[offset] = static_cast<uint8_t>(value);
            header[offset + 1] = static_cast<uint8_t>(value >> 8);
            header[offset + 2] = static_cast<uint8_t>(value >> 16);
            header[offset + 3] = static_cast<uint8_t>(value >> 24);
        };
        put32(2, 54u + imageBytes);
        put32(10, 54u);
        put32(14, 40u);
        put32(18, w);
        put32(22, h);
        header[26] = 1;
        header[28] = 24;
        put32(34, imageBytes);
        std::fwrite(header, 1, 54, fp);
        std::vector<uint8_t> scan(rowBytes, 0u);
        for (int y = static_cast<int>(h) - 1; y >= 0; --y)
        {
            const uint8_t *src = frame.pixels.data() + static_cast<uint32_t>(y) * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < w; ++x)
            {
                scan[x * 3u] = src[x * 4u + 2u];
                scan[x * 3u + 1u] = src[x * 4u + 1u];
                scan[x * 3u + 2u] = src[x * 4u];
            }
            std::fwrite(scan.data(), 1, rowBytes, fp);
        }
        std::fclose(fp);
        return frame;
    }
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    m_clut = m_clutPool[0].data();
    m_clutSlot = 0u;
    m_clutRetire[0] = ~uint64_t{0};
    Reset();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    if (vram && vramSize < GSMem::MEMORY_SIZE)
        throw std::invalid_argument("GS CPU backend requires at least 4 MiB of VRAM");

    std::lock_guard<std::mutex> lock(m_mutex);
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    WaitForWorkersUnlocked();
    std::fill(m_clut, m_clut + 512, uint16_t{0});
    m_clutCbp.fill(0u);
    m_clutLoad = {};
    ++m_memoryStamp;
    m_texturePageCache.Invalidate();
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

bool GSCpuBackend::DrawBox(const GSPrimitiveBatch &batch, int &x0, int &y0, int &x1, int &y1)
{
    const auto &ctx = batch.state.context;
    const int ofx = static_cast<int>(ctx.xyoffset.ofx >> 4);
    const int ofy = static_cast<int>(ctx.xyoffset.ofy >> 4);
    const int sx0 = static_cast<int>(ctx.scissor.x0);
    const int sx1 = static_cast<int>(ctx.scissor.x1);
    const int sy0 = static_cast<int>(ctx.scissor.y0);
    const int sy1 = static_cast<int>(ctx.scissor.y1);
    const auto &prim = batch.state.prim;

    if (prim.type == GS_PRIM_SPRITE)
    {
        if (batch.vertexCount < 2u)
            return false;
        if (alphaTestDiscardsPrimitive(ctx.test, prim.tme, batch.vertices[1].a))
            return false;
        float fx0 = batch.vertices[0].x - static_cast<float>(ofx);
        float fy0 = batch.vertices[0].y - static_cast<float>(ofy);
        float fx1 = batch.vertices[1].x - static_cast<float>(ofx);
        float fy1 = batch.vertices[1].y - static_cast<float>(ofy);
        if (fx0 > fx1)
            std::swap(fx0, fx1);
        if (fy0 > fy1)
            std::swap(fy0, fy1);
        const int ux0 = static_cast<int>(std::ceil(fx0));
        const int uy0 = static_cast<int>(std::ceil(fy0));
        const int ux1 = std::max(ux0, static_cast<int>(std::ceil(fx1)) - 1);
        const int uy1 = std::max(uy0, static_cast<int>(std::ceil(fy1)) - 1);
        // DrawSprite returns before any read or write when the rectangle misses the scissor.
        if (ux1 < sx0 || ux0 > sx1 || uy1 < sy0 || uy0 > sy1)
            return false;
        x0 = clampInt(ux0, sx0, sx1);
        x1 = clampInt(ux1, sx0, sx1);
        y0 = clampInt(uy0, sy0, sy1);
        y1 = clampInt(uy1, sy0, sy1);
        return x1 >= x0 && y1 >= y0;
    }

    if (prim.type == GS_PRIM_TRIANGLE || prim.type == GS_PRIM_TRISTRIP || prim.type == GS_PRIM_TRIFAN)
    {
        if (batch.vertexCount < 3u)
            return false;
        const float fx0 = batch.vertices[0].x - static_cast<float>(ofx);
        const float fy0 = batch.vertices[0].y - static_cast<float>(ofy);
        const float fx1 = batch.vertices[1].x - static_cast<float>(ofx);
        const float fy1 = batch.vertices[1].y - static_cast<float>(ofy);
        const float fx2 = batch.vertices[2].x - static_cast<float>(ofx);
        const float fy2 = batch.vertices[2].y - static_cast<float>(ofy);
        const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
        if (std::fabs(denom) < 0.001f)
            return false; // DrawTriangle returns before the loop
        // Same inclusive bounds as DrawTriangle, including the scissor clamp it applies even when the
        // triangle sits outside (that clamp still visits the border column).
        x0 = clampInt(static_cast<int>(std::floor(std::min({fx0, fx1, fx2}))), sx0, sx1);
        x1 = clampInt(static_cast<int>(std::ceil(std::max({fx0, fx1, fx2}))), sx0, sx1);
        y0 = clampInt(static_cast<int>(std::floor(std::min({fy0, fy1, fy2}))), sy0, sy1);
        y1 = clampInt(static_cast<int>(std::ceil(std::max({fy0, fy1, fy2}))), sy0, sy1);
        return x1 >= x0 && y1 >= y0;
    }

    if (prim.type == GS_PRIM_LINE || prim.type == GS_PRIM_LINESTRIP)
    {
        if (batch.vertexCount < 2u)
            return false;
        const int ax = static_cast<int>(batch.vertices[0].x) - ofx;
        const int ay = static_cast<int>(batch.vertices[0].y) - ofy;
        const int bx = static_cast<int>(batch.vertices[1].x) - ofx;
        const int by = static_cast<int>(batch.vertices[1].y) - ofy;
        const int lx0 = std::min(ax, bx);
        const int ly0 = std::min(ay, by);
        const int lx1 = std::max(ax, bx);
        const int ly1 = std::max(ay, by);
        if (lx1 < sx0 || lx0 > sx1 || ly1 < sy0 || ly0 > sy1)
            return false;
        x0 = clampInt(lx0, sx0, sx1);
        x1 = clampInt(lx1, sx0, sx1);
        y0 = clampInt(ly0, sy0, sy1);
        y1 = clampInt(ly1, sy0, sy1);
        return x1 >= x0 && y1 >= y0;
    }

    if (prim.type == GS_PRIM_POINT)
    {
        if (batch.vertexCount < 1u)
            return false;
        const int px = static_cast<int>(batch.vertices[0].x) - ofx;
        const int py = static_cast<int>(batch.vertices[0].y) - ofy;
        if (px < sx0 || px > sx1 || py < sy0 || py > sy1)
            return false;
        x0 = x1 = px;
        y0 = y1 = py;
        return true;
    }
    return false;
}

bool GSCpuBackend::AppendSampledPages(const GSPrimitiveBatch &batch, PageRect *rects, int &count, bool &full) const
{
    count = 0;
    full = false;
    const auto &prim = batch.state.prim;
    if (!prim.tme)
        return false;
    const bool sprite = prim.type == GS_PRIM_SPRITE;
    const bool triangle = prim.type == GS_PRIM_TRIANGLE || prim.type == GS_PRIM_TRISTRIP || prim.type == GS_PRIM_TRIFAN;
    if (!sprite && !triangle)
        return false; // DrawLine and points never sample
    if (sprite)
    {
        if (batch.vertexCount < 2u || alphaTestDiscardsPrimitive(batch.state.context.test, true, batch.vertices[1].a))
            return false;
        int dx0, dy0, dx1, dy1;
        if (!DrawBox(batch, dx0, dy0, dx1, dy1))
            return false; // clipped away before the texture loop
    }
    else if (batch.vertexCount < 3u)
    {
        return false;
    }
    else
    {
        const float ofx = static_cast<float>(batch.state.context.xyoffset.ofx >> 4);
        const float ofy = static_cast<float>(batch.state.context.xyoffset.ofy >> 4);
        const float fx0 = batch.vertices[0].x - ofx, fy0 = batch.vertices[0].y - ofy;
        const float fx1 = batch.vertices[1].x - ofx, fy1 = batch.vertices[1].y - ofy;
        const float fx2 = batch.vertices[2].x - ofx, fy2 = batch.vertices[2].y - ofy;
        const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
        if (std::fabs(denom) < 0.001f)
            return false;
    }

    const auto &tex = batch.state.context.tex0;
    const int texW = std::max<int>(batch.state.textureWidth, 1);
    const int texH = std::max<int>(batch.state.textureHeight, 1);
    auto markWhole = [&]() {
        full = true;
        rects[count++] = MakePageRect(tex.tbp0, tex.tbw, tex.psm, 0, 0, texW - 1, texH - 1);
    };

    const uint64_t clamp = batch.state.context.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const int minU = static_cast<int>((clamp >> 4) & 0x3FFu);
    const int maxU = static_cast<int>((clamp >> 14) & 0x3FFu);
    const int minV = static_cast<int>((clamp >> 24) & 0x3FFu);
    const int maxV = static_cast<int>((clamp >> 34) & 0x3FFu);
    const int n = sprite ? 2 : 3;
    float uMin = 1.0e30f, uMax = -1.0e30f, vMin = 1.0e30f, vMax = -1.0e30f;
    bool bounded = true;
    if (!prim.fst)
    {
        float qMin = 1.0e30f, qMax = -1.0e30f;
        for (int i = 0; i < n; ++i)
        {
            qMin = std::min(qMin, batch.vertices[i].q);
            qMax = std::max(qMax, batch.vertices[i].q);
        }
        // Covered pixels use affine weights >= -1e-4. Q must stay the same sign across that extrapolation,
        // which holds when every vertex Q has that sign and the largest is at most 512x the smallest
        // (a near-zero Q, or a perspective that crosses 0, falls back to the whole texture).
        const float qAbsMin = std::fabs(qMin);
        const float qAbsMax = std::fabs(qMax);
        if (!(qMin > 1.0e-20f || qMax < -1.0e-20f) || qAbsMax > qAbsMin * 512.0f)
            bounded = false;
        else
        {
            for (int i = 0; i < n; ++i)
            {
                const float q = batch.vertices[i].q;
                uMin = std::min(uMin, batch.vertices[i].s / q * static_cast<float>(texW));
                uMax = std::max(uMax, batch.vertices[i].s / q * static_cast<float>(texW));
                vMin = std::min(vMin, batch.vertices[i].t / q * static_cast<float>(texH));
                vMax = std::max(vMax, batch.vertices[i].t / q * static_cast<float>(texH));
            }
        }
    }
    else if (sprite)
    {
        for (int i = 0; i < n; ++i)
        {
            const float u = static_cast<float>(batch.vertices[i].u) / 16.0f;
            const float v = static_cast<float>(batch.vertices[i].v) / 16.0f;
            uMin = std::min(uMin, u);
            uMax = std::max(uMax, u);
            vMin = std::min(vMin, v);
            vMax = std::max(vMax, v);
        }
    }
    else
    {
        // DrawTriangle casts the interpolated 12.4 UV to uint16 before dividing by 16. A weight of -1e-4
        // (the coverage epsilon) can push that cast through zero, which wraps. Fall back when it might.
        float fMin = 1.0e30f, fMax = -1.0e30f, gMin = 1.0e30f, gMax = -1.0e30f;
        for (int i = 0; i < 3; ++i)
        {
            fMin = std::min(fMin, static_cast<float>(batch.vertices[i].u));
            fMax = std::max(fMax, static_cast<float>(batch.vertices[i].u));
            gMin = std::min(gMin, static_cast<float>(batch.vertices[i].v));
            gMax = std::max(gMax, static_cast<float>(batch.vertices[i].v));
        }
        const float du = (fMax - fMin) * 2.0e-4f + 2.0f;
        const float dv = (gMax - gMin) * 2.0e-4f + 2.0f;
        if (fMin - du < 0.0f || fMax + du > 65535.0f || gMin - dv < 0.0f || gMax + dv > 65535.0f)
            bounded = false;
        else
        {
            uMin = (fMin - du) / 16.0f;
            uMax = (fMax + du) / 16.0f;
            vMin = (gMin - dv) / 16.0f;
            vMax = (gMax + dv) / 16.0f;
        }
    }

    if (!bounded || !(uMin > -1.0e7f && uMax < 1.0e7f && vMin > -1.0e7f && vMax < 1.0e7f))
    {
        markWhole();
        return true;
    }

    const float growU = (prim.fst && !sprite) ? 0.05f : (uMax - uMin) * 1.0e-3f + 0.05f;
    const float growV = (prim.fst && !sprite) ? 0.05f : (vMax - vMin) * 1.0e-3f + 0.05f;
    int iU0 = floorToInt(uMin - growU);
    int iU1 = floorToInt(uMax + growU);
    int iV0 = floorToInt(vMin - growV);
    int iV1 = floorToInt(vMax + growV);
    if (batch.state.linearFilter)
    {
        // Sample at uv-0.5 and also fetch the +1 texel. The weight-0 skip only drops fetches.
        iU0 -= 1;
        iU1 += 1;
        iV0 -= 1;
        iV1 += 1;
    }

    AxisCover uCover, vCover;
    coverAxis(iU0, iU1, texW, wrapU, minU, maxU, uCover);
    coverAxis(iV0, iV1, texH, wrapV, minV, maxV, vCover);
    if (uCover.full || vCover.full)
    {
        markWhole();
        return true;
    }

    auto push = [&](int ua, int ub, int va, int vb) {
        if (count >= 4 || ub < ua || vb < va)
            return;
        rects[count++] = MakePageRect(tex.tbp0, tex.tbw, tex.psm, ua, va, ub, vb);
    };
    push(uCover.a0, uCover.a1, vCover.a0, vCover.a1);
    if (uCover.b1 >= uCover.b0)
        push(uCover.b0, uCover.b1, vCover.a0, vCover.a1);
    if (vCover.b1 >= vCover.b0)
        push(uCover.a0, uCover.a1, vCover.b0, vCover.b1);
    if (uCover.b1 >= uCover.b0 && vCover.b1 >= vCover.b0)
        push(uCover.b0, uCover.b1, vCover.b0, vCover.b1);
    return true;
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || batch.vertexCount == 0u)
        return;
    if (!m_workersConfigured)
        StartWorkersIfConfigured();
    if (m_workers.empty())
    {
        DrawPrimitive(batch);
        // The draw has landed. A later CLUT load of the same registers must re-read GS memory.
        ++m_memoryStamp;
        return;
    }
    RetireFinishedJobs();

    // The bands run ahead of each other, so a queued draw and a queued sample must not share a GS page,
    // and one primitive must not sample a page it is itself still writing from another band.
    // Drawn pages are the rectangle the rasterizer visits. Sampled pages are the texels that rectangle can
    // read (the whole texture only when the UV range is not a safe subset: Q crosses 0, REGION_REPEAT over
    // a long span, or a 12.4 coordinate that can wrap through the uint16 cast).
    const auto &ctx = batch.state.context;
    int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
    const bool drawn = DrawBox(batch, x0, y0, x1, y1);
    PageRect sampled[4];
    int sampledCount = 0;
    bool textureFull = false;
    const bool samples = AppendSampledPages(batch, sampled, sampledCount, textureFull);
    if (!drawn && !samples)
        return;
    const uint32_t fbw = std::max<uint32_t>(ctx.frame.fbw, 1u);
    const bool zTouched = drawn && (!ctx.zbuf.zmask || ((ctx.test >> 17) & 3u) != 1u);
    const PageRect frame = drawn ? MakePageRect(ctx.frame.fbp * 32u, fbw, ctx.frame.psm, x0, y0, x1, y1) : PageRect{};
    const PageRect z = zTouched ? MakePageRect(ctx.zbuf.zbp * 32u, fbw, ctx.zbuf.psm, x0, y0, x1, y1) : PageRect{};
    bool sampleHit = false;
    for (int i = 0; i < sampledCount && !sampleHit; ++i)
        sampleHit = AnyPages(m_queuedWritePages, sampled[i]);
    const bool writeHit = !sampleHit && (AnyPages(m_queuedReadPages, frame) || AnyPages(m_queuedReadPages, z));
    if (m_drainDetail >= 2 && !sampleHit && !writeHit && samples)
    {
        const PageRect wide = MakePageRect(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0, 0,
                                           std::max<int>(batch.state.textureWidth, 1) - 1,
                                           std::max<int>(batch.state.textureHeight, 1) - 1);
        if (AnyPages(m_queuedWritePages, wide))
            ++m_savedSample;
        else if (AnyPages(m_queuedReadPagesWide, frame) || AnyPages(m_queuedReadPagesWide, z))
            ++m_savedWrite;
    }
    auto logConflict = [&](const char *kind) {
        if (m_drainDetail < 2 || m_detailPrints >= 12u)
            return;
        ++m_detailPrints;
        std::cout << "[gs:conflict] " << kind
                  << " tbp=" << ctx.tex0.tbp0
                  << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                  << " tpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                  << " tex=" << batch.state.textureWidth << "x" << batch.state.textureHeight
                  << " full=" << (textureFull ? 1 : 0)
                  << " fbp=" << ctx.frame.fbp
                  << " fbw=" << fbw
                  << " fpsm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                  << " zbp=" << ctx.zbuf.zbp
                  << " box=" << x0 << "," << y0 << "-" << x1 << "," << y1
                  << " prim=" << static_cast<uint32_t>(batch.state.prim.type)
                  << " fst=" << static_cast<uint32_t>(batch.state.prim.fst)
                  << " uv0=" << (batch.vertices[0].u / 16u) << "," << (batch.vertices[0].v / 16u)
                  << " stq0=" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q
                  << std::endl;
    };
    // Own frame/Z pages are not in the queued masks yet. Split across bands, a lower band can sample a row
    // an upper band has not written. Drain first, then draw every row on this thread (its band count is 1).
    PageMask ownPages{};
    MarkPages(ownPages, frame);
    MarkPages(ownPages, z);
    bool selfHit = false;
    for (int i = 0; i < sampledCount && !selfHit; ++i)
        selfHit = AnyPages(ownPages, sampled[i]);
    if (selfHit)
    {
        logConflict("self");
        WaitForWorkersUnlocked(DrainReason::SampleConflict);
        g_gsSelfOverlapSerial.fetch_add(1u, std::memory_order_relaxed);
        DrawPrimitive(batch);
        ++m_memoryStamp;
        return;
    }
    if (sampleHit)
    {
        logConflict("sample");
        WaitForWorkersUnlocked(DrainReason::SampleConflict);
    }
    else if (writeHit)
    {
        logConflict("write");
        WaitForWorkersUnlocked(DrainReason::WriteConflict);
    }
    MarkPages(m_queuedWritePages, frame);
    MarkPages(m_queuedWritePages, z);
    for (int i = 0; i < sampledCount; ++i)
        MarkPages(m_queuedReadPages, sampled[i]);
    if (m_drainDetail >= 2 && samples)
    {
        if (textureFull)
            ++m_texFull;
        else
            ++m_texTight;
        const PageRect wide = MakePageRect(ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0, 0,
                                           std::max<int>(batch.state.textureWidth, 1) - 1,
                                           std::max<int>(batch.state.textureHeight, 1) - 1);
        MarkPages(m_queuedReadPagesWide, wide);
    }
    ++m_statPrimitives;

    uint64_t published = m_published.load(std::memory_order_relaxed); // this thread is the only writer
    if ((published & 0xFFFFFFFFu) == kJobChunkSize * kMaxJobChunks)
    {
        // The wait clears the page counts, including the marks just above. Put this job's pages back.
        WaitForWorkersUnlocked(DrainReason::QueueFull);
        published = m_published.load(std::memory_order_relaxed);
        MarkPages(m_queuedWritePages, frame);
        MarkPages(m_queuedWritePages, z);
        for (int i = 0; i < sampledCount; ++i)
            MarkPages(m_queuedReadPages, sampled[i]);
    }
    const uint32_t slot = static_cast<uint32_t>(published & 0xFFFFFFFFu);
    std::unique_ptr<Job[]> &chunk = m_jobChunks[slot / kJobChunkSize];
    if (!chunk)
        chunk = std::make_unique<Job[]>(kJobChunkSize);
    Job &job = chunk[slot % kJobChunkSize];
    job.batch = batch;
    job.clut = m_clut;
    job.epoch = m_epoch;
    job.frame = frame;
    job.depth = z;
    job.sampledCount = static_cast<uint8_t>(std::min(sampledCount, 4));
    for (int i = 0; i < job.sampledCount; ++i)
        job.sampled[i] = sampled[i];
    job.boxX0 = x0;
    job.boxY0 = y0;
    job.boxX1 = x1;
    job.boxY1 = y1;
    job.frameBase = ctx.frame.fbp * 32u;
    job.depthBase = ctx.zbuf.zbp * 32u;
    job.bufferWidth = fbw;
    job.framePsm = ctx.frame.psm;
    job.depthPsm = ctx.zbuf.psm;
    job.drew = drawn;
    job.depthTouched = zTouched;
    SetPageBits(job.frameBits, job.frame);
    SetPageBits(job.depthBits, job.depth);
    m_published.store(published + 1u); // seq_cst: pairs with the sleeper count (see WorkerLoop)
    // Waking threads costs more than a primitive: let a few queue up first (a wait for the workers wakes them too).
    if (slot + 1u - m_lastWakeCount >= 16u)
        WakeWorkers();
}

void GSCpuBackend::WakeWorkers() const
{
    m_lastWakeCount = static_cast<uint32_t>(m_published.load(std::memory_order_relaxed) & 0xFFFFFFFFu);
    if (m_sleepers.load() == 0u)
        return;
    std::lock_guard<std::mutex> jobLock(m_jobMutex);
    m_jobCv.notify_all();
}

GSCpuBackend::PageRect GSCpuBackend::MakePageRect(uint32_t blockBase, uint32_t bufferWidth, uint32_t psm, int x0,
                                                  int y0, int x1, int y1)
{
    PageRect rect;
    x0 = std::max(x0, 0);
    y0 = std::max(y0, 0);
    if (x1 < x0 || y1 < y0)
        return rect;
    uint32_t pageWidth = 64u, pageHeight = 32u; // 32-bit formats, and 8H/4HL/4HH stored in 32-bit pages
    switch (psm)
    {
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        pageHeight = 64u;
        break;
    case GS_PSM_T8:
        pageWidth = 128u;
        pageHeight = 64u;
        break;
    case GS_PSM_T4:
        pageWidth = 128u;
        pageHeight = 128u;
        break;
    default:
        break;
    }
    rect.base = blockBase / 32u;
    rect.extra = (blockBase % 32u) != 0u ? 1u : 0u;
    // PixelStorageTraits::PageId uses (bw * 64) / pageWidth, truncating. Rounding up marked pages the
    // sampler and the writer never touch (odd buffer widths on PSMT8 / PSMT4, where a page is 128 wide).
    rect.stride = pageWidth == 0u ? 0u : (bufferWidth * 64u) / pageWidth;
    rect.row0 = static_cast<uint32_t>(y0) / pageHeight;
    rect.row1 = std::min(static_cast<uint32_t>(y1) / pageHeight, rect.row0 + kPageCount - 1u);
    rect.col0 = static_cast<uint32_t>(x0) / pageWidth;
    rect.col1 = std::min(static_cast<uint32_t>(x1) / pageWidth, rect.col0 + kPageCount - 1u);
    rect.empty = false;
    return rect;
}

uint32_t GSCpuBackend::PagesCoveringTexel(uint32_t blockBase, uint32_t bufferWidth, uint32_t psm, int x, int y,
                                         uint16_t *pages, uint32_t cap)
{
    if (!pages || cap == 0u)
        return 0u;
    const PageRect rect = MakePageRect(blockBase, bufferWidth, psm, x, y, x, y);
    if (rect.empty)
        return 0u;
    uint8_t seen[kPageCount] = {};
    uint32_t count = 0u;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
        {
            const uint32_t index = page % kPageCount;
            if (seen[index])
                continue;
            seen[index] = 1u;
            if (count < cap)
                pages[count++] = static_cast<uint16_t>(index);
        }
    }
    return count;
}

void GSCpuBackend::MarkPages(PageMask &mask, const PageRect &rect)
{
    if (rect.empty)
        return;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
        {
            uint16_t &count = mask[page % kPageCount];
            if (count != 0xFFFFu)
                ++count;
        }
    }
}

void GSCpuBackend::UnmarkPages(PageMask &mask, const PageRect &rect)
{
    if (rect.empty)
        return;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
        {
            uint16_t &count = mask[page % kPageCount];
            if (count != 0u)
                --count;
        }
    }
}

bool GSCpuBackend::AnyPages(const PageMask &mask, const PageRect &rect)
{
    if (rect.empty)
        return false;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
            if (mask[page % kPageCount] != 0u)
                return true;
    }
    return false;
}

void GSCpuBackend::SetPageBits(uint64_t bits[8], const PageRect &rect)
{
    std::memset(bits, 0, sizeof(uint64_t) * 8u);
    if (rect.empty)
        return;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
        {
            const uint32_t index = page % kPageCount;
            bits[index >> 6] |= uint64_t{1} << (index & 63u);
        }
    }
}

void GSCpuBackend::RetireFinishedJobs() const
{
    if (m_workers.empty() || !m_progress || !m_jobChunks)
        return;
    const uint64_t published = m_published.load(std::memory_order_acquire);
    const uint32_t gen = static_cast<uint32_t>(published >> 32);
    const uint32_t count = static_cast<uint32_t>(published);
    uint32_t minDone = count;
    for (size_t i = 0; i < m_workers.size(); ++i)
    {
        const uint64_t done = m_progress[i].done.load(std::memory_order_acquire);
        if (static_cast<uint32_t>(done >> 32) != gen)
        {
            minDone = 0u;
            break;
        }
        minDone = std::min(minDone, static_cast<uint32_t>(done));
    }
    // Jobs [0, minDone) have been drawn by every worker. A worker still on the previous generation forces minDone
    // to 0, and Wait already cleared the counts, so m_retired is 0 too.
    while (m_retired < minDone)
    {
        const uint32_t slot = m_retired++;
        const Job &job = m_jobChunks[slot / kJobChunkSize][slot % kJobChunkSize];
        UnmarkPages(m_queuedWritePages, job.frame);
        UnmarkPages(m_queuedWritePages, job.depth);
        for (uint8_t s = 0; s < job.sampledCount && s < 4u; ++s)
            UnmarkPages(m_queuedReadPages, job.sampled[s]);
    }
}

bool GSCpuBackend::UploadMissesInFlightPixels(int ux0, int uy0, int ux1, int uy1, uint32_t dbp, uint32_t dbw,
                                              uint8_t dpsm) const
{
    const uint32_t count = static_cast<uint32_t>(m_published.load(std::memory_order_relaxed));
    if (count < m_retired || !m_jobChunks)
        return false;
    if (count - m_retired > kBoxScanMax)
    {
        ++m_boxFailQueue;
        return false;
    }
    dbw = std::max<uint32_t>(dbw, 1u);
    // DBW is in 64-pixel units. X past that width is the same pixel as X - width on the next page row,
    // so a raw box test would miss a draw of the alias. Keep the page wait in that case.
    const int bufferPixels = static_cast<int>(dbw * 64u);
    if (ux0 < 0 || ux1 >= bufferPixels)
    {
        ++m_boxFailAlias;
        return false;
    }
    // 4-bit pixels share a byte with a horizontal neighbour. Pad so a disjoint test cannot split that byte.
    const int pad = (dpsm == GS_PSM_T4 || dpsm == GS_PSM_T4HL || dpsm == GS_PSM_T4HH) ? 1 : 0;
    const int ix0 = ux0 > pad ? ux0 - pad : 0;
    const int ix1 = ux1 + pad;
    uint64_t uploadBits[8];
    SetPageBits(uploadBits, MakePageRect(dbp, dbw, dpsm, ux0, uy0, ux1, uy1));
    for (uint32_t slot = m_retired; slot < count; ++slot)
    {
        const Job &job = m_jobChunks[slot / kJobChunkSize][slot % kJobChunkSize];
        if (!job.drew)
            continue;
        bool frameHit = false;
        bool depthHit = false;
        for (int word = 0; word < 8; ++word)
        {
            if ((job.frameBits[word] & uploadBits[word]) != 0u)
                frameHit = true;
            if ((job.depthBits[word] & uploadBits[word]) != 0u)
                depthHit = true;
        }
        if (!frameHit && !depthHit)
            continue;
        // This job writes a page the upload covers. Its box is not a pixel test when X wraps the buffer.
        if (job.boxX0 < 0 || job.boxX1 >= static_cast<int>(job.bufferWidth * 64u))
        {
            ++m_boxFailAlias;
            return false;
        }
        const bool boxHits = !(ix1 < job.boxX0 || job.boxX1 < ix0 || uy1 < job.boxY0 || job.boxY1 < uy0);
        if (frameHit)
        {
            const bool same = dbp == job.frameBase && dbw == job.bufferWidth && dpsm == job.framePsm;
            if (!same)
            {
                ++m_boxFailBuf;
                return false;
            }
            if (boxHits)
            {
                ++m_boxFailOverlap;
                return false;
            }
        }
        if (depthHit)
        {
            const bool same = dbp == job.depthBase && dbw == job.bufferWidth && dpsm == job.depthPsm;
            if (!same)
            {
                ++m_boxFailBuf;
                return false;
            }
            if (boxHits)
            {
                ++m_boxFailOverlap;
                return false;
            }
        }
    }
    return true;
}

bool GSCpuBackend::PagesQueued(const PageRect &rect) const
{
    return AnyPages(m_queuedWritePages, rect) || AnyPages(m_queuedReadPages, rect);
}

uint32_t GSCpuBackend::CollectPages(const PageRect &rect, uint16_t *pages, uint32_t cap) const
{
    if (rect.empty || !pages || cap == 0u)
        return 0u;
    uint8_t seen[kPageCount] = {};
    uint32_t count = 0u;
    for (uint32_t row = rect.row0; row <= rect.row1; ++row)
    {
        const uint32_t first = rect.base + row * rect.stride + rect.col0;
        const uint32_t end = std::min(rect.base + row * rect.stride + rect.col1 + 1u + rect.extra, first + kPageCount);
        for (uint32_t page = first; page < end; ++page)
        {
            const uint32_t index = page % kPageCount;
            if (seen[index])
                continue;
            seen[index] = 1u;
            if (count < cap)
                pages[count++] = static_cast<uint16_t>(index);
        }
    }
    return count;
}

uint32_t GSCpuBackend::PublishReadShadows(const PageRect &rect, uint16_t *pages, uint32_t cap)
{
    if (!m_pageShadows || !m_vram || !pages)
        return 0u;
    const uint32_t count = CollectPages(rect, pages, cap);
    uint32_t kept = 0u;
    for (uint32_t i = 0u; i < count; ++i)
    {
        const uint32_t index = pages[i];
        if (m_queuedReadPages[index] == 0u)
            continue;
        GSPageShadows::Node *node = nullptr;
        if (!m_pageShadows->freeNodes.empty())
        {
            node = m_pageShadows->freeNodes.back();
            m_pageShadows->freeNodes.pop_back();
        }
        else
        {
            m_pageShadows->owned.push_back(std::make_unique<GSPageShadows::Node>());
            node = m_pageShadows->owned.back().get();
        }
        node->epoch = m_epoch;
        node->older = m_pageShadows->head[index].load(std::memory_order_relaxed);
        std::memcpy(node->bytes, m_vram + static_cast<size_t>(index) * 8192u, 8192u);
        m_pageShadows->head[index].store(node, std::memory_order_release);
        pages[kept++] = static_cast<uint16_t>(index);
    }
    return kept;
}

void GSCpuBackend::BumpPageSeq(const uint16_t *pages, uint32_t count) const
{
    if (!m_pageShadows || !pages)
        return;
    for (uint32_t i = 0u; i < count; ++i)
        m_pageShadows->seq[pages[i]].fetch_add(1u, std::memory_order_release);
}

void GSCpuBackend::ReleasePageShadows() const
{
    if (!m_pageShadows)
        return;
    for (uint32_t i = 0u; i < kPageCount; ++i)
        m_pageShadows->head[i].store(nullptr, std::memory_order_relaxed);
    m_pageShadows->freeNodes.clear();
    m_pageShadows->freeNodes.reserve(m_pageShadows->owned.size());
    for (const std::unique_ptr<GSPageShadows::Node> &node : m_pageShadows->owned)
        m_pageShadows->freeNodes.push_back(node.get());
}

uint32_t GS_TakeSelfOverlapSerialDraws()
{
    return g_gsSelfOverlapSerial.exchange(0u, std::memory_order_relaxed);
}

GSCpuBackend::~GSCpuBackend()
{
    StopWorkers();
}

void GSCpuBackend::StartWorkersIfConfigured()
{
    m_workersConfigured = true;
    const char *setting = std::getenv("PS2X_GS_THREADS");
    if (!setting)
        return;
    unsigned count = static_cast<unsigned>(std::strtoul(setting, nullptr, 10));
    if (std::strcmp(setting, "auto") == 0)
    {
        const unsigned hardware = std::thread::hardware_concurrency();
        count = hardware > 2u ? hardware - 2u : 0u; // leave the EE and the window thread a core each
    }
    if (count < 2u)
        return;
    count = std::min(count, 16u);
    const char *stats = std::getenv("PS2X_GS_DRAIN_STATS");
    m_drainStats = stats && *stats && *stats != '0';
    m_drainDetail = m_drainStats ? (stats[0] >= '1' && stats[0] <= '9' ? stats[0] - '0' : 1) : 0;
    m_bandCount = count;
    m_stopWorkers = false;
    m_jobChunks = std::make_unique<std::unique_ptr<Job[]>[]>(kMaxJobChunks);
    m_published = 0u;
    m_progress = std::make_unique<WorkerProgress[]>(count);
    m_workerCaches.clear();
    if (!m_pageShadows)
        m_pageShadows = std::make_unique<GSPageShadows>();
    for (unsigned i = 0; i < count; ++i)
        m_workerCaches.push_back(std::make_unique<GSMem::TexturePageCache>());
    for (unsigned i = 0; i < count; ++i)
        m_workers.emplace_back([this, i]() { WorkerLoop(i); });
}

void GSCpuBackend::StopWorkers()
{
    if (m_workers.empty())
        return;
    {
        std::lock_guard<std::mutex> jobLock(m_jobMutex);
        m_stopWorkers = true;
    }
    m_jobCv.notify_all();
    for (std::thread &worker : m_workers)
        worker.join();
    m_workers.clear();
}

void GSCpuBackend::WaitForWorkersUnlocked(DrainReason reason) const
{
    if (m_workers.empty())
        return;
    if (m_drainStats)
    {
        const int64_t second = std::chrono::duration_cast<std::chrono::seconds>(
                                   std::chrono::steady_clock::now().time_since_epoch()).count();
        if (second != m_statSecond)
        {
            if (m_statSecond >= 0)
            {
                std::cout << "[gs:drains] primitives " << m_statPrimitives << " other " << m_drainCounts[0]
                          << " sample " << m_drainCounts[1] << " write " << m_drainCounts[2] << " upload "
                          << m_drainCounts[3] << " clut " << m_drainCounts[4] << " full " << m_drainCounts[5]
                          << " shadow " << m_shadowUploads << " box " << m_boxSkip
                          << " bq " << m_boxFailQueue << " ba " << m_boxFailAlias
                          << " bb " << m_boxFailBuf << " bo " << m_boxFailOverlap
                          << " nw " << m_uploadNoWrite;
                if (m_drainDetail >= 2)
                    std::cout << " saved " << m_savedSample << " " << m_savedWrite << " " << m_savedUpload
                              << " texfull " << m_texFull << " textight " << m_texTight
                              << " clutload " << m_clutLoads << " clutskip " << m_clutSkipped;
                std::cout << std::endl;
            }
            m_statSecond = second;
            m_statPrimitives = 0u;
            m_drainCounts.fill(0u);
            m_savedSample = m_savedWrite = m_savedUpload = 0u;
            m_texFull = m_texTight = m_clutLoads = m_clutSkipped = 0u;
            m_shadowUploads = 0u;
            m_boxSkip = 0u;
            m_boxFailQueue = m_boxFailAlias = m_boxFailBuf = m_boxFailOverlap = 0u;
            m_uploadNoWrite = 0u;
            m_detailPrints = 0u;
        }
    }
    const uint64_t target = m_published.load(std::memory_order_relaxed);
    if ((target & 0xFFFFFFFFu) != 0u)
    {
        ++m_drainCounts[static_cast<size_t>(reason)];
        auto allDone = [this, target]()
        {
            for (size_t i = 0; i < m_workers.size(); ++i)
                if (m_progress[i].done.load() != target)
                    return false;
            return true;
        };
        WakeWorkers();
        for (int spin = 0; spin < 4096 && !allDone(); ++spin)
            GS_CPU_RELAX();
        if (!allDone())
        {
            m_drainWaiting = true; // seq_cst: a worker finishing after this sees it and notifies
            std::unique_lock<std::mutex> jobLock(m_jobMutex);
            m_idleCv.wait(jobLock, allDone);
            m_drainWaiting = false;
        }
        // New generation: the next primitive goes to slot 0 (the workers notice the generation change).
        m_published.store(((target >> 32) + 1u) << 32);
        m_lastWakeCount = 0u;
        ++m_memoryStamp; // queued draws have landed; a CLUT loaded before them must be read again
    }
    m_queuedWritePages.fill(0u);
    m_queuedReadPages.fill(0u);
    m_retired = 0u;
    if (m_drainDetail >= 2)
        m_queuedReadPagesWide.fill(0u);
    // Whatever runs next on this thread may change GS memory: the workers drop their texture caches at the next
    // primitive, and the EE thread's own cache (used for CLUT loads) starts fresh too.
    ++m_epoch;
    m_texturePageCache.Invalidate();
    if (m_clutSlot != 0u)
    {
        std::memcpy(m_clutPool[0].data(), m_clut, 512u * sizeof(uint16_t));
        m_clutSlot = 0u;
        m_clut = m_clutPool[0].data();
    }
    m_clutRetire.fill(0u);
    m_clutRetire[0] = ~uint64_t{0};
    ReleasePageShadows();
}

void GSCpuBackend::WorkerLoop(unsigned index)
{
    t_bandIndex = index;
    t_bandCount = m_bandCount;
    t_textureCache = m_workerCaches[index].get();
    t_pageShadows = m_pageShadows.get();
    uint64_t done = 0u; // generation << 32 | jobs of that generation drawn
    uint32_t epoch = ~0u;
    int idleSpins = 0;
    for (;;)
    {
        if (m_stopWorkers.load(std::memory_order_relaxed)) // shutting down: queued primitives are dropped
            return;
        const uint64_t published = m_published.load(std::memory_order_acquire);
        if ((published >> 32) != (done >> 32))
            done = published & ~uint64_t{0xFFFFFFFFu};
        if ((done & 0xFFFFFFFFu) < (published & 0xFFFFFFFFu))
        {
            const uint32_t slot = static_cast<uint32_t>(done & 0xFFFFFFFFu);
            const Job &job = m_jobChunks[slot / kJobChunkSize][slot % kJobChunkSize];
            if (job.epoch != epoch)
            {
                t_textureCache->Invalidate();
                epoch = job.epoch;
            }
            t_clut = job.clut;
            t_jobEpoch = job.epoch;
            DrawPrimitive(job.batch);
            ++done;
            m_progress[index].done.store(done); // seq_cst: pairs with m_drainWaiting
            if (m_drainWaiting.load())
            {
                std::lock_guard<std::mutex> jobLock(m_jobMutex);
                m_idleCv.notify_all();
            }
            idleSpins = 0;
            continue;
        }
        // Caught up: spin a moment (the next primitive is usually close behind), then sleep until more is queued.
        if (++idleSpins < 512)
        {
            GS_CPU_RELAX();
            continue;
        }
        idleSpins = 0;
        std::unique_lock<std::mutex> jobLock(m_jobMutex);
        ++m_sleepers; // seq_cst, then the predicate reads m_published: Submit stores it, then reads m_sleepers
        m_jobCv.wait(jobLock, [this, published]() { return m_stopWorkers || m_published.load() != published; });
        --m_sleepers;
        if (m_stopWorkers)
            return;
    }
}

void GSCpuBackend::LoadClut(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || (!isFourBitIndexedPsm(tex0.psm) && !isEightBitIndexedPsm(tex0.psm)))
        return;

    switch (tex0.cld)
    {
    case 0u:
    case 6u:
    case 7u:
        return;
    case 1u:
        break;
    case 2u:
        m_clutCbp[0] = tex0.cbp;
        break;
    case 3u:
        m_clutCbp[1] = tex0.cbp;
        break;
    case 4u:
        if (m_clutCbp[0] == tex0.cbp)
            return;
        m_clutCbp[0] = tex0.cbp;
        break;
    case 5u:
        if (m_clutCbp[1] == tex0.cbp)
            return;
        m_clutCbp[1] = tex0.cbp;
        break;
    default:
        return;
    }

    const uint32_t clutBw = tex0.csm == 0u ? 1u : std::max<uint32_t>(texclut.cbw, 1u);
    const PageRect source =
        tex0.csm == 0u ? MakePageRect(tex0.cbp, clutBw, tex0.cpsm, 0, 0, 15, 15)
                       : MakePageRect(tex0.cbp, clutBw, tex0.cpsm, static_cast<int>(texclut.cou) * 16,
                                      static_cast<int>(texclut.cov), static_cast<int>(texclut.cou) * 16 + 255,
                                      static_cast<int>(texclut.cov));
    ClutLoadKey key;
    key.cbp = tex0.cbp;
    key.packed = static_cast<uint32_t>(tex0.cpsm) | (static_cast<uint32_t>(tex0.csm) << 8u) |
                 (static_cast<uint32_t>(tex0.csa) << 16u) | (static_cast<uint32_t>(tex0.psm) << 24u);
    key.cbw = clutBw;
    key.cou = texclut.cou;
    key.cov = texclut.cov;
    if (!m_workers.empty())
        RetireFinishedJobs();
    const bool queuedWrite = !m_workers.empty() && AnyPages(m_queuedWritePages, source);
    // Same registers and no write into the CLUT pages since the last load: the table bytes are unchanged.
    if (!queuedWrite && key.cbp == m_clutLoad.cbp && key.packed == m_clutLoad.packed && key.cbw == m_clutLoad.cbw &&
        key.cou == m_clutLoad.cou && key.cov == m_clutLoad.cov && m_clutLoad.stamp == m_memoryStamp)
    {
        if (m_drainStats)
            ++m_clutSkipped;
        return;
    }

    if (!m_workers.empty())
    {
        if (queuedWrite)
            WaitForWorkersUnlocked(DrainReason::ClutSource);
        // Queued primitives keep the table they were queued with: load into the next slot.
        if ((m_published.load(std::memory_order_relaxed) & 0xFFFFFFFFu) != 0u)
        {
            // Jobs already queued keep the current table. Reuse a slot whose jobs have all been drawn.
            auto slotFree = [this](uint32_t slot) {
                const uint64_t retire = m_clutRetire[slot];
                if (retire == ~uint64_t{0})
                    return false;
                const uint32_t gen = static_cast<uint32_t>(retire >> 32);
                const uint32_t need = static_cast<uint32_t>(retire);
                for (size_t i = 0; i < m_workers.size(); ++i)
                {
                    const uint64_t done = m_progress[i].done.load();
                    const uint32_t dgen = static_cast<uint32_t>(done >> 32);
                    if (dgen == gen)
                    {
                        if (static_cast<uint32_t>(done) < need)
                            return false;
                    }
                    else if (dgen < gen)
                        return false;
                }
                return true;
            };
            m_clutRetire[m_clutSlot] = m_published.load(std::memory_order_relaxed);
            uint32_t fresh = kClutSlots;
            for (uint32_t s = 0; s < kClutSlots; ++s)
            {
                if (s != m_clutSlot && slotFree(s))
                {
                    fresh = s;
                    break;
                }
            }
            if (fresh == kClutSlots)
                WaitForWorkersUnlocked(DrainReason::Other);
            if ((m_published.load(std::memory_order_relaxed) & 0xFFFFFFFFu) != 0u)
            {
                if (fresh == kClutSlots)
                {
                    // The wait collapsed the live table into slot 0. Publishing resumed, so that slot is
                    // in use again and the new table has to go somewhere else.
                    m_clutRetire[0] = m_published.load(std::memory_order_relaxed);
                    fresh = 1u;
                }
                std::memcpy(m_clutPool[fresh].data(), m_clut, 512u * sizeof(uint16_t));
                m_clutSlot = fresh;
                m_clut = m_clutPool[fresh].data();
                m_clutRetire[fresh] = ~uint64_t{0};
            }
        }
    }
    LoadClutUnlocked(tex0, texclut);
    m_clutLoad = key;
    m_clutLoad.stamp = m_memoryStamp;
    if (m_drainStats)
        ++m_clutLoads;
}

void GSCpuBackend::LoadClutUnlocked(const GSTex0Reg &tex0, const GSTexClutReg &texclut)
{
    const bool fourBit = isFourBitIndexedPsm(tex0.psm);
    const bool sixteenBit = tex0.cpsm == GS_PSM_CT16 || tex0.cpsm == GS_PSM_CT16S;
    const bool thirtyTwoBit = tex0.cpsm == GS_PSM_CT32 || tex0.cpsm == GS_PSM_CT24;
    if (!sixteenBit && !thirtyTwoBit)
        return;

    const uint32_t entryCount = fourBit ? 16u : 256u;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t destinationBase = (static_cast<uint32_t>(tex0.csa) & csaMask) << 4u;

    const bool loadCsm1Suffix = tex0.csm == 0u && thirtyTwoBit && !fourBit;
    const uint32_t firstEntry = loadCsm1Suffix ? destinationBase : 0u;

    for (uint32_t entry = firstEntry; entry < entryCount; ++entry)
    {
        uint32_t sourceX = 0u;
        uint32_t sourceY = 0u;
        uint32_t sourceWidth = 1u;

        if (tex0.csm == 0u)
        {
            const uint32_t sourceIndex = swizzleClutIndexCSM1(entry);
            sourceX = sourceIndex & 0x0Fu;
            sourceY = sourceIndex >> 4u;
        }
        else
        {
            sourceWidth = texclut.cbw != 0u ? static_cast<uint32_t>(texclut.cbw) : 1u;
            sourceX = (static_cast<uint32_t>(texclut.cou) << 4u) + entry;
            sourceY = static_cast<uint32_t>(texclut.cov);
        }

        const uint32_t raw = ReadTextureVramUnlocked(tex0.cpsm, tex0.cbp, sourceWidth, sourceX, sourceY); 
        const uint32_t destination = (loadCsm1Suffix ? entry : destinationBase + entry) & (sixteenBit ? 0x1FFu : 0x0FFu);
        if (sixteenBit)
        {
            m_clut[destination] = static_cast<uint16_t>(raw);
        }
        else
        {
            m_clut[destination] = static_cast<uint16_t>(raw & 0xFFFFu);
            m_clut[destination + 256u] = static_cast<uint16_t>(raw >> 16u);
        }
    }
}

void GSCpuBackend::Flush()
{
    // CPU backend is immediate. GPU backends may submit command buffers here.
}

void GSCpuBackend::TextureFlush()
{
    // Uploads have already reached GS memory. Queued primitives keep the page snapshot from before an upload
    // that overlapped their texture; only primitives queued from now on need fresh texture caches.
    std::lock_guard<std::mutex> lock(m_mutex);
    ++m_epoch;
    m_texturePageCache.Invalidate();
}

void GSCpuBackend::Sync(GSSyncReason)
{
    // Without drawing threads the CPU backend is immediate; with them, wait until they are done.
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
    return ReadVramUnlocked(psm, base, bw, x, y);
}

GS_FORCE_INLINE uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    // Direct calls so the swizzle inlines into the pixel loop. Same bytes as the function-pointer table.
    switch (psm & 0x3Fu)
    {
    case GS_PSM_CT32: return GSMem::PixelStorageTraits<GSMem::C32>::Read(m_vram, base, bw, x, y);
    case GS_PSM_CT24: return GSMem::PixelStorageTraits<GSMem::C24>::Read(m_vram, base, bw, x, y);
    case GS_PSM_CT16: return GSMem::PixelStorageTraits<GSMem::C16>::Read(m_vram, base, bw, x, y);
    case GS_PSM_CT16S: return GSMem::PixelStorageTraits<GSMem::C16S>::Read(m_vram, base, bw, x, y);
    case GS_PSM_T8: return GSMem::PixelStorageTraits<GSMem::P8>::Read(m_vram, base, bw, x, y);
    case GS_PSM_T4: return GSMem::PixelStorageTraits<GSMem::P4>::Read(m_vram, base, bw, x, y);
    case GS_PSM_T8H: return GSMem::PixelStorageTraits<GSMem::P8H>::Read(m_vram, base, bw, x, y);
    case GS_PSM_T4HL: return GSMem::PixelStorageTraits<GSMem::P4HL>::Read(m_vram, base, bw, x, y);
    case GS_PSM_T4HH: return GSMem::PixelStorageTraits<GSMem::P4HH>::Read(m_vram, base, bw, x, y);
    case GS_PSM_Z32: return GSMem::PixelStorageTraits<GSMem::Z32>::Read(m_vram, base, bw, x, y);
    case GS_PSM_Z24: return GSMem::PixelStorageTraits<GSMem::Z24>::Read(m_vram, base, bw, x, y);
    case GS_PSM_Z16: return GSMem::PixelStorageTraits<GSMem::Z16>::Read(m_vram, base, bw, x, y);
    case GS_PSM_Z16S: return GSMem::PixelStorageTraits<GSMem::Z16S>::Read(m_vram, base, bw, x, y);
    default: return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
    }
}

GS_FORCE_INLINE uint32_t GSCpuBackend::ReadTextureFast(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    if (!m_vram)
        return 0u;
    GSMem::TexturePageCache &cache = t_textureCache ? *t_textureCache : m_texturePageCache;
    switch (psm & 0x3Fu)
    {
    case GS_PSM_CT32: return GSMem::PixelStorageTraits<GSMem::C32>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_CT24: return GSMem::PixelStorageTraits<GSMem::C24>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_CT16: return GSMem::PixelStorageTraits<GSMem::C16>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_CT16S: return GSMem::PixelStorageTraits<GSMem::C16S>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_T8: return GSMem::PixelStorageTraits<GSMem::P8>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_T4: return GSMem::PixelStorageTraits<GSMem::P4>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_T8H: return GSMem::PixelStorageTraits<GSMem::P8H>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_T4HL: return GSMem::PixelStorageTraits<GSMem::P4HL>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_T4HH: return GSMem::PixelStorageTraits<GSMem::P4HH>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_Z32: return GSMem::PixelStorageTraits<GSMem::Z32>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_Z24: return GSMem::PixelStorageTraits<GSMem::Z24>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_Z16: return GSMem::PixelStorageTraits<GSMem::Z16>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    case GS_PSM_Z16S: return GSMem::PixelStorageTraits<GSMem::Z16S>::ReadThroughCache(cache, m_vram, base, bw, x, y);
    default: return GSMem::ReadTexture(cache, m_vram, psm, base, bw, x, y);
    }
}

uint32_t GSCpuBackend::ReadTextureVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y)
{
    return ReadTextureFast(psm, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
    WriteVramUnlocked(psm, base, bw, x, y, value);
    ++m_memoryStamp;
}

GS_FORCE_INLINE void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    switch (psm & 0x3Fu)
    {
    case GS_PSM_CT32: GSMem::PixelStorageTraits<GSMem::C32>::Write(m_vram, base, bw, x, y, value); return;
    case GS_PSM_CT24: GSMem::PixelStorageTraits<GSMem::C24>::Write(m_vram, base, bw, x, y, value); return;
    case GS_PSM_CT16: GSMem::PixelStorageTraits<GSMem::C16>::Write(m_vram, base, bw, x, y, static_cast<uint16_t>(value)); return;
    case GS_PSM_CT16S: GSMem::PixelStorageTraits<GSMem::C16S>::Write(m_vram, base, bw, x, y, static_cast<uint16_t>(value)); return;
    case GS_PSM_T8: GSMem::PixelStorageTraits<GSMem::P8>::Write(m_vram, base, bw, x, y, static_cast<uint8_t>(value)); return;
    case GS_PSM_T4: GSMem::PixelStorageTraits<GSMem::P4>::Write(m_vram, base, bw, x, y, static_cast<uint8_t>(value)); return;
    case GS_PSM_T8H: GSMem::PixelStorageTraits<GSMem::P8H>::Write(m_vram, base, bw, x, y, static_cast<uint8_t>(value)); return;
    case GS_PSM_T4HL: GSMem::PixelStorageTraits<GSMem::P4HL>::Write(m_vram, base, bw, x, y, static_cast<uint8_t>(value)); return;
    case GS_PSM_T4HH: GSMem::PixelStorageTraits<GSMem::P4HH>::Write(m_vram, base, bw, x, y, static_cast<uint8_t>(value)); return;
    case GS_PSM_Z32: GSMem::PixelStorageTraits<GSMem::Z32>::Write(m_vram, base, bw, x, y, value); return;
    case GS_PSM_Z24: GSMem::PixelStorageTraits<GSMem::Z24>::Write(m_vram, base, bw, x, y, value); return;
    case GS_PSM_Z16: GSMem::PixelStorageTraits<GSMem::Z16>::Write(m_vram, base, bw, x, y, static_cast<uint16_t>(value)); return;
    case GS_PSM_Z16S: GSMem::PixelStorageTraits<GSMem::Z16S>::Write(m_vram, base, bw, x, y, static_cast<uint16_t>(value)); return;
    default: m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value); return;
    }
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 64u)
        {
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        if (ownsRow(py))
            WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

// The part of WritePixel that depends only on the draw state.
struct GSCpuBackend::PixelPipeline
{
    explicit PixelPipeline(const GSDrawState &state)
    {
        const auto &ctx = state.context;
        scissorX0 = ctx.scissor.x0;
        scissorX1 = ctx.scissor.x1;
        scissorY0 = ctx.scissor.y0;
        scissorY1 = ctx.scissor.y1;
        fog = state.prim.fge;
        fogR = state.fogR;
        fogG = state.fogG;
        fogB = state.fogB;
        fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
        fbw = std::max<u32>(ctx.frame.fbw, 1u);
        fpsm = ctx.frame.psm;
        sixteenBitFrame = fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S || fpsm == GS_PSM_Z16 || fpsm == GS_PSM_Z16S;
        zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
        zpsm = ctx.zbuf.psm;
        zmask = ctx.zbuf.zmask;
        test = ctx.test;
        ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
        alphaBlend = state.prim.abe;
        pabe = state.pabe;
        destinationAlphaTestNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u &&
                                        (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);
        fbmsk = ctx.frame.fbmsk;
        forceAlphaMsb = (ctx.fba & 0x1ull) != 0ull && fpsm != GS_PSM_CT24;
        asel = static_cast<uint8_t>(ctx.alpha & 3u);
        bsel = static_cast<uint8_t>((ctx.alpha >> 2) & 3u);
        csel = static_cast<uint8_t>((ctx.alpha >> 4) & 3u);
        dsel = static_cast<uint8_t>((ctx.alpha >> 6) & 3u);
        fix = static_cast<uint8_t>((ctx.alpha >> 32) & 0xFFu);
    }

    int scissorX0, scissorX1, scissorY0, scissorY1;
    bool fog;
    uint8_t fogR, fogG, fogB;
    u32 fbp, fbw, fpsm, zbp, zpsm;
    bool sixteenBitFrame, zmask;
    uint64_t test;
    uint32_t ztestMethod;
    bool alphaBlend, pabe, destinationAlphaTestNeedsRead, forceAlphaMsb;
    u32 fbmsk;
    uint8_t asel, bsel, csel, dsel, fix;
};

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    WritePixel(PixelPipeline(state), x, y, z, r, g, b, a, fog);
}

bool GSCpuBackend::DepthRejects(const PixelPipeline &p, int x, int y, int z) const
{
    // WritePixel returns with no store when the depth test fails, including after an alpha-test failure
    // (that path returns earlier, also with no store). Skipping the texture sample matches those bytes.
    if (p.ztestMethod == 1u)
        return false;
    if (p.ztestMethod == 0u)
        return true;
    const uint32_t storedZ = ReadVramUnlocked(p.zpsm, p.zbp, p.fbw, static_cast<uint32_t>(x), static_cast<uint32_t>(y));
    const uint32_t zz = static_cast<uint32_t>(z);
    if (p.ztestMethod == 2u)
        return zz < storedZ;
    return zz <= storedZ;
}

void GSCpuBackend::WritePixel(const PixelPipeline &p, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog,
                              bool zAlreadyPassed)
{
    if (x < p.scissorX0 || x > p.scissorX1 || y < p.scissorY0 || y > p.scissorY1)
        return;

    if (p.fog)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, p.fogR);
        g = applyFog(g, p.fogG);
        b = applyFog(b, p.fogB);
    }

    const PixelWriteMask writeMask = classifyAlphaTest(p.test, a, static_cast<uint8_t>(p.fpsm));
    if (!writeMask.writesAnything())
    {
        return;
    }

    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && p.fpsm == GS_PSM_CT32;

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = p.destinationAlphaTestNeedsRead ||
                      (writeMask.writesFramebuffer() && ((p.fbmsk != 0) || p.alphaBlend || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(p.fpsm, p.fbp, p.fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (p.sixteenBitFrame)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (p.fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(p.test, static_cast<uint8_t>(p.fpsm), rawFramebufferPixel))
    {
        return;
    }

    bool zpass = zAlreadyPassed;
    uint32_t storedZ = 0u;
    if (!zAlreadyPassed)
    {
        switch (p.ztestMethod)
        {
        case 0:
            zpass = false;
            break;
        case 1:
            zpass = true;
            break;
        case 2:
            storedZ = ReadVramUnlocked(p.zpsm, p.zbp, p.fbw, x, y);
            zpass = static_cast<uint32_t>(z) >= storedZ;
            break;
        case 3:
            storedZ = ReadVramUnlocked(p.zpsm, p.zbp, p.fbw, x, y);
            zpass = static_cast<uint32_t>(z) > storedZ;
            break;
        }
    }

    if (!zpass)
    {
        return;
    }

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (p.alphaBlend)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(p.pabe && (a & 0x80u) == 0u))
            {
                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (p.csel == 0) ? a : (p.csel == 1) ? da
                                                               : p.fix;

                r = clampU8(((pickRGB(p.asel, r, dr) - pickRGB(p.bsel, r, dr)) * cAlpha >> 7) + pickRGB(p.dsel, r, dr));
                g = clampU8(((pickRGB(p.asel, g, dg) - pickRGB(p.bsel, g, dg)) * cAlpha >> 7) + pickRGB(p.dsel, g, dg));
                b = clampU8(((pickRGB(p.asel, b, db) - pickRGB(p.bsel, b, db)) * cAlpha >> 7) + pickRGB(p.dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && p.forceAlphaMsb)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (p.fbmsk != 0)
        {
            pixel = (pixel & ~p.fbmsk) | (fbrgba & p.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (p.sixteenBitFrame)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(p.fpsm, p.fbp, p.fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !p.zmask)
    {
        WriteVramUnlocked(p.zpsm, p.zbp, p.fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const bool sixteenBit = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
    const uint32_t csaMask = sixteenBit ? 0x1Fu : 0x0Fu;
    const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;
    const uint32_t sourceIndex = isFourBitIndexedPsm(sourcePsm)
                                     ? (static_cast<uint32_t>(index) & 0x0Fu)
                                     : static_cast<uint32_t>(index);

    uint32_t clutIndex = (clutBase + sourceIndex) & (sixteenBit ? 0x1FFu : 0x0FFu);
    if (!sixteenBit && csm == 0u && isEightBitIndexedPsm(sourcePsm))
    {
        const uint32_t block = std::min((sourceIndex & 0xF0u) + clutBase, 240u);
        clutIndex = block + (sourceIndex & 0x0Fu);
    }

    const uint16_t *clut = t_clut ? t_clut : m_clut;
    switch (cpsm)
    {
    case GS_PSM_CT32:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw);
    }
    case GS_PSM_CT24:
    {
        const uint32_t raw = static_cast<uint32_t>(clut[clutIndex]) | (static_cast<uint32_t>(clut[clutIndex + 256u]) << 16u);
        return applyTexa(state.texa, cpsm, raw & 0x00FFFFFFu);
    }
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(clut[clutIndex]));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

GS_FORCE_INLINE uint32_t GSCpuBackend::SamplePoint(const GSDrawState &state, int sampleU, int sampleV, int texW, int texH,
                                                   uint8_t wrapU, uint8_t wrapV, uint16_t minU, uint16_t maxU,
                                                   uint16_t minV, uint16_t maxV)
{
    const auto &tex = state.context.tex0;
    sampleU = wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU);
    sampleV = wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV);
    const u32 out = ReadTextureFast(tex.psm, tex.tbp0, tex.tbw, static_cast<uint32_t>(sampleU),
                                    static_cast<uint32_t>(sampleV));
    switch (tex.psm)
    {
    case GS_PSM_CT32:
    case GS_PSM_Z32:
    case GS_PSM_CT24:
    case GS_PSM_Z24:
        return applyTexa(state.texa, tex.psm, out);
    case GS_PSM_CT16:
    case GS_PSM_CT16S:
    case GS_PSM_Z16:
    case GS_PSM_Z16S:
        return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
    case GS_PSM_T8:
    case GS_PSM_T8H:
    case GS_PSM_T4:
    case GS_PSM_T4HL:
    case GS_PSM_T4HH:
        return LookupCLUT(state, static_cast<u8>(out), tex.cpsm, tex.csm, tex.csa, tex.psm);
    }
    return 0xFFFF00FFu;
}

GS_FORCE_INLINE uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u,
                                                       uint16_t v, TexelReuse *reuse)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    auto fetch = [&](int su, int sv) {
        return SamplePoint(state, su, sv, texW, texH, wrapU, wrapV, minU, maxU, minV, maxV);
    };

    if (!state.linearFilter)
    {
        const int su = static_cast<int>(texUf);
        const int sv = static_cast<int>(texVf);
        if (reuse && (reuse->valid & 0x1u) != 0u && reuse->keyU == su && reuse->keyV == sv)
            return reuse->c00;
        const uint32_t color = fetch(su, sv);
        if (reuse)
        {
            reuse->keyU = su;
            reuse->keyV = sv;
            reuse->c00 = color;
            reuse->valid = 0x1u;
        }
        return color;
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = floorToInt(sampleU);
    const int v0 = floorToInt(sampleV);
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    // A texel with weight 0 does not change the result: copies that sample on texel centres (fx or fy = 0, e.g.
    // a screen copy at a half-pixel offset) need only one or two fetches. Reused corners are fetches from the
    // previous pixel, never the weight-0 copies (those bits stay clear in valid).
    uint32_t c00 = 0u, c10 = 0u, c01 = 0u, c11 = 0u;
    uint8_t have = 0u;
    if (reuse && reuse->keyU == u0 && reuse->keyV == v0)
    {
        c00 = reuse->c00;
        c10 = reuse->c10;
        c01 = reuse->c01;
        c11 = reuse->c11;
        have = reuse->valid;
    }
    else if (reuse && reuse->keyU + 1 == u0 && reuse->keyV == v0 && (reuse->valid & 0xAu) == 0xAu)
    {
        c00 = reuse->c10;
        c01 = reuse->c11;
        have = 0x5u;
    }
    if ((have & 0x1u) == 0u)
        c00 = fetch(u0, v0);
    if (fx == 0.0f && fy == 0.0f)
    {
        if (reuse)
        {
            reuse->keyU = u0;
            reuse->keyV = v0;
            reuse->c00 = c00;
            reuse->valid = 0x1u;
        }
        return c00;
    }
    if (fx != 0.0f)
    {
        if ((have & 0x2u) == 0u)
            c10 = fetch(u1, v0);
    }
    else
        c10 = c00;
    if (fy != 0.0f)
    {
        if ((have & 0x4u) == 0u)
            c01 = fetch(u0, v1);
    }
    else
        c01 = c00;
    if (fx != 0.0f && fy != 0.0f)
    {
        if ((have & 0x8u) == 0u)
            c11 = fetch(u1, v1);
    }
    else
        c11 = fx != 0.0f ? c10 : c01;
    if (reuse)
    {
        reuse->keyU = u0;
        reuse->keyV = v0;
        reuse->c00 = c00;
        reuse->c10 = c10;
        reuse->c01 = c01;
        reuse->c11 = c11;
        uint8_t real = 0x1u;
        if (fx != 0.0f)
            real |= 0x2u;
        if (fy != 0.0f)
            real |= 0x4u;
        if (fx != 0.0f && fy != 0.0f)
            real |= 0x8u;
        reuse->valid = real;
    }

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    u32 z1 = static_cast<u32>(v1.z);

    // Vertices may come in either order; the texture coordinates stay with their vertex (swapped below).
    const bool flipX = fx0 > fx1;
    const bool flipY = fy0 > fy1;
    if (flipX)
        std::swap(fx0, fx1);
    if (flipY)
        std::swap(fy0, fy1);

    // A sprite covers the pixels X0 <= x < X1 (pixel centres at integer positions), so a right edge at 63.5
    // still draws column 63: games tile the screen with such strips (x 0..63.5, 64..127.5, ...).
    const int unclippedX0 = static_cast<int>(std::ceil(fx0));
    const int unclippedY0 = static_cast<int>(std::ceil(fy0));
    const int unclippedX1 = std::max(unclippedX0, static_cast<int>(std::ceil(fx1)) - 1);
    const int unclippedY1 = std::max(unclippedY0, static_cast<int>(std::ceil(fy1)) - 1);
    const float spanXf = std::max(1.0f, fx1 - fx0);
    const float spanYf = std::max(1.0f, fy1 - fy0);

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;
    if (alphaTestDiscardsPrimitive(ctx.test, state.prim.tme, a))
        return;
    const PixelPipeline pipe(state);

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u) / 16.0f; // UV is 10.4 fixed point
            v0f = static_cast<float>(v0.v) / 16.0f;
            u1f = static_cast<float>(v1.u) / 16.0f;
            v1f = static_cast<float>(v1.v) / 16.0f;
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }
        if (flipX)
            std::swap(u0f, u1f);
        if (flipY)
            std::swap(v0f, v1f);

        TexelReuse reuse;
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsRow(y))
                continue;
            // The GS evaluates UV at the pixel's own position (pixel centres are integer).
            float ty = (static_cast<float>(y) - fy0) / spanYf;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                if (DepthRejects(pipe, x, y, static_cast<int>(z1)))
                    continue;
                float tx = (static_cast<float>(x) - fx0) / spanXf;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(state, 0.0f, 0.0f, 1.0f, sampleU, sampleV, &reuse);
                }
                else
                {
                    texel = SampleTexture(state, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u, &reuse);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixel(pipe, x, y, z1, color.r, color.g, color.b, color.a, v1.fog, true);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!ownsRow(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
            {
                if (DepthRejects(pipe, x, y, static_cast<int>(z1)))
                    continue;
                WritePixel(pipe, x, y, z1, r, g, b, a, v1.fog, true);
            }
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const PixelPipeline pipe(state);
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;
    // Half-plane x range in double, padded so it is a superset of the float test below. Weights inside the
    // span are still the same float expression as the old per-pixel loop (an incremental add would not be).
    const double A0 = static_cast<double>(fy1 - fy2) * static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double C0 = static_cast<double>(fx2 - fx1) * static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double A1 = static_cast<double>(fy2 - fy0) * static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double C1 = static_cast<double>(fx0 - fx2) * static_cast<double>(winding) * static_cast<double>(invAbsDenom);
    const double A2 = -(A0 + A1);
    const double C2 = -(C0 + C1);
    const double fx2d = static_cast<double>(fx2);
    const double fy2d = static_cast<double>(fy2);
    TexelReuse reuse;

    for (int y = minY; y <= maxY; ++y)
    {
        if (!ownsRow(y))
            continue;
        float py = static_cast<float>(y) + 0.5f;
        const double dpy = static_cast<double>(py) - fy2d;
        double spanLo = static_cast<double>(minX);
        double spanHi = static_cast<double>(maxX);
        auto clipHalfPlane = [&](double A, double K) {
            constexpr double kEps = 1.0e-4;
            constexpr double kPad = 2.0;
            if (std::fabs(A) < 1.0e-8)
            {
                if (K < -kEps - 1.0e-3)
                {
                    spanLo = 1.0;
                    spanHi = 0.0;
                }
                return;
            }
            const double xBound = ((-kEps - K) / A) - 0.5;
            if (A > 0.0)
                spanLo = std::max(spanLo, xBound - kPad);
            else
                spanHi = std::min(spanHi, xBound + kPad);
        };
        clipHalfPlane(A0, C0 * dpy - A0 * fx2d);
        clipHalfPlane(A1, C1 * dpy - A1 * fx2d);
        clipHalfPlane(A2, 1.0 + C2 * dpy - A2 * fx2d);
        if (!(spanLo <= spanHi))
            continue;
        const int xBegin = clampInt(static_cast<int>(std::floor(spanLo)), minX, maxX);
        const int xEnd = clampInt(static_cast<int>(std::ceil(spanHi)), minX, maxX);
        for (int x = xBegin; x <= xEnd; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;
            const int zi = static_cast<int>(static_cast<uint32_t>(z + 0.5));
            if (DepthRejects(pipe, x, y, zi))
                continue;

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(state, is, it, iq, iu, iv, &reuse);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            WritePixel(pipe, x, y, zi, r, g, b, a, fog, true);
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const PixelPipeline pipe(state);
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        if (ownsRow(y0))
            WritePixel(pipe, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (command.direction != 0u) // host -> local only touches GS memory in UploadImage, which checks its pages
        WaitForWorkersUnlocked();
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // A call that stores nothing must not drain the workers. The fight issues these against the previous
    // BITBLTBUF, which still shares pages with queued draws, and the old path waited then returned.
    const bool canWrite = data && sizeBytes != 0u && m_vram && m_transferState.direction == 0u &&
                          m_transfer.trxreg.rrw != 0u && m_transfer.trxreg.rrh != 0u &&
                          m_transferState.totalPixels != 0u &&
                          m_transferState.copiedPixels < m_transferState.totalPixels;
    if (!canWrite)
    {
        ++m_uploadNoWrite;
        return;
    }
    uint16_t shadowPages[kPageCount];
    uint32_t shadowCount = 0u;
    if (!m_workers.empty())
    {
        // A queued draw into these pages has to finish first (it owns the bytes). A queued sample only
        // needs the bytes from before this upload: keep those in a snapshot and write live memory now.
        const int dx = static_cast<int>(m_transfer.trxpos.dsax), dy = static_cast<int>(m_transfer.trxpos.dsay);
        const uint32_t dbwPages = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
        const PageRect destination =
            MakePageRect(m_transfer.bitbltbuf.dbp, dbwPages, m_transfer.bitbltbuf.dpsm, dx, dy,
                         dx + static_cast<int>(m_transfer.trxreg.rrw) - 1, dy + static_cast<int>(m_transfer.trxreg.rrh) - 1);
        RetireFinishedJobs();
        const bool writeHit = AnyPages(m_queuedWritePages, destination);
        // A page shared with a queued draw is not a write conflict when the pixel boxes miss. It can still be a
        // sample of this upload: those jobs need a snapshot, same as a plain read hit.
        const bool boxMiss = writeHit &&
                             UploadMissesInFlightPixels(dx, dy, dx + static_cast<int>(m_transfer.trxreg.rrw) - 1,
                                                        dy + static_cast<int>(m_transfer.trxreg.rrh) - 1,
                                                        m_transfer.bitbltbuf.dbp, dbwPages, m_transfer.bitbltbuf.dpsm);
        const bool realWrite = writeHit && !boxMiss;
        const bool readHit = !realWrite && AnyPages(m_queuedReadPages, destination);
        if (realWrite)
        {
            if (m_drainDetail >= 2 && m_detailPrints < 12u)
            {
                ++m_detailPrints;
                std::cout << "[gs:conflict] upload dbp=" << m_transfer.bitbltbuf.dbp
                          << " dbw=" << static_cast<uint32_t>(m_transfer.bitbltbuf.dbw)
                          << " dpsm=0x" << std::hex << static_cast<uint32_t>(m_transfer.bitbltbuf.dpsm) << std::dec
                          << " rect=" << dx << "," << dy << "+" << m_transfer.trxreg.rrw << "x" << m_transfer.trxreg.rrh
                          << " hit=" << (realWrite ? "write" : "read")
                          << std::endl;
            }
            WaitForWorkersUnlocked(DrainReason::Upload);
        }
        else if (readHit)
        {
            if (boxMiss)
                ++m_boxSkip;
            shadowCount = PublishReadShadows(destination, shadowPages, kPageCount);
            ++m_shadowUploads;
        }
        else if (boxMiss)
            ++m_boxSkip;
        else if (m_drainDetail >= 2 && AnyPages(m_queuedReadPagesWide, destination))
            ++m_savedUpload;
        ++m_epoch; // primitives queued after this upload must not sample stale cached pages
    }
    m_texturePageCache.Invalidate();
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;
    // Serial runs have no workers, so the bump cannot live in the block above. One bump per upload.
    ++m_memoryStamp;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
    if (rrw == 0u || m_transferState.copiedPixels >= m_transferState.totalPixels)
        return;
    // Inlined swizzle stores. Same pixels as the per-pixel WriteVram loop, including a PSMT4 byte whose
    // second nibble is dropped when it would pass the end of the rectangle. Seqlock the snapshotted pages
    // so a sample that misses the snapshot retries instead of copying a half-written page.
    if (shadowCount != 0u)
        BumpPageSeq(shadowPages, shadowCount);
    const GSMem::ImageTransferWrite wrote = GSMem::WriteImageTransfer(
        m_vram, dbp, dbw, dpsm, m_transferState.x, m_transferState.y, dsax, rrw, data, sizeBytes,
        m_transferState.totalPixels - m_transferState.copiedPixels);
    if (shadowCount != 0u)
        BumpPageSeq(shadowPages, shadowCount);
    if (wrote.pixels == 0u)
        return;
    m_transferState.copiedPixels =
        std::min(m_transferState.totalPixels, m_transferState.copiedPixels + wrote.pixels);
    if (m_transferState.copiedPixels >= m_transferState.totalPixels)
    {
        m_transferState.direction = 3u;
        m_transferState.totalPixels = 0u;
        return;
    }
    m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
    m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    ++m_memoryStamp;
    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    WaitForWorkersUnlocked();
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        ++m_memoryStamp;
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        ++m_memoryStamp;
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVram(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const uint32_t lineRepeat1 = decodeDisplayLineRepeat(request.display1, smode2);
    const uint32_t lineRepeat2 = decodeDisplayLineRepeat(request.display2, smode2);
    // In interlaced FRAME mode the buffer holds one field; a TV shows the odd one a line lower. Place it by the
    // field it was drawn for, not the current one: a game below 60 fps keeps a field on screen for several.
    const uint32_t fieldOffset = (smode2.interlaced && smode2.frameMode && request.displayField == 1) ? 1u : 0u;
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;

    // Reads the buffer lines a circuit shows (displayHeight / lineRepeat of them) and repeats each one to fill
    // its display height.
    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t displayHeight,
                          uint32_t lineRepeat,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        const uint32_t height = (displayHeight + lineRepeat - 1u) / lineRepeat;
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        repeatPresentationLines(pixels, width, displayHeight, lineRepeat, lineRepeat > 1u ? fieldOffset : 0u);
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, height1, lineRepeat1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, height2, lineRepeat2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < height2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < height1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return dumpPresented(std::move(result));
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    if (!copySource(displayFrame, origin, result.width, result.height, valid1 ? lineRepeat1 : lineRepeat2, true, false,
                    selected, result.pixels, result.usedPreferred))
        return {};
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return dumpPresented(std::move(result));
}
