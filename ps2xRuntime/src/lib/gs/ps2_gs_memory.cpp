#include <array>
#include <cstring>
#include <mutex>

#include "runtime/gs/ps2_gs_memory.h"

namespace GSMem
{
    using C32Traits  = PixelStorageTraits<C32>;
    using Z32Traits  = PixelStorageTraits<Z32>;
    using C16Traits  = PixelStorageTraits<C16>;
    using C16STraits = PixelStorageTraits<C16S>;
    using Z16Traits  = PixelStorageTraits<Z16>;
    using Z16STraits = PixelStorageTraits<Z16S>;
    using P8Traits   = PixelStorageTraits<P8>;
    using P4Traits   = PixelStorageTraits<P4>;

    using C32PageLookupTable    = C32Traits::PageLookupTableT;
    using C32BlockLookupTableT  = C32Traits::BlockLookupTableT;
    using Z32PageLookupTableT = Z32Traits::PageLookupTableT;
    using Z32BlockLookupTableT = Z32Traits::BlockLookupTableT;

    // column table is shared
    using C32ColumnLookupTableT = C32Traits::ColumnLookupTableT;

    using C16PageLookupTable   = C16Traits::PageLookupTableT;
    using C16BlockLookupTable  = C16Traits::BlockLookupTableT;
    using Z16PageLookupTable   = Z16Traits::PageLookupTableT;
    using Z16BlockLookupTable  = Z16Traits::BlockLookupTableT;

    using C16SPageLookupTable  = C16STraits::PageLookupTableT;
    using C16SBlockLookupTable = C16STraits::BlockLookupTableT;
    using Z16SPageLookupTable  = Z16STraits::PageLookupTableT;
    using Z16SBlockLookupTable = Z16STraits::BlockLookupTableT;

    // column table is shared
    using C16ColumnLookupTable = C16Traits::ColumnLookupTableT;

    using P8PageLookupTable   = P8Traits::PageLookupTableT;
    using P8BlockLookupTable  = P8Traits::BlockLookupTableT;
    using P8ColumnLookupTable = P8Traits::ColumnLookupTableT;

    using P4PageLookupTable   = P4Traits::PageLookupTableT;
    using P4BlockLookupTable  = P4Traits::BlockLookupTableT;
    using P4ColumnLookupTable = P4Traits::ColumnLookupTableT;


    // this is going to be massive (an entire page of addess lookups)
    static C32PageLookupTable  PageTableC32{ };
    static Z32PageLookupTableT PageTableZ32{ };
    static C16PageLookupTable  PageTableC16{ };
    static C16SPageLookupTable PageTableC16S{ };
    static Z16PageLookupTable  PageTableZ16{ };
    static Z16SPageLookupTable PageTableZ16S{ };
    static P8PageLookupTable   PageTableP8{ };
    static P4PageLookupTable   PageTableP4{ };

    void InitLookupTables()
    {
        // Once: the CPU backend, the GL backend's texture store and gs_swizzle all call this, from different threads.
        static std::once_flag once;
        std::call_once(once, [] {
            // 32 bit
            PixelStorageTraits<C32>::InitPageLookupTable(PageTableC32, BlockTableC32, ColumnTable32);
            PixelStorageTraits<Z32>::InitPageLookupTable(PageTableZ32, BlockTableZ32, ColumnTable32);

            // 16 bit
            PixelStorageTraits<C16>::InitPageLookupTable(PageTableC16, BlockTableC16, ColumnTable16);
            PixelStorageTraits<C16S>::InitPageLookupTable(PageTableC16S, BlockTableC16S, ColumnTable16);
            PixelStorageTraits<Z16>::InitPageLookupTable(PageTableZ16, BlockTableZ16, ColumnTable16);
            PixelStorageTraits<Z16S>::InitPageLookupTable(PageTableZ16S, BlockTableZ16S, ColumnTable16);

            // 8 bit
            PixelStorageTraits<P8>::InitPageLookupTable(PageTableP8, BlockTableP8, ColumnTable8);

            // 4 bit
            PixelStorageTraits<P4>::InitPageLookupTable(PageTableP4, BlockTableP4, ColumnTable4);
        });
    }

    // GL backend (from the God of War port, adapted from Taylor N. Albarnaz / LightVelox, sotc-port ac9efa0, GPL-3.0):
    // the page tables its shaders and texture store address GS memory with.
    const u16* PageTableData(PixelStorageMode psm)
    {
        switch (psm)
        {
        case C32:
        case C24:
        case P8H:
        case P4HL:
        case P4HH:
            return &PageTableC32[0][0][0];
        case Z32:
        case Z24:
            return &PageTableZ32[0][0][0];
        case C16:
            return &PageTableC16[0][0][0];
        case C16S:
            return &PageTableC16S[0][0][0];
        case Z16:
            return &PageTableZ16[0][0][0];
        case Z16S:
            return &PageTableZ16S[0][0][0];
        case P8:
            return &PageTableP8[0][0][0];
        case P4:
            return &PageTableP4[0][0][0];
        default:
            return nullptr;
        }
    }

    bool SwizzleMatchesLookup()
    {
        InitLookupTables();
        auto check = [](auto psmTag, const auto &table) {
            constexpr PixelStorageMode psm = decltype(psmTag)::value;
            using Traits = PixelStorageTraits<psm>;
            const u32 bps[] = {0u, 1u, 17u, 32u, 100u, 511u};
            const u32 bws[] = {1u, 2u, 5u, 10u};
            for (u32 bp : bps)
                for (u32 bw : bws)
                    for (u32 y = 0u; y < 160u; y += 3u)
                        for (u32 x = 0u; x < 200u; x += 3u)
                            if (SwizzlePixel<psm>(bp, bw, x, y) != static_cast<u32>(Traits::Address(table, bp, bw, x, y)))
                                return false;
            return true;
        };
        using C32T = std::integral_constant<PixelStorageMode, C32>;
        using Z32T = std::integral_constant<PixelStorageMode, Z32>;
        using C24T = std::integral_constant<PixelStorageMode, C24>;
        using Z24T = std::integral_constant<PixelStorageMode, Z24>;
        using C16T = std::integral_constant<PixelStorageMode, C16>;
        using C16ST = std::integral_constant<PixelStorageMode, C16S>;
        using Z16T = std::integral_constant<PixelStorageMode, Z16>;
        using Z16ST = std::integral_constant<PixelStorageMode, Z16S>;
        using P8T = std::integral_constant<PixelStorageMode, P8>;
        using P4T = std::integral_constant<PixelStorageMode, P4>;
        using P8HT = std::integral_constant<PixelStorageMode, P8H>;
        using P4HLT = std::integral_constant<PixelStorageMode, P4HL>;
        using P4HHT = std::integral_constant<PixelStorageMode, P4HH>;
        return check(C32T{}, PageTableC32) && check(Z32T{}, PageTableZ32) && check(C24T{}, PageTableC32) &&
               check(Z24T{}, PageTableZ32) && check(C16T{}, PageTableC16) && check(C16ST{}, PageTableC16S) &&
               check(Z16T{}, PageTableZ16) && check(Z16ST{}, PageTableZ16S) && check(P8T{}, PageTableP8) &&
               check(P4T{}, PageTableP4) && check(P8HT{}, PageTableC32) && check(P4HLT{}, PageTableC32) &&
               check(P4HHT{}, PageTableC32);
    }

    u32 ReadTexture(TexturePageCache& cache, const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y)
    {
        switch (static_cast<PixelStorageMode>(psm & 0x3Fu))
        {
        case C32:
            return PixelStorageTraits<C32>::ReadThroughCache(cache, data, bp, bw, x, y);
        case C24:
            return PixelStorageTraits<C24>::ReadThroughCache(cache, data, bp, bw, x, y);
        case C16:
            return PixelStorageTraits<C16>::ReadThroughCache(cache, data, bp, bw, x, y);
        case C16S:
            return PixelStorageTraits<C16S>::ReadThroughCache(cache, data, bp, bw, x, y);
        case P8:
            return PixelStorageTraits<P8>::ReadThroughCache(cache, data, bp, bw, x, y);
        case P4:
            return PixelStorageTraits<P4>::ReadThroughCache(cache, data, bp, bw, x, y);
        case P8H:
            return PixelStorageTraits<P8H>::ReadThroughCache(cache, data, bp, bw, x, y);
        case P4HL:
            return PixelStorageTraits<P4HL>::ReadThroughCache(cache, data, bp, bw, x, y);
        case P4HH:
            return PixelStorageTraits<P4HH>::ReadThroughCache(cache, data, bp, bw, x, y);
        case Z32:
            return PixelStorageTraits<Z32>::ReadThroughCache(cache, data, bp, bw, x, y);
        case Z24:
            return PixelStorageTraits<Z24>::ReadThroughCache(cache, data, bp, bw, x, y);
        case Z16:
            return PixelStorageTraits<Z16>::ReadThroughCache(cache, data, bp, bw, x, y);
        case Z16S:
            return PixelStorageTraits<Z16S>::ReadThroughCache(cache, data, bp, bw, x, y);
        default:
            return 0u;
        }
    }

    void WriteCT32(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<C32>::Write(data, bp, bw, x, y, value);
    }

    void WriteCT24(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<C24>::Write(data, bp, bw, x, y, value);
    }

    void WriteZ32(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<Z32>::Write(data, bp, bw, x, y, value);
    }

    void WriteZ24(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<Z24>::Write(data, bp, bw, x, y, value);
    }

    void WriteCT16(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<C16>::Write(data, bp, bw, x, y, value);
    }

    void WriteCT16S(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<C16S>::Write(data, bp, bw, x, y, value);
    }

    void WriteZ16(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<Z16>::Write(data, bp, bw, x, y, value);
    }

    void WriteZ16S(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<Z16S>::Write(data, bp, bw, x, y, value);
    }

    void WriteP8(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<P8>::Write(data, bp, bw, x, y, value);
    }

    void WriteP8H(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<P8H>::Write(data, bp, bw, x, y, value);
    }

    void WriteP4(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<P4>::Write(data, bp, bw, x, y, value);
    }

    void WriteP4HL(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<P4HL>::Write(data, bp, bw, x, y, value);
    }

    void WriteP4HH(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
        PixelStorageTraits<P4HH>::Write(data, bp, bw, x, y, value);
    }

    void WriteNull(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value)
    {
    }

    // Row runs for the GL backend's texture store (God of War port): same addresses as the per-pixel Write/Read.
    namespace
    {
        template<PixelStorageMode psm, typename Fetch>
        void writeRunT(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 count, Fetch fetch)
        {
            using Packed = typename PixelStorageTraits<psm>::PackedT;
            for (u32 i = 0; i < count; ++i)
                PixelStorageTraits<psm>::Write(data, bp, bw, x + i, y, static_cast<Packed>(fetch(i)));
        }

        // Page lookup table instead of the per-pixel swizzle formula (SwizzleMatchesLookup ties the two together):
        // the GL backend decodes whole index textures with this, its hottest path in Tekken's menus.
        template<PixelStorageMode psm, typename Table>
        void readIndexRunT(const Table& table, const u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 count, u8* out)
        {
            using Traits = PixelStorageTraits<psm>;
            constexpr u32 width = static_cast<u32>(UnpackedBitWidth(psm));
            constexpr u32 offset = static_cast<u32>(Traits::BitOffset());
            constexpr u32 mask = (psm == P4 || psm == P4HL || psm == P4HH) ? 0x0Fu : 0xFFu;
            for (u32 i = 0; i < count; ++i)
            {
                const u32 bits = Traits::Address(table, bp, bw, x + i, y) * width + offset;
                out[i] = static_cast<u8>((data[(bits >> 3) & (MEMORY_SIZE - 1u)] >> (bits & 7u)) & mask);
            }
        }

        template<PixelStorageMode psm>
        void readColorRunT(const u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 count, u32* out)
        {
            for (u32 i = 0; i < count; ++i)
                out[i] = static_cast<u32>(PixelStorageTraits<psm>::Read(data, bp, bw, x + i, y));
        }
    }

    void ReadIndexRun(const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, u32 count, u8* out)
    {
        switch (static_cast<PixelStorageMode>(psm & 0x3Fu))
        {
        case P8: readIndexRunT<P8>(PageTableP8, data, bp, bw, x, y, count, out); break;
        case P8H: readIndexRunT<P8H>(PageTableC32, data, bp, bw, x, y, count, out); break;
        case P4: readIndexRunT<P4>(PageTableP4, data, bp, bw, x, y, count, out); break;
        case P4HL: readIndexRunT<P4HL>(PageTableC32, data, bp, bw, x, y, count, out); break;
        case P4HH: readIndexRunT<P4HH>(PageTableC32, data, bp, bw, x, y, count, out); break;
        default: std::memset(out, 0, count); break;
        }
    }

    bool ReadColorRun(const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, u32 count, u32* out)
    {
        switch (static_cast<PixelStorageMode>(psm & 0x3Fu))
        {
        case C32: readColorRunT<C32>(data, bp, bw, x, y, count, out); return true;
        case Z32: readColorRunT<Z32>(data, bp, bw, x, y, count, out); return true;
        case C24: readColorRunT<C24>(data, bp, bw, x, y, count, out); return true;
        case Z24: readColorRunT<Z24>(data, bp, bw, x, y, count, out); return true;
        case C16: readColorRunT<C16>(data, bp, bw, x, y, count, out); return true;
        case C16S: readColorRunT<C16S>(data, bp, bw, x, y, count, out); return true;
        case Z16: readColorRunT<Z16>(data, bp, bw, x, y, count, out); return true;
        case Z16S: readColorRunT<Z16S>(data, bp, bw, x, y, count, out); return true;
        default: return false;
        }
    }

    void WriteRun(u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, const u8* src, u32 srcPixel, u32 count)
    {
        auto f32 = [src, srcPixel](u32 i) { u32 v; std::memcpy(&v, src + (srcPixel + i) * 4u, 4u); return v; };
        auto f24 = [src, srcPixel](u32 i) { const u8* p = src + (srcPixel + i) * 3u; return u32(p[0]) | (u32(p[1]) << 8) | (u32(p[2]) << 16); };
        auto f16 = [src, srcPixel](u32 i) { u16 v; std::memcpy(&v, src + (srcPixel + i) * 2u, 2u); return u32(v); };
        auto f8 = [src, srcPixel](u32 i) { return u32(src[srcPixel + i]); };
        auto f4 = [src, srcPixel](u32 i) { const u32 n = srcPixel + i; return u32(src[n >> 1] >> ((n & 1u) * 4u)) & 0x0Fu; };
        switch (static_cast<PixelStorageMode>(psm & 0x3Fu))
        {
        case C32: writeRunT<C32>(data, bp, bw, x, y, count, f32); break;
        case Z32: writeRunT<Z32>(data, bp, bw, x, y, count, f32); break;
        case C24: writeRunT<C24>(data, bp, bw, x, y, count, f24); break;
        case Z24: writeRunT<Z24>(data, bp, bw, x, y, count, f24); break;
        case C16: writeRunT<C16>(data, bp, bw, x, y, count, f16); break;
        case C16S: writeRunT<C16S>(data, bp, bw, x, y, count, f16); break;
        case Z16: writeRunT<Z16>(data, bp, bw, x, y, count, f16); break;
        case Z16S: writeRunT<Z16S>(data, bp, bw, x, y, count, f16); break;
        case P8: writeRunT<P8>(data, bp, bw, x, y, count, f8); break;
        case P8H: writeRunT<P8H>(data, bp, bw, x, y, count, f8); break;
        case P4: writeRunT<P4>(data, bp, bw, x, y, count, f4); break;
        case P4HL: writeRunT<P4HL>(data, bp, bw, x, y, count, f4); break;
        case P4HH: writeRunT<P4HH>(data, bp, bw, x, y, count, f4); break;
        default: break;
        }
    }

    template<PixelStorageMode PSM>
    ImageTransferWrite writeImageTransfer(const typename PixelStorageTraits<PSM>::PageLookupTableT& table, u8* data,
                                          u32 bp, u32 bw, u32 x, u32 y, u32 xOrigin, u32 rowWidth, const u8* src,
                                          u32 srcBytes, u32 pixelBudget, u32 bytesPerPixel, bool pairNibbles)
    {
        (void)table;
        ImageTransferWrite wrote{};
        if (!data || !src || rowWidth == 0u || pixelBudget == 0u)
            return wrote;
        u32 col = x >= xOrigin ? x - xOrigin : 0u;
        if (col >= rowWidth)
        {
            col = 0u;
            x = xOrigin;
        }
        auto advance = [&]() {
            ++wrote.pixels;
            if (++col == rowWidth)
            {
                col = 0u;
                ++y;
                x = xOrigin;
            }
            else
            {
                ++x;
            }
        };
        if (pairNibbles)
        {
            (void)bytesPerPixel;
            while (wrote.pixels < pixelBudget && wrote.bytes < srcBytes)
            {
                const u8 packed = src[wrote.bytes++];
                PixelStorageTraits<PSM>::Write(data, bp, bw, x, y, static_cast<u8>(packed & 0x0Fu));
                advance();
                if (wrote.pixels >= pixelBudget)
                    break;
                PixelStorageTraits<PSM>::Write(data, bp, bw, x, y, static_cast<u8>((packed >> 4) & 0x0Fu));
                advance();
            }
            return wrote;
        }
        while (wrote.pixels < pixelBudget && srcBytes - wrote.bytes >= bytesPerPixel)
        {
            u32 value = 0u;
            if (bytesPerPixel == 3u)
            {
                value = static_cast<u32>(src[wrote.bytes]) | (static_cast<u32>(src[wrote.bytes + 1u]) << 8u) |
                        (static_cast<u32>(src[wrote.bytes + 2u]) << 16u);
            }
            else
            {
                std::memcpy(&value, src + wrote.bytes, bytesPerPixel);
            }
            wrote.bytes += bytesPerPixel;
            PixelStorageTraits<PSM>::Write(data, bp, bw, x, y,
                                           static_cast<typename PixelStorageTraits<PSM>::PackedT>(value));
            advance();
        }
        return wrote;
    }

    ImageTransferWrite WriteImageTransfer(u8* data, u32 bp, u32 bw, u8 psm, u32 x, u32 y, u32 xOrigin, u32 rowWidth,
                                          const u8* src, u32 srcBytes, u32 pixelBudget)
    {
        switch (static_cast<PixelStorageMode>(psm))
        {
        case C32:
            return writeImageTransfer<C32>(PageTableC32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 4u, false);
        case Z32:
            return writeImageTransfer<Z32>(PageTableZ32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 4u, false);
        case C24:
            return writeImageTransfer<C24>(PageTableC32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 3u, false);
        case Z24:
            return writeImageTransfer<Z24>(PageTableZ32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 3u, false);
        case C16:
            return writeImageTransfer<C16>(PageTableC16, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 2u, false);
        case C16S:
            return writeImageTransfer<C16S>(PageTableC16S, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                            pixelBudget, 2u, false);
        case Z16:
            return writeImageTransfer<Z16>(PageTableZ16, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 2u, false);
        case Z16S:
            return writeImageTransfer<Z16S>(PageTableZ16S, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                            pixelBudget, 2u, false);
        case P8:
            return writeImageTransfer<P8>(PageTableP8, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                          pixelBudget, 1u, false);
        case P8H:
            return writeImageTransfer<P8H>(PageTableC32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                           pixelBudget, 1u, false);
        case P4:
            return writeImageTransfer<P4>(PageTableP4, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                          pixelBudget, 1u, true);
        case P4HL:
            return writeImageTransfer<P4HL>(PageTableC32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                            pixelBudget, 1u, true);
        case P4HH:
            return writeImageTransfer<P4HH>(PageTableC32, data, bp, bw, x, y, xOrigin, rowWidth, src, srcBytes,
                                            pixelBudget, 1u, true);
        default:
            return {};
        }
    }

    u32 ReadCT32(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<C32>::Read(data, bp, bw, x, y);
    }

    u32 ReadZ32(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<Z32>::Read(data, bp, bw, x, y);
    }

    u32 ReadCT24(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<C24>::Read(data, bp, bw, x, y);
    }

    u32 ReadZ24(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<Z24>::Read(data, bp, bw, x, y);
    }

    u32 ReadCT16(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<C16>::Read(data, bp, bw, x, y);
    }

    u32 ReadCT16S(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<C16S>::Read(data, bp, bw, x, y);
    }

    u32 ReadZ16(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<Z16>::Read(data, bp, bw, x, y);
    }

    u32 ReadZ16S(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<Z16S>::Read(data, bp, bw, x, y);
    }

    u32 ReadP8(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<P8>::Read(data, bp, bw, x, y);
    }

    u32 ReadP8H(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<P8H>::Read(data, bp, bw, x, y);
    }

    u32 ReadP4(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<P4>::Read(data, bp, bw, x, y);
    }

    u32 ReadP4HL(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<P4HL>::Read(data, bp, bw, x, y);
    }

    u32 ReadP4HH(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return PixelStorageTraits<P4HH>::Read(data, bp, bw, x, y);
    }

    u32 ReadNull(u8* data, u32 bp, u32 bw, u32 x, u32 y)
    {
        return 0;
    }
}


