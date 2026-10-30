#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <array>
#include <span>

#include "types.h"
#include "runtime/gs/gs_texture_page_cache.h"

namespace GSMem
{
	constexpr usz MEMORY_SIZE = 4_mb;
	constexpr usz GS_PAGE_SIZE = TexturePageCache::kPageSize;

	// these are all the same regardless of storage mode
	constexpr usz BLOCKS_PER_PAGE = 32;
	constexpr usz COLUMNS_PER_BLOCK = 4;

	// note:
	// there are others (FFX uses 0x11 as an alias for CT24)
	// these are just the official ones
	enum class PixelStorageMode
	{
		C32 = 0x00, // 32 bit color
		C24 = 0x01, // 24 bit color
		C16 = 0x02, // 16 bit color
		C16S = 0x0A, // 16 bit color (32 bit compatible memory layout)
		P8 = 0x13, // 8 bit index
		P4 = 0x14, // 4 bit index
		P8H = 0x1B, // 8 bit index (stored in upper 8 bits of 32 bit)
		P4HL = 0x24, // 4 bit index (stored in low nibble of upper 8 bits of 32 bit)
		P4HH = 0x2C, // 4 bit index (stored in high nibble of upper 8 bits of 32 bit)
		Z32 = 0x30, // 32 bit depth
		Z24 = 0x31, // 24 bit depth
		Z16 = 0x32, // 16 bit depth
		Z16S = 0x3A, // 16 bit depth (32 bit compatible memory layout)
		Max = 0x3F // 6 bits of values
	};


	inline constexpr PixelStorageMode C32 = PixelStorageMode::C32;
	inline constexpr PixelStorageMode C24 = PixelStorageMode::C24;
	inline constexpr PixelStorageMode C16 = PixelStorageMode::C16;
	inline constexpr PixelStorageMode C16S = PixelStorageMode::C16S;
	inline constexpr PixelStorageMode P8 = PixelStorageMode::P8;
	inline constexpr PixelStorageMode P4 = PixelStorageMode::P4;
	inline constexpr PixelStorageMode P8H = PixelStorageMode::P8H;
	inline constexpr PixelStorageMode P4HL = PixelStorageMode::P4HL;
	inline constexpr PixelStorageMode P4HH = PixelStorageMode::P4HH;
	inline constexpr PixelStorageMode Z32 = PixelStorageMode::Z32;
	inline constexpr PixelStorageMode Z24 = PixelStorageMode::Z24;
	inline constexpr PixelStorageMode Z16 = PixelStorageMode::Z16;
	inline constexpr PixelStorageMode Z16S = PixelStorageMode::Z16S;

	struct Extent2D
	{
		u32 x{ 0 };
		u32 y{ 0 };
	};

	template<typename T, usz Width, usz Height>
	using LookupTable = std::array<std::array<T, Width>, Height>;

	constexpr bool IsValidPsm(PixelStorageMode psm)
	{
		switch (psm)
		{
		case C32:
		case Z32:
		case C24:
		case Z24:
		case C16:
		case C16S:
		case Z16:
		case Z16S:
		case P8:
		case P8H:
		case P4:
		case P4HH:
		case P4HL:
			return true;
		}

		return false;
	}

	// Bits per pixel in unpacked format (GS memory)
	constexpr usz UnpackedBitWidth(PixelStorageMode psm)
	{
		switch (psm)
		{
		case C32:
		case Z32:
		case C24:
		case Z24:
		case P8H:  // unpacked to alpha of 32
		case P4HH: // unpacked to alpha of 32
		case P4HL: // unpacked to alpha of 32
			return 32;
		case C16:
		case C16S:
		case Z16:
		case Z16S:
			return 16;
		case P8:
			return 8;
		case P4:
			return 4;
		}

		return 32;
	}

	// bits per pixel (packed width)
	constexpr usz BitsPerPixel(PixelStorageMode psm)
	{
		switch (psm)
		{
		case C32:
		case Z32:
			return 32;
		case C24:
		case Z24:
			return 24;
		case C16:
		case C16S:
		case Z16:
		case Z16S:
			return 16;
		case P8:
		case P8H:
			return 8;
		case P4:
		case P4HH:
		case P4HL:
			return 4;
		default:
			break;
		}

		return 32;
	}

	constexpr bool IsDepth(PixelStorageMode psm)
	{
		switch (psm)
		{
		case Z16:
		case Z16S:
		case Z24:
		case Z32:
			return true;
		default:
			break;
		}

		return false;
	}

	constexpr bool IsColor(PixelStorageMode psm)
	{
		return !IsDepth(psm);
	}

	constexpr bool IsPaletted(PixelStorageMode psm)
	{
		switch (psm)
		{
		case P8:
		case P8H:
		case P4:
		case P4HL:
		case P4HH:
			return true;
		}

		return false;
	}

	template<usz BitCount>
	struct BitStorageHelper;

	template<>
	struct BitStorageHelper<4>
	{
		using Type = u8;
	};

	template<>
	struct BitStorageHelper<8>
	{
		using Type = u8;
	};

	template<>
	struct BitStorageHelper<16>
	{
		using Type = u16;
	};

	template<>
	struct BitStorageHelper<24>
	{
		using Type = u32;
	};

	template<>
	struct BitStorageHelper<32>
	{
		using Type = u32;
	};

	template<PixelStorageMode psm>
	struct PixelStorageTraits
	{
		// page extent, in units of pixels
		static constexpr Extent2D PageExtent();

		// block size in a page, in units of blocks
		static constexpr Extent2D BlockExtent();

		// column extent in a block, in units of pixels
		static constexpr Extent2D ColumnExtent();

		// block count in a page (always 32 for valid psm)
		static constexpr usz BlocksPerPage();

		// pixel count in a block
		static constexpr usz PixelsPerBlock();

		// pixel count in a page
		static constexpr usz PixelsPerPage();

		using BlockLookupTableT = LookupTable<u8, BlockExtent().x, BlockExtent().y>;
		using ColumnLookupTableT = LookupTable<u16, ColumnExtent().x, ColumnExtent().y>;
		using PageLookupTableT = std::array<LookupTable<u16, PageExtent().x, PageExtent().y>, BlocksPerPage()>;

		// calculates the column offset
		static constexpr usz ColumnId(const ColumnLookupTableT& table, u32 x, u32 y);

		// calculates the block offset
		static constexpr usz BlockId(const BlockLookupTableT& table, u32 x, u32 y);

		// calculates the page offset
		static constexpr usz PageId(u32 base, u32 bw, u32 x, u32 y);

		// sets up the lookup table which precomputes a bunch of math for us
		static constexpr void InitPageLookupTable(PageLookupTableT& table, const BlockLookupTableT& block_table, const ColumnLookupTableT& column_table);

		// gets a pixel address
		static constexpr u32 Address(const PageLookupTableT& table, u32 block, u32 bw, u32 x, u32 y);

		using PackedT = BitStorageHelper<BitsPerPixel(psm)>::Type;
		using UnapckedT = BitStorageHelper<UnpackedBitWidth(psm)>::Type;

		// returns a offset into a 32 bit word (P8H, P4HH, P4HL)
		static constexpr usz BitOffset();

		// writes the pixel. Address comes from SwizzlePixel (the page LUT is the same formula, checked in tests).
		static constexpr void Write(u8* data, u32 block, u32 bw, u32 x, u32 y, PackedT value);

		// Frame and Z reads. Texture samples use ReadThroughCache so a page copied before TEXFLUSH
		// stays visible until that flush (GS texture page buffer).
		static constexpr auto Read(const u8* data, u32 block, u32 bw, u32 x, u32 y) -> PackedT;
		static auto ReadThroughCache(TexturePageCache& cache, const u8* data, u32 block, u32 bw, u32 x, u32 y) -> PackedT;

		static_assert(BlocksPerPage() == BLOCKS_PER_PAGE);
		static_assert(IsValidPsm(psm));
	};

	template<PixelStorageMode psm>
	constexpr Extent2D PixelStorageTraits<psm>::PageExtent()
	{
		switch (psm)
		{
		case C32:
		case Z32:
		case C24:
		case Z24:
		case P8H:
		case P4HL:
		case P4HH:
			return { 64, 32 };
		case C16:
		case C16S:
		case Z16:
		case Z16S:
			return { 64, 64 };
		case P8:
			return { 128, 64 };
		case P4:
			return { 128, 128 };
		}
	}

	template<PixelStorageMode psm>
	constexpr Extent2D PixelStorageTraits<psm>::BlockExtent()
	{
		switch (psm)
		{
		case C32:
		case C24:
		case P8H:
		case P4HL:
		case P4HH:
		case Z32:
		case Z24:
		case P8:
			return { 8, 4 };
		case C16:
		case C16S:
		case Z16:
		case Z16S:
		case P4:
			return { 4, 8 };
		default:
			break;
		}

		return { 0, 0 };
	}

	template<PixelStorageMode psm>
	constexpr Extent2D PixelStorageTraits<psm>::ColumnExtent()
	{
		switch (psm)
		{
		case C32:
		case Z32:
		case C24:
		case P8H:
		case P4HL:
		case P4HH:
			return { 8, 8 };
		case C16:
		case C16S:
		case Z16:
		case Z16S:
			return { 16, 8 };
		case P8:
			return { 16, 16 };
		case P4:
			return { 32, 16 };
		}

		return { 0, 0 };
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::BlocksPerPage()
	{
		const auto extent = BlockExtent();

		return static_cast<usz>(extent.x) * static_cast<usz>(extent.y);
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::PixelsPerBlock()
	{
		constexpr auto extent = ColumnExtent();

		return static_cast<usz>(extent.x) * extent.y;
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::PixelsPerPage()
	{
		constexpr auto extent = PageExtent();

		return static_cast<usz>(extent.x) * extent.y;
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::ColumnId(const ColumnLookupTableT& table, u32 x, u32 y)
	{
		constexpr auto extent = ColumnExtent();

		// wrap along the column extent
		y = y % extent.y;
		x = x % extent.x;

		return table[y][x];
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::BlockId(const BlockLookupTableT& table, u32 x, u32 y)
	{
		constexpr auto block_extent = BlockExtent();
		constexpr auto column_extent = ColumnExtent();

		y = (y / column_extent.y) % block_extent.y;
		x = (x / column_extent.x) % block_extent.x;

		return table[y][x];
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::PageId(u32 base, u32 bw, u32 x, u32 y)
	{
		constexpr auto page_extent = PageExtent();

		const u32 base_page = base / BlocksPerPage();
		const u32 row = (y / page_extent.y) * ((bw * 64) / page_extent.x); // T8 and T4 need a / 2 of bw
		const u32 col = x / page_extent.x;

		return static_cast<usz>(base_page) + row + col;
	}

	template<PixelStorageMode psm>
	constexpr void PixelStorageTraits<psm>::InitPageLookupTable(PageLookupTableT& table, const BlockLookupTableT& block_table, const ColumnLookupTableT& column_table)
	{
		constexpr auto page_extent = PageExtent();
		constexpr auto pixels_per_block = PixelsPerBlock();
		constexpr auto blocks_per_page = BlocksPerPage();

		for (usz block = 0; block < blocks_per_page; ++block)
		{
			for (usz y = 0; y < page_extent.y; ++y)
			{
				for (usz x = 0; x < page_extent.x; ++x)
				{
					table[block][y][x] = ((block + BlockId(block_table, x, y)) * PixelsPerBlock()) + ColumnId(column_table, x, y);
				}
			}
		}
	}

	template<PixelStorageMode psm>
	constexpr u32 PixelStorageTraits<psm>::Address(const PageLookupTableT& table, u32 block, u32 bw, u32 x, u32 y)
	{
		constexpr auto block_count = BlocksPerPage();
		constexpr auto page_extent = PageExtent();
		constexpr auto pixels_per_page = PixelsPerPage();

		const u32 page = PageId(block, bw, x, y);

		block = block % block_count;
		y = y % page_extent.y;
		x = x % page_extent.x;

		return (page * PixelsPerPage()) + table[block][y][x];
	}

	template<PixelStorageMode psm>
	constexpr usz PixelStorageTraits<psm>::BitOffset()
	{
		switch (psm)
		{
		case P8H:
		case P4HL:
			return 24;
		case P4HH:
			return 28;
		default:
			break;
		}

		return 0;
	}

    inline constexpr PixelStorageTraits<C32>::BlockLookupTableT BlockTableC32
    {{
        {  0,  1,  4,  5, 16, 17, 20, 21 },
        {  2,  3,  6,  7, 18, 19, 22, 23 },
        {  8,  9, 12, 13, 24, 25, 28, 29 },
        { 10, 11, 14, 15, 26, 27, 30, 31 }
    }};

    inline constexpr PixelStorageTraits<Z32>::BlockLookupTableT BlockTableZ32
    {{
        { 24, 25, 28, 29,  8,  9, 12, 13 },
        { 26, 27, 30, 31, 10, 11, 14, 15 },
        { 16, 17, 20, 21,  0,  1,  4,  5 },
        { 18, 19, 22, 23,  2,  3,  6,  7 }
    }};

    inline constexpr PixelStorageTraits<C16>::BlockLookupTableT BlockTableC16
    {{
        {  0,  2,  8, 10 },
        {  1,  3,  9, 11 },
        {  4,  6, 12, 14 },
        {  5,  7, 13, 15 },
        { 16, 18, 24, 26 },
        { 17, 19, 25, 27 },
        { 20, 22, 28, 30 },
        { 21, 23, 29, 31 }
    }};

    inline constexpr PixelStorageTraits<C16S>::BlockLookupTableT BlockTableC16S
    {{
        {  0,  2, 16, 18 },
        {  1,  3, 17, 19 },
        {  8, 10, 24, 26 },
        {  9, 11, 25, 27 },
        {  4,  6, 20, 22 },
        {  5,  7, 21, 23 },
        { 12, 14, 28, 30 },
        { 13, 15, 29, 31 }
    }};

    inline constexpr PixelStorageTraits<Z16>::BlockLookupTableT BlockTableZ16
    {{
        { 24, 26, 16, 18 },
        { 25, 27, 17, 19 },
        { 28, 30, 20, 22 },
        { 29, 31, 21, 23 },
        {  8, 10,  0,  2 },
        {  9, 11,  1,  3 },
        { 12, 14,  4,  6 },
        { 13, 15,  5,  7 }
    }};

    inline constexpr PixelStorageTraits<Z16S>::BlockLookupTableT BlockTableZ16S
    {{
        { 24, 26,  8, 10 },
        { 25, 27,  9, 11 },
        { 16, 18,  0,  2 },
        { 17, 19,  1,  3 },
        { 28, 30, 12, 14 },
        { 29, 31, 13, 15 },
        { 20, 22,  4,  6 },
        { 21, 23,  5,  7 }
    }};

    inline constexpr PixelStorageTraits<P8>::BlockLookupTableT BlockTableP8
    {{
        {  0,  1,  4,  5, 16, 17, 20, 21},
        {  2,  3,  6,  7, 18, 19, 22, 23},
        {  8,  9, 12, 13, 24, 25, 28, 29},
        { 10, 11, 14, 15, 26, 27, 30, 31}
    }};

    inline constexpr PixelStorageTraits<P4>::BlockLookupTableT BlockTableP4
    {{
        {  0,  2,  8, 10 },
        {  1,  3,  9, 11 },
        {  4,  6, 12, 14 },
        {  5,  7, 13, 15 },
        { 16, 18, 24, 26 },
        { 17, 19, 25, 27 },
        { 20, 22, 28, 30 },
        { 21, 23, 29, 31 }
    }};

    // column layout is shared between c32 and z32
    inline constexpr PixelStorageTraits<C32>::ColumnLookupTableT ColumnTable32
    {{
        {  0,  1,  4,  5,  8,  9, 12, 13 },
        {  2,  3,  6,  7, 10, 11, 14, 15 },
        { 16, 17, 20, 21, 24, 25, 28, 29 },
        { 18, 19, 22, 23, 26, 27, 30, 31 },
        { 32, 33, 36, 37, 40, 41, 44, 45 },
        { 34, 35, 38, 39, 42, 43, 46, 47 },
        { 48, 49, 52, 53, 56, 57, 60, 61 },
        { 50, 51, 54, 55, 58, 59, 62, 63 },
    }};

    inline constexpr PixelStorageTraits<C16>::ColumnLookupTableT ColumnTable16
    {{
        {   0,   2,   8,  10,  16,  18,  24,  26,   1,   3,   9,  11,  17,  19,  25,  27 },
        {   4,   6,  12,  14,  20,  22,  28,  30,   5,   7,  13,  15,  21,  23,  29,  31 },
        {  32,  34,  40,  42,  48,  50,  56,  58,  33,  35,  41,  43,  49,  51,  57,  59 },
        {  36,  38,  44,  46,  52,  54,  60,  62,  37,  39,  45,  47,  53,  55,  61,  63 },
        {  64,  66,  72,  74,  80,  82,  88,  90,  65,  67,  73,  75,  81,  83,  89,  91 },
        {  68,  70,  76,  78,  84,  86,  92,  94,  69,  71,  77,  79,  85,  87,  93,  95 },
        {  96,  98, 104, 106, 112, 114, 120, 122,  97,  99, 105, 107, 113, 115, 121, 123 },
        { 100, 102, 108, 110, 116, 118, 124, 126, 101, 103, 109, 111, 117, 119, 125, 127 }
    }};

    inline constexpr PixelStorageTraits<P8>::ColumnLookupTableT ColumnTable8
    {{
        {   0,   4,  16,  20,  32,  36,  48,  52,   2,   6,  18,  22,  34,  38,  50,  54 },
        {   8,  12,  24,  28,  40,  44,  56,  60,  10,  14,  26,  30,  42,  46,  58,  62 },
        {  33,  37,  49,  53,   1,   5,  17,  21,  35,  39,  51,  55,   3,   7,  19,  23 },
        {  41,  45,  57,  61,   9,  13,  25,  29,  43,  47,  59,  63,  11,  15,  27,  31 },
        {  96, 100, 112, 116,  64,  68,  80,  84,  98, 102, 114, 118,  66,  70,  82,  86 },
        { 104, 108, 120, 124,  72,  76,  88,  92, 106, 110, 122, 126,  74,  78,  90,  94 },
        {  65,  69,  81,  85,  97, 101, 113, 117,  67,  71,  83,  87,  99, 103, 115, 119 },
        {  73,  77,  89,  93, 105, 109, 121, 125,  75,  79,  91,  95, 107, 111, 123, 127 },
        { 128, 132, 144, 148, 160, 164, 176, 180, 130, 134, 146, 150, 162, 166, 178, 182 },
        { 136, 140, 152, 156, 168, 172, 184, 188, 138, 142, 154, 158, 170, 174, 186, 190 },
        { 161, 165, 177, 181, 129, 133, 145, 149, 163, 167, 179, 183, 131, 135, 147, 151 },
        { 169, 173, 185, 189, 137, 141, 153, 157, 171, 175, 187, 191, 139, 143, 155, 159 },
        { 224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214 },
        { 232, 236, 248, 252, 200, 204, 216, 220, 234, 238, 250, 254, 202, 206, 218, 222 },
        { 193, 197, 209, 213, 225, 229, 241, 245, 195, 199, 211, 215, 227, 231, 243, 247 },
        { 201, 205, 217, 221, 233, 237, 249, 253, 203, 207, 219, 223, 235, 239, 251, 255 }
    }};

    inline constexpr PixelStorageTraits<P4>::ColumnLookupTableT ColumnTable4
    {{
        {   0,   8,  32,  40,  64,  72,  96, 104,   2,  10,  34,  42,  66,  74,  98, 106,   4,  12,  36,  44,  68,  76, 100, 108,   6,  14,  38,  46,  70,  78, 102, 110 },
        {  16,  24,  48,  56,  80,  88, 112, 120,  18,  26,  50,  58,  82,  90, 114, 122,  20,  28,  52,  60,  84,  92, 116, 124,  22,  30,  54,  62,  86,  94, 118, 126 },
        {  65,  73,  97, 105,   1,   9,  33,  41,  67,  75,  99, 107,   3,  11,  35,  43,  69,  77, 101, 109,   5,  13,  37,  45,  71,  79, 103, 111,   7,  15,  39,  47 },
        {  81,  89, 113, 121,  17,  25,  49,  57,  83,  91, 115, 123,  19,  27,  51,  59,  85,  93, 117, 125,  21,  29,  53,  61,  87,  95, 119, 127,  23,  31,  55,  63 },
        { 192, 200, 224, 232, 128, 136, 160, 168, 194, 202, 226, 234, 130, 138, 162, 170, 196, 204, 228, 236, 132, 140, 164, 172, 198, 206, 230, 238, 134, 142, 166, 174 },
        { 208, 216, 240, 248, 144, 152, 176, 184, 210, 218, 242, 250, 146, 154, 178, 186, 212, 220, 244, 252, 148, 156, 180, 188, 214, 222, 246, 254, 150, 158, 182, 190 },
        { 129, 137, 161, 169, 193, 201, 225, 233, 131, 139, 163, 171, 195, 203, 227, 235, 133, 141, 165, 173, 197, 205, 229, 237, 135, 143, 167, 175, 199, 207, 231, 239 },
        { 145, 153, 177, 185, 209, 217, 241, 249, 147, 155, 179, 187, 211, 219, 243, 251, 149, 157, 181, 189, 213, 221, 245, 253, 151, 159, 183, 191, 215, 223, 247, 255 },
        { 256, 264, 288, 296, 320, 328, 352, 360, 258, 266, 290, 298, 322, 330, 354, 362, 260, 268, 292, 300, 324, 332, 356, 364, 262, 270, 294, 302, 326, 334, 358, 366 },
        { 272, 280, 304, 312, 336, 344, 368, 376, 274, 282, 306, 314, 338, 346, 370, 378, 276, 284, 308, 316, 340, 348, 372, 380, 278, 286, 310, 318, 342, 350, 374, 382 },
        { 321, 329, 353, 361, 257, 265, 289, 297, 323, 331, 355, 363, 259, 267, 291, 299, 325, 333, 357, 365, 261, 269, 293, 301, 327, 335, 359, 367, 263, 271, 295, 303 },
        { 337, 345, 369, 377, 273, 281, 305, 313, 339, 347, 371, 379, 275, 283, 307, 315, 341, 349, 373, 381, 277, 285, 309, 317, 343, 351, 375, 383, 279, 287, 311, 319 },
        { 448, 456, 480, 488, 384, 392, 416, 424, 450, 458, 482, 490, 386, 394, 418, 426, 452, 460, 484, 492, 388, 396, 420, 428, 454, 462, 486, 494, 390, 398, 422, 430 },
        { 464, 472, 496, 504, 400, 408, 432, 440, 466, 474, 498, 506, 402, 410, 434, 442, 468, 476, 500, 508, 404, 412, 436, 444, 470, 478, 502, 510, 406, 414, 438, 446 },
        { 385, 393, 417, 425, 449, 457, 481, 489, 387, 395, 419, 427, 451, 459, 483, 491, 389, 397, 421, 429, 453, 461, 485, 493, 391, 399, 423, 431, 455, 463, 487, 495 },
        { 401, 409, 433, 441, 465, 473, 497, 505, 403, 411, 435, 443, 467, 475, 499, 507, 405, 413, 437, 445, 469, 477, 501, 509, 407, 415, 439, 447, 471, 479, 503, 511 },
    }};


	// Pixel index inside the swizzle. Same value as Address() on the page LUT (the LUT is that formula
	// expanded). Page, block and column sizes are powers of two, so the index is shifts into the small
	// block and column tables instead of a load from a table the size of a page.
	template<u32 pageWShift, u32 pageHShift, u32 colWShift, u32 colHShift, u32 blockWMask, u32 blockHMask, typename BlockT, typename ColumnT>
	constexpr u32 swizzleBoxes(const BlockT& blocks, const ColumnT& columns, u32 bp, u32 bw, u32 x, u32 y)
	{
		constexpr u32 pageW = 1u << pageWShift;
		constexpr u32 pageH = 1u << pageHShift;
		constexpr u32 colW = 1u << colWShift;
		constexpr u32 colH = 1u << colHShift;
		constexpr u32 pixBlock = colW * colH;
		constexpr u32 pixPage = pageW * pageH;
		const u32 pageMul = bw >> (pageWShift - 6u);
		const u32 page = (bp >> 5) + ((y >> pageHShift) * pageMul) + (x >> pageWShift);
		const u32 lx = x & (pageW - 1u);
		const u32 ly = y & (pageH - 1u);
		const u32 bx = (lx >> colWShift) & blockWMask;
		const u32 by = (ly >> colHShift) & blockHMask;
		const u32 block = (bp & 31u) + static_cast<u32>(blocks[by][bx]);
		const u32 column = static_cast<u32>(columns[ly & (colH - 1u)][lx & (colW - 1u)]);
		return page * pixPage + block * pixBlock + column;
	}

	template<PixelStorageMode psm>
	constexpr u32 SwizzlePixel(u32 bp, u32 bw, u32 x, u32 y)
	{
		if constexpr (psm == C32 || psm == C24 || psm == P8H || psm == P4HL || psm == P4HH)
			return swizzleBoxes<6, 5, 3, 3, 7, 3>(BlockTableC32, ColumnTable32, bp, bw, x, y);
		else if constexpr (psm == Z32 || psm == Z24)
			return swizzleBoxes<6, 5, 3, 3, 7, 3>(BlockTableZ32, ColumnTable32, bp, bw, x, y);
		else if constexpr (psm == C16)
			return swizzleBoxes<6, 6, 4, 3, 3, 7>(BlockTableC16, ColumnTable16, bp, bw, x, y);
		else if constexpr (psm == C16S)
			return swizzleBoxes<6, 6, 4, 3, 3, 7>(BlockTableC16S, ColumnTable16, bp, bw, x, y);
		else if constexpr (psm == Z16)
			return swizzleBoxes<6, 6, 4, 3, 3, 7>(BlockTableZ16, ColumnTable16, bp, bw, x, y);
		else if constexpr (psm == Z16S)
			return swizzleBoxes<6, 6, 4, 3, 3, 7>(BlockTableZ16S, ColumnTable16, bp, bw, x, y);
		else if constexpr (psm == P8)
			return swizzleBoxes<7, 6, 4, 4, 7, 3>(BlockTableP8, ColumnTable8, bp, bw, x, y);
		else if constexpr (psm == P4)
			return swizzleBoxes<7, 7, 5, 4, 3, 7>(BlockTableP4, ColumnTable4, bp, bw, x, y);
		else
			return 0u;
	}

	template<PixelStorageMode psm>
	constexpr void PixelStorageTraits<psm>::Write(u8* data, u32 block, u32 bw, u32 x, u32 y, PackedT value)
	{
		const u32 pixel_addr = SwizzlePixel<psm>(block, bw, x, y);
		const u32 bits = pixel_addr * UnpackedBitWidth(psm) + BitOffset();
		const u32 byte_addr = (bits / 8) & (MEMORY_SIZE - sizeof(PackedT));
		const u32 shift = bits % 8;

		u8* ptr = &data[byte_addr];

		PackedT old;
		std::memcpy(&old, ptr, sizeof(PackedT));

		switch (psm)
		{
		case C32:
		case Z32:
		case C16:
		case C16S:
		case Z16:
		case Z16S:
		case P8:
		case P8H:
			std::memcpy(ptr, &value, sizeof(PackedT));
			break;
		case C24:
		case Z24:
			// RMW
			value = (old & 0xFF000000) | (value & 0x00FFFFFF);
			std::memcpy(ptr, &value, sizeof(PackedT));
			break;
		case P4:
		case P4HL:
		case P4HH:
			// RMW
			value = (old &~(0x0F << shift)) | ((value & 0x0F) << shift);
			std::memcpy(ptr, &value, sizeof(PackedT));
			break;
		default:
			break;
		}
	}

	template<PixelStorageMode psm>
	constexpr auto PixelStorageTraits<psm>::Read(const u8* data, u32 block, u32 bw, u32 x, u32 y) -> PackedT
	{
		const u32 pixel_addr = SwizzlePixel<psm>(block, bw, x, y);
		const u32 bits = pixel_addr * UnpackedBitWidth(psm) + BitOffset();
		const u32 byte_addr = (bits / 8) & (MEMORY_SIZE - sizeof(PackedT));
		const u32 shift = bits % 8;

		const u8* source = data + byte_addr;
		PackedT v;
		std::memcpy(&v, source, sizeof(PackedT));

		switch (psm)
		{
		case C32:
		case Z32:
		case C16:
		case C16S:
		case Z16:
		case Z16S:
		case P8:
		case P8H:
			return v;
		case C24:
		case Z24:
			return v & 0x00FFFFFF;
		case P4:
		case P4HL:
		case P4HH:
			return (v >> shift) & 0x0F;
		default:
			break;
		}

		return static_cast<PackedT>(0xFFFF00FFu);
	}

	template<PixelStorageMode psm>
	auto PixelStorageTraits<psm>::ReadThroughCache(TexturePageCache& cache, const u8* data, u32 block, u32 bw, u32 x, u32 y) -> PackedT
	{
		const u32 pixel_addr = SwizzlePixel<psm>(block, bw, x, y);
		const u32 bits = pixel_addr * UnpackedBitWidth(psm) + BitOffset();
		const u32 byte_addr = (bits / 8) & (MEMORY_SIZE - sizeof(PackedT));
		const u32 shift = bits % 8;

		const u8* source = cache.Resolve(data, byte_addr);
		PackedT v;
		std::memcpy(&v, source, sizeof(PackedT));

		switch (psm)
		{
		case C32:
		case Z32:
		case C16:
		case C16S:
		case Z16:
		case Z16S:
		case P8:
		case P8H:
			return v;
		case C24:
		case Z24:
			return v & 0x00FFFFFF;
		case P4:
		case P4HL:
		case P4HH:
			return (v >> shift) & 0x0F;
		default:
			break;
		}

		return static_cast<PackedT>(0xFFFF00FFu);
	}

	void InitLookupTables();
	bool SwizzleMatchesLookup();
	// GL backend: the page lookup table of a format (the CT32 one for P8H/P4HL/P4HH), nullptr for others.
	const u16* PageTableData(PixelStorageMode psm);

    // Shares swizzle, VRAM wrapping, and lane extraction with the direct reads.
    u32 ReadTexture(TexturePageCache& cache, const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y);

	void WriteCT32(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteZ32(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	void WriteCT24(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteZ24(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	void WriteCT16(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteCT16S(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteZ16(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteZ16S(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	void WriteP8(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteP8H(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	void WriteP4(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteP4HH(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);
	void WriteP4HL(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	void WriteNull(u8* data, u32 bp, u32 bw, u32 x, u32 y, u32 value);

	// GL backend row runs (God of War port): `count` pixels of row y from x, same addresses as Write/Read.
	// WriteRun takes pixel srcPixel+i of a packed stream (32/24/16/8 bit LE, 4 bit low nibble first).
	void WriteRun(u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, const u8* src, u32 srcPixel, u32 count);
	// Paletted formats (T8/T8H/T4/T4HL/T4HH) as 8 bit indices.
	void ReadIndexRun(const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, u32 count, u8* out);
	// Direct colour / depth formats as raw values; false for other formats.
	bool ReadColorRun(const u8* data, u32 psm, u32 bp, u32 bw, u32 x, u32 y, u32 count, u32* out);

	// One host->local transfer in raster order (x across rowWidth, then y+1 at xOrigin).
	// Same stored bytes as the per-pixel Write* for that psm. A source byte that holds two PSMT4
	// pixels is consumed whole; the high nibble is stored only when another pixel is still in budget.
	struct ImageTransferWrite
	{
		u32 pixels;
		u32 bytes;
	};
	ImageTransferWrite WriteImageTransfer(u8* data, u32 bp, u32 bw, u8 psm, u32 x, u32 y, u32 xOrigin,
	                                      u32 rowWidth, const u8* src, u32 srcBytes, u32 pixelBudget);

	u32 ReadCT32(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadZ32(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadCT24(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadZ24(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadCT24(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadZ24(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadCT16(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadCT16S(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadZ16(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadZ16S(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadP8(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadP8H(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadP4(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadP4HL(u8* data, u32 bp, u32 bw, u32 x, u32 y);
	u32 ReadP4HH(u8* data, u32 bp, u32 bw, u32 x, u32 y);

	u32 ReadNull(u8* data, u32 bp, u32 bw, u32 x, u32 y);
}