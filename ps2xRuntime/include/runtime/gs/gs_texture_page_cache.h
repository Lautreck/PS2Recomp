#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace GSMem
{
    // Copies one 8 KiB page into dst. Drawing threads use a snapshot when an upload has replaced the
    // live page underneath a primitive that was already queued (see GSCpuBackend). Other callers copy vram.
    void CopyStablePage(uint8_t *dst, const uint8_t *vram, uint32_t pageBase) noexcept;

    class TexturePageCache
    {
    public:
        static constexpr uint32_t kPageSize = 8192u;

        void Invalidate() noexcept
        {
            m_pageBase = UINT32_MAX;
        }

        // byteAddress is the wrapped, swizzled VRAM address. The returned
        // pointer is valid only until the next miss or invalidation.
        const uint8_t* Resolve(const uint8_t* vram, uint32_t byteAddress) noexcept
        {
            const uint32_t pageBase = byteAddress & ~(kPageSize - 1u);
            if (m_pageBase != pageBase)
            {
                CopyStablePage(m_bytes.data(), vram, pageBase);
                m_pageBase = pageBase;
            }
            return m_bytes.data() + (byteAddress & (kPageSize - 1u));
        }

    private:
        alignas(64) std::array<uint8_t, kPageSize> m_bytes{};
        uint32_t m_pageBase = UINT32_MAX;
    };
}
