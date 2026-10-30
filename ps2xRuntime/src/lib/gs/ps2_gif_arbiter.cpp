#include "runtime/gs/ps2_gif_arbiter.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
// TEKKEN4_GT_STATS=1. Counts only; the GS still sees the same bytes.
struct GifCopyStats
{
    bool on = false;
    uint64_t submits = 0;
    uint64_t immediate = 0;
    uint64_t copied = 0;
    uint64_t bytes = 0;

    GifCopyStats()
    {
        const char *env = std::getenv("TEKKEN4_GT_STATS");
        on = env != nullptr && env[0] != '\0' && env[0] != '0';
    }

    void note(bool direct, uint32_t sizeBytes)
    {
        if (!on)
            return;
        ++submits;
        bytes += sizeBytes;
        if (direct)
            ++immediate;
        else
            ++copied;
        if ((submits & 0x3FFFu) == 0u)
            dump("snap");
    }

    void dump(const char *tag) const
    {
        if (!on)
            return;
        std::fprintf(stderr, "[gt-stats] gif %s submits=%llu immediate=%llu copied=%llu bytes=%llu\n", tag,
                     static_cast<unsigned long long>(submits), static_cast<unsigned long long>(immediate),
                     static_cast<unsigned long long>(copied), static_cast<unsigned long long>(bytes));
    }

    ~GifCopyStats() { dump("final"); }
};

GifCopyStats g_gifCopyStats;
} // namespace

GifArbiter::GifArbiter(ProcessPacketFn processFn)
    : m_processFn(std::move(processFn))
{
}

bool GifArbiter::isImagePacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes < 16u)
        return false;

    uint64_t tagLo = 0;
    std::memcpy(&tagLo, data, sizeof(tagLo));
    const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
    return flg == 2u;
}

void GifArbiter::submit(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool path2DirectHl,
                       bool drainImmediately)
{
    if (!data || sizeBytes < 16 || !m_processFn)
        return;

    if (drainImmediately && m_queue.empty())
    {
        m_processFn(data, sizeBytes);
        g_gifCopyStats.note(true, sizeBytes);
        return;
    }

    GifArbiterPacket pkt;
    pkt.pathId = pathId;
    pkt.path2DirectHl = (pathId == GifPathId::Path2) && path2DirectHl;
    pkt.path3Image = (pathId == GifPathId::Path3) && isImagePacket(data, sizeBytes);
    pkt.data.resize(sizeBytes);
    std::memcpy(pkt.data.data(), data, sizeBytes);
    m_queue.push_back(std::move(pkt));
    g_gifCopyStats.note(false, sizeBytes);
}

void GifArbiter::drain()
{
    if (!m_processFn)
        return;

    std::stable_sort(m_queue.begin(), m_queue.end(),
                     [](const GifArbiterPacket &a, const GifArbiterPacket &b)
                     {
                         // DIRECTHL cannot preempt PATH3 IMAGE transfers.
                         if (a.path2DirectHl != b.path2DirectHl || a.path3Image != b.path3Image)
                         {
                             if (a.path3Image && b.path2DirectHl)
                                 return true;
                             if (a.path2DirectHl && b.path3Image)
                                 return false;
                         }
                         return pathPriority(a.pathId) < pathPriority(b.pathId);
                     });

    for (size_t i = 0; i < m_queue.size(); ++i)
    {
        auto &pkt = m_queue[i];
        if (!pkt.data.empty())
        {
            m_processFn(pkt.data.data(), static_cast<uint32_t>(pkt.data.size()));
        }
    }
    m_queue.clear();
}

uint8_t GifArbiter::pathPriority(GifPathId id)
{
    return static_cast<uint8_t>(id);
}
