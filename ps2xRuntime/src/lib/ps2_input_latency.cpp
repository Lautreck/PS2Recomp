// Includes windows.h and not raylib (their names clash), so it is its own
// translation unit in the CMake source list.
#include "runtime/ps2_input_latency.h"
#include "runtime/ps2_graphics_settings.h"
#include "runtime/ps2_hud.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace
{
    constexpr int kSlots = 32;

    struct Press
    {
        uint32_t seq = 0;
        uint32_t vk = 0;
        uint16_t bit = 0;
        uint64_t t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
        uint64_t tickAtT2 = 0;
        uint64_t hashBefore = 0;
        uint64_t hashAtT5 = 0;
        int stage = 0; // 1 t1, 2 t2, 3 t3, 4 t4, 5 printed
        int latchesAfterT2 = 0;
        bool sampleSent = false;
        bool counted = false;
    };

    std::atomic<uint16_t> g_posted{0};
    std::atomic<bool> g_suppressKeys{false};
    std::atomic<bool> g_blockGamepad{false};
    std::atomic<uint64_t> g_vkHeld[4]{};
    std::atomic<int> g_open{0};
    std::atomic<uintptr_t> g_hwnd{0};
    std::atomic<uint16_t> g_asyncCache{0};
    std::atomic<uint64_t> g_asyncCacheQpc{0};
    std::atomic<uint64_t> g_lastHash{0};
    WNDPROC g_oldProc = nullptr;
    bool g_timerPeriod = false;
    bool g_latched = false;

    std::mutex g_pressMu;
    Press g_press[kSlots];
    uint32_t g_seq = 0;

    bool envIs1(const char *name)
    {
        const char *v = std::getenv(name);
        return v && v[0] == '1';
    }

    uint64_t qpc()
    {
        LARGE_INTEGER c;
        QueryPerformanceCounter(&c);
        return static_cast<uint64_t>(c.QuadPart);
    }

    uint64_t qpcFreq()
    {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return static_cast<uint64_t>(f.QuadPart);
    }

    // Published keyboard table (ps2_controls). One load of the snapshot pointer;
    // the 200 us async cache below still does the GetAsyncKeyState calls.
    uint16_t vkToPad(WPARAM vk)
    {
        if (vk >= 256)
            return 0;
        const ps2_controls::Snap *snap = ps2_controls::publishedSnap();
        return snap ? snap->vkMask[vk] : 0;
    }

    uint16_t asyncPadMask()
    {
        const ps2_controls::Snap *snap = ps2_controls::publishedSnap();
        if (!snap)
            return 0;
        uint16_t m = 0;
        for (int i = 0; i < snap->pollCount; ++i)
        {
            const unsigned vk = snap->pollVk[i];
            if ((GetAsyncKeyState(static_cast<int>(vk)) & 0x8000) != 0)
                m = static_cast<uint16_t>(m | snap->vkMask[vk]);
        }
        return m;
    }

    void clearHeldKeys()
    {
        g_posted.store(0, std::memory_order_release);
        for (std::atomic<uint64_t> &word : g_vkHeld)
            word.store(0, std::memory_order_release);
    }

    void setVkHeld(unsigned vk, bool down)
    {
        if (vk >= 256)
            return;
        std::atomic<uint64_t> &word = g_vkHeld[vk >> 6];
        const uint64_t bit = 1ull << (vk & 63u);
        if (down)
            word.fetch_or(bit, std::memory_order_release);
        else
            word.fetch_and(~bit, std::memory_order_release);
    }

    void printPress(const Press &p)
    {
        std::fprintf(stderr,
                     "[input-lag] seq=%u vk=%u t1=%llu t2=%llu t3=%llu t4=%llu t5=%llu before=%llu hash=%llu\n",
                     p.seq, p.vk,
                     static_cast<unsigned long long>(p.t1),
                     static_cast<unsigned long long>(p.t2),
                     static_cast<unsigned long long>(p.t3),
                     static_cast<unsigned long long>(p.t4),
                     static_cast<unsigned long long>(p.t5),
                     static_cast<unsigned long long>(p.hashBefore),
                     static_cast<unsigned long long>(p.hashAtT5));
        std::fflush(stderr);
    }

    void openPress(uint32_t vk, uint16_t bit)
    {
        std::lock_guard<std::mutex> lock(g_pressMu);
        Press *slot = nullptr;
        for (Press &p : g_press)
        {
            if (p.stage == 0 || p.stage == 5)
            {
                slot = &p;
                break;
            }
        }
        if (!slot)
            slot = &g_press[0];
        if (slot->counted)
        {
            slot->counted = false;
            g_open.fetch_sub(1, std::memory_order_relaxed);
        }
        *slot = Press{};
        slot->seq = ++g_seq;
        slot->vk = vk;
        slot->bit = bit;
        slot->t1 = qpc();
        slot->hashBefore = g_lastHash.load(std::memory_order_relaxed);
        slot->stage = 1;
        slot->counted = true;
        g_open.fetch_add(1, std::memory_order_release);
    }

    void releaseSlot(Press &p, int stage)
    {
        if (p.counted)
        {
            p.counted = false;
            g_open.fetch_sub(1, std::memory_order_release);
        }
        p.stage = stage;
    }

    // Game thread. scePadRead calls notePad only when TEKKEN4_INPUT_LAG=1
    // (Pad.cpp, not this task's files). The fast path already samples the pad
    // here, so t2 is stamped from that read as well.
    void stampSeen(uint16_t pressedMask, uint64_t vsyncTick)
    {
        if (pressedMask == 0 || g_open.load(std::memory_order_acquire) == 0)
            return;
        std::lock_guard<std::mutex> lock(g_pressMu);
        const uint64_t now = qpc();
        for (Press &p : g_press)
        {
            if (p.stage == 1 && (pressedMask & p.bit) != 0)
            {
                p.t2 = now;
                p.tickAtT2 = vsyncTick;
                p.stage = 2;
            }
        }
    }

    LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
    {
        if (msg == WM_KEYDOWN || msg == WM_KEYUP || msg == WM_SYSKEYDOWN || msg == WM_SYSKEYUP)
        {
            const bool down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
            const bool repeat = down && ((lp & (static_cast<LPARAM>(1) << 30)) != 0);
            if (down && !repeat && wp == VK_F3)
                ps2_hud::toggle();
            // A rebind or an ImGui text field must not reach the game. F3 still toggles above.
            if (!g_suppressKeys.load(std::memory_order_relaxed))
            {
                if (wp < 256)
                    setVkHeld(static_cast<unsigned>(wp), down);
                const uint16_t bit = vkToPad(wp);
                if (bit != 0)
                {
                    if (down)
                        g_posted.fetch_or(bit, std::memory_order_release);
                    else
                        g_posted.fetch_and(static_cast<uint16_t>(~bit), std::memory_order_release);
                    // One stamp per physical press. No hash. Idle path does not get here.
                    if (down && !repeat)
                        openPress(static_cast<uint32_t>(wp), bit);
                }
            }
        }
        if (g_oldProc)
            return CallWindowProcW(g_oldProc, hwnd, msg, wp, lp);
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

    void setTimerPeriod(bool on)
    {
        const auto begin = reinterpret_cast<UINT(WINAPI *)(UINT)>(GetProcAddress(GetModuleHandleA("winmm.dll"), "timeBeginPeriod"));
        const auto end = reinterpret_cast<UINT(WINAPI *)(UINT)>(GetProcAddress(GetModuleHandleA("winmm.dll"), "timeEndPeriod"));
        HMODULE winmm = nullptr;
        auto beginFn = begin;
        auto endFn = end;
        if (!beginFn || !endFn)
        {
            winmm = LoadLibraryA("winmm.dll");
            beginFn = reinterpret_cast<UINT(WINAPI *)(UINT)>(GetProcAddress(winmm, "timeBeginPeriod"));
            endFn = reinterpret_cast<UINT(WINAPI *)(UINT)>(GetProcAddress(winmm, "timeEndPeriod"));
        }
        if (on && beginFn && !g_timerPeriod)
        {
            if (beginFn(1) == 0)
                g_timerPeriod = true;
        }
        else if (!on && endFn && g_timerPeriod)
        {
            endFn(1);
            g_timerPeriod = false;
        }
        (void)winmm;
    }

    uint64_t hashImage(const uint8_t *rgba, uint32_t width, uint32_t height)
    {
        // 16x16 block-mean luma, quantised. A menu cursor moves several blocks.
        // A one-line interlace shift does not, which a per-pixel sample did.
        uint64_t h = 14695981039346656037ull;
        if (!rgba || width < 16u || height < 16u)
            return 0;
        for (uint32_t y = 0; y + 16u <= height; y += 16u)
        {
            for (uint32_t x = 0; x + 16u <= width; x += 16u)
            {
                uint32_t sum = 0;
                for (uint32_t by = 0; by < 16u; by += 2u)
                {
                    const uint8_t *row = rgba + (static_cast<size_t>(y + by) * width + x) * 4u;
                    for (uint32_t bx = 0; bx < 16u; bx += 2u)
                    {
                        const uint8_t *p = row + static_cast<size_t>(bx) * 4u;
                        sum += (static_cast<unsigned>(p[0]) * 3u + static_cast<unsigned>(p[1]) * 4u +
                                static_cast<unsigned>(p[2])) >>
                               3;
                    }
                }
                const unsigned q = (sum / 64u) >> 4;
                h ^= q;
                h *= 1099511628211ull;
            }
        }
        return h == 0 ? 1 : h;
    }
}

namespace ps2_input_latency
{
    bool enabled()
    {
        static const bool on = envIs1("TEKKEN4_INPUT_LAG");
        return on;
    }

    bool legacy()
    {
        static const bool on = envIs1("TEKKEN4_INPUT_LEGACY");
        return on;
    }

    bool presentOnTick()
    {
        static const bool tick = []
        {
            const char *v = std::getenv("TEKKEN4_INPUT_PRESENT");
            return v && std::strcmp(v, "tick") == 0;
        }();
        return tick;
    }

    bool hookInstalled()
    {
        return g_oldProc != nullptr;
    }

    void attachWindow(void *hwnd)
    {
        if (!hwnd)
            return;
        // Load PS2X_SETTINGS before the hook sees a key, so a remap is already published.
        ps2_graphics::get();
        if (legacy() && !enabled())
            return;
        auto window = static_cast<HWND>(hwnd);
        g_hwnd.store(reinterpret_cast<uintptr_t>(window), std::memory_order_release);
        if (!legacy())
            setTimerPeriod(true);
        if (!g_oldProc)
        {
            g_oldProc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(window, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(wndProc)));
        }
        if (enabled())
        {
            std::fprintf(stderr, "[input-lag] enabled legacy=%d present=%s qpc_freq=%llu\n",
                         legacy() ? 1 : 0, presentOnTick() ? "tick" : "paced",
                         static_cast<unsigned long long>(qpcFreq()));
            std::fflush(stderr);
        }
    }

    void detachWindow()
    {
        const uintptr_t raw = g_hwnd.exchange(0, std::memory_order_acq_rel);
        if (raw && g_oldProc)
        {
            SetWindowLongPtrW(reinterpret_cast<HWND>(raw), GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_oldProc));
            g_oldProc = nullptr;
        }
        setTimerPeriod(false);
    }

    void waitMessage(unsigned timeoutMs)
    {
        MsgWaitForMultipleObjects(0, nullptr, FALSE, timeoutMs, QS_ALLINPUT);
    }

    void setPadPostingSuppressed(bool suppressed)
    {
        const bool prev = g_suppressKeys.exchange(suppressed, std::memory_order_acq_rel);
        if (suppressed && !prev)
            clearHeldKeys();
    }

    bool padPostingSuppressed()
    {
        return g_suppressKeys.load(std::memory_order_acquire);
    }

    void setGamepadButtonsBlocked(bool blocked)
    {
        g_blockGamepad.store(blocked, std::memory_order_release);
    }

    bool gamepadButtonsBlocked()
    {
        return g_blockGamepad.load(std::memory_order_acquire);
    }

    bool sampleKeyboard(uint16_t &padDownMask, float *stickLx, float *stickLy, float *stickRx, float *stickRy,
                        bool *stickActive)
    {
        if (stickActive)
            *stickActive = false;
        if (legacy())
            return false;
        const uintptr_t raw = g_hwnd.load(std::memory_order_acquire);
        if (raw == 0)
            return false;
        if (g_suppressKeys.load(std::memory_order_acquire))
        {
            padDownMask = 0;
            return true;
        }
        uint16_t mask = g_posted.load(std::memory_order_acquire);
        const uint64_t now = qpc();
        const uint64_t cachedAt = g_asyncCacheQpc.load(std::memory_order_acquire);
        const uint64_t freq = qpcFreq();
        const uint64_t budget = freq / 5000u; // 200 us
        uint16_t asyncMask = g_asyncCache.load(std::memory_order_acquire);
        if (cachedAt == 0 || now - cachedAt >= budget)
        {
            const bool focused = GetForegroundWindow() == reinterpret_cast<HWND>(raw);
            asyncMask = focused ? asyncPadMask() : 0;
            g_asyncCache.store(asyncMask, std::memory_order_release);
            g_asyncCacheQpc.store(now, std::memory_order_release);
        }
        padDownMask = static_cast<uint16_t>(mask | asyncMask);
        const ps2_controls::Snap *snap = ps2_controls::publishedSnap();
        if (snap && snap->stickCount > 0 && stickLx && stickLy && stickRx && stickRy)
        {
            float acc[2][2] = {};
            bool any = false;
            const bool focused = GetForegroundWindow() == reinterpret_cast<HWND>(raw);
            for (int i = 0; i < snap->stickCount; ++i)
            {
                const ps2_controls::Snap::StickBind &bind = snap->sticks[i];
                bool down = false;
                if (bind.vk > 0 && bind.vk < 256)
                {
                    const uint64_t word = g_vkHeld[static_cast<unsigned>(bind.vk) >> 6].load(std::memory_order_acquire);
                    down = ((word >> (bind.vk & 63)) & 1ull) != 0;
                    if (!down && focused)
                        down = (GetAsyncKeyState(bind.vk) & 0x8000) != 0;
                }
                if (!down)
                    continue;
                acc[bind.stick][bind.axis] += bind.value;
                any = true;
            }
            if (any)
            {
                *stickLx = acc[0][0];
                *stickLy = acc[0][1];
                *stickRx = acc[1][0];
                *stickRy = acc[1][1];
                if (stickActive)
                    *stickActive = true;
            }
        }
        if (padDownMask != 0)
            stampSeen(padDownMask, 0);
        return true;
    }

    void notePad(uint16_t pressedMask, uint64_t vsyncTick)
    {
        stampSeen(pressedMask, vsyncTick);
    }

    void onLatchBegin(uint64_t vsyncTick)
    {
        (void)vsyncTick;
        g_latched = true;
        if (g_open.load(std::memory_order_acquire) == 0)
            return;
        std::lock_guard<std::mutex> lock(g_pressMu);
        const uint64_t now = qpc();
        for (Press &p : g_press)
        {
            if (p.stage == 2)
            {
                p.t3 = now;
                p.stage = 3;
            }
        }
    }

    void onLatchPixels(const uint8_t *rgba, uint32_t width, uint32_t height)
    {
        if (!enabled())
            return;
        g_lastHash.store(hashImage(rgba, width, height), std::memory_order_relaxed);
    }

    void onSwapReturned()
    {
        const bool latched = g_latched;
        g_latched = false;
        if (g_open.load(std::memory_order_acquire) == 0)
            return;
        const bool full = enabled();
        const uint64_t now = qpc();
        const uint64_t hash = full ? g_lastHash.load(std::memory_order_relaxed) : 0;
        const uint64_t freq = qpcFreq();
        std::lock_guard<std::mutex> lock(g_pressMu);
        for (Press &p : g_press)
        {
            if (p.stage == 0 || p.stage == 5)
                continue;
            if (p.stage == 3 && latched)
            {
                p.t4 = now;
                p.stage = 4;
                if (!p.sampleSent && p.t1 != 0 && p.t4 >= p.t1 && freq != 0)
                {
                    p.sampleSent = true;
                    const double ms = (static_cast<double>(p.t4 - p.t1) * 1000.0) / static_cast<double>(freq);
                    ps2_hud::noteSample(ms);
                }
            }
            if (!full)
            {
                // No frame hash on this path. Drop a press the guest never reads.
                const bool timedOut = p.t1 != 0 && freq != 0 && (now - p.t1) > freq * 2u;
                if (p.stage == 4 || timedOut)
                    releaseSlot(p, 0);
                continue;
            }
            // Pixels hashed for this swap were latched in onLatchBegin. That
            // latch is only after the pad read once t3 is set. A swap that
            // was already in flight when the key arrived must not become t5.
            if (latched && p.t3 != 0 && p.t5 == 0 && p.hashBefore != 0 && hash != 0 && hash != p.hashBefore)
            {
                p.t5 = now;
                p.hashAtT5 = hash;
            }
            if (latched)
                ++p.latchesAfterT2;
            const bool done = p.t4 != 0 && p.t5 != 0;
            const bool gaveUp = p.latchesAfterT2 > 45;
            if ((done || gaveUp) && p.stage < 5)
            {
                printPress(p);
                releaseSlot(p, 5);
            }
        }
    }
}

#else

namespace
{
    std::atomic<bool> g_suppressKeys{false};
    std::atomic<bool> g_blockGamepad{false};
}

namespace ps2_input_latency
{
    bool enabled() { return false; }
    bool legacy() { return false; }
    bool hookInstalled() { return false; }
    bool presentOnTick() { return false; }
    void attachWindow(void *) {}
    void detachWindow() {}
    void waitMessage(unsigned) {}
    bool sampleKeyboard(uint16_t &, float *, float *, float *, float *, bool *) { return false; }
    void setPadPostingSuppressed(bool suppressed) { g_suppressKeys.store(suppressed); }
    bool padPostingSuppressed() { return g_suppressKeys.load(); }
    void setGamepadButtonsBlocked(bool blocked) { g_blockGamepad.store(blocked); }
    bool gamepadButtonsBlocked() { return g_blockGamepad.load(); }
    void notePad(uint16_t, uint64_t) {}
    void onLatchBegin(uint64_t) {}
    void onLatchPixels(const uint8_t *, uint32_t, uint32_t) {}
    void onSwapReturned() {}
}

#endif
