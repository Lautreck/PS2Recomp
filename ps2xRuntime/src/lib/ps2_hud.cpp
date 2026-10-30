#include "runtime/ps2_hud.h"

#include "raylib.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace
{
    struct State
    {
        bool inited = false;
        bool visible = true;
        bool log = false;
        bool started = false;
        bool showHave = false;
        uint32_t gameAcc = 0;
        uint32_t hostAcc = 0;
        uint32_t prevGame = 0;
        uint32_t prevHost = 0;
        double prevSec = 0.0;
        int showGame = 0;
        int showHost = 0;
        int showInput = 0;
        int showAvg = 0;
        int showN = 0;
        double showInputMs = -1.0;
        double showAvgMs = -1.0;
        double lastMs = 0.0;
        bool haveLast = false;
        double ring[10] = {};
        int ringN = 0;
        int ringI = 0;
        int logS = 0;
        std::chrono::steady_clock::time_point halfStart{};
        std::chrono::steady_clock::time_point nextLog{};
    };

    State g;
    // Frames reported by a game override (game thread). Once any arrive they replace the vsync-tick fallback.
    std::atomic<uint32_t> g_reportedFrames{0};
    std::atomic<bool> g_gameReports{false};

    void ensure()
    {
        if (g.inited)
            return;
        g.inited = true;
        const char *hud = std::getenv("TEKKEN4_HUD");
        if (hud && std::strcmp(hud, "0") == 0)
            g.visible = false;
        const char *log = std::getenv("TEKKEN4_HUD_LOG");
        g.log = log && std::strcmp(log, "1") == 0;
    }

    void refresh(std::chrono::steady_clock::time_point now)
    {
        if (!g.started)
            return;
        const double sec = std::chrono::duration<double>(now - g.halfStart).count();
        if (sec < 0.5)
            return;
        if (g_gameReports.load(std::memory_order_relaxed))
            g.gameAcc = g_reportedFrames.exchange(0, std::memory_order_relaxed);
        const double span = sec + g.prevSec;
        const double gfps = (g.prevSec > 0.0) ? (static_cast<double>(g.gameAcc + g.prevGame) / span)
                                               : (static_cast<double>(g.gameAcc) / sec);
        const double hfps = (g.prevSec > 0.0) ? (static_cast<double>(g.hostAcc + g.prevHost) / span)
                                               : (static_cast<double>(g.hostAcc) / sec);
        g.showGame = static_cast<int>(std::lround(gfps));
        g.showHost = static_cast<int>(std::lround(hfps));
        if (g.ringN > 0 && g.haveLast)
        {
            double sum = 0.0;
            for (int i = 0; i < g.ringN; ++i)
                sum += g.ring[i];
            g.showInputMs = g.lastMs;
            g.showAvgMs = sum / static_cast<double>(g.ringN);
            g.showInput = static_cast<int>(std::lround(g.showInputMs));
            g.showAvg = static_cast<int>(std::lround(g.showAvgMs));
            g.showN = g.ringN;
            g.showHave = true;
        }
        g.prevGame = g.gameAcc;
        g.prevHost = g.hostAcc;
        g.prevSec = sec;
        g.gameAcc = 0;
        g.hostAcc = 0;
        g.halfStart = now;
    }

    void logLine(std::chrono::steady_clock::time_point now)
    {
        if (!g.log || !g.started || now < g.nextLog)
            return;
        if (now - g.nextLog > std::chrono::seconds(1))
            g.nextLog = now;
        g.nextLog += std::chrono::seconds(1);
        ++g.logS;
        if (g.showHave)
        {
            std::fprintf(stdout, "[hud] s=%d game=%d host=%d input=%d avg=%d input_ms=%.1f avg_ms=%.1f n=%d\n",
                         g.logS, g.showGame, g.showHost, g.showInput, g.showAvg, g.showInputMs, g.showAvgMs, g.showN);
        }
        else
        {
            std::fprintf(stdout, "[hud] s=%d game=%d host=%d input=- avg=- input_ms=-1 avg_ms=-1 n=0\n",
                         g.logS, g.showGame, g.showHost);
        }
        std::fflush(stdout);
    }
}

namespace ps2_hud
{
    void onHostPresent(bool newGuestFrame)
    {
        ensure();
        const auto now = std::chrono::steady_clock::now();
        if (!g.started)
        {
            g.started = true;
            g.halfStart = now;
            g.nextLog = now + std::chrono::seconds(1);
        }
        ++g.hostAcc;
        if (newGuestFrame)
            ++g.gameAcc;
    }

    void noteGameFrame()
    {
        g_reportedFrames.fetch_add(1, std::memory_order_relaxed);
        g_gameReports.store(true, std::memory_order_relaxed);
    }

    void noteSample(double ms)
    {
        ensure();
        if (!(ms >= 0.0))
            return;
        g.lastMs = ms;
        g.haveLast = true;
        g.ring[g.ringI] = ms;
        g.ringI = (g.ringI + 1) % 10;
        if (g.ringN < 10)
            ++g.ringN;
    }

    void toggle()
    {
        ensure();
        g.visible = !g.visible;
    }

    void draw()
    {
        ensure();
        const auto now = std::chrono::steady_clock::now();
        refresh(now);
        logLine(now);
        if (!g.visible)
            return;

        char line0[64];
        char line1[64];
        char line2[64];
        char line3[64];
        std::snprintf(line0, sizeof(line0), "GAME %d fps", g.showGame);
        std::snprintf(line1, sizeof(line1), "HOST %d fps", g.showHost);
        if (g.showHave)
        {
            std::snprintf(line2, sizeof(line2), "INPUT %d ms", g.showInput);
            std::snprintf(line3, sizeof(line3), "AVG %d ms", g.showAvg);
        }
        else
        {
            std::snprintf(line2, sizeof(line2), "INPUT -- ms");
            std::snprintf(line3, sizeof(line3), "AVG -- ms");
        }
        const char *lines[4] = {line0, line1, line2, line3};
        constexpr int font = 18;
        constexpr int lineH = 22;
        constexpr int pad = 8;
        int maxW = 0;
        for (const char *line : lines)
        {
            const int w = MeasureText(line, font);
            if (w > maxW)
                maxW = w;
        }
        const int boxX = 6;
        const int boxY = 6;
        const int boxW = maxW + pad * 2;
        const int boxH = 4 * lineH + pad;
        DrawRectangle(boxX, boxY, boxW, boxH, Color{0, 0, 0, 190});
        for (int i = 0; i < 4; ++i)
            DrawText(lines[i], boxX + pad, boxY + 4 + i * lineH, font, WHITE);
    }
}
