#include "MiniTest.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "Stubs/Pad.h"
#include "runtime/ps2_graphics_settings.h"

#include <vector>
#include <cstdint>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>


namespace
{
    constexpr uint32_t kPadDataAddr = 0x1000;

    constexpr uint16_t kPadBtnSelect = 1u << 0;
    constexpr uint16_t kPadBtnL3 = 1u << 1;
    constexpr uint16_t kPadBtnR3 = 1u << 2;
    constexpr uint16_t kPadBtnStart = 1u << 3;
    constexpr uint16_t kPadBtnUp = 1u << 4;
    constexpr uint16_t kPadBtnRight = 1u << 5;
    constexpr uint16_t kPadBtnDown = 1u << 6;
    constexpr uint16_t kPadBtnLeft = 1u << 7;
    constexpr uint16_t kPadBtnL2 = 1u << 8;
    constexpr uint16_t kPadBtnR2 = 1u << 9;
    constexpr uint16_t kPadBtnL1 = 1u << 10;
    constexpr uint16_t kPadBtnR1 = 1u << 11;
    constexpr uint16_t kPadBtnTriangle = 1u << 12;
    constexpr uint16_t kPadBtnCircle = 1u << 13;
    constexpr uint16_t kPadBtnCross = 1u << 14;
    constexpr uint16_t kPadBtnSquare = 1u << 15;

    void setRegU32(R5900Context &ctx, int reg, uint32_t value)
    {
        ctx.r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    void openPadPort(R5900Context &ctx, std::vector<uint8_t> &rdram, uint32_t port = 0, uint32_t slot = 0)
    {
        setRegU32(ctx, 4, port);
        setRegU32(ctx, 5, slot);
        setRegU32(ctx, 6, kPadDataAddr + 0x200u);
        ps2_stubs::scePadPortOpen(rdram.data(), &ctx, nullptr);
    }

    void closePadPort(R5900Context &ctx, std::vector<uint8_t> &rdram, uint32_t port = 0, uint32_t slot = 0)
    {
        setRegU32(ctx, 4, port);
        setRegU32(ctx, 5, slot);
        ps2_stubs::scePadPortClose(rdram.data(), &ctx, nullptr);
    }

    void runPadRead(R5900Context &ctx, std::vector<uint8_t> &rdram)
    {
        setRegU32(ctx, 4, 0u);
        setRegU32(ctx, 5, 0u);
        setRegU32(ctx, 6, kPadDataAddr); // a2
        ps2_stubs::scePadRead(rdram.data(), &ctx, nullptr);
    }

    uint16_t readButtons(const std::vector<uint8_t> &rdram)
    {
        const uint8_t *data = rdram.data() + kPadDataAddr;
        return static_cast<uint16_t>(data[2] | (data[3] << 8));
    }

    bool nearf(float a, float b)
    {
        return std::fabs(a - b) < 1e-4f;
    }

    bool controlsFieldsEqual(const ps2_controls::Controls &a, const ps2_controls::Controls &b, std::string &why)
    {
        char ka[80];
        char va[192];
        char kb[80];
        char vb[192];
        for (int i = 0;; ++i)
        {
            const bool aa = ps2_controls::controlsField(a, i, ka, 80, va, 192);
            const bool bb = ps2_controls::controlsField(b, i, kb, 80, vb, 192);
            if (!aa && !bb)
                return true;
            if (!aa || !bb || std::strcmp(ka, kb) != 0 || std::strcmp(va, vb) != 0)
            {
                why = std::string(aa ? ka : "-") + "=" + (aa ? va : "-") + " vs " + (bb ? kb : "-") + "=" + (bb ? vb : "-");
                return false;
            }
        }
    }

    ps2_controls::Controls roundTrip(const ps2_controls::Controls &src, bool &ok)
    {
        char text[16384];
        int used = 0;
        text[0] = '\0';
        ok = true;
        char key[80];
        char value[192];
        for (int i = 0; ps2_controls::controlsField(src, i, key, 80, value, 192); ++i)
        {
            const int n = std::snprintf(text + used, sizeof(text) - static_cast<std::size_t>(used), "%s=%s\n", key, value);
            if (n < 0 || used + n >= static_cast<int>(sizeof(text)))
            {
                ok = false;
                break;
            }
            used += n;
        }
        ps2_controls::Controls dst = ps2_controls::defaultControls();
        if (ok)
            ps2_controls::applyControlsText(dst, text);
        return dst;
    }

    std::string fieldValue(const ps2_controls::Controls &controls, const char *want)
    {
        char key[80];
        char value[192];
        for (int i = 0; ps2_controls::controlsField(controls, i, key, 80, value, 192); ++i)
        {
            if (std::strcmp(key, want) == 0)
                return value;
        }
        return std::string();
    }
}

void register_pad_input_tests()
{
    MiniTest::Case("PadInput", [](TestCase &tc)
                   {
        tc.Run("scePadRead uses override state", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            const uint16_t buttons = static_cast<uint16_t>(0xFFFFu & ~kPadBtnCross & ~kPadBtnStart);
            ps2_stubs::setPadOverrideState(buttons, 0x00, 0xFF, 0x10, 0xEE);

            runPadRead(ctx, rdram);

            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadRead should return 1");
            t.Equals(readButtons(rdram), buttons, "button bitmask should match override state");
            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[4], static_cast<uint8_t>(0x10), "rx should match override");
            t.Equals(data[5], static_cast<uint8_t>(0xEE), "ry should match override");
            t.Equals(data[6], static_cast<uint8_t>(0x00), "lx should match override");
            t.Equals(data[7], static_cast<uint8_t>(0xFF), "ly should match override");

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("scePadRead button bits are active-low", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            struct ButtonCase
            {
                uint16_t mask;
                const char *name;
            };

            const ButtonCase cases[] = {
                {kPadBtnSelect, "select"},
                {kPadBtnL3, "l3"},
                {kPadBtnR3, "r3"},
                {kPadBtnStart, "start"},
                {kPadBtnUp, "up"},
                {kPadBtnRight, "right"},
                {kPadBtnDown, "down"},
                {kPadBtnLeft, "left"},
                {kPadBtnL2, "l2"},
                {kPadBtnR2, "r2"},
                {kPadBtnL1, "l1"},
                {kPadBtnR1, "r1"},
                {kPadBtnTriangle, "triangle"},
                {kPadBtnCircle, "circle"},
                {kPadBtnCross, "cross"},
                {kPadBtnSquare, "square"}};

            for (const auto &entry : cases)
            {
                const uint16_t buttons = static_cast<uint16_t>(0xFFFFu & ~entry.mask);
                ps2_stubs::setPadOverrideState(buttons, 0x80, 0x80, 0x80, 0x80);
                runPadRead(ctx, rdram);

                t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadRead should succeed for opened ports");
                const uint16_t mask = readButtons(rdram);
                t.IsTrue((mask & entry.mask) == 0, std::string("button should be active-low: ").append(entry.name));
            }

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("scePadGetButtonMask returns all buttons", [](TestCase &t)
               {
            R5900Context ctx;
            ps2_stubs::scePadGetButtonMask(nullptr, &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0xFFFF), "button mask should be 0xFFFF");
        });

        tc.Run("basic pad init/port/state functions return expected values", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInit should succeed");

            ps2_stubs::scePadInit2(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInit2 should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "closed port should report DISCONNECTED");

            openPadPort(ctx, rdram);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadPortOpen should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "scePadGetState should return STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetReqState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadGetReqState should return completed");

            ps2_stubs::scePadGetPortMax(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(2), "scePadGetPortMax should be 2");

            setRegU32(ctx, 4, 0u);
            ps2_stubs::scePadGetSlotMax(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadGetSlotMax should be 1");

            ps2_stubs::scePadGetModVersion(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0x0200), "scePadGetModVersion should be 0x0200");

            closePadPort(ctx, rdram);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadPortClose should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "closed port should return DISCONNECTED after close");
        });

        tc.Run("pad command state reports EXECCMD once before returning STABLE", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(5), "first state after mode command should be EXECCMD");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "second state after mode command should return STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(5), "first state after press-mode command should be EXECCMD");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "second state after press-mode command should return STABLE");

            closePadPort(ctx, rdram);
        });

        tc.Run("pad info and mode helpers return consistent values", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 6, static_cast<uint32_t>(-1));
            ps2_stubs::scePadInfoAct(rdram.data(), &ctx, nullptr);
            t.IsTrue(static_cast<uint32_t>(getRegU32(&ctx, 2)) >= 1u, "scePadInfoAct should report at least one actuator descriptor");

            ps2_stubs::scePadInfoComb(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadInfoComb should return 0");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "scePadInfoMode CURID should return digital at open");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 4);
            setRegU32(ctx, 7, static_cast<uint32_t>(-1));
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.IsTrue(static_cast<uint32_t>(getRegU32(&ctx, 2)) >= 1u, "scePadInfoMode table count should be non-zero");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadInfoPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadInfoPressMode should report pressure support");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should accept digital mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 0u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "CURID should switch to digital mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should accept analog mode");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 4);
            setRegU32(ctx, 7, 0u);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(7), "mode table entry should return DualShock in analog mode");

            closePadPort(ctx, rdram);
        });

        tc.Run("pads open in digital mode and switch to analog on scePadSetMainMode", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(6), "freshly opened port should report STABLE");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(4), "scePadInfoMode CURID should return digital at open");

            runPadRead(ctx, rdram);
            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[1], static_cast<uint8_t>(0x41), "mode byte should be 0x41 (digital) at open");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 3);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed switching to analog");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1);
            setRegU32(ctx, 7, 0);
            ps2_stubs::scePadInfoMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(7), "scePadInfoMode CURID should return analog after SetMainMode");

            // scePadSetMainMode queues a one-shot EXECCMD transient state; pump scePadGetState
            // once so the port settles back to STABLE before reading, mirroring the existing
            // "pad command state reports EXECCMD once before returning STABLE" test.
            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadGetState(rdram.data(), &ctx, nullptr);

            runPadRead(ctx, rdram);
            t.Equals(data[1], static_cast<uint8_t>(0x73), "mode byte should be 0x73 (analog) after SetMainMode");

            closePadPort(ctx, rdram);
        });

        tc.Run("pad setters return success", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetActAlign(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetActAlign should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetActDirect(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetActDirect should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0xFFFFu);
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 1u);
            setRegU32(ctx, 7, 3u);
            ps2_stubs::scePadSetMainMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetMainMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetReqState(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetReqState should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetVrefParam(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetVrefParam should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadSetWarningLevel(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(0), "scePadSetWarningLevel should return 0");

            ps2_stubs::scePadEnd(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnd should succeed");

            openPadPort(ctx, rdram);
            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should succeed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadExitPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadExitPressMode should succeed");

            closePadPort(ctx, rdram);
        });

        tc.Run("scePadRead fills pressure bytes and honors button info mask", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            ps2_stubs::scePadInit(rdram.data(), &ctx, nullptr);
            openPadPort(ctx, rdram);

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, 0xFFFFu);
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should accept all buttons");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            ps2_stubs::scePadEnterPressMode(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadEnterPressMode should enable pressure data");

            const uint16_t pressedButtons = static_cast<uint16_t>(0xFFFFu &
                                                                   ~kPadBtnLeft &
                                                                   ~kPadBtnUp &
                                                                   ~kPadBtnTriangle &
                                                                   ~kPadBtnCross &
                                                                   ~kPadBtnL1 &
                                                                   ~kPadBtnR2);
            ps2_stubs::setPadOverrideState(pressedButtons, 0x80, 0x80, 0x80, 0x80);
            runPadRead(ctx, rdram);

            const uint8_t *data = rdram.data() + kPadDataAddr;
            t.Equals(data[8], static_cast<uint8_t>(0x00), "right pressure should be clear when not pressed");
            t.Equals(data[9], static_cast<uint8_t>(0xFF), "left pressure should be populated when pressed");
            t.Equals(data[10], static_cast<uint8_t>(0xFF), "up pressure should be populated when pressed");
            t.Equals(data[11], static_cast<uint8_t>(0x00), "down pressure should be clear when not pressed");
            t.Equals(data[12], static_cast<uint8_t>(0xFF), "triangle pressure should be populated when pressed");
            t.Equals(data[13], static_cast<uint8_t>(0x00), "circle pressure should be clear when not pressed");
            t.Equals(data[14], static_cast<uint8_t>(0xFF), "cross pressure should be populated when pressed");
            t.Equals(data[15], static_cast<uint8_t>(0x00), "square pressure should be clear when not pressed");
            t.Equals(data[16], static_cast<uint8_t>(0xFF), "L1 pressure should be populated when pressed");
            t.Equals(data[17], static_cast<uint8_t>(0x00), "L2 pressure should be clear when not pressed");
            t.Equals(data[18], static_cast<uint8_t>(0x00), "R1 pressure should be clear when not pressed");
            t.Equals(data[19], static_cast<uint8_t>(0xFF), "R2 pressure should be populated when pressed");

            setRegU32(ctx, 4, 0u);
            setRegU32(ctx, 5, 0u);
            setRegU32(ctx, 6, static_cast<uint32_t>(kPadBtnL1 | kPadBtnR2));
            ps2_stubs::scePadSetButtonInfo(rdram.data(), &ctx, nullptr);
            t.Equals(static_cast<uint32_t>(getRegU32(&ctx, 2)), static_cast<uint32_t>(1), "scePadSetButtonInfo should narrow the enabled pressure mask");

            runPadRead(ctx, rdram);

            t.Equals(data[9], static_cast<uint8_t>(0x00), "masked-out direction pressure should clear");
            t.Equals(data[10], static_cast<uint8_t>(0x00), "masked-out direction pressure should clear");
            t.Equals(data[12], static_cast<uint8_t>(0x00), "masked-out face-button pressure should clear");
            t.Equals(data[14], static_cast<uint8_t>(0x00), "masked-out face-button pressure should clear");
            t.Equals(data[16], static_cast<uint8_t>(0xFF), "enabled L1 pressure should remain populated");
            t.Equals(data[19], static_cast<uint8_t>(0xFF), "enabled R2 pressure should remain populated");

            ps2_stubs::clearPadOverrideState();
            closePadPort(ctx, rdram);
        });

        tc.Run("pad string helpers map state codes", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            setRegU32(ctx, 4, 1);
            setRegU32(ctx, 5, kPadDataAddr);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr)).find("FINDPAD") != std::string::npos,
                     "state 1 should map to FINDPAD");

            setRegU32(ctx, 4, 0);
            setRegU32(ctx, 5, kPadDataAddr + 64);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 64)).find("DISCONNECTED") != std::string::npos,
                     "state 0 should map to DISCONNECTED");

            setRegU32(ctx, 4, 1);
            setRegU32(ctx, 5, kPadDataAddr + 128);
            ps2_stubs::scePadReqIntToStr(rdram.data(), &ctx, nullptr);
            t.IsTrue(std::string(reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 128)).find("BUSY") != std::string::npos,
                     "req state 1 should map to BUSY");
        });
        tc.Run("scePadGetFrameCount increments", [](TestCase &t)
               {
            R5900Context ctx;
            ps2_stubs::scePadGetFrameCount(nullptr, &ctx, nullptr);
            const uint32_t first = getRegU32(&ctx, 2);
            ps2_stubs::scePadGetFrameCount(nullptr, &ctx, nullptr);
            const uint32_t second = getRegU32(&ctx, 2);
            t.Equals(second, first + 1, "frame count should increment");
        });

        tc.Run("scePadStateIntToStr and scePadReqIntToStr write strings", [](TestCase &t)
               {
            std::vector<uint8_t> rdram(PS2_RAM_SIZE, 0);
            R5900Context ctx;

            setRegU32(ctx, 4, 6);
            setRegU32(ctx, 5, kPadDataAddr);
            ps2_stubs::scePadStateIntToStr(rdram.data(), &ctx, nullptr);
            const char *stateStr = reinterpret_cast<const char *>(rdram.data() + kPadDataAddr);
            t.IsTrue(std::string(stateStr).find("STABLE") != std::string::npos, "state string should include STABLE");

            setRegU32(ctx, 4, 0);
            setRegU32(ctx, 5, kPadDataAddr + 64);
            ps2_stubs::scePadReqIntToStr(rdram.data(), &ctx, nullptr);
            const char *reqStr = reinterpret_cast<const char *>(rdram.data() + kPadDataAddr + 64);
            t.IsTrue(std::string(reqStr).find("COMPLETE") != std::string::npos, "req string should include COMPLETE");
        });
    });

    MiniTest::Case("ControlsMap", [](TestCase &tc)
                   {
        tc.Run("presets share the position map except Nintendo labels", [](TestCase &t)
               {
            const ps2_controls::Layout same[] = {ps2_controls::Layout::Playstation, ps2_controls::Layout::Xbox,
                                                 ps2_controls::Layout::Switch, ps2_controls::Layout::Keyboard};
            for (ps2_controls::Layout layout : same)
            {
                for (int b = 0; b < ps2_controls::kButtonCount; ++b)
                {
                    t.Equals(ps2_controls::presetSource(layout, b),
                             ps2_controls::presetSource(ps2_controls::Layout::Playstation, b), "positional preset");
                }
            }
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::Playstation, ps2_controls::Cross), 7, "cross is face down");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::Playstation, ps2_controls::Circle), 6, "circle is face right");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::Playstation, ps2_controls::Square), 8, "square is face left");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::Playstation, ps2_controls::Triangle), 5, "triangle is face up");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::Playstation, ps2_controls::Select), 13, "select is middle left");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::SwitchLabels, ps2_controls::Cross), 6, "nintendo A is cross");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::SwitchLabels, ps2_controls::Circle), 7, "nintendo B is circle");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::SwitchLabels, ps2_controls::Square), 5, "nintendo X is square");
            t.Equals(ps2_controls::presetSource(ps2_controls::Layout::SwitchLabels, ps2_controls::Triangle), 8, "nintendo Y is triangle");
            for (int b = 0; b < ps2_controls::kButtonCount; ++b)
            {
                if (b == ps2_controls::Cross || b == ps2_controls::Circle || b == ps2_controls::Square || b == ps2_controls::Triangle)
                    continue;
                t.Equals(ps2_controls::presetSource(ps2_controls::Layout::SwitchLabels, b),
                         ps2_controls::presetSource(ps2_controls::Layout::Playstation, b), "labels keep the other buttons");
            }
            ps2_controls::Controls keys = ps2_controls::defaultControls();
            t.Equals(keys.keyCount[ps2_controls::Up], 1, "defaults already have a key");
            ps2_controls::applyLayout(keys, ps2_controls::Layout::Keyboard);
            t.IsTrue(keys.ignoreGamepads, "keyboard preset ignores gamepads");
            t.Equals(keys.keyCount[ps2_controls::Up], 1, "layout change keeps keys");
            t.IsFalse(ps2_controls::defaultControls().ignoreGamepads, "position default uses the gamepad");
        });

        tc.Run("default key table matches today's keyboard", [](TestCase &t)
               {
            const ps2_controls::Controls controls = ps2_controls::defaultControls();
            ps2_controls::Snap snap;
            ps2_controls::fillSnap(snap, controls);
            t.Equals(snap.pollCount, 22, "22 default keys");
            t.Equals(snap.rlCount, 22, "22 raylib keys");
            t.Equals(snap.stickCount, 0, "no default stick keys");
            t.IsTrue(nearf(snap.deadzoneLeft, 0.15f) && nearf(snap.deadzoneRight, 0.15f), "deadzone 0.15");
            t.IsFalse(snap.invertLeft || snap.invertRight || snap.stickSwap || snap.ignoreGamepads, "no invert swap or ignore");
            t.IsTrue(snap.triggerThreshold < 0.f, "digital triggers");
            const struct
            {
                unsigned vk;
                uint16_t mask;
            } rows[] = {
                {0x57, 0x0010}, {0x53, 0x0040}, {0x41, 0x0080}, {0x44, 0x0020},
                {0x58, 0x4000}, {0x43, 0x2000}, {0x5A, 0x8000}, {0x56, 0x1000},
                {0x51, 0x0400}, {0x45, 0x0800}, {0xA0, 0x0100}, {0x10, 0x0100}, {0xA1, 0x0200},
                {0x0D, 0x0008}, {0x09, 0x0001}, {0x28, 0x4000}, {0x20, 0x4000},
                {0x27, 0x2000}, {0x1B, 0x2000}, {0x25, 0x8000}, {0x60, 0x8000},
                {0x26, 0x1000}, {0x61, 0x1000}, {0x42, 0},
            };
            for (const auto &row : rows)
                t.Equals(snap.vkMask[row.vk], row.mask, "default vk mask");
            t.Equals(fieldValue(controls, "controls_key_CROSS"), std::string("DOWN,X,SPACE"), "cross keys");
            t.Equals(fieldValue(controls, "controls_key_CIRCLE"), std::string("RIGHT,C,ESCAPE"), "circle keys");
            t.Equals(controls.keyCount[ps2_controls::L3], 0, "no L3 key");
            t.Equals(controls.keyCount[ps2_controls::R3], 0, "no R3 key");
            int l3 = 0;
            int r3 = 0;
            for (int v = 0; v < 256; ++v)
            {
                if ((snap.vkMask[v] & 0x0002u) != 0)
                    ++l3;
                if ((snap.vkMask[v] & 0x0004u) != 0)
                    ++r3;
            }
            t.Equals(l3, 0, "L3 vk unused");
            t.Equals(r3, 0, "R3 vk unused");
        });

        tc.Run("remapped keys or into the fast-path table", [](TestCase &t)
               {
            ps2_controls::Controls controls = ps2_controls::defaultControls();
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_key_CROSS", "DOWN,C,SPACE"), "cross remap");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_key_CIRCLE", "RIGHT,X,ESCAPE"), "circle remap");
            ps2_controls::Snap snap;
            ps2_controls::fillSnap(snap, controls);
            t.Equals(snap.vkMask[0x58], static_cast<uint16_t>(0x2000), "X is circle");
            t.Equals(snap.vkMask[0x43], static_cast<uint16_t>(0x4000), "C is cross");
            t.Equals(snap.vkMask[0x28], static_cast<uint16_t>(0x4000), "down arrow stays cross");
            t.Equals(snap.vkMask[0x57], static_cast<uint16_t>(0x0010), "W stays up");
            ps2_controls::Controls shared = ps2_controls::defaultControls();
            t.IsTrue(ps2_controls::applyControlsField(shared, "controls_key_UP", "W,X"), "share X");
            ps2_controls::fillSnap(snap, shared);
            t.Equals(snap.vkMask[0x58], static_cast<uint16_t>(0x4010), "X ors up and cross");
            t.Equals(snap.vkMask[0x57], static_cast<uint16_t>(0x0010), "W still up");
        });

        tc.Run("bad control fields are ignored", [](TestCase &t)
               {
            ps2_controls::Controls controls = ps2_controls::defaultControls();
            const ps2_controls::Controls saved = controls;
            std::string why;
            const char *badKeys[] = {"controls_layout", "controls_deadzone", "controls_deadzone", "controls_invert_y",
                                    "controls_gp_CROSS", "controls_key_CROSS", "controls_trigger", "controls_rumble",
                                    "controls_gp_FOO", "controls_key_CROSS"};
            const char *badVals[] = {"gamecube", "1.2", "-0.1", "sideways", "NOT_A_BUTTON", "NOT_A_KEY", "2", "1",
                                    "LEFT_FACE_UP", "DOWN,NOT_A_KEY"};
            for (int i = 0; i < 10; ++i)
            {
                t.IsFalse(ps2_controls::applyControlsField(controls, badKeys[i], badVals[i]), "bad field rejected");
                t.IsTrue(controlsFieldsEqual(controls, saved, why), why.empty() ? "bad field changed state" : why);
            }
            t.IsFalse(ps2_controls::applyControlsField(controls, "Controls_layout", "xbox"), "prefix is case-sensitive");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_key_UP", ""), "empty list clears");
            t.Equals(controls.keyCount[ps2_controls::Up], 0, "UP cleared");
            t.Equals(controls.keyCount[ps2_controls::Down], 1, "DOWN kept");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_deadzone", "0,20"), "comma decimal");
            t.Equals(fieldValue(controls, "controls_deadzone_left"), std::string("0.200"), "deadzone 0.200");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_trigger", "off"), "trigger off");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_trigger", "0"), "trigger 0 is off");
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_trigger", "null"), "trigger null is off");
            t.IsTrue(controls.triggerThreshold < 0.f, "threshold stays off");
        });

        tc.Run("radial deadzone invert and swap", [](TestCase &t)
               {
            float x = 0.1f;
            float y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(x == 0.f && y == 0.f, "inside deadzone");
            x = -0.1f;
            y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(x == 0.f && y == 0.f, "negative inside deadzone");
            x = 0.f;
            y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(x == 0.f && y == 0.f, "origin");
            x = 1.f;
            y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(nearf(x, 1.f) && y == 0.f, "full deflection");
            x = -1.f;
            y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(nearf(x, -1.f) && y == 0.f, "full left");
            x = 0.5f;
            y = 0.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(nearf(x, 0.4117647f), "half rescaled");
            x = 3.f;
            y = 4.f;
            ps2_controls::applyRadialDeadzone(x, y, 0.15f);
            t.IsTrue(nearf(x, 0.6f) && nearf(y, 0.8f), "over-range unit clamp");
            x = 0.25f;
            y = -0.4f;
            ps2_controls::applyRadialDeadzone(x, y, 0.f);
            t.IsTrue(nearf(x, 0.25f) && nearf(y, -0.4f), "zero deadzone keeps the vector");

            ps2_controls::Snap snap{};
            snap.deadzoneLeft = 0.f;
            snap.deadzoneRight = 0.f;
            snap.invertRight = true;
            ps2_controls::StickFloats sticks;
            sticks.ry = 0.5f;
            ps2_controls::invertAndDeadzone(snap, sticks);
            t.IsTrue(nearf(sticks.ry, -0.5f) && sticks.ly == 0.f, "invert right only");

            snap = ps2_controls::Snap{};
            snap.invertLeft = true;
            snap.deadzoneLeft = 0.15f;
            sticks = ps2_controls::StickFloats{};
            sticks.ly = 0.5f;
            ps2_controls::invertAndDeadzone(snap, sticks);
            t.IsTrue(nearf(sticks.ly, -0.4117647f), "invert then deadzone");

            snap = ps2_controls::Snap{};
            snap.deadzoneLeft = 0.f;
            snap.deadzoneRight = 0.f;
            snap.stickSwap = true;
            snap.invertRight = true;
            sticks = ps2_controls::StickFloats{};
            sticks.lx = 0.2f;
            sticks.ly = 0.3f;
            sticks.rx = -0.4f;
            sticks.ry = 0.8f;
            ps2_controls::swapSticksIf(snap, sticks);
            ps2_controls::invertAndDeadzone(snap, sticks);
            t.IsTrue(nearf(sticks.lx, -0.4f) && nearf(sticks.ly, 0.8f) && nearf(sticks.rx, 0.2f) && nearf(sticks.ry, -0.3f),
                     "swap then invert right");
        });

        tc.Run("gamepad map and trigger threshold", [](TestCase &t)
               {
            ps2_controls::Controls controls = ps2_controls::defaultControls();
            ps2_controls::Snap snap;
            ps2_controls::fillSnap(snap, controls);
            bool down[18] = {};
            float axes[6] = {};
            bool signedL = false;
            bool signedR = false;
            ps2_controls::StickFloats sticks;
            down[7] = true;
            uint16_t pressed = ps2_controls::mapGamepad(snap, down, axes, signedL, signedR, sticks, true);
            t.Equals(pressed, static_cast<uint16_t>(0x4000), "face down is cross");
            down[7] = false;
            axes[0] = 0.4f;
            pressed = ps2_controls::mapGamepad(snap, down, axes, signedL, signedR, sticks, false);
            t.Equals(pressed, static_cast<uint16_t>(0), "blocked buttons");
            t.IsTrue(nearf(sticks.lx, 0.4f), "sticks still copy");

            ps2_controls::applyLayout(controls, ps2_controls::Layout::SwitchLabels);
            ps2_controls::fillSnap(snap, controls);
            down[6] = true;
            axes[0] = 0.f;
            pressed = ps2_controls::mapGamepad(snap, down, axes, signedL, signedR, sticks, true);
            t.Equals(pressed, static_cast<uint16_t>(0x4000), "nintendo A is cross");
            down[6] = false;
            down[7] = true;
            pressed = ps2_controls::mapGamepad(snap, down, axes, signedL, signedR, sticks, true);
            t.Equals(pressed, static_cast<uint16_t>(0x2000), "nintendo B is circle");

            controls = ps2_controls::defaultControls();
            ps2_controls::fillSnap(snap, controls);
            bool digital[18] = {};
            digital[10] = true;
            float idle[6] = {};
            signedL = false;
            pressed = ps2_controls::mapGamepad(snap, digital, idle, signedL, signedR, sticks, true);
            t.Equals(pressed & static_cast<uint16_t>(0x0100), static_cast<uint16_t>(0x0100), "digital L2");
            digital[10] = false;
            idle[4] = 1.f;
            pressed = ps2_controls::mapGamepad(snap, digital, idle, signedL, signedR, sticks, true);
            t.Equals(pressed & static_cast<uint16_t>(0x0100), static_cast<uint16_t>(0), "axis ignored while digital");

            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_gp_L2", "LEFT_TRIGGER_AXIS"), "bind axis");
            ps2_controls::fillSnap(snap, controls);
            idle[4] = 1.f;
            signedL = false;
            pressed = ps2_controls::mapGamepad(snap, digital, idle, signedL, signedR, sticks, true);
            t.Equals(pressed & static_cast<uint16_t>(0x0100), static_cast<uint16_t>(0x0100), "axis at 1 uses 0.5");
            idle[4] = 0.4f;
            pressed = ps2_controls::mapGamepad(snap, digital, idle, signedL, signedR, sticks, true);
            t.Equals(pressed & static_cast<uint16_t>(0x0100), static_cast<uint16_t>(0), "axis 0.4 below 0.5");

            controls = ps2_controls::defaultControls();
            t.IsTrue(ps2_controls::applyControlsField(controls, "controls_trigger", "0.4"), "threshold 0.4");
            t.Equals(fieldValue(controls, "controls_trigger"), std::string("0.400"), "trigger field");
            ps2_controls::fillSnap(snap, controls);
            bool sawAxis = false;
            bool sawDigital = false;
            for (int i = 0; i < snap.gpCount; ++i)
            {
                if (snap.gp[i].mask != 0x0100u)
                    continue;
                if (snap.gp[i].src == ps2_controls::kSrcAxisLeft)
                    sawAxis = true;
                if (snap.gp[i].src == ps2_controls::kGpLeftTrigger2)
                    sawDigital = true;
            }
            t.IsTrue(sawAxis, "L2 rewritten to the axis");
            t.IsFalse(sawDigital, "digital L2 removed");
            float trig[6] = {};
            bool none[18] = {};
            signedL = false;
            trig[4] = 1.f;
            pressed = ps2_controls::mapGamepad(snap, none, trig, signedL, signedR, sticks, true);
            t.IsTrue((pressed & 0x0100u) != 0 && !signedL, "unsigned full press");
            trig[4] = -1.f;
            pressed = ps2_controls::mapGamepad(snap, none, trig, signedL, signedR, sticks, true);
            t.IsTrue((pressed & 0x0100u) == 0 && signedL, "signed rest is clear");
            trig[4] = 0.f;
            pressed = ps2_controls::mapGamepad(snap, none, trig, signedL, signedR, sticks, true);
            t.IsTrue((pressed & 0x0100u) != 0 && signedL, "signed centre is half");
            trig[4] = -0.3f;
            pressed = ps2_controls::mapGamepad(snap, none, trig, signedL, signedR, sticks, true);
            t.IsTrue((pressed & 0x0100u) == 0, "signed 0.35 is under 0.4");
        });

        tc.Run("settings text round-trips", [](TestCase &t)
               {
            bool ok = false;
            std::string why;
            const ps2_controls::Controls defaults = ps2_controls::defaultControls();
            const ps2_controls::Controls back = roundTrip(defaults, ok);
            t.IsTrue(ok, "default text fits");
            t.IsTrue(controlsFieldsEqual(defaults, back, why), why.empty() ? "default round trip" : why);

            ps2_controls::Controls modified = ps2_controls::defaultControls();
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_layout", "switch-labels"), "layout");
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_deadzone", "0.25"), "deadzone");
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_invert_y", "both"), "invert");
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_stick_swap", "1"), "swap");
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_key_CROSS", "DOWN,C,SPACE"), "cross");
            t.IsTrue(ps2_controls::applyControlsField(modified, "controls_key_CIRCLE", "RIGHT,X,ESCAPE"), "circle");
            const ps2_controls::Controls again = roundTrip(modified, ok);
            t.IsTrue(ok, "modified text fits");
            t.IsTrue(controlsFieldsEqual(modified, again, why), why.empty() ? "modified round trip" : why);
            t.IsTrue(again.ignoreGamepads == false, "switch-labels still uses gamepads");
            t.Equals(again.gp[ps2_controls::Cross][0], 6, "round trip keeps nintendo A");
        });
                   });
}
