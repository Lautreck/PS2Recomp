#include "runtime/ps2_pad.h"
#include "runtime/ps2_graphics_settings.h"
#include "runtime/ps2_input_latency.h"
#include "ps2_host_backend.h"
#include <cstring>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    // Port 0's historical byte: 128 + axis*127 (full left is 1, not 0).
    uint8_t stickByte(float v)
    {
        if (v < -1.f)
            v = -1.f;
        else if (v > 1.f)
            v = 1.f;
        return static_cast<uint8_t>(128.f + v * 127.f);
    }

    bool g_triggerSignedL = false;
    bool g_triggerSignedR = false;

    void overlayKeySticks(const ps2_controls::Snap &snap, const ps2_controls::StickFloats &keySticks, bool active,
                          ps2_controls::StickFloats &sticks)
    {
        if (!active)
            return;
        if (keySticks.lx != 0.f || keySticks.ly != 0.f)
        {
            sticks.lx = keySticks.lx;
            sticks.ly = keySticks.ly;
        }
        if (keySticks.rx != 0.f || keySticks.ry != 0.f)
        {
            sticks.rx = keySticks.rx;
            sticks.ry = keySticks.ry;
        }
        (void)snap;
    }
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    const ps2_controls::Snap *snap = ps2_controls::publishedSnap();
    if (!snap)
        return true;

    uint16_t pressed = 0;
    ps2_controls::StickFloats sticks;
    const bool blockButtons = ps2_input_latency::gamepadButtonsBlocked();
    if (!snap->ignoreGamepads && IsGamepadAvailable(0))
    {
        bool down[18] = {};
        if (!blockButtons)
        {
            for (int b = 1; b <= 17; ++b)
                down[b] = IsGamepadButtonDown(0, b);
        }
        float axes[6];
        for (int a = 0; a < 6; ++a)
            axes[a] = GetGamepadAxisMovement(0, a);
        pressed = ps2_controls::mapGamepad(*snap, down, axes, g_triggerSignedL, g_triggerSignedR, sticks, !blockButtons);
    }
    ps2_controls::swapSticksIf(*snap, sticks);

    if (!ps2_input_latency::padPostingSuppressed())
    {
        uint16_t hostKeys = 0;
        float klx = 0.f, kly = 0.f, krx = 0.f, kry = 0.f;
        bool keyStick = false;
        if (ps2_input_latency::sampleKeyboard(hostKeys, &klx, &kly, &krx, &kry, &keyStick))
        {
            pressed = static_cast<uint16_t>(pressed | hostKeys);
            ps2_controls::StickFloats fromKeys;
            fromKeys.lx = klx;
            fromKeys.ly = kly;
            fromKeys.rx = krx;
            fromKeys.ry = kry;
            overlayKeySticks(*snap, fromKeys, keyStick, sticks);
        }
        else
        {
            for (int i = 0; i < snap->rlCount; ++i)
            {
                if (snap->rlKeys[i].key > 0 && IsKeyDown(snap->rlKeys[i].key))
                    pressed = static_cast<uint16_t>(pressed | snap->rlKeys[i].mask);
            }
            ps2_controls::StickFloats fromKeys;
            bool any = false;
            for (int i = 0; i < snap->stickCount; ++i)
            {
                const ps2_controls::Snap::StickBind &bind = snap->sticks[i];
                if (bind.key > 0 && IsKeyDown(bind.key))
                {
                    if (bind.stick == 0 && bind.axis == 0)
                        fromKeys.lx += bind.value;
                    else if (bind.stick == 0)
                        fromKeys.ly += bind.value;
                    else if (bind.axis == 0)
                        fromKeys.rx += bind.value;
                    else
                        fromKeys.ry += bind.value;
                    any = true;
                }
            }
            overlayKeySticks(*snap, fromKeys, any, sticks);
        }
    }

    ps2_controls::invertAndDeadzone(*snap, sticks);
    const uint16_t activeLow = static_cast<uint16_t>(~pressed);
    data[2] = static_cast<uint8_t>(activeLow & 0xFF);
    data[3] = static_cast<uint8_t>(activeLow >> 8);
    data[6] = stickByte(sticks.lx);
    data[7] = stickByte(sticks.ly);
    data[4] = stickByte(sticks.rx);
    data[5] = stickByte(sticks.ry);
    return true;
}
