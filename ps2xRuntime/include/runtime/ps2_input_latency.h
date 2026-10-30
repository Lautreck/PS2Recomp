#pragma once

#include <cstdint>

// Host input-latency path. The QPC log and frame hash run only when
// TEKKEN4_INPUT_LAG=1. t1 (key arrival) and t4 (first swap after the guest
// read the key) are always stamped, with no hash and no work when no press
// is open, so the HUD can show latency. TEKKEN4_INPUT_LEGACY=1 keeps the old
// once-per-host-frame raylib poll. The fast path samples posted keys from the
// window thread and physical keys with GetAsyncKeyState at pad-read time.
namespace ps2_input_latency
{
    bool enabled();
    bool legacy();
    // Win32 window hook is installed (F3 is handled there).
    bool hookInstalled();
    // "paced" (default): present on a 16.667 ms clock.
    // "tick" (TEKKEN4_INPUT_PRESENT=tick): present when the guest vsync tick
    // changes. Measured slower (readback lands while the GS is still drawing).
    // Ignored when legacy() is set.
    bool presentOnTick();

    void attachWindow(void *hwnd);
    void detachWindow();

    // Window thread. Returns after a queued input message or timeoutMs.
    void waitMessage(unsigned timeoutMs);

    // Game thread. Active-high pad bits. False: caller uses raylib IsKeyDown.
    // Stick pointers are filled only when a keyboard stick binding is down.
    // While pad posting is suppressed, returns true with an empty mask so the
    // legacy IsKeyDown path does not run either.
    bool sampleKeyboard(uint16_t &padDownMask, float *stickLx = nullptr, float *stickLy = nullptr,
                        float *stickRx = nullptr, float *stickRy = nullptr, bool *stickActive = nullptr);

    // Rebind capture or an ImGui text field. The hook does not post pad bits.
    void setPadPostingSuppressed(bool suppressed);
    bool padPostingSuppressed();
    // Rebind capture: gamepad buttons are not applied. Sticks still are.
    void setGamepadButtonsBlocked(bool blocked);
    bool gamepadButtonsBlocked();

    // Game thread, after scePadRead wrote the pad packet. pressedMask is active-high.
    void notePad(uint16_t pressedMask, uint64_t vsyncTick);

    // Window thread, around the latch that actually calls GS present.
    void onLatchBegin(uint64_t vsyncTick);
    void onLatchPixels(const uint8_t *rgba, uint32_t width, uint32_t height);
    void onSwapReturned();
}
