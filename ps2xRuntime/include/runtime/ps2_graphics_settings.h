#pragma once

// User graphics settings: the F2 settings window (ps2_debug_panel.cpp) changes them while the game runs, the host
// present path and the GS backend read them. PS2X_SETTINGS=<file> loads them at start and saves every change; without
// it the defaults hold and nothing is written (test runs stay reproducible).
//
// Controls (padmap-compatible, Windows): layout preset, per-button gamepad and keyboard bindings, radial deadzone,
// invert Y, stick swap, analog trigger threshold. The pad read path uses a published snapshot, not this mutex.
#include <cstdint>

namespace ps2_controls
{
    constexpr int kButtonCount = 16;
    constexpr int kStickDirCount = 8;
    constexpr int kMaxSources = 4;
    constexpr int kMaxKeys = 6;

    // DS2 order used by the settings file and the F2 list.
    enum Button : int
    {
        Up = 0,
        Down,
        Left,
        Right,
        Cross,
        Circle,
        Square,
        Triangle,
        L1,
        R1,
        L2,
        R2,
        L3,
        R3,
        Start,
        Select
    };

    // Active-high bits, same positions as the active-low report.
    constexpr uint16_t kButtonMask[kButtonCount] = {
        0x0010u, 0x0040u, 0x0080u, 0x0020u, 0x4000u, 0x2000u, 0x8000u, 0x1000u,
        0x0400u, 0x0800u, 0x0100u, 0x0200u, 0x0002u, 0x0004u, 0x0008u, 0x0001u};

    enum StickDir : int
    {
        LStickUp = 0,
        LStickDown,
        LStickLeft,
        LStickRight,
        RStickUp,
        RStickDown,
        RStickLeft,
        RStickRight
    };

    enum class Layout : int
    {
        Playstation = 0,
        Xbox = 1,
        Switch = 2,
        SwitchLabels = 3,
        Keyboard = 4
    };

    // Gamepad source: raylib GamepadButton 1..17, or a trigger axis (raylib 5.5 values).
    constexpr int kSrcEmpty = 0;
    constexpr int kSrcAxisLeft = -5;  // -(GAMEPAD_AXIS_LEFT_TRIGGER + 1)
    constexpr int kSrcAxisRight = -6; // -(GAMEPAD_AXIS_RIGHT_TRIGGER + 1)
    constexpr int kGpLeftTrigger2 = 10;
    constexpr int kGpRightTrigger2 = 12;

    struct Controls
    {
        Layout layout = Layout::Playstation;
        bool ignoreGamepads = false;
        int gp[kButtonCount][kMaxSources]{};
        int gpCount[kButtonCount]{};
        int keys[kButtonCount][kMaxKeys]{}; // raylib KeyboardKey
        int keyCount[kButtonCount]{};
        int stickKeys[kStickDirCount][kMaxSources]{};
        int stickKeyCount[kStickDirCount]{};
        float deadzoneLeft = 0.15f;
        float deadzoneRight = 0.15f;
        bool invertLeft = false;
        bool invertRight = false; // padmap invert_y true
        bool stickSwap = false;
        float triggerThreshold = -1.f; // < 0: digital trigger buttons
    };

    struct StickFloats
    {
        float lx = 0.f;
        float ly = 0.f;
        float rx = 0.f;
        float ry = 0.f;
    };

    struct Snap
    {
        uint16_t vkMask[256]{};
        uint8_t pollVk[96]{};
        int pollCount = 0;
        struct RlKey
        {
            int key;
            uint16_t mask;
        };
        RlKey rlKeys[96]{};
        int rlCount = 0;
        struct GpSrc
        {
            int src;
            uint16_t mask;
        };
        GpSrc gp[64]{};
        int gpCount = 0;
        float deadzoneLeft = 0.15f;
        float deadzoneRight = 0.15f;
        bool invertLeft = false;
        bool invertRight = false;
        bool stickSwap = false;
        bool ignoreGamepads = false;
        float triggerThreshold = -1.f;
        struct StickBind
        {
            int key;
            int vk;
            int stick;
            int axis;
            float value;
        };
        StickBind sticks[32]{};
        int stickCount = 0;
    };

    Controls defaultControls();
    // Resets gamepad sources to the preset. Keyboard bindings are left alone.
    // Keyboard layout sets ignoreGamepads.
    void applyLayout(Controls &controls, Layout layout);
    int presetSource(Layout layout, int button);

    const char *layoutName(Layout layout);
    const char *buttonName(int button);
    const char *stickDirName(int dir);
    const char *sourceName(int src);
    const char *keyName(int raylibKey);

    int catalogKeyCount();
    int catalogKeyRaylib(int index);

    // Unknown keys and bad values return false and leave controls unchanged.
    // An empty list (controls_key_UP=) is valid and clears that binding.
    bool applyControlsField(Controls &controls, const char *key, const char *value);
    // Each line is key=value. Unknown lines are skipped. Returns how many fields applied.
    int applyControlsText(Controls &controls, const char *text);

    // One settings-file field. Returns false when index is past the end.
    bool controlsField(const Controls &controls, int index, char *key, int keyCap, char *value, int valueCap);

    void fillSnap(Snap &snap, const Controls &controls);
    const Snap *publishedSnap();
    void publishControls(const Controls &controls);

    void applyRadialDeadzone(float &x, float &y, float deadzone);
    float triggerAmount(float raw, bool &signedRange);
    // buttonsLive false: sticks only (a rebind press must not reach the game).
    // down[18] indexed by raylib GamepadButton. axes[6] are raylib axes.
    uint16_t mapGamepad(const Snap &snap, const bool down[18], const float axes[6], bool &signedL, bool &signedR,
                        StickFloats &sticks, bool buttonsLive = true);
    void swapSticksIf(const Snap &snap, StickFloats &sticks);
    void invertAndDeadzone(const Snap &snap, StickFloats &sticks);
}

namespace ps2_graphics
{
    enum class Aspect : int
    {
        Ratio4x3 = 0,  // a PS2 on a 4:3 TV (PCSX2's default)
        Ratio16x9 = 1, // stretches a 4:3 picture unless the game renders widescreen
        Stretch = 2,   // fill the window
        Native = 3,    // the frame's own pixel ratio (640x448 -> 1.43:1), the old behaviour
    };

    enum class TextureFilter : int
    {
        Game = 0,     // the game's TEX1 filter (the PS2's look)
        Bilinear = 1, // forced for 3D textures, sprites keep the game's
    };

    struct Settings
    {
        Aspect aspect = Aspect::Ratio4x3;
        bool smoothScreen = false; // bilinear filtering when the frame is scaled to the window
        int internalScale = 1;     // GL GS renderer resolution, 1..4 x the PS2's (PS2X_GS_HW_SCALE wins until changed)
        bool widescreen = false;   // runtime switch "widescreen": the game's 16:9 patch ([patches] in the recompiler config)
        bool noInterlace = false;  // runtime switch "no_interlace": PCSX2 No-Interlacing (boot GS mode 480p; removes field bob)
        TextureFilter textureFilter = TextureFilter::Game;
        bool fxaa = false;         // FXAA on the frame as it is drawn to the window
        int sharpen = 0;           // 0..100 %, contrast-limited sharpening in the same pass
        ps2_controls::Controls controls = ps2_controls::defaultControls();
    };
    constexpr int kMaxInternalScale = 4;

    Settings get();
    void set(const Settings &settings);

    // Width / height of the picture on screen for a window of that size (Stretch -> the window's own ratio).
    float displayAspect(const Settings &settings, float frameWidth, float frameHeight, float windowWidth,
                        float windowHeight);
}
