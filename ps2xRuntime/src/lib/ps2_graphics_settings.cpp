#include "runtime/ps2_graphics_settings.h"
#include "runtime/ps2_switches.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace ps2_controls
{
    namespace
    {
        struct KeyInfo
        {
            const char *name;
            int raylib;
            int vk;
        };

        // raylib 5.5 KeyboardKey values (cmake pins raylib GIT_TAG 5.5) and the Win32 VK the hook sees.
        // VK 0: no fast-path code (KP_ENTER would alias Enter).
        constexpr KeyInfo kKeys[] = {
            {"APOSTROPHE", 39, 0xDE}, {"COMMA", 44, 0xBC}, {"MINUS", 45, 0xBD}, {"PERIOD", 46, 0xBE},
            {"SLASH", 47, 0xBF},      {"ZERO", 48, '0'},   {"ONE", 49, '1'},     {"TWO", 50, '2'},
            {"THREE", 51, '3'},       {"FOUR", 52, '4'},   {"FIVE", 53, '5'},    {"SIX", 54, '6'},
            {"SEVEN", 55, '7'},       {"EIGHT", 56, '8'},  {"NINE", 57, '9'},    {"SEMICOLON", 59, 0xBA},
            {"EQUAL", 61, 0xBB},      {"A", 65, 'A'},      {"B", 66, 'B'},       {"C", 67, 'C'},
            {"D", 68, 'D'},           {"E", 69, 'E'},      {"F", 70, 'F'},       {"G", 71, 'G'},
            {"H", 72, 'H'},           {"I", 73, 'I'},      {"J", 74, 'J'},       {"K", 75, 'K'},
            {"L", 76, 'L'},           {"M", 77, 'M'},      {"N", 78, 'N'},       {"O", 79, 'O'},
            {"P", 80, 'P'},           {"Q", 81, 'Q'},      {"R", 82, 'R'},       {"S", 83, 'S'},
            {"T", 84, 'T'},           {"U", 85, 'U'},      {"V", 86, 'V'},       {"W", 87, 'W'},
            {"X", 88, 'X'},           {"Y", 89, 'Y'},      {"Z", 90, 'Z'},       {"LEFT_BRACKET", 91, 0xDB},
            {"BACKSLASH", 92, 0xDC},  {"RIGHT_BRACKET", 93, 0xDD}, {"GRAVE", 96, 0xC0}, {"SPACE", 32, 0x20},
            {"ESCAPE", 256, 0x1B},    {"ENTER", 257, 0x0D}, {"TAB", 258, 0x09},  {"BACKSPACE", 259, 0x08},
            {"INSERT", 260, 0x2D},    {"DELETE", 261, 0x2E}, {"RIGHT", 262, 0x27}, {"LEFT", 263, 0x25},
            {"DOWN", 264, 0x28},      {"UP", 265, 0x26},   {"PAGE_UP", 266, 0x21}, {"PAGE_DOWN", 267, 0x22},
            {"HOME", 268, 0x24},      {"END", 269, 0x23},  {"CAPS_LOCK", 280, 0x14}, {"SCROLL_LOCK", 281, 0x91},
            {"NUM_LOCK", 282, 0x90},  {"PRINT_SCREEN", 283, 0x2C}, {"PAUSE", 284, 0x13}, {"F1", 290, 0x70},
            {"F2", 291, 0x71},        {"F3", 292, 0x72},   {"F4", 293, 0x73},    {"F5", 294, 0x74},
            {"F6", 295, 0x75},        {"F7", 296, 0x76},   {"F8", 297, 0x77},    {"F9", 298, 0x78},
            {"F10", 299, 0x79},       {"F11", 300, 0x7A},  {"F12", 301, 0x7B},   {"LEFT_SHIFT", 340, 0xA0},
            {"LEFT_CONTROL", 341, 0xA2}, {"LEFT_ALT", 342, 0xA4}, {"LEFT_SUPER", 343, 0x5B},
            {"RIGHT_SHIFT", 344, 0xA1}, {"RIGHT_CONTROL", 345, 0xA3}, {"RIGHT_ALT", 346, 0xA5},
            {"RIGHT_SUPER", 347, 0x5C}, {"KB_MENU", 348, 0x5D}, {"KP_0", 320, 0x60}, {"KP_1", 321, 0x61},
            {"KP_2", 322, 0x62},      {"KP_3", 323, 0x63}, {"KP_4", 324, 0x64},  {"KP_5", 325, 0x65},
            {"KP_6", 326, 0x66},      {"KP_7", 327, 0x67}, {"KP_8", 328, 0x68},  {"KP_9", 329, 0x69},
            {"KP_DECIMAL", 330, 0x6E}, {"KP_DIVIDE", 331, 0x6F}, {"KP_MULTIPLY", 332, 0x6A},
            {"KP_SUBTRACT", 333, 0x6D}, {"KP_ADD", 334, 0x6B}, {"KP_ENTER", 335, 0}, {"KP_EQUAL", 336, 0},
            {"BACK", 4, 0},           {"MENU", 5, 0},      {"VOLUME_UP", 24, 0}, {"VOLUME_DOWN", 25, 0},
        };
        constexpr int kKeyCatalog = static_cast<int>(sizeof(kKeys) / sizeof(kKeys[0]));

        struct GpName
        {
            const char *name;
            int src;
        };
        constexpr GpName kGpNames[] = {
            {"LEFT_FACE_UP", 1},       {"LEFT_FACE_RIGHT", 2},    {"LEFT_FACE_DOWN", 3},
            {"LEFT_FACE_LEFT", 4},     {"RIGHT_FACE_UP", 5},      {"RIGHT_FACE_RIGHT", 6},
            {"RIGHT_FACE_DOWN", 7},    {"RIGHT_FACE_LEFT", 8},    {"LEFT_TRIGGER_1", 9},
            {"LEFT_TRIGGER_2", 10},    {"RIGHT_TRIGGER_1", 11},   {"RIGHT_TRIGGER_2", 12},
            {"MIDDLE_LEFT", 13},       {"MIDDLE", 14},            {"MIDDLE_RIGHT", 15},
            {"LEFT_THUMB", 16},        {"RIGHT_THUMB", 17},       {"LEFT_TRIGGER_AXIS", kSrcAxisLeft},
            {"RIGHT_TRIGGER_AXIS", kSrcAxisRight},
        };

        constexpr const char *kButtonNames[kButtonCount] = {
            "UP", "DOWN", "LEFT", "RIGHT", "CROSS", "CIRCLE", "SQUARE", "TRIANGLE",
            "L1", "R1", "L2", "R2", "L3", "R3", "START", "SELECT"};
        constexpr const char *kStickNames[kStickDirCount] = {
            "LSTICK_UP", "LSTICK_DOWN", "LSTICK_LEFT", "LSTICK_RIGHT",
            "RSTICK_UP", "RSTICK_DOWN", "RSTICK_LEFT", "RSTICK_RIGHT"};
        // stick, axis (0=x, 1=y), value. Matches padmap STICK_DIRS.
        constexpr int kStickStick[kStickDirCount] = {0, 0, 0, 0, 1, 1, 1, 1};
        constexpr int kStickAxis[kStickDirCount] = {1, 1, 0, 0, 1, 1, 0, 0};
        constexpr float kStickValue[kStickDirCount] = {-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f};

        // Today's ps2_pad.cpp / vkToPad bindings. Not padmap's keyboard preset (that one drives sticks).
        struct DefKey
        {
            int button;
            int raylib;
        };
        constexpr DefKey kDefaultKeys[] = {
            {Up, 87},        {Down, 83},       {Left, 65},        {Right, 68},
            {Cross, 264},    {Cross, 88},      {Cross, 32},
            {Circle, 262},   {Circle, 67},     {Circle, 256},
            {Square, 263},   {Square, 90},     {Square, 320},
            {Triangle, 265}, {Triangle, 86},   {Triangle, 321},
            {L1, 81},        {R1, 69},         {L2, 340},         {R2, 344},
            {Start, 257},    {Select, 258},
        };

        constexpr int kVkShift = 0x10;
        constexpr int kVkLShift = 0xA0;

        Snap g_snaps[3];
        std::atomic<int> g_snapIndex{0};
        std::once_flag g_snapOnce;

        int asciiUpper(int c)
        {
            if (c >= 'a' && c <= 'z')
                return c - ('a' - 'A');
            return c;
        }

        bool sameName(const char *a, const char *b)
        {
            if (!a || !b)
                return false;
            while (*a && *b)
            {
                if (asciiUpper(static_cast<unsigned char>(*a)) != asciiUpper(static_cast<unsigned char>(*b)))
                    return false;
                ++a;
                ++b;
            }
            return *a == '\0' && *b == '\0';
        }

        void trimInPlace(char *s)
        {
            if (!s)
                return;
            char *start = s;
            while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n')
                ++start;
            if (start != s)
                std::memmove(s, start, std::strlen(start) + 1);
            std::size_t n = std::strlen(s);
            while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t' || s[n - 1] == '\r' || s[n - 1] == '\n'))
            {
                s[n - 1] = '\0';
                --n;
            }
        }

        const KeyInfo *findKeyName(const char *name)
        {
            if (!name || !name[0])
                return nullptr;
            if ((name[0] == 'K' || name[0] == 'k') && (name[1] == 'E' || name[1] == 'e') &&
                (name[2] == 'Y' || name[2] == 'y') && name[3] == '_')
                name += 4;
            for (const KeyInfo &info : kKeys)
            {
                if (sameName(info.name, name))
                    return &info;
            }
            return nullptr;
        }

        const KeyInfo *findKeyCode(int raylib)
        {
            for (const KeyInfo &info : kKeys)
            {
                if (info.raylib == raylib)
                    return &info;
            }
            return nullptr;
        }

        int findGpName(const char *name, int &src)
        {
            if (!name || !name[0])
                return 0;
            if ((name[0] == 'G' || name[0] == 'g') && sameName(name, "GAMEPAD_BUTTON_UNKNOWN"))
                return 0;
            const char *body = name;
            const char prefix[] = "GAMEPAD_BUTTON_";
            bool prefixed = true;
            for (int i = 0; prefix[i]; ++i)
            {
                if (asciiUpper(static_cast<unsigned char>(name[i])) != prefix[i])
                {
                    prefixed = false;
                    break;
                }
            }
            if (prefixed)
                body = name + sizeof(prefix) - 1;
            for (const GpName &gp : kGpNames)
            {
                if (sameName(gp.name, body))
                {
                    src = gp.src;
                    return 1;
                }
            }
            return 0;
        }

        int findButtonName(const char *name)
        {
            for (int i = 0; i < kButtonCount; ++i)
            {
                if (sameName(kButtonNames[i], name))
                    return i;
            }
            return -1;
        }

        int findStickName(const char *name)
        {
            for (int i = 0; i < kStickDirCount; ++i)
            {
                if (sameName(kStickNames[i], name))
                    return i;
            }
            return -1;
        }

        int positionalSource(int button)
        {
            switch (button)
            {
            case Up: return 1;
            case Down: return 3;
            case Left: return 4;
            case Right: return 2;
            case Cross: return 7;
            case Circle: return 6;
            case Square: return 8;
            case Triangle: return 5;
            case L1: return 9;
            case R1: return 11;
            case L2: return 10;
            case R2: return 12;
            case L3: return 16;
            case R3: return 17;
            case Start: return 15;
            case Select: return 13;
            default: return kSrcEmpty;
            }
        }

        bool parseUnit(const char *text, float &out)
        {
            if (!text || !text[0])
                return false;
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s", text);
            trimInPlace(buf);
            for (char *p = buf; *p; ++p)
            {
                if (*p == ',')
                    *p = '.';
            }
            char *end = nullptr;
            const float v = std::strtof(buf, &end);
            if (end == buf)
                return false;
            while (*end == ' ' || *end == '\t')
                ++end;
            if (*end != '\0')
                return false;
            out = v;
            return true;
        }

        void formatUnit(char *dst, int cap, float v)
        {
            if (!(v > -10.f && v < 10.f))
                v = 0.f;
            const bool neg = v < 0.f;
            const float mag = neg ? -v : v;
            int milli = static_cast<int>(std::lround(mag * 1000.f));
            if (milli < 0)
                milli = 0;
            std::snprintf(dst, static_cast<std::size_t>(cap), "%s%d.%03d", neg ? "-" : "", milli / 1000, milli % 1000);
        }

        bool parseList(const char *text, int *out, int max, int &count, bool keys)
        {
            count = 0;
            if (!text)
                return false;
            char buf[192];
            std::snprintf(buf, sizeof(buf), "%s", text);
            trimInPlace(buf);
            if (buf[0] == '\0')
                return true;
            char *cursor = buf;
            while (*cursor)
            {
                while (*cursor == ' ' || *cursor == '\t' || *cursor == ',')
                    ++cursor;
                if (!*cursor)
                    break;
                char *start = cursor;
                while (*cursor && *cursor != ',')
                    ++cursor;
                if (*cursor == ',')
                {
                    *cursor = '\0';
                    ++cursor;
                }
                trimInPlace(start);
                if (!start[0])
                    continue;
                if (count >= max)
                    return false;
                if (keys)
                {
                    const KeyInfo *info = findKeyName(start);
                    if (!info)
                        return false;
                    out[count++] = info->raylib;
                }
                else
                {
                    int src = 0;
                    if (!findGpName(start, src))
                        return false;
                    out[count++] = src;
                }
            }
            return true;
        }

        void formatKeyList(const int *keys, int count, char *dst, int cap)
        {
            dst[0] = '\0';
            int used = 0;
            for (int i = 0; i < count; ++i)
            {
                const char *name = keyName(keys[i]);
                if (!name)
                    continue;
                const int n = std::snprintf(dst + used, static_cast<std::size_t>(cap - used), "%s%s",
                                            used ? "," : "", name);
                if (n < 0 || used + n >= cap)
                {
                    dst[0] = '\0';
                    return;
                }
                used += n;
            }
        }

        void formatGpList(const int *src, int count, char *dst, int cap)
        {
            dst[0] = '\0';
            int used = 0;
            for (int i = 0; i < count; ++i)
            {
                const char *name = sourceName(src[i]);
                if (!name || src[i] == kSrcEmpty)
                    continue;
                const int n = std::snprintf(dst + used, static_cast<std::size_t>(cap - used), "%s%s",
                                            used ? "," : "", name);
                if (n < 0 || used + n >= cap)
                {
                    dst[0] = '\0';
                    return;
                }
                used += n;
            }
        }

        const char *invertToken(const Controls &controls)
        {
            if (controls.invertLeft && controls.invertRight)
                return "both";
            if (controls.invertLeft)
                return "left";
            if (controls.invertRight)
                return "true";
            return "false";
        }

        void seedDefaultSnaps()
        {
            Snap snap;
            fillSnap(snap, defaultControls());
            g_snaps[0] = snap;
            g_snaps[1] = snap;
            g_snaps[2] = snap;
        }
    }

    Controls defaultControls()
    {
        Controls controls;
        applyLayout(controls, Layout::Playstation);
        controls.deadzoneLeft = 0.15f;
        controls.deadzoneRight = 0.15f;
        controls.invertLeft = false;
        controls.invertRight = false;
        controls.stickSwap = false;
        controls.triggerThreshold = -1.f;
        controls.ignoreGamepads = false;
        for (const DefKey &entry : kDefaultKeys)
        {
            const int b = entry.button;
            if (controls.keyCount[b] < kMaxKeys)
                controls.keys[b][controls.keyCount[b]++] = entry.raylib;
        }
        return controls;
    }

    void applyLayout(Controls &controls, Layout layout)
    {
        controls.layout = layout;
        controls.ignoreGamepads = layout == Layout::Keyboard;
        for (int b = 0; b < kButtonCount; ++b)
        {
            controls.gpCount[b] = 1;
            controls.gp[b][0] = presetSource(layout, b);
            for (int s = 1; s < kMaxSources; ++s)
                controls.gp[b][s] = kSrcEmpty;
        }
    }

    int presetSource(Layout layout, int button)
    {
        if (button < 0 || button >= kButtonCount)
            return kSrcEmpty;
        if (layout == Layout::SwitchLabels)
        {
            switch (button)
            {
            case Cross: return 6;     // RIGHT_FACE_RIGHT, Nintendo A
            case Circle: return 7;    // RIGHT_FACE_DOWN, Nintendo B
            case Square: return 5;    // RIGHT_FACE_UP, Nintendo X
            case Triangle: return 8;  // RIGHT_FACE_LEFT, Nintendo Y
            default: break;
            }
        }
        return positionalSource(button);
    }

    const char *layoutName(Layout layout)
    {
        switch (layout)
        {
        case Layout::Playstation: return "playstation";
        case Layout::Xbox: return "xbox";
        case Layout::Switch: return "switch";
        case Layout::SwitchLabels: return "switch-labels";
        case Layout::Keyboard: return "keyboard";
        }
        return "playstation";
    }

    const char *buttonName(int button)
    {
        if (button < 0 || button >= kButtonCount)
            return nullptr;
        return kButtonNames[button];
    }

    const char *stickDirName(int dir)
    {
        if (dir < 0 || dir >= kStickDirCount)
            return nullptr;
        return kStickNames[dir];
    }

    const char *sourceName(int src)
    {
        for (const GpName &gp : kGpNames)
        {
            if (gp.src == src)
                return gp.name;
        }
        return nullptr;
    }

    const char *keyName(int raylibKey)
    {
        const KeyInfo *info = findKeyCode(raylibKey);
        return info ? info->name : nullptr;
    }

    int catalogKeyCount()
    {
        return kKeyCatalog;
    }

    int catalogKeyRaylib(int index)
    {
        if (index < 0 || index >= kKeyCatalog)
            return 0;
        return kKeys[index].raylib;
    }

    bool applyControlsField(Controls &controls, const char *key, const char *value)
    {
        if (!key || !value)
            return false;
        char kbuf[96];
        char vbuf[192];
        std::snprintf(kbuf, sizeof(kbuf), "%s", key);
        std::snprintf(vbuf, sizeof(vbuf), "%s", value);
        trimInPlace(kbuf);
        trimInPlace(vbuf);
        const char *prefix = "controls_";
        if (std::strncmp(kbuf, prefix, 9) != 0)
            return false;
        const char *body = kbuf + 9;

        if (sameName(body, "layout"))
        {
            Layout layout = controls.layout;
            if (sameName(vbuf, "playstation"))
                layout = Layout::Playstation;
            else if (sameName(vbuf, "xbox"))
                layout = Layout::Xbox;
            else if (sameName(vbuf, "switch"))
                layout = Layout::Switch;
            else if (sameName(vbuf, "switch-labels") || sameName(vbuf, "switch_labels"))
                layout = Layout::SwitchLabels;
            else if (sameName(vbuf, "keyboard"))
                layout = Layout::Keyboard;
            else
                return false;
            applyLayout(controls, layout);
            return true;
        }
        if (sameName(body, "deadzone"))
        {
            float v = 0.f;
            if (!parseUnit(vbuf, v) || v < 0.f || v > 0.9f)
                return false;
            controls.deadzoneLeft = v;
            controls.deadzoneRight = v;
            return true;
        }
        if (sameName(body, "deadzone_left") || sameName(body, "deadzone_right"))
        {
            float v = 0.f;
            if (!parseUnit(vbuf, v) || v < 0.f || v > 0.9f)
                return false;
            if (sameName(body, "deadzone_left"))
                controls.deadzoneLeft = v;
            else
                controls.deadzoneRight = v;
            return true;
        }
        if (sameName(body, "invert_y"))
        {
            if (sameName(vbuf, "false") || sameName(vbuf, "none") || sameName(vbuf, "0"))
            {
                controls.invertLeft = false;
                controls.invertRight = false;
            }
            else if (sameName(vbuf, "true") || sameName(vbuf, "right") || sameName(vbuf, "1"))
            {
                controls.invertLeft = false;
                controls.invertRight = true;
            }
            else if (sameName(vbuf, "left"))
            {
                controls.invertLeft = true;
                controls.invertRight = false;
            }
            else if (sameName(vbuf, "both"))
            {
                controls.invertLeft = true;
                controls.invertRight = true;
            }
            else
                return false;
            return true;
        }
        if (sameName(body, "stick_swap"))
        {
            if (sameName(vbuf, "1") || sameName(vbuf, "true"))
                controls.stickSwap = true;
            else if (sameName(vbuf, "0") || sameName(vbuf, "false"))
                controls.stickSwap = false;
            else
                return false;
            return true;
        }
        if (sameName(body, "trigger") || sameName(body, "trigger_threshold"))
        {
            if (vbuf[0] == '\0' || sameName(vbuf, "off") || sameName(vbuf, "none") || sameName(vbuf, "null") ||
                sameName(vbuf, "-1"))
            {
                controls.triggerThreshold = -1.f;
                return true;
            }
            float v = 0.f;
            if (!parseUnit(vbuf, v))
                return false;
            if (v == 0.f)
            {
                controls.triggerThreshold = -1.f;
                return true;
            }
            if (!(v > 0.f && v <= 1.f))
                return false;
            controls.triggerThreshold = v;
            return true;
        }
        if (std::strncmp(body, "gp_", 3) == 0)
        {
            const int button = findButtonName(body + 3);
            if (button < 0)
                return false;
            int src[kMaxSources];
            int count = 0;
            if (!parseList(vbuf, src, kMaxSources, count, false))
                return false;
            controls.gpCount[button] = count;
            for (int i = 0; i < kMaxSources; ++i)
                controls.gp[button][i] = i < count ? src[i] : kSrcEmpty;
            return true;
        }
        if (std::strncmp(body, "key_", 4) == 0)
        {
            const char *name = body + 4;
            const int stick = findStickName(name);
            int codes[kMaxKeys];
            int count = 0;
            if (!parseList(vbuf, codes, stick >= 0 ? kMaxSources : kMaxKeys, count, true))
                return false;
            if (stick >= 0)
            {
                controls.stickKeyCount[stick] = count;
                for (int i = 0; i < kMaxSources; ++i)
                    controls.stickKeys[stick][i] = i < count ? codes[i] : 0;
                return true;
            }
            const int button = findButtonName(name);
            if (button < 0)
                return false;
            controls.keyCount[button] = count;
            for (int i = 0; i < kMaxKeys; ++i)
                controls.keys[button][i] = i < count ? codes[i] : 0;
            return true;
        }
        return false;
    }

    int applyControlsText(Controls &controls, const char *text)
    {
        if (!text)
            return 0;
        int applied = 0;
        const char *cursor = text;
        while (*cursor)
        {
            char line[256];
            int n = 0;
            while (*cursor && *cursor != '\n' && n < 255)
                line[n++] = *cursor++;
            line[n] = '\0';
            if (*cursor == '\n')
                ++cursor;
            trimInPlace(line);
            char *eq = std::strchr(line, '=');
            if (!eq)
                continue;
            *eq = '\0';
            if (applyControlsField(controls, line, eq + 1))
                ++applied;
        }
        return applied;
    }

    bool controlsField(const Controls &controls, int index, char *key, int keyCap, char *value, int valueCap)
    {
        if (!key || !value || keyCap < 8 || valueCap < 8 || index < 0)
            return false;
        value[0] = '\0';
        auto putKey = [&](const char *name)
        { std::snprintf(key, static_cast<std::size_t>(keyCap), "controls_%s", name); };
        if (index == 0)
        {
            putKey("layout");
            std::snprintf(value, static_cast<std::size_t>(valueCap), "%s", layoutName(controls.layout));
            return true;
        }
        if (index == 1)
        {
            putKey("deadzone_left");
            formatUnit(value, valueCap, controls.deadzoneLeft);
            return true;
        }
        if (index == 2)
        {
            putKey("deadzone_right");
            formatUnit(value, valueCap, controls.deadzoneRight);
            return true;
        }
        if (index == 3)
        {
            putKey("invert_y");
            std::snprintf(value, static_cast<std::size_t>(valueCap), "%s", invertToken(controls));
            return true;
        }
        if (index == 4)
        {
            putKey("stick_swap");
            std::snprintf(value, static_cast<std::size_t>(valueCap), "%d", controls.stickSwap ? 1 : 0);
            return true;
        }
        if (index == 5)
        {
            putKey("trigger");
            if (!(controls.triggerThreshold > 0.f))
                std::snprintf(value, static_cast<std::size_t>(valueCap), "-1");
            else
                formatUnit(value, valueCap, controls.triggerThreshold);
            return true;
        }
        if (index < 6 + kButtonCount)
        {
            const int button = index - 6;
            std::snprintf(key, static_cast<std::size_t>(keyCap), "controls_gp_%s", kButtonNames[button]);
            formatGpList(controls.gp[button], controls.gpCount[button], value, valueCap);
            return true;
        }
        if (index < 6 + kButtonCount * 2)
        {
            const int button = index - (6 + kButtonCount);
            std::snprintf(key, static_cast<std::size_t>(keyCap), "controls_key_%s", kButtonNames[button]);
            formatKeyList(controls.keys[button], controls.keyCount[button], value, valueCap);
            return true;
        }
        if (index < 6 + kButtonCount * 2 + kStickDirCount)
        {
            const int dir = index - (6 + kButtonCount * 2);
            std::snprintf(key, static_cast<std::size_t>(keyCap), "controls_key_%s", kStickNames[dir]);
            formatKeyList(controls.stickKeys[dir], controls.stickKeyCount[dir], value, valueCap);
            return true;
        }
        return false;
    }

    void fillSnap(Snap &snap, const Controls &controls)
    {
        snap = Snap{};
        snap.deadzoneLeft = controls.deadzoneLeft;
        snap.deadzoneRight = controls.deadzoneRight;
        snap.invertLeft = controls.invertLeft;
        snap.invertRight = controls.invertRight;
        snap.stickSwap = controls.stickSwap;
        snap.ignoreGamepads = controls.ignoreGamepads;
        snap.triggerThreshold = controls.triggerThreshold;
        bool polled[256] = {};
        for (int b = 0; b < kButtonCount; ++b)
        {
            for (int s = 0; s < controls.gpCount[b] && s < kMaxSources; ++s)
            {
                int src = controls.gp[b][s];
                if (controls.triggerThreshold > 0.f)
                {
                    if (b == L2 && src == kGpLeftTrigger2)
                        src = kSrcAxisLeft;
                    else if (b == R2 && src == kGpRightTrigger2)
                        src = kSrcAxisRight;
                }
                if (src == kSrcEmpty || snap.gpCount >= 64)
                    continue;
                snap.gp[snap.gpCount++] = Snap::GpSrc{src, kButtonMask[b]};
            }
            for (int k = 0; k < controls.keyCount[b] && k < kMaxKeys; ++k)
            {
                const KeyInfo *info = findKeyCode(controls.keys[b][k]);
                if (!info || snap.rlCount >= 96)
                    continue;
                snap.rlKeys[snap.rlCount++] = Snap::RlKey{info->raylib, kButtonMask[b]};
                if (info->vk > 0 && info->vk < 256)
                {
                    snap.vkMask[info->vk] = static_cast<uint16_t>(snap.vkMask[info->vk] | kButtonMask[b]);
                    if (!polled[info->vk] && snap.pollCount < 96)
                    {
                        polled[info->vk] = true;
                        snap.pollVk[snap.pollCount++] = static_cast<uint8_t>(info->vk);
                    }
                }
            }
        }
        if (snap.vkMask[kVkLShift] != 0 && snap.vkMask[kVkShift] == 0)
            snap.vkMask[kVkShift] = snap.vkMask[kVkLShift];
        for (int d = 0; d < kStickDirCount; ++d)
        {
            for (int k = 0; k < controls.stickKeyCount[d] && k < kMaxSources; ++k)
            {
                const KeyInfo *info = findKeyCode(controls.stickKeys[d][k]);
                if (!info || snap.stickCount >= 32)
                    continue;
                Snap::StickBind bind;
                bind.key = info->raylib;
                bind.vk = info->vk;
                bind.stick = kStickStick[d];
                bind.axis = kStickAxis[d];
                bind.value = kStickValue[d];
                snap.sticks[snap.stickCount++] = bind;
                if (info->vk > 0 && info->vk < 256 && !polled[info->vk] && snap.pollCount < 96)
                {
                    polled[info->vk] = true;
                    snap.pollVk[snap.pollCount++] = static_cast<uint8_t>(info->vk);
                }
            }
        }
    }

    const Snap *publishedSnap()
    {
        std::call_once(g_snapOnce, seedDefaultSnaps);
        return &g_snaps[g_snapIndex.load(std::memory_order_acquire)];
    }

    void publishControls(const Controls &controls)
    {
        std::call_once(g_snapOnce, seedDefaultSnaps);
        const int cur = g_snapIndex.load(std::memory_order_relaxed);
        const int next = (cur + 1) % 3;
        fillSnap(g_snaps[next], controls);
        g_snapIndex.store(next, std::memory_order_release);
    }

    void applyRadialDeadzone(float &x, float &y, float deadzone)
    {
        const float magnitude = std::sqrt(x * x + y * y);
        if (magnitude <= deadzone || magnitude <= 0.f)
        {
            x = 0.f;
            y = 0.f;
            return;
        }
        const float clamped = magnitude > 1.f ? 1.f : magnitude;
        const float span = 1.f - deadzone;
        const float scale = span > 0.f ? (clamped - deadzone) / span / magnitude : 0.f;
        x = x * scale;
        y = y * scale;
        if (x < -1.f)
            x = -1.f;
        else if (x > 1.f)
            x = 1.f;
        if (y < -1.f)
            y = -1.f;
        else if (y > 1.f)
            y = 1.f;
    }

    float triggerAmount(float raw, bool &signedRange)
    {
        if (raw < -0.01f)
            signedRange = true;
        if (signedRange)
            return (raw + 1.f) * 0.5f;
        return raw;
    }

    uint16_t mapGamepad(const Snap &snap, const bool down[18], const float axes[6], bool &signedL, bool &signedR,
                        StickFloats &sticks, bool buttonsLive)
    {
        sticks.lx = axes ? axes[0] : 0.f;
        sticks.ly = axes ? axes[1] : 0.f;
        sticks.rx = axes ? axes[2] : 0.f;
        sticks.ry = axes ? axes[3] : 0.f;
        uint16_t pressed = 0;
        if (!buttonsLive || !axes)
            return pressed;
        for (int i = 0; i < snap.gpCount; ++i)
        {
            const int src = snap.gp[i].src;
            const uint16_t mask = snap.gp[i].mask;
            if (src > 0 && src < 18)
            {
                if (down && down[src])
                    pressed = static_cast<uint16_t>(pressed | mask);
                continue;
            }
            if (src != kSrcAxisLeft && src != kSrcAxisRight)
                continue;
            const int axis = src == kSrcAxisLeft ? 4 : 5;
            bool &sig = axis == 4 ? signedL : signedR;
            const float amount = triggerAmount(axes[axis], sig);
            float threshold = snap.triggerThreshold;
            if (!(threshold > 0.f))
                threshold = 0.5f;
            if (amount >= threshold)
                pressed = static_cast<uint16_t>(pressed | mask);
        }
        return pressed;
    }

    void swapSticksIf(const Snap &snap, StickFloats &sticks)
    {
        if (!snap.stickSwap)
            return;
        const float lx = sticks.lx;
        const float ly = sticks.ly;
        sticks.lx = sticks.rx;
        sticks.ly = sticks.ry;
        sticks.rx = lx;
        sticks.ry = ly;
    }

    void invertAndDeadzone(const Snap &snap, StickFloats &sticks)
    {
        if (snap.invertLeft)
            sticks.ly = -sticks.ly;
        if (snap.invertRight)
            sticks.ry = -sticks.ry;
        applyRadialDeadzone(sticks.lx, sticks.ly, snap.deadzoneLeft);
        applyRadialDeadzone(sticks.rx, sticks.ry, snap.deadzoneRight);
    }
}

namespace ps2_graphics
{
    namespace
    {
        std::mutex g_mutex;
        Settings g_settings;
        bool g_loaded = false;

        const char *settingsPath()
        {
            const char *path = std::getenv("PS2X_SETTINGS");
            return (path && path[0]) ? path : nullptr;
        }

        void logControlsLocked()
        {
            const ps2_controls::Controls &controls = g_settings.controls;
            char cross[96];
            char circle[96];
            char key[64];
            char value[96];
            cross[0] = '\0';
            circle[0] = '\0';
            for (int i = 0; ps2_controls::controlsField(controls, i, key, 64, value, 96); ++i)
            {
                if (std::strcmp(key, "controls_key_CROSS") == 0)
                    std::snprintf(cross, sizeof(cross), "%s", value);
                else if (std::strcmp(key, "controls_key_CIRCLE") == 0)
                    std::snprintf(circle, sizeof(circle), "%s", value);
            }
            char dzL[16];
            char dzR[16];
            char trig[16];
            // Reuse the file tokens via the field walker above; format threshold here for the log.
            std::snprintf(dzL, sizeof(dzL), "%.3f", static_cast<double>(controls.deadzoneLeft));
            std::snprintf(dzR, sizeof(dzR), "%.3f", static_cast<double>(controls.deadzoneRight));
            if (controls.triggerThreshold > 0.f)
                std::snprintf(trig, sizeof(trig), "%.3f", static_cast<double>(controls.triggerThreshold));
            else
                std::snprintf(trig, sizeof(trig), "off");
            const char *inv = "false";
            if (controls.invertLeft && controls.invertRight)
                inv = "both";
            else if (controls.invertLeft)
                inv = "left";
            else if (controls.invertRight)
                inv = "true";
            std::fprintf(stderr,
                         "[controls] layout=%s dz=%s/%s invert=%s swap=%d trigger=%s cross_keys=%s circle_keys=%s\n",
                         ps2_controls::layoutName(controls.layout), dzL, dzR, inv, controls.stickSwap ? 1 : 0, trig,
                         cross[0] ? cross : "-", circle[0] ? circle : "-");
            std::fflush(stderr);
        }

        // key=value lines; unknown keys and bad values are ignored.
        void loadLocked()
        {
            g_loaded = true;
            const char *path = settingsPath();
            if (!path)
                return;
            FILE *file = std::fopen(path, "r");
            if (!file)
                return;
            char line[256];
            while (std::fgets(line, sizeof(line), file))
            {
                char *eq = std::strchr(line, '=');
                if (!eq)
                    continue;
                *eq = '\0';
                const char *raw = eq + 1;
                if (std::strncmp(line, "controls_", 9) == 0)
                {
                    ps2_controls::applyControlsField(g_settings.controls, line, raw);
                    continue;
                }
                const std::string key(line);
                const int value = std::atoi(raw);
                if (key == "aspect" && value >= 0 && value <= static_cast<int>(Aspect::Native))
                    g_settings.aspect = static_cast<Aspect>(value);
                else if (key == "smooth_screen")
                    g_settings.smoothScreen = value != 0;
                else if (key == "internal_scale" && value >= 1 && value <= kMaxInternalScale)
                    g_settings.internalScale = value;
                else if (key == "widescreen")
                    g_settings.widescreen = value != 0;
                else if (key == "texture_filter" && (value == 0 || value == 1))
                    g_settings.textureFilter = static_cast<TextureFilter>(value);
                else if (key == "fxaa")
                    g_settings.fxaa = value != 0;
                else if (key == "sharpen" && value >= 0 && value <= 100)
                    g_settings.sharpen = value;
            }
            std::fclose(file);
            ps2_controls::publishControls(g_settings.controls);
            logControlsLocked();
        }

        // Settings that recompiled code reads through runtime switches.
        void applySwitchesLocked()
        {
            ps2_switches::set("widescreen", g_settings.widescreen);
        }

        void saveLocked()
        {
            const char *path = settingsPath();
            if (!path)
                return;
            FILE *file = std::fopen(path, "w");
            if (!file)
                return;
            std::fprintf(file, "aspect=%d\n", static_cast<int>(g_settings.aspect));
            std::fprintf(file, "smooth_screen=%d\n", g_settings.smoothScreen ? 1 : 0);
            std::fprintf(file, "internal_scale=%d\n", g_settings.internalScale);
            std::fprintf(file, "widescreen=%d\n", g_settings.widescreen ? 1 : 0);
            std::fprintf(file, "texture_filter=%d\n", static_cast<int>(g_settings.textureFilter));
            std::fprintf(file, "fxaa=%d\n", g_settings.fxaa ? 1 : 0);
            std::fprintf(file, "sharpen=%d\n", g_settings.sharpen);
            char key[64];
            char value[192];
            for (int i = 0; ps2_controls::controlsField(g_settings.controls, i, key, 64, value, 192); ++i)
                std::fprintf(file, "%s=%s\n", key, value);
            std::fclose(file);
        }
    }

    Settings get()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_loaded)
        {
            loadLocked();
            applySwitchesLocked();
            if (!settingsPath())
                ps2_controls::publishControls(g_settings.controls);
        }
        return g_settings;
    }

    void set(const Settings &settings)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_loaded = true;
        g_settings = settings;
        applySwitchesLocked();
        saveLocked();
        ps2_controls::publishControls(g_settings.controls);
    }

    float displayAspect(const Settings &settings, float frameWidth, float frameHeight, float windowWidth,
                        float windowHeight)
    {
        switch (settings.aspect)
        {
        case Aspect::Ratio4x3:
            return 4.0f / 3.0f;
        case Aspect::Ratio16x9:
            return 16.0f / 9.0f;
        case Aspect::Stretch:
            return windowHeight > 0.0f ? windowWidth / windowHeight : 4.0f / 3.0f;
        case Aspect::Native:
        default:
            return frameHeight > 0.0f ? frameWidth / frameHeight : 4.0f / 3.0f;
        }
    }
}
