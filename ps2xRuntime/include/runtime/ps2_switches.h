#pragma once

#include <atomic>

// Runtime switches of recompiled code: a [patches] instruction with switch = "<name>" is emitted as
// if (PS2_SWITCH("<name>")) { patched } else { original }, so an option (the settings window) can turn a game patch
// on and off while the game runs.
namespace ps2_switches
{
    // The flag of a switch, created off on first use and kept for the process.
    std::atomic<bool> &flag(const char *name);

    inline void set(const char *name, bool on)
    {
        flag(name).store(on, std::memory_order_relaxed);
    }
}

// One lookup per call site, then a relaxed load.
#define PS2_SWITCH(name)                                                  \
    ([]() -> std::atomic<bool> & {                                        \
        static std::atomic<bool> &switchFlag = ::ps2_switches::flag(name); \
        return switchFlag;                                                \
    }()                                                                   \
         .load(std::memory_order_relaxed))
