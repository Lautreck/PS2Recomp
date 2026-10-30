#include "runtime/ps2_switches.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ps2_switches
{
    std::atomic<bool> &flag(const char *name)
    {
        static std::mutex mutex;
        static std::unordered_map<std::string, std::unique_ptr<std::atomic<bool>>> flags;
        std::lock_guard<std::mutex> lock(mutex);
        std::unique_ptr<std::atomic<bool>> &slot = flags[name];
        if (!slot)
            slot = std::make_unique<std::atomic<bool>>(false);
        return *slot;
    }
}
