#pragma once

#include "ps2x/iop/iop_types.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace ps2x::iop
{
    enum class HostPathKind : uint32_t
    {
        ElfDirectory = 0,
        CdRoot = 1,
        CdImage = 2,
        HostRoot = 3,
        MemoryCardRoot = 4,
    };

    enum class LogLevel : uint32_t
    {
        Debug = 0,
        Info = 1,
        Warning = 2,
        Error = 3,
    };

    enum class MemoryCardOperation : uint32_t
    {
        Init,
        GetInfo,
        Open,
        Close,
        Seek,
        Read,
        Write,
        Flush,
        Chdir,
        GetDir,
        SetFileInfo,
        Delete,
        Format,
        Unformat,
        Mkdir,
    };

    struct MemoryCardRequest
    {
        MemoryCardOperation operation = MemoryCardOperation::Init;
        // The fifth argument is carried in $t0 by the EE n32 ABI
        std::array<uint32_t, 5> arguments{};
    };

    class IopHost
    {
    public:
        virtual ~IopHost() = default;

        virtual bool readGuest(uint32_t address, void *destination, size_t size) const = 0;
        virtual bool writeGuest(uint32_t address, const void *source, size_t size) = 0;
        virtual bool zeroGuest(uint32_t address, size_t size) = 0;
        virtual bool normalizeGuestAddress(uint32_t address, uint32_t &normalized) const = 0;

        // IOP RAM is a distinct address space from the EE guest. TODO remove this later
        virtual bool readIopMemory(uint32_t address, void *destination, size_t size) const
        {
            (void)address;
            (void)destination;
            (void)size;
            return false;
        }
        virtual bool writeIopMemory(uint32_t address, const void *source, size_t size)
        {
            (void)address;
            (void)source;
            (void)size;
            return false;
        }
        virtual bool zeroIopMemory(uint32_t address, size_t size)
        {
            (void)address;
            (void)size;
            return false;
        }
        virtual bool normalizeIopAddress(uint32_t address, uint32_t &normalized) const
        {
            (void)address;
            normalized = 0u;
            return false;
        }
        virtual uint32_t allocateIopHandle(IopHandleKind kind) = 0;
        virtual uint32_t allocateGuest(uint32_t size, uint32_t alignment) = 0;
        virtual void freeGuest(uint32_t address) = 0;

        virtual void audioCommand(uint32_t sid, uint32_t function, GuestBuffer send, GuestBuffer receive) = 0;

        virtual std::string hostPath(HostPathKind kind) const = 0;
        virtual std::string translateGuestPath(std::string_view path) const = 0;
        virtual uint64_t openHostFile(std::string_view path) = 0;
        virtual bool hostFileSize(uint64_t handle, uint64_t &size) const = 0;
        virtual bool readHostFile(uint64_t handle,
                                  uint64_t offset,
                                  void *destination,
                                  size_t size,
                                  size_t &bytesRead) = 0;
        virtual void closeHostFile(uint64_t handle) = 0;

        virtual int32_t memoryCard(const MemoryCardRequest &request) = 0;

        virtual bool hasGuestFunction(uint32_t address) const = 0;
        virtual bool invokeGuestFunction(uint64_t callToken,
                                         uint32_t address,
                                         uint32_t a0,
                                         uint32_t a1,
                                         uint32_t a2,
                                         uint32_t a3,
                                         uint32_t *resultAddress) = 0;

        // Deliver an IOP -> EE SIF command packet. The default keeps hosts
        // which do not emulate the EE command dispatcher source-compatible.
        virtual bool sendSifCommand(uint32_t commandId,
                                    const void *packet,
                                    size_t packetSize)
        {
            (void)commandId;
            (void)packet;
            (void)packetSize;
            return false;
        }

        virtual void log(LogLevel level, std::string_view message) = 0;
    };

    // Host mix from the IOP sound engine (48 kHz interleaved s16 stereo). The slot is one per process.
    // Empty until the runtime registers a device. Playback is skipped when no function is set.
    using MixedAudioFn = void (*)(const int16_t *interleavedStereo, size_t frames);
    inline std::atomic<MixedAudioFn> &mixedAudioFnSlot()
    {
        static std::atomic<MixedAudioFn> fn{nullptr};
        return fn;
    }
    inline void setMixedAudioFn(MixedAudioFn fn)
    {
        mixedAudioFnSlot().store(fn, std::memory_order_release);
    }

    // 32 SIF software registers (ps2sdk SIF_SREG, indices 0..31). The EE writes them with
    // SIF_CMD_SET_SREG (cid 0x80000001, packet+16 = index, packet+20 = value). IOP sifcmd
    // GetSreg/SetSreg (ordinals 6/7) read the same store. zsiopdrv spins until register 31 is non-zero.
    void setSifSoftwareRegister(uint32_t index, uint32_t value) noexcept;
    uint32_t sifSoftwareRegister(uint32_t index) noexcept;

    // EE sceSifSendCmd with a user cid (bit 31 clear): runs the handler the IOP registered with
    // sceSifAddCmdHandler on a copy of the 16..112-byte packet. The extra payload must already be in IOP RAM.
    // False when no handler is registered for that cid.
#define PS2X_IOP_HAS_USER_SIF_CMD 1
    bool deliverEeUserSifCmd(uint32_t commandId, const void *packet, size_t packetSize) noexcept;
}
