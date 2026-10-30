#include "iop_emulator.h"
#include "imports/iop_cdvd.h"
#include "core/iop_cpu.h"
#include "imports/iop_heaplib.h"
#include "imports/iop_imports.h"
#include "imports/iop_intrman.h"
#include "imports/iop_ioman.h"
#include "core/iop_kernel.h"
#include "imports/iop_loadcore.h"
#include "core/iop_memory.h"
#include "core/sound_engine.h"
#include "services/iop_module_loader.h"
#include "services/iop_rpc.h"
#include "imports/iop_stdio.h"
#include "imports/iop_sysclib.h"
#include "imports/iop_sysmem.h"
#include "imports/iop_timrman.h"
#include "imports/iop_vblank.h"
#include "iop_emulator_const.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        // The emulator deliverEeUserSifCmd reaches (one per process, like the SIF software registers).
        IopEmulator *g_userCmdEmulator = nullptr;
        constexpr uint32_t kRamSize = IopMemory::RamSize;
        constexpr uint32_t kKernelHeapBase = IopMemory::HeapBase;
        constexpr uint32_t kKernelHeapLimit = IopMemory::HeapLimit;
        constexpr uint32_t kCallStackBase = kKernelHeapLimit;
        constexpr uint32_t kCallStackLimit = 0x001FFF00u;
        constexpr uint32_t kCallStackSize = 0x2000u;
        constexpr uint32_t kCallStackCapacity = (kCallStackLimit - kCallStackBase) / kCallStackSize;
        constexpr uint64_t kCdvdCompletionCycles = 128u;

        uint32_t physicalAddress(uint32_t address)
        {
            return IopMemory::physicalAddress(address);
        }

        int32_t sign16(uint32_t value)
        {
            return static_cast<int16_t>(value & 0xFFFFu);
        }

        bool iequals(std::string_view lhs, std::string_view rhs)
        {
            if (lhs.size() != rhs.size())
                return false;
            for (size_t i = 0; i < lhs.size(); ++i)
            {
                if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
                    std::tolower(static_cast<unsigned char>(rhs[i])))
                    return false;
            }
            return true;
        }

        // ELF image sitting in IOP RAM (the bytes sceSifSetDma copied). IopHost::readGuest is EE RAM
        // on the runtime adapter and the test host's own buffer in unit tests, so the loader tries this first.
        bool readElfImageFromIopRam(IopMemory &memory, uint32_t address, std::vector<uint8_t> &bytes)
        {
            uint8_t hdr[52]{};
            if (!memory.readRam(address, hdr, sizeof(hdr)))
                return false;
            if (hdr[0] != 0x7Fu || hdr[1] != 'E' || hdr[2] != 'L' || hdr[3] != 'F' || hdr[4] != 1u || hdr[5] != 1u)
                return false;
            const auto ru16 = [](const uint8_t *p) -> uint16_t {
                return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8u));
            };
            const auto ru32 = [](const uint8_t *p) -> uint32_t {
                return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8u) |
                       (static_cast<uint32_t>(p[2]) << 16u) | (static_cast<uint32_t>(p[3]) << 24u);
            };
            if (ru16(hdr + 18) != 8u || ru16(hdr + 40) < 52u)
                return false;
            const uint32_t phoff = ru32(hdr + 28);
            const uint32_t shoff = ru32(hdr + 32);
            const uint16_t phentsize = ru16(hdr + 42);
            const uint16_t phnum = ru16(hdr + 44);
            const uint16_t shentsize = ru16(hdr + 46);
            const uint16_t shnum = ru16(hdr + 48);
            constexpr uint64_t kMaxImageSize = 2ull * 1024ull * 1024ull;
            uint64_t required = sizeof(hdr);
            required = std::max<uint64_t>(required, static_cast<uint64_t>(phoff) + static_cast<uint64_t>(phentsize) * phnum);
            required = std::max<uint64_t>(required, static_cast<uint64_t>(shoff) + static_cast<uint64_t>(shentsize) * shnum);
            if (required > kMaxImageSize)
                return false;
            bytes.resize(static_cast<size_t>(required));
            if (!memory.readRam(address, bytes.data(), bytes.size()))
                return false;
            constexpr uint32_t kShtNobits = 8u;
            if (shnum != 0u && shentsize >= 40u)
            {
                for (uint16_t i = 0; i < shnum; ++i)
                {
                    const size_t off = static_cast<size_t>(shoff) + static_cast<size_t>(i) * shentsize;
                    if (off + 40u > bytes.size())
                        return false;
                    const uint8_t *section = bytes.data() + off;
                    if (ru32(section + 4) != kShtNobits)
                        required = std::max<uint64_t>(required, static_cast<uint64_t>(ru32(section + 16)) + ru32(section + 20));
                }
            }
            if (phnum != 0u && phentsize >= 32u)
            {
                for (uint16_t i = 0; i < phnum; ++i)
                {
                    const size_t off = static_cast<size_t>(phoff) + static_cast<size_t>(i) * phentsize;
                    if (off + 32u > bytes.size())
                        return false;
                    const uint8_t *program = bytes.data() + off;
                    required = std::max<uint64_t>(required, static_cast<uint64_t>(ru32(program + 4)) + ru32(program + 16));
                }
            }
            if (required > kMaxImageSize)
                return false;
            bytes.resize(static_cast<size_t>(required));
            return memory.readRam(address, bytes.data(), bytes.size());
        }

    }

    class IopEmulator::Impl final : public IopGuestExecutor
    {
    public:
        using CpuState = IopCpuState;

        struct Module
        {
            int id = 0;
            std::string path;
            std::string name;
            uint32_t base = 0;
            uint32_t size = 0;
            uint32_t entry = 0;
            uint32_t gp = 0;
            bool resident = false;
        };

        struct GuestCallback
        {
            uint32_t function = 0;
            uint32_t gp = 0;
        };

        struct ScheduledGuestCallback
        {
            uint32_t function = 0u;
            uint32_t gp = 0u;
            uint32_t argument = 0u;
        };

        explicit Impl(IopHost &hostRef)
            : host(hostRef),
              sysmem(host, memory),
              kernel(memory),
              cdvd(host, memory, kernel),
              vblank(kernel),
              rpc(host, memory, kernel),
              sysclib(memory),
              stdio(host, memory),
              heaplib(memory),
              intrman(memory),
              timrman(),
              ioman(hostRef, memory),
              cpuCore(memory),
              imports(memory),
              loadcore(memory, imports)
        {
            reset();
        }

        void reset()
        {
            memory.reset();
            kernel.reset();
            modules.clear();
            imports.reset();
            rpc.reset();
            cdvd.reset();
            intrman.reset();
            timrman.reset();
            ioman.reset();
            m_libsd = {};
            m_sound.reset();
            m_soundRoute = -1;
            pendingDmaInterrupts.clear();
            pendingGuestCallbacks.clear();
            nextModuleId = 1;
            moduleCursor = kModuleLoadBase;
            totalCycles = 0;
            totalInstructions = 0;
            eeCycleCarry = 0;
            m_idleValid = false;
            m_idleUntil = 0;
            m_idleEpoch = 0;
            m_idleSkipCount = 0;
            activeCpu = nullptr;
            lastError.clear();
            servicingDmaInterrupts = false;
            servicingGuestCallbacks = false;
            callDepth = 0u;
            secrMcCommandHandler = {};
            secrMcDevIdHandler = {};
            checkKelfPathCallback = {};
        }

        uint8_t read8(uint32_t address) const
        {
            return memory.read8(address);
        }

        uint16_t read16(uint32_t address) const
        {
            return memory.read16(address);
        }

        uint32_t read32(uint32_t address) const
        {
            return memory.read32(address);
        }

        void write8(uint32_t address, uint8_t value)
        {
            memory.write8(address, value);
            schedulePendingDma();
        }

        void write16(uint32_t address, uint16_t value)
        {
            memory.write16(address, value);
            schedulePendingDma();
        }

        void write32(uint32_t address, uint32_t value)
        {
            memory.write32(address, value);
            schedulePendingDma();
        }

        void schedulePendingDma()
        {
            if (const auto dma = memory.takeDmaStart())
            {
                pendingDmaInterrupts[dma->irq] = totalCycles + dma->delayCycles;
                // A DMA completion can run a guest handler that readies a thread. Clearing here,
                // not inside that handler, is what stops a sealed idle skip from jumping over the IRQ.
                m_idleValid = false;
            }
        }

        bool readRam(uint32_t address, void *destination, size_t size) const
        {
            return memory.readRam(address, destination, size);
        }

        bool writeRam(uint32_t address, const void *source, size_t size)
        {
            return memory.writeRam(address, source, size);
        }

        bool zeroRam(uint32_t address, size_t size)
        {
            return memory.zeroRam(address, size);
        }

        bool isHardwareAddress(uint32_t phys) const
        {
            return memory.isHardwareAddress(phys);
        }

        uint32_t allocate(uint32_t size, uint32_t alignment = 16u, std::optional<uint32_t> fixed = std::nullopt)
        {
            return memory.allocate(size, alignment, fixed);
        }

        bool freeAllocation(uint32_t address)
        {
            return memory.freeAllocation(address);
        }

        void log(LogLevel level, std::string_view text)
        {
            host.log(level, text);
        }

        bool checkInterrupt(CpuState &cpu)
        {
            const uint32_t status = cpu.cop0[12];
            if ((status & 1u) == 0u)
                return false;
            if ((status & 0x2u) != 0u)
                return false;
            const bool pending = memory.interruptControl() != 0u && (memory.interruptStatus() & memory.interruptMask()) != 0u;
            if (!pending)
                return false;
            cpu.cop0[13] |= 0x400u;
            cpuCore.raiseException(cpu, 0u, cpu.pc, false);
            return true;
        }

        enum class ImportDisposition
        {
            Handled,
            JumpToGuest,
            Missing,
        };

        [[nodiscard]] bool soundRouteOn()
        {
            if (m_soundRoute < 0)
            {
                m_soundRoute = host.hostPath(HostPathKind::CdImage).empty() ? 0 : 1;
                if (m_soundRoute == 1)
                    log(LogLevel::Info, "[libsd] sound route on");
            }
            return m_soundRoute == 1;
        }

        [[nodiscard]] uint32_t stackArg(const CpuState &cpu, uint32_t slot) const
        {
            const uint32_t sp = cpu.gpr[29];
            if (sp < 16u)
                return 0u;
            return memory.read32(sp + 16u + slot * 4u);
        }

        void libsdInit()
        {
            m_libsd = {};
            for (uint32_t core = 0; core < 2u; ++core)
            {
                m_sound.initCore(core);
                m_sound.setSwitch((0x16u << 8) | core, 0xFFFFFFu); // KOFF
                for (uint32_t id : {0x18u, 0x19u, 0x1Au, 0x1Bu})
                    m_sound.setSwitch((id << 8) | core, 0xFFFFFFu); // VMIX*
                m_sound.setParam((0x08u << 8) | core, core == 0u ? 0x0FF0u : 0x0FFCu);
                for (uint32_t id = 0x09u; id <= 0x10u; ++id)
                {
                    // Both masters start at unity. step() multiplies core 0 by core 1's master, so an
                    // untouched core would mute the mix. The driver overwrites these when it wants to.
                    const bool master = id == 0x09u || id == 0x0Au;
                    uint16_t value = master ? 0x3FFFu : 0u;
                    if (core == 1u && (id == 0x0Du || id == 0x0Eu))
                        value = 0x7FFFu;
                    m_sound.setParam((id << 8) | 0x80u | core, value);
                }
                for (uint32_t voice = 0; voice < SoundEngine::VoicesPerCore; ++voice)
                {
                    const uint32_t packed = (voice << 1) | core;
                    m_sound.setParam((0x02u << 8) | packed, 0x3FFFu);
                    m_sound.setAddr((0x20u << 8) | packed, 0u);
                }
            }
            const uint8_t silentLoop[16] = {0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
                                            0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07};
            m_sound.writeSampleRam(0x5000u, silentLoop, sizeof(silentLoop));
        }

        bool dispatchLibsd(uint16_t ordinal, CpuState &cpu)
        {
            if (!soundRouteOn())
                return false;
            const auto ret = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };
            const uint32_t a0 = cpu.gpr[4];
            const uint32_t a1 = cpu.gpr[5];
            const uint32_t a2 = cpu.gpr[6];
            const uint32_t a3 = cpu.gpr[7];
            switch (ordinal)
            {
            case 4:
                libsdInit();
                log(LogLevel::Info, "[libsd] sceSdInit");
                ret(0u);
                return true;
            case 5:
                m_sound.setParam(a0, static_cast<uint16_t>(a1));
                ret(0u);
                return true;
            case 6:
                ret(m_sound.getParam(a0));
                return true;
            case 7:
                if (((a0 >> 8) & 0xFFu) == 0x15u && a1 != 0u)
                {
                    static int keys = 0;
                    if (keys < 8)
                    {
                        ++keys;
                        std::ostringstream out;
                        out << "[libsd] KON entry=0x" << std::hex << a0 << " mask=0x" << a1;
                        log(LogLevel::Info, out.str());
                    }
                }
                m_sound.setSwitch(a0, a1);
                ret(0u);
                return true;
            case 8:
                ret(m_sound.getSwitch(a0));
                return true;
            case 9:
                if (((a0 >> 8) & 0xFFu) == 0x1Fu)
                    logLibsdIrq("irqa", a0, a1, cpu.gpr[31]);
                m_sound.setAddr(a0, a1);
                ret(0u);
                return true;
            case 10:
                ret(m_sound.getAddr(a0));
                return true;
            case 11:
                if ((a0 & ~1u) == 4u)
                    logLibsdIrq("coreattr", a0, a1, cpu.gpr[31]);
                m_sound.setCoreAttr(a0, a1);
                ret(0u);
                return true;
            case 12:
                ret(m_sound.getCoreAttr(a0));
                return true;
            case 17:
            {
                const uint32_t channel = a0 & 1u;
                const uint32_t mode = a1;
                uint32_t size = stackArg(cpu, 0);
                if (size == 0u || size > 0x200000u)
                {
                    const uint32_t viaT0 = cpu.gpr[8];
                    if (viaT0 != 0u && viaT0 <= 0x200000u)
                        size = viaT0;
                    else
                        size = 0u;
                }
                std::vector<uint8_t> data(size);
                if ((mode & 3u) == 0u)
                {
                    if (size != 0u && !memory.readRam(a2, data.data(), size))
                        data.assign(size, 0u);
                    if (size != 0u)
                        m_sound.writeSampleRam(a3, data.data(), size);
                }
                else if (size != 0u)
                {
                    m_sound.readSampleRam(a3, data.data(), size);
                    (void)memory.writeRam(a2, data.data(), size);
                }
                if ((mode & 8u) == 0u && m_libsd.transHandler[channel] != 0u)
                {
                    (void)callFunction(m_libsd.transHandler[channel], channel, m_libsd.transArg[channel], 0u, 0u,
                                       m_libsd.transGp[channel], 100000u);
                }
                static int transfers = 0;
                if (transfers < 12)
                {
                    ++transfers;
                    std::ostringstream out;
                    out << "[libsd] VoiceTrans ch=" << channel << " mode=0x" << std::hex << mode
                        << " iop=0x" << a2 << " spu=0x" << a3 << " size=" << std::dec << size;
                    log(LogLevel::Info, out.str());
                }
                ret((size + 63u) & ~63u);
                return true;
            }
            case 18:
            {
                const uint32_t channel = a0 & 1u;
                const uint32_t mode = a1;
                const uint32_t iopBuffer = a2;
                const uint32_t size = a3;
                if ((mode & 3u) == 2u)
                {
                    m_sound.stopStream(channel);
                    m_libsd.blockActive[channel] = false;
                    ret(0u);
                    return true;
                }
                const uint32_t physical = IopMemory::physicalAddress(iopBuffer);
                if (size == 0u || physical >= IopMemory::RamSize || size > IopMemory::RamSize - physical)
                {
                    ret(static_cast<uint32_t>(-100));
                    return true;
                }
                m_sound.startStream(channel, memory.ram().data() + physical, size, (mode & 0x10u) != 0u);
                m_libsd.blockActive[channel] = true;
                m_libsd.blockBuffer[channel] = iopBuffer;
                m_libsd.blockSize[channel] = size;
                std::ostringstream out;
                out << "[libsd] BlockTrans ch=" << channel << " mode=0x" << std::hex << mode
                    << " buf=0x" << iopBuffer << " size=" << std::dec << size;
                log(LogLevel::Info, out.str());
                ret(0u);
                return true;
            }
            case 19:
                ret(1u);
                return true;
            case 20:
            {
                const uint32_t channel = a0 & 1u;
                if (!m_libsd.blockActive[channel])
                {
                    ret(0u);
                    return true;
                }
                const uint32_t position = m_sound.streamPosition(channel);
                const uint32_t side = m_libsd.blockSize[channel] != 0u && position >= m_libsd.blockSize[channel] / 2u ? 1u : 0u;
                ret((side << 24) | ((m_libsd.blockBuffer[channel] + position) & 0xFFFFFFu));
                return true;
            }
            case 23:
                if (a1 != 0u)
                {
                    // sceSdEffectAttr: int core, int mode, short depth_L, short depth_R, int delay, int feedback.
                    const uint32_t core = a0 & 1u;
                    const int32_t mode = static_cast<int32_t>(memory.read32(a1 + 4u));
                    const int16_t depthL = static_cast<int16_t>(memory.read16(a1 + 8u));
                    const int16_t depthR = static_cast<int16_t>(memory.read16(a1 + 10u));
                    const int32_t delay = static_cast<int32_t>(memory.read32(a1 + 12u));
                    const int32_t feedback = static_cast<int32_t>(memory.read32(a1 + 16u));
                    m_sound.setEffectAttr(core, mode, depthL, depthR, delay, feedback);
                }
                ret(0u);
                return true;
            case 25:
            {
                static constexpr uint32_t kWorkHalfwords[10] = {0x02u, 0x4D8u, 0x3E8u, 0x908u, 0xDFCu, 0x15BCu, 0x1ED8u, 0x3008u, 0x3008u, 0x780u};
                const uint32_t core = a0 & 1u;
                const uint32_t mode = a2 < 10u ? a2 : 0u;
                const uint32_t end = 0x1FFFFFu;
                const uint32_t start = end - kWorkHalfwords[mode] * 8u + 1u;
                m_sound.setAddr((0x1Du << 8) | core, end);
                m_sound.setAddr((0x1Cu << 8) | core, start);
                m_sound.clearSampleRam(start, 64u);
                ret(0u);
                return true;
            }
            case 21:
            case 26:
            {
                const uint32_t channel = a0 & 1u;
                m_libsd.transHandler[channel] = a1;
                m_libsd.transArg[channel] = a2;
                m_libsd.transGp[channel] = cpu.gpr[28];
                ret(0u);
                return true;
            }
            case 22: // sceSdSetIRQCallback(cb): freesd calls cb(0) only when no ordinal-27 handler is set
            {
                const uint32_t old = m_libsd.irqCallback;
                m_libsd.irqCallback = a0;
                m_libsd.irqCallbackGp = cpu.gpr[28];
                ret(old);
                return true;
            }
            case 27: // sceSdSetSpu2IntrHandler(handler, arg): handler(core_bit, arg)
            {
                const uint32_t old = m_libsd.spu2Handler[0];
                m_libsd.spu2Handler[0] = a0;
                m_libsd.spu2Arg[0] = a1;
                m_libsd.spu2Gp[0] = cpu.gpr[28];
                ret(old);
                return true;
            }
            default:
            {
                static uint32_t seen = 0u;
                if (ordinal < 32u && (seen & (1u << ordinal)) == 0u)
                {
                    seen |= 1u << ordinal;
                    std::ostringstream out;
                    out << "[libsd] ordinal " << ordinal << " stubbed";
                    log(LogLevel::Info, out.str());
                }
                ret(0u);
                return true;
            }
            }
        }

        // TEKKEN4_LIBSD_LOG=1: sceSdSetCoreAttr(SD_CORE_IRQ_ENABLE) and sceSdSetAddr(SD_ADDR_IRQA).
        // Off unless the env is exactly "1". First 48 lines, then every 256th, so a fight log stays bounded.
        void logLibsdIrq(const char *what, uint32_t entry, uint32_t value, uint32_t ra)
        {
            static const bool on = [] {
                const char *env = std::getenv("TEKKEN4_LIBSD_LOG");
                return env != nullptr && env[0] == '1' && env[1] == '\0';
            }();
            if (!on)
                return;
            static uint32_t lines = 0u;
            static uint32_t attrOn = 0u;
            static uint32_t attrOff = 0u;
            static uint32_t irqaN = 0u;
            if (what[0] == 'c')
            {
                if (value != 0u)
                    ++attrOn;
                else
                    ++attrOff;
            }
            else
                ++irqaN;
            ++lines;
            if (lines > 48u && (lines % 256u) != 0u)
                return;
            std::fprintf(stderr,
                         "[libsd] %s n=%u entry=%x val=%x ra=%x inirq=%d pump=%d attrOn=%u attrOff=%u irqa=%u\n", what,
                         lines, entry, value, ra, m_inSpu2Irq && !m_inSpu2Pump ? 1 : 0, m_inSpu2Pump ? 1 : 0, attrOn,
                         attrOff, irqaN);
        }

        // True when every core in coreBit has ATTR bit 6 set again.
        bool irqCoresRearmed(uint32_t coreBit) const
        {
            if ((coreBit & 1u) != 0u && m_sound.getCoreAttr(4u) == 0u)
                return false;
            if ((coreBit & 2u) != 0u && m_sound.getCoreAttr(5u) == 0u)
                return false;
            return true;
        }

        // SPU2 IRQ: advanceTo stops on the sample that crosses a core's IRQA. takeSpu2Irq clears ATTR bit 6
        // the way libsd's interrupt does, then the user handler runs. That handler only wakes a worker; the
        // worker re-arms. Run ready threads before the rest of this catch-up so a long slice cannot play
        // past the next IRQA with the enable still clear. Cap 32 crosses per slice.
        void serviceSpu2Irq()
        {
            for (int guard = 0; guard < 32; ++guard)
            {
                m_sound.advanceTo(totalCycles);
                const uint32_t coreBit = m_sound.takeSpu2Irq();
                if (coreBit == 0u)
                    return;
                const bool intrHandler = m_libsd.spu2Handler[0] != 0u;
                const uint32_t handler = intrHandler ? m_libsd.spu2Handler[0] : m_libsd.irqCallback;
                if (handler != 0u)
                {
                    m_inSpu2Irq = true;
                    if (intrHandler)
                        (void)callFunction(handler, coreBit, m_libsd.spu2Arg[0], 0u, 0u, m_libsd.spu2Gp[0]);
                    else
                        (void)callFunction(handler, 0u, 0u, 0u, 0u, m_libsd.irqCallbackGp);
                    m_inSpu2Irq = false;
                }
                if (irqCoresRearmed(coreBit))
                    continue;
                // m_inSpu2Irq stays set so a thread we run here cannot re-enter drainMixedAudio.
                m_inSpu2Irq = true;
                m_inSpu2Pump = true;
                for (int slice = 0; slice < 24 && !irqCoresRearmed(coreBit); ++slice)
                {
                    IopThread *next = kernel.beginNextReady(totalCycles);
                    if (next == nullptr)
                        break;
                    runCpu(next->cpu, kDefaultSlice);
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                }
                m_inSpu2Pump = false;
                m_inSpu2Irq = false;
                m_idleValid = false;
            }
        }

        void drainMixedAudio()
        {
            // The IRQ handler's own libsd calls can land here: no nested drain.
            if (!soundRouteOn() || m_inSpu2Irq)
                return;
            serviceSpu2Irq();
            int16_t block[1024 * 2];
            for (;;)
            {
                const size_t frames = m_sound.drainOutput(block, 1024u);
                if (frames == 0u)
                    break;
                if (const MixedAudioFn sink = mixedAudioFnSlot().load(std::memory_order_acquire))
                    sink(block, frames);
            }
        }

        ImportDisposition dispatchImport(const IopImportCall &call, CpuState &cpu)
        {
            const uint32_t a0 = cpu.gpr[4];
            auto setV0 = [&](uint32_t value)
            {
                cpu.gpr[2] = value;
            };

            if (iequals(call.library, "sysmem") && sysmem.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "cdvdman") && cdvd.dispatchImport(call.ordinal, cpu))
            {
                if (const auto callback = cdvd.takeCompletionCallback())
                {
                    pendingGuestCallbacks.emplace(
                        totalCycles + kCdvdCompletionCycles,
                        ScheduledGuestCallback{
                            callback->address,
                            callback->gp,
                            callback->reason,
                        });
                    m_idleValid = false;
                }
                return ImportDisposition::Handled;
            }

            if (iequals(call.library, "loadcore") && loadcore.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            if (iequals(call.library, "thbase") || iequals(call.library, "threadman"))
            {
                return kernel.dispatchThreadImport(call.ordinal, cpu, totalCycles)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thsemap"))
            {
                return kernel.dispatchSemaphoreImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "thevent"))
            {
                return kernel.dispatchEventImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "sifcmd"))
            {
                // Sound route only. Default IOP GetSreg stays the HLE that returns 0.
                // zsiopdrv's 0x8000 handler spins at sceSifGetSreg(31) until the EE's SET_SREG lands.
                if (soundRouteOn() && (call.ordinal == 6u || call.ordinal == 7u))
                {
                    const uint32_t index = cpu.gpr[4];
                    const uint32_t previous = sifSoftwareRegister(index);
                    if (call.ordinal == 7u)
                        setSifSoftwareRegister(index, cpu.gpr[5]);
                    if (call.ordinal == 6u && previous != 0u)
                    {
                        static int logged = 0;
                        if (logged < 4)
                        {
                            ++logged;
                            std::ostringstream out;
                            out << "[sifcmd] GetSreg " << index << " = 0x" << std::hex << previous;
                            log(LogLevel::Info, out.str());
                        }
                    }
                    setV0(previous);
                    return ImportDisposition::Handled;
                }
                return rpc.dispatchSifCmdImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "intrman") && intrman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "secrman"))
            {
                switch (call.ordinal)
                {
                case 4: // SecrSetMcCommandHandler
                    secrMcCommandHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                case 5: // SecrSetMcDevIDHandler
                    secrMcDevIdHandler = {a0, cpu.gpr[28]};
                    setV0(0);
                    return ImportDisposition::Handled;
                default:
                    break;
                }
            }
            if (iequals(call.library, "modload") && call.ordinal == 13u)
            {
                checkKelfPathCallback = {a0, cpu.gpr[28]};
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "ioman") && ioman.dispatchImport(call.ordinal, cpu, *this))
                return ImportDisposition::Handled;
            if (iequals(call.library, "libsd") && dispatchLibsd(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sifman"))
            {
                return rpc.dispatchSifManImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "vblank") && vblank.dispatchImport(call.ordinal, cpu, totalCycles))
                return ImportDisposition::Handled;
            if (iequals(call.library, "timrman") && timrman.dispatchImport(call.ordinal, cpu, totalCycles))
            {
                // Any timrman import can arm, stop, or move compare/overflow. GetSystemTime is thbase.
                m_idleValid = false;
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "dmacman"))
            {
                setV0(0);
                return ImportDisposition::Handled;
            }
            if (iequals(call.library, "stdio") && stdio.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;
            if (iequals(call.library, "sysclib"))
            {
                return sysclib.dispatchImport(call.ordinal, cpu)
                           ? ImportDisposition::Handled
                           : ImportDisposition::Missing;
            }
            if (iequals(call.library, "heaplib") && heaplib.dispatchImport(call.ordinal, cpu))
                return ImportDisposition::Handled;

            const uint32_t target = imports.resolve(call.library, call.ordinal, call.version);
            if (target != 0u)
            {
                cpu.pc = target;
                cpu.branchPending = false;
                return ImportDisposition::JumpToGuest;
            }

            std::ostringstream out;
            out << "[IOP] unhandled import " << call.library << ':' << call.ordinal
                << " version=0x" << std::hex << call.version << " pc=0x" << cpu.pc;
            log(LogLevel::Warning, out.str());
            setV0(0);
            return ImportDisposition::Missing;
        }

        bool step(CpuState &cpu)
        {
            if (cpu.stopped)
                return false;
            if (cpu.pc == kThreadReturnSentinel || cpu.pc == kCallReturnSentinel)
            {
                cpu.stopped = true;
                return false;
            }
            if (physicalAddress(cpu.pc) >= kRamSize)
            {
                std::ostringstream out;
                out << "[IOP] execution outside RAM pc=0x" << std::hex << cpu.pc;
                log(LogLevel::Error, out.str());
                cpu.stopped = true;
                return false;
            }
            if (checkInterrupt(cpu))
                return true;

            if (const auto import = imports.decode(cpu.pc))
            {
                const ImportDisposition disposition = dispatchImport(*import, cpu);
                ++totalInstructions;
                ++totalCycles;
                if (disposition == ImportDisposition::JumpToGuest)
                    return true;
                cpu.pc = cpu.gpr[31];
                cpu.branchPending = false;
                return !cpu.stopped;
            }

            const bool running = cpuCore.executeInstruction(cpu);
            // No IOP BIOS is mapped, so the vector at 0x80000080 is low RAM and a taken
            // exception spins until the RPC budget dies (sound-run-6: pc=0x80000a00).
            // On the sound route, resume past the faulting instruction and keep the driver going.
            if (cpu.exception && soundRouteOn())
            {
                const uint32_t epc = cpu.cop0[14];
                const uint32_t cause = cpu.cop0[13];
                const uint32_t code = (cause >> 2u) & 0x1Fu;
                static uint32_t lastEpc = 0xFFFFFFFFu;
                static int same = 0;
                static int logged = 0;
                if (epc == lastEpc)
                    ++same;
                else
                {
                    same = 0;
                    lastEpc = epc;
                }
                if (logged < 12 || same == 64)
                {
                    ++logged;
                    std::ostringstream out;
                    out << "[IOP] exception epc=0x" << std::hex << epc << " cause=0x" << cause
                        << " code=" << std::dec << code << " bad=0x" << std::hex << cpu.cop0[8]
                        << " insn=0x" << memory.read32(epc) << " same=" << std::dec << same;
                    log(LogLevel::Warning, out.str());
                }
                if (same >= 64)
                {
                    cpu.stopped = true;
                    return false;
                }
                const uint32_t status = cpu.cop0[12];
                cpu.cop0[12] = (status & ~0x3Fu) | ((status >> 2u) & 0xFu);
                const bool delaySlot = (cause & 0x80000000u) != 0u;
                cpu.pc = epc + (delaySlot ? 8u : 4u);
                cpu.branchPending = false;
                cpu.exception = false;
                schedulePendingDma();
                ++totalInstructions;
                ++totalCycles;
                return !cpu.stopped;
            }
            schedulePendingDma();
            ++totalInstructions;
            ++totalCycles;
            return running;
        }

        uint32_t runCpu(CpuState &cpu, uint32_t instructionBudget)
        {
            CpuState *previous = activeCpu;
            activeCpu = &cpu;
            const uint64_t start = totalInstructions;
            while (!cpu.stopped && !cpu.yielded && totalInstructions - start < instructionBudget)
            {
                if (!step(cpu))
                    break;
                if (!servicingDmaInterrupts && !pendingDmaInterrupts.empty())
                    servicePendingDmaInterrupts();
                if (!servicingGuestCallbacks && !pendingGuestCallbacks.empty())
                    servicePendingGuestCallbacks();
            }
            activeCpu = previous;
            return static_cast<uint32_t>(totalInstructions - start);
        }

        uint32_t callFunction(uint32_t address,
                              uint32_t a0,
                              uint32_t a1,
                              uint32_t a2,
                              uint32_t a3,
                              uint32_t gp,
                              uint32_t budget = kMaxCallInstructions)
        {
            struct CallDepthGuard
            {
                uint32_t &depth;
                ~CallDepthGuard() { --depth; }
            };

            const uint32_t depth = callDepth++;
            const CallDepthGuard depthGuard{callDepth};
            CpuState cpu{};
            cpu.pc = address;
            cpu.gpr[4] = a0;
            cpu.gpr[5] = a1;
            cpu.gpr[6] = a2;
            cpu.gpr[7] = a3;
            cpu.gpr[28] = gp;
            if (depth < kCallStackCapacity)
            {
                const uint32_t stackTop = kCallStackLimit - depth * kCallStackSize;
                cpu.gpr[29] = stackTop - 32u;
            }
            else if (activeCpu && activeCpu->gpr[29] > kCallStackBase + kStackGuardBytes)
            {
                // Extremely deep re-entrancy borrows unused space below the
                // suspended caller's live frame. Stack growth remains away
                // from the caller, so its saved registers stay intact.
                cpu.gpr[29] = (activeCpu->gpr[29] - kStackGuardBytes) & ~15u;
            }
            else
            {
                cpu.gpr[29] = kCallStackBase - 32u;
            }
            cpu.gpr[31] = kCallReturnSentinel;
            const uint32_t ran = runCpu(cpu, budget);
            if (soundRouteOn() && ran >= budget && cpu.pc != kCallReturnSentinel && !cpu.stopped)
            {
                static int aborts = 0;
                if (aborts < 8)
                {
                    ++aborts;
                    std::ostringstream out;
                    out << "[IOP] call budget hit entry=0x" << std::hex << address << " pc=0x" << cpu.pc
                        << " ran=" << std::dec << ran;
                    log(LogLevel::Warning, out.str());
                }
            }
            return cpu.gpr[2];
        }

        // EE sceSifSendCmd with a user cid: the IOP sifcmd handler runs with the packet (its extra payload is
        // already in IOP RAM).
        bool deliverUserSifCmd(uint32_t commandId, const void *packet, uint32_t packetSize)
        {
            uint32_t address = 0u;
            uint32_t handler = 0u;
            uint32_t harg = 0u;
            uint32_t gp = 0u;
            if (!rpc.stageUserCommand(commandId, packet, packetSize, address, handler, harg, gp))
                return false;
            (void)callFunction(handler, address, harg, 0u, 0u, gp);
            return true;
        }

        uint32_t executeGuestFunction(uint32_t address,
                                      uint32_t a0,
                                      uint32_t a1,
                                      uint32_t a2,
                                      uint32_t a3,
                                      uint32_t gp) override
        {
            return callFunction(address, a0, a1, a2, a3, gp);
        }

        uint32_t executeGuestFunctionWithBudget(uint32_t address,
                                                uint32_t a0,
                                                uint32_t a1,
                                                uint32_t a2,
                                                uint32_t a3,
                                                uint32_t gp,
                                                uint32_t instructionBudget) override
        {
            return callFunction(address, a0, a1, a2, a3, gp, instructionBudget);
        }

        // Not that good to use exception handling for control flow but will do for now
        void servicePendingDmaInterrupts()
        {
            if (servicingDmaInterrupts || pendingDmaInterrupts.empty())
                return;

            servicingDmaInterrupts = true;

            std::vector<int> completed;
            for (auto it = pendingDmaInterrupts.begin(); it != pendingDmaInterrupts.end();)
            {
                if (it->second > totalCycles)
                {
                    ++it;
                    continue;
                }
                completed.push_back(it->first);
                it = pendingDmaInterrupts.erase(it);
            }
            try
            {
                for (const int irq : completed)
                    (void)intrman.dispatchInterrupt(irq, *this);
            }
            catch (...)
            {
                servicingDmaInterrupts = false;
                throw;
            }
            servicingDmaInterrupts = false;
        }

        void servicePendingGuestCallbacks()
        {
            if (servicingGuestCallbacks || pendingGuestCallbacks.empty())
                return;

            std::vector<ScheduledGuestCallback> callbacks;
            for (auto it = pendingGuestCallbacks.begin(); it != pendingGuestCallbacks.end();)
            {
                if (it->first > totalCycles)
                    break;
                callbacks.push_back(it->second);
                it = pendingGuestCallbacks.erase(it);
            }
            if (callbacks.empty())
                return;

            servicingGuestCallbacks = true;
            try
            {
                for (const ScheduledGuestCallback &callback : callbacks)
                {
                    if (callback.function != 0u)
                    {
                        (void)callFunction(callback.function,
                                           callback.argument,
                                           0u,
                                           0u,
                                           0u,
                                           callback.gp,
                                           100000u);
                    }
                }
            }
            catch (...)
            {
                servicingGuestCallbacks = false;
                throw;
            }
            servicingGuestCallbacks = false;
        }

        void runCycles(uint64_t cycles) noexcept
        {
            try
            {
                const uint64_t target = totalCycles + cycles;
                while (totalCycles < target)
                {
                    // Sealed while ready and delay queues cannot grow a runnable thread.
                    // Land on m_idleUntil and fall through so an event exactly at that cycle
                    // is serviced, matching the old jump-then-continue loop.
                    if (m_idleCacheEnabled && m_idleValid && kernel.schedEpoch() == m_idleEpoch &&
                        totalCycles < m_idleUntil)
                    {
                        const uint64_t stop = std::min(target, m_idleUntil);
                        if (stop > totalCycles)
                            ++m_idleSkipCount;
                        totalCycles = stop;
                        if (totalCycles >= target)
                            break;
                    }

                    servicePendingDmaInterrupts();
                    servicePendingGuestCallbacks();
                    timrman.serviceDue(totalCycles, *this);
                    IopThread *next = kernel.beginNextReady(totalCycles);
                    if (!next)
                    {
                        // Horizon past this call's target, so the next small runEeCycles can skip
                        // the kernel entirely. Do not pass target into nextWakeCycle.
                        uint64_t horizon = kernel.nextWakeCycle(~uint64_t{0});
                        for (const auto &[irq, completionCycle] : pendingDmaInterrupts)
                            horizon = std::min(horizon, completionCycle);
                        if (!pendingGuestCallbacks.empty())
                            horizon = std::min(horizon, pendingGuestCallbacks.begin()->first);
                        horizon = timrman.nextEventCycle(horizon);
                        if (m_idleCacheEnabled && horizon > totalCycles)
                        {
                            m_idleUntil = horizon;
                            m_idleEpoch = kernel.schedEpoch();
                            m_idleValid = true;
                        }
                        else
                            m_idleValid = false;
                        totalCycles = std::max(totalCycles + 1u, std::min(target, horizon));
                        continue;
                    }
                    m_idleValid = false;
                    const uint64_t before = totalCycles;
                    runCpu(next->cpu, static_cast<uint32_t>(std::min<uint64_t>(kDefaultSlice, target - totalCycles)));
                    kernel.endTimeslice(*next, kThreadReturnSentinel);
                    if (totalCycles == before)
                        ++totalCycles;
                }
                drainMixedAudio();
            }
            catch (...)
            {
                // Runtime scheduling must never throw through EeScheduler::accountCycles().
            }
        }

        ModuleLoadResult loadImage(std::string path, std::span<const uint8_t> image, const void *arguments, uint32_t argumentSize)
        {
            ModuleLoadResult result{true, -1, -1};
            const IopImageLoadResult loaded = IopModuleLoader::load(image, memory, moduleCursor);
            moduleCursor = loaded.nextModuleCursor;
            if (!loaded)
            {
                if (loaded.error == IopImageLoadError::InvalidElf)
                    log(LogLevel::Error, "[IOP] rejected invalid/non-MIPS IRX ELF");
                else if (loaded.error == IopImageLoadError::ArenaExhausted)
                    log(LogLevel::Error, "[IOP] module arena exhausted");
                return result;
            }
            if (!loaded.relocationsComplete)
                log(LogLevel::Warning, "[IOP] one or more IRX relocations were unsupported");

            Module module;
            module.id = nextModuleId++;
            module.path = std::move(path);
            const size_t slash = module.path.find_last_of("/\\:");
            module.name = slash == std::string::npos ? module.path : module.path.substr(slash + 1u);
            module.base = loaded.base;
            module.size = loaded.size;
            module.entry = loaded.entry;
            module.gp = loaded.gp;

            uint32_t args = 0u;
            if (arguments && argumentSize)
            {
                args = allocate(argumentSize + 1u, 16u);
                if (args)
                {
                    writeRam(args, arguments, argumentSize);
                    write8(args + argumentSize, 0u);
                }
            }
            const uint32_t startResult = callFunction(module.entry, argumentSize, args, 0u, 0u, module.gp);
            if (args)
                freeAllocation(args);
            module.resident = startResult == 0u || startResult == 2u;
            result.moduleId = module.id;
            result.startResult = static_cast<int32_t>(startResult);
            modules[module.id] = std::move(module);

            std::ostringstream out;
            out << "[IOP] loaded IRX id=" << result.moduleId
                << " entry=0x" << std::hex << modules[result.moduleId].entry
                << " base=0x" << modules[result.moduleId].base
                << " start=" << std::dec << result.startResult;
            log(LogLevel::Info, out.str());
            // callFunction runs only the entry. An IRX that CreateThread's its RPC server and returns
            // would otherwise still be unregistered when sceSifLoadModuleBuffer returns to the EE.
            runCycles(2000000u);
            return result;
        }

        ModuleLoadResult loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            if (!IopModuleLoader::readWholeHostFile(host, path, image))
            {
                log(LogLevel::Warning, std::string("[IOP] failed to open IRX '") + std::string(path) + "'");
                return {true, -1, -1};
            }
            return loadImage(std::string(path), image, arguments, argumentSize);
        }

        ModuleLoadResult loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
        {
            std::vector<uint8_t> image;
            // The game DMAs the IRX package into IOP RAM, then passes that IOP address.
            // readGuest on the runtime host reads EE RAM (unit tests plant the image there instead).
            if (!readElfImageFromIopRam(memory, guestAddress, image) &&
                !IopModuleLoader::readElfFromGuest(host, guestAddress, image))
            {
                uint8_t mag[4]{};
                (void)memory.readRam(guestAddress, mag, sizeof(mag));
                std::ostringstream out;
                out << "[IOP] loadModuleBuffer 0x" << std::hex << guestAddress << " not an ELF, iop bytes "
                    << static_cast<unsigned>(mag[0]) << " " << static_cast<unsigned>(mag[1]) << " "
                    << static_cast<unsigned>(mag[2]) << " " << static_cast<unsigned>(mag[3]);
                log(LogLevel::Warning, out.str());
                return {true, -1, -1};
            }
            std::ostringstream tag;
            tag << "buffer@0x" << std::hex << guestAddress;
            return loadImage(tag.str(), image, arguments, argumentSize);
        }

        bool stopModule(int32_t moduleId, int32_t *result)
        {
            auto it = modules.find(moduleId);
            if (it == modules.end())
                return false;
            // A removable IRX normally exposes a stop entry through module metadata. We do not guess it; terminate owned execution and release the image cleanly.
            kernel.terminateThreadsInRange(it->second.base, it->second.size);
            rpc.removeServersInRange(it->second.base, it->second.size);
            imports.eraseRange(it->second.base, it->second.size);
            modules.erase(it);
            kernel.cleanupDeadThreads();
            if (result)
                *result = 0;
            return true;
        }

        IopHost &host;
        IopMemory memory;
        IopSysmem sysmem;
        IopKernel kernel;
        IopCdvd cdvd;
        IopVblank vblank;
        IopRpcBridge rpc;
        IopSysclib sysclib;
        IopStdio stdio;
        IopHeaplib heaplib;
        IopIntrman intrman;
        IopTimrman timrman;
        IopIoman ioman;
        struct LibsdState
        {
            std::array<uint32_t, 2> transHandler{};
            std::array<uint32_t, 2> transGp{};
            std::array<uint32_t, 2> transArg{};
            std::array<uint32_t, 2> spu2Handler{};
            std::array<uint32_t, 2> spu2Gp{};
            std::array<uint32_t, 2> spu2Arg{};
            uint32_t irqCallback = 0;
            uint32_t irqCallbackGp = 0;
            std::array<bool, 2> blockActive{};
            std::array<uint32_t, 2> blockBuffer{};
            std::array<uint32_t, 2> blockSize{};
        };
        LibsdState m_libsd{};
        SoundEngine m_sound;
        bool m_inSpu2Irq = false;
        bool m_inSpu2Pump = false;
        int m_soundRoute = -1;
        // TEKKEN4_IOP_IDLE_SKIP=0 turns the idle-skip cache off (paired speed runs with one exe).
        bool m_idleCacheEnabled = []
        {
            const char *env = std::getenv("TEKKEN4_IOP_IDLE_SKIP");
            return !(env && env[0] == '0' && env[1] == '\0');
        }();
        bool m_idleValid = false;
        uint64_t m_idleUntil = 0;
        uint64_t m_idleEpoch = 0;
        uint64_t m_idleSkipCount = 0;
        IopCpuCore cpuCore;
        IopImportRegistry imports;
        IopLoadcore loadcore;
        std::map<int, Module> modules;
        std::map<int, uint64_t> pendingDmaInterrupts;
        std::multimap<uint64_t, ScheduledGuestCallback> pendingGuestCallbacks;
        uint32_t nextModuleId = 1;
        uint32_t moduleCursor = kModuleLoadBase;
        uint64_t totalCycles = 0;
        uint64_t totalInstructions = 0;
        uint64_t eeCycleCarry = 0;
        CpuState *activeCpu = nullptr;
        std::string lastError;
        bool servicingDmaInterrupts = false;
        bool servicingGuestCallbacks = false;
        uint32_t callDepth = 0u;
        GuestCallback secrMcCommandHandler;
        GuestCallback secrMcDevIdHandler;
        GuestCallback checkKelfPathCallback;
    };

    IopEmulator::IopEmulator(IopHost &host)
        : m_impl(std::make_unique<Impl>(host))
    {
        g_userCmdEmulator = this;
    }

    IopEmulator::~IopEmulator()
    {
        if (g_userCmdEmulator == this)
            g_userCmdEmulator = nullptr;
    }

    void IopEmulator::reset()
    {
        m_impl->reset();
    }

    ModuleLoadResult IopEmulator::loadModule(std::string_view path, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModule(path, arguments, argumentSize);
    }

    ModuleLoadResult IopEmulator::loadModuleBuffer(uint32_t guestAddress, const void *arguments, uint32_t argumentSize)
    {
        return m_impl->loadModuleBuffer(guestAddress, arguments, argumentSize);
    }

    bool IopEmulator::stopModule(int32_t moduleId, int32_t *result)
    {
        return m_impl->stopModule(moduleId, result);
    }

    void IopEmulator::runEeCycles(uint64_t eeCycles) noexcept
    {
        // TEKKEN4_IOP_DUMP=<dir>: IOP RAM to <dir>/iop_<s>s.bin every 2 s of wall time, to diff against the
        // iopMemory.bin of a PCSX2 savestate taken on the same screen (tools\iop_diff.py).
        static const char *dumpDir = std::getenv("TEKKEN4_IOP_DUMP");
        if (dumpDir && dumpDir[0])
        {
            using Clock = std::chrono::steady_clock;
            static const Clock::time_point start = Clock::now();
            static int64_t nextSecond = 2;
            const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - start).count();
            if (now >= nextSecond)
            {
                std::vector<uint8_t> ram(kRamSize);
                const std::string path = std::string(dumpDir) + "/iop_" + std::to_string(now) + "s.bin";
                if (m_impl->memory.readRam(0, ram.data(), ram.size()))
                    if (FILE *f = std::fopen(path.c_str(), "wb"))
                    {
                        std::fwrite(ram.data(), 1, ram.size(), f);
                        std::fclose(f);
                    }
                nextSecond = now + 2;
            }
        }
        const uint64_t total = m_impl->eeCycleCarry + eeCycles;
        const uint64_t iopCycles = total / 8u;
        m_impl->eeCycleCarry = total % 8u;
        if (iopCycles)
            m_impl->runCycles(iopCycles);
    }

    RpcResult IopEmulator::handleRpc(const RpcRequest &request)
    {
        return m_impl->rpc.handleRpc(request, *m_impl);
    }

    bool IopEmulator::deliverUserSifCmd(uint32_t commandId, const void *packet, uint32_t packetSize)
    {
        return m_impl->deliverUserSifCmd(commandId, packet, packetSize);
    }

    bool IopEmulator::hasRpcServer(uint32_t sid) const noexcept
    {
        return m_impl->rpc.hasServer(sid);
    }

    void IopEmulator::onSifTransfer(const SifTransfer &transfer)
    {
        m_impl->rpc.onSifTransfer(transfer);
    }

    uint32_t IopEmulator::allocateMemory(uint32_t size, uint32_t alignment)
    {
        return m_impl->memory.allocate(size, alignment);
    }

    bool IopEmulator::freeMemory(uint32_t address)
    {
        return m_impl->memory.freeAllocation(address);
    }

    bool IopEmulator::readMemory(uint32_t address, void *destination, size_t size) const
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.readRam(address, destination, size);
    }

    bool IopEmulator::writeMemory(uint32_t address, const void *source, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.writeRam(address, source, size);
    }

    bool IopEmulator::zeroMemory(uint32_t address, size_t size)
    {
        return isMemoryRange(address, size) &&
               m_impl->memory.zeroRam(address, size);
    }

    bool IopEmulator::isMemoryRange(uint32_t address, size_t size) const
    {
        const bool physicalSegment = address < IopMemory::RamSize;
        const bool cachedSegment = address >= 0x80000000u && address < 0x80200000u;
        const bool uncachedSegment = address >= 0xA0000000u && address < 0xA0200000u;
        if (!physicalSegment && !cachedSegment && !uncachedSegment)
            return false;
        const uint32_t physical = IopMemory::physicalAddress(address);
        return physical <= IopMemory::RamSize && size <= IopMemory::RamSize - physical;
    }

    uint64_t IopEmulator::cycles() const noexcept
    {
        return m_impl->totalCycles;
    }

    void IopEmulator::setIdleSkipCache(bool enabled) noexcept
    {
        m_impl->m_idleCacheEnabled = enabled;
        if (!enabled)
            m_impl->m_idleValid = false;
    }

    void IopEmulator::testSealIdleSkip(uint64_t untilCycle) noexcept
    {
        if (!m_impl->m_idleCacheEnabled)
            return;
        m_impl->m_idleUntil = untilCycle;
        m_impl->m_idleEpoch = m_impl->kernel.schedEpoch();
        m_impl->m_idleValid = true;
    }

    uint64_t IopEmulator::testIdleSkipCount() const noexcept
    {
        return m_impl->m_idleSkipCount;
    }

    uint64_t IopEmulator::instructions() const noexcept
    {
        return m_impl->totalInstructions;
    }

    uint32_t IopEmulator::loadedModuleCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->modules.size());
    }

    uint32_t IopEmulator::threadCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->kernel.threadCount());
    }

    uint32_t IopEmulator::rpcServerCount() const noexcept
    {
        return static_cast<uint32_t>(m_impl->rpc.serverCount());
    }

}

namespace ps2x::iop
{
    namespace
    {
        std::array<uint32_t, 32> g_sifSoftwareRegisters{};
    }

    void setSifSoftwareRegister(uint32_t index, uint32_t value) noexcept
    {
        if (index < g_sifSoftwareRegisters.size())
            g_sifSoftwareRegisters[index] = value;
    }

    uint32_t sifSoftwareRegister(uint32_t index) noexcept
    {
        return index < g_sifSoftwareRegisters.size() ? g_sifSoftwareRegisters[index] : 0u;
    }

    bool deliverEeUserSifCmd(uint32_t commandId, const void *packet, size_t packetSize) noexcept
    {
        if (detail::g_userCmdEmulator == nullptr || packet == nullptr || packetSize < 16u || packetSize > 112u)
            return false;
        return detail::g_userCmdEmulator->deliverUserSifCmd(commandId, packet, static_cast<uint32_t>(packetSize));
    }
}
