// Ported from God-Of-War-Native-Port origin/runtime ps2xIOP/src/emulator/core/sound_engine.cpp (GPL-3.0).
// SPU2 mixer used at the LIBSD import boundary. Reverb runs from the libsd effect presets.
#include "sound_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr int32_t kAdpcmFilters[5][2] = {{0, 0}, {60, 0}, {115, -52}, {98, -55}, {122, -60}};
        constexpr size_t kMaxBufferedFrames = 48000u * 2u; // 2 s; without a consumer the oldest samples are dropped

        // TEKKEN4_SOUND_TRACE=1. Off unless that env is exactly "1". One line per mixer second plus KON/KOFF.
        bool soundTrace()
        {
            static const bool on = [] {
                const char *value = std::getenv("TEKKEN4_SOUND_TRACE");
                return value != nullptr && value[0] == '1' && value[1] == '\0';
            }();
            return on;
        }

        struct TraceState
        {
            uint64_t stepped = 0;
            uint64_t dropped = 0;
            uint64_t maxAdvance = 0;
            uint64_t irqHits = 0;
            int peak = 0;
            int lastActive = -1;
            std::chrono::steady_clock::time_point t0{};
            bool started = false;
        };

        TraceState &traceState()
        {
            static TraceState state;
            return state;
        }

        double traceMixSeconds()
        {
            return static_cast<double>(traceState().stepped) / 48000.0;
        }

        double traceWallSeconds()
        {
            TraceState &state = traceState();
            if (!state.started)
                return 0.0;
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - state.t0).count();
        }

        // libsd ids (entry >> 8), in the order of LIBSD's register table (LIBSD.IRX 0x4930, docs/sound.md)
        enum : uint32_t
        {
            IdVolL = 0x00, IdVolR = 0x01, IdPitch = 0x02, IdAdsr1 = 0x03, IdAdsr2 = 0x04, IdEnvx = 0x05, IdVolXL = 0x06,
            IdVolXR = 0x07, IdMmix = 0x08, IdMvolL = 0x09, IdMvolR = 0x0A, IdEvolL = 0x0B, IdEvolR = 0x0C, IdAvolL = 0x0D,
            IdAvolR = 0x0E, IdBvolL = 0x0F, IdBvolR = 0x10, IdMvolXL = 0x11, IdMvolXR = 0x12,
            IdPmon = 0x13, IdNon = 0x14, IdKon = 0x15, IdKoff = 0x16, IdEndx = 0x17, IdVmixL = 0x18, IdVmixEL = 0x19,
            IdVmixR = 0x1A, IdVmixER = 0x1B,
            IdEsa = 0x1C, IdEea = 0x1D, IdTsa = 0x1E, IdIrqa = 0x1F, IdSsa = 0x20, IdLsa = 0x21, IdNax = 0x22,
        };

        int16_t clamp16(int32_t value) noexcept
        {
            return static_cast<int16_t>(std::clamp(value, -32768, 32767));
        }

        // voice / core volume: fixed mode (bit 15 = 0) = 15-bit signed; sweep mode applies its target at once
        int32_t volumeOf(uint16_t value) noexcept
        {
            if ((value & 0x8000u) == 0u)
                return static_cast<int16_t>(value << 1u) >> 1; // -0x4000..0x3FFF
            return (value & 0x2000u) != 0u ? 0 : 0x3FFF;
        }

        uint32_t idOf(uint32_t entry) noexcept { return (entry >> 8u) & 0xFFu; }
        uint32_t coreOf(uint32_t entry) noexcept { return entry & 1u; }
        uint32_t voiceOf(uint32_t entry) noexcept { return (entry >> 1u) & 0x1Fu; }
    }

    SoundEngine::SoundEngine() : m_ram(RamHalfwords)
    {
        reset();
    }

    void SoundEngine::reset()
    {
        std::fill(m_ram.begin(), m_ram.end(), uint16_t{0});
        m_cores = {};
        m_stream = {};
        m_streamHalf = {};
        m_irqLatched = {};
        for (Reverb &reverb : m_reverb)
        {
            reverb.enabled = false;
            reverb.coef.fill(0);
            reverb.pos = 0;
            reverb.half = false;
            reverb.accL = reverb.accR = reverb.holdL = reverb.holdR = 0;
            std::fill(reverb.buf.begin(), reverb.buf.end(), int16_t{0});
        }
        {
            std::lock_guard<std::mutex> lock(m_outputMutex);
            m_output.clear();
        }
        m_lastCycle = 0;
    }

    void SoundEngine::initCore(uint32_t core)
    {
        Core &c = m_cores[core & 1u];
        const std::array<Voice, VoicesPerCore> voices{};
        c = Core{};
        c.voices = voices;
        m_stream[core & 1u] = {};
        m_irqLatched[core & 1u] = false;
    }

    // ---- parameters ----------------------------------------------------------------------------------------------

    void SoundEngine::setParam(uint32_t entry, uint16_t value)
    {
        Core &c = m_cores[coreOf(entry)];
        const uint32_t voice = voiceOf(entry);
        // GOW_SND_REGLOG=1: core-level writes (MMIX, master/effect/input volumes) as the driver sets them
        static const bool regLog = std::getenv("GOW_SND_REGLOG") != nullptr;
        if (regLog && idOf(entry) >= IdMmix && idOf(entry) <= IdBvolR)
            std::fprintf(stderr, "[snd-reg] core=%u id=%02x value=%04x\n", coreOf(entry), idOf(entry), value);
        // GOW_SND_CALLLOG=1: voice volume / pitch writes that change the value
        static const bool callLog = std::getenv("GOW_SND_CALLLOG") != nullptr;
        if (callLog && voice < VoicesPerCore && idOf(entry) <= IdPitch)
        {
            const Voice &v = c.voices[voice];
            const uint16_t old = idOf(entry) == IdVolL ? v.volL : idOf(entry) == IdVolR ? v.volR : v.pitch;
            if (old != value)
                std::fprintf(stderr, "[snd-call] core=%u voice=%u id=%02x %04x -> %04x\n", coreOf(entry), voice, idOf(entry), old, value);
        }
        switch (idOf(entry))
        {
        case IdVolL: if (voice < VoicesPerCore) c.voices[voice].volL = value; break;
        case IdVolR: if (voice < VoicesPerCore) c.voices[voice].volR = value; break;
        case IdPitch: if (voice < VoicesPerCore) c.voices[voice].pitch = value; break;
        case IdAdsr1: if (voice < VoicesPerCore) c.voices[voice].adsr1 = value; break;
        case IdAdsr2: if (voice < VoicesPerCore) c.voices[voice].adsr2 = value; break;
        case IdMmix: c.mmix = value; break;
        case IdMvolL: c.mvolL = value; break;
        case IdMvolR: c.mvolR = value; break;
        case IdEvolL: c.evolL = value; break;
        case IdEvolR: c.evolR = value; break;
        case IdAvolL: c.avolL = value; break;
        case IdAvolR: c.avolR = value; break;
        case IdBvolL: c.bvolL = value; break;
        case IdBvolR: c.bvolR = value; break;
        default: break; // ENVX / VOLX / MVOLX are read-only
        }
        const uint32_t id = idOf(entry);
        if (id >= IdMmix && id <= IdBvolR)
        {
            static uint32_t logged = 0;
            static uint16_t seen[2][9] = {};
            const uint32_t core = coreOf(entry);
            const uint32_t slot = id - IdMmix;
            if (core < 2u && slot < 9u && seen[core][slot] != value && logged < 32u)
            {
                seen[core][slot] = value;
                ++logged;
                std::fprintf(stderr, "[snd-mix] core=%u id=%02x value=%04x\n", core, id, value);
            }
        }
    }

    uint16_t SoundEngine::getParam(uint32_t entry) const
    {
        const Core &c = m_cores[coreOf(entry)];
        const uint32_t voice = std::min(voiceOf(entry), VoicesPerCore - 1u);
        const Voice &v = c.voices[voice];
        switch (idOf(entry))
        {
        case IdVolL: return v.volL;
        case IdVolR: return v.volR;
        case IdPitch: return v.pitch;
        case IdAdsr1: return v.adsr1;
        case IdAdsr2: return v.adsr2;
        case IdEnvx: return static_cast<uint16_t>(v.envelope);
        case IdVolXL: return static_cast<uint16_t>(volumeOf(v.volL) << 1);
        case IdVolXR: return static_cast<uint16_t>(volumeOf(v.volR) << 1);
        case IdMmix: return c.mmix;
        case IdMvolL: return c.mvolL;
        case IdMvolR: return c.mvolR;
        case IdEvolL: return c.evolL;
        case IdEvolR: return c.evolR;
        case IdAvolL: return c.avolL;
        case IdAvolR: return c.avolR;
        case IdBvolL: return c.bvolL;
        case IdBvolR: return c.bvolR;
        case IdMvolXL: return static_cast<uint16_t>(volumeOf(c.mvolL) << 1);
        case IdMvolXR: return static_cast<uint16_t>(volumeOf(c.mvolR) << 1);
        default: return 0;
        }
    }

    // ---- switches (24-bit voice masks) ---------------------------------------------------------------------------

    void SoundEngine::setSwitch(uint32_t entry, uint32_t mask)
    {
        Core &c = m_cores[coreOf(entry)];
        mask &= 0xFFFFFFu;
        static const bool callLog = std::getenv("GOW_SND_CALLLOG") != nullptr;
        if (callLog && (idOf(entry) == IdKon || idOf(entry) == IdKoff) && mask != 0u)
            std::fprintf(stderr, "[snd-call] core=%u %s %06x\n", coreOf(entry), idOf(entry) == IdKon ? "KON" : "KOFF", mask);
        if (soundTrace() && (idOf(entry) == IdKon || idOf(entry) == IdKoff) && mask != 0u)
            std::fprintf(stderr, "[snd-trace] mix=%.3f wall=%.3f %s core=%u mask=%06x\n", traceMixSeconds(),
                         traceWallSeconds(), idOf(entry) == IdKon ? "KON" : "KOFF", coreOf(entry), mask);
        switch (idOf(entry))
        {
        case IdPmon: c.pmon = mask; break;
        case IdNon: c.non = mask; break;
        case IdKon: keyOn(c, mask); break;
        case IdKoff: keyOff(c, mask); break;
        case IdEndx: c.endx = 0; break; // writing ENDX clears it
        case IdVmixL: c.vmixL = mask; break;
        case IdVmixEL: c.vmixEL = mask; break;
        case IdVmixR: c.vmixR = mask; break;
        case IdVmixER: c.vmixER = mask; break;
        default: break;
        }
    }

    uint32_t SoundEngine::getSwitch(uint32_t entry) const
    {
        const Core &c = m_cores[coreOf(entry)];
        switch (idOf(entry))
        {
        case IdPmon: return c.pmon;
        case IdNon: return c.non;
        case IdEndx: return c.endx;
        case IdVmixL: return c.vmixL;
        case IdVmixEL: return c.vmixEL;
        case IdVmixR: return c.vmixR;
        case IdVmixER: return c.vmixER;
        default: return 0; // KON / KOFF read back as 0
        }
    }

    // ---- addresses (libsd: byte addresses, 16-byte aligned; stored as halfwords) ---------------------------------

    void SoundEngine::setAddr(uint32_t entry, uint32_t byteAddress)
    {
        Core &c = m_cores[coreOf(entry)];
        const uint32_t halfwords = (byteAddress >> 1u) & (RamHalfwords - 1u) & ~7u;
        const uint32_t voice = voiceOf(entry);
        switch (idOf(entry))
        {
        case IdEsa: c.esa = halfwords; break;
        case IdEea: c.eea = halfwords; break;
        case IdTsa: c.tsa = halfwords; break;
        case IdIrqa: c.irqa = halfwords; break;
        case IdSsa:
            if (voice < VoicesPerCore)
                c.voices[voice].ssa = halfwords;
            break;
        case IdLsa:
            if (voice < VoicesPerCore)
            {
                c.voices[voice].lsa = halfwords;
                c.voices[voice].lsaFromDriver = true; // the driver's loop start wins over block flags
            }
            break;
        case IdNax:
            if (voice < VoicesPerCore)
            {
                Voice &v = c.voices[voice];
                v.nax = halfwords;
                v.playBlock = halfwords; // reads back as written until the next block is decoded
                v.playStarted = false;
            }
            break;
        default: break;
        }
    }

    uint32_t SoundEngine::getAddr(uint32_t entry) const
    {
        const Core &c = m_cores[coreOf(entry)];
        const Voice &v = c.voices[std::min(voiceOf(entry), VoicesPerCore - 1u)];
        uint32_t halfwords = 0;
        switch (idOf(entry))
        {
        case IdEsa: halfwords = c.esa; break;
        case IdEea: halfwords = c.eea; break;
        case IdTsa: halfwords = c.tsa; break;
        case IdIrqa: halfwords = c.irqa; break;
        case IdSsa: halfwords = v.ssa; break;
        case IdLsa: halfwords = v.lsa; break;
        case IdNax: halfwords = naxOf(v); break;
        default: break;
        }
        return halfwords << 1u;
    }

    // NAX as PCSX2 reports NextA: always inside the block being played (block + 1 .. block + 7). 'nax' is the next block
    // to decode; during the last block of a streaming half it already lies past the half, and 989snd (989NOMID.IRX
    // 0x13258) then aborts the stream (error 0x35): the music restarted every few seconds.
    uint32_t SoundEngine::naxOf(const Voice &v) noexcept
    {
        if (!v.playStarted)
            return (v.playBlock | 1u) & (RamHalfwords - 1u);
        const uint32_t played = std::min<uint32_t>(v.decodedIndex, 27u) / 4u; // 0..6 (4 samples per halfword)
        return (v.playBlock + 1u + played) & (RamHalfwords - 1u);
    }

    // sceSdSetCoreAttr ids (entry & ~1): 2 effect enable, 4 IRQ enable, 6 mute, 8 noise clock, 0xA SPDIF mode
    void SoundEngine::setCoreAttr(uint32_t entry, uint32_t value)
    {
        Core &c = m_cores[coreOf(entry)];
        switch (entry & ~1u)
        {
        case 2: c.attr = static_cast<uint16_t>((c.attr & ~0x0080u) | (value ? 0x0080u : 0u)); break;
        case 4: c.attr = static_cast<uint16_t>((c.attr & ~0x0040u) | (value ? 0x0040u : 0u)); break;
        case 6: c.attr = static_cast<uint16_t>((c.attr & ~0x4000u) | (value ? 0x4000u : 0u)); break;
        case 8: c.attr = static_cast<uint16_t>((c.attr & ~0x3F00u) | ((value & 0x3Fu) << 8u)); break;
        default: break;
        }
    }

    uint32_t SoundEngine::getCoreAttr(uint32_t entry) const
    {
        const Core &c = m_cores[coreOf(entry)];
        switch (entry & ~1u)
        {
        case 2: return (c.attr >> 7u) & 1u;
        case 4: return (c.attr >> 6u) & 1u;
        case 6: return (c.attr >> 14u) & 1u;
        case 8: return (c.attr >> 8u) & 0x3Fu;
        default: return 0u;
        }
    }

    // libsd effect presets: 10 modes x 32 SPU reverb words. The packed words are the table in ps2sdk
    // iop/sound/libsd/src/effect.c (published libsd). Little-endian halves are the psx-spx register order:
    // dAPF1 dAPF2 vIIR vCOMB1..4 vWALL vAPF1 vAPF2 mLSAME mRSAME mLCOMB1 mRCOMB1 mLCOMB2 mRCOMB2
    // dLSAME dRSAME mLDIFF mRDIFF mLCOMB3 mRCOMB3 mLCOMB4 mRCOMB4 dLDIFF dRDIFF mLAPF1 mRAPF1 mLAPF2 mRAPF2 vLIN vRIN.
    // Mode 9 is pipe (the source comment on that row says CLEAR; 0x100 is the clear flag, not a row).
    static const uint32_t kEffectPacked[10][16] = {
        {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00010001, 0x00000000, 0x00000000,
         0x00000000, 0x00010001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000},
        {0x005B007D, 0x54B86D80, 0x0000BED0, 0xBA800000, 0x53005800, 0x033304D6, 0x022703F0, 0x01EF0374,
         0x01B70336, 0x01B60335, 0x01B50334, 0x01B50334, 0x01B50334, 0x013601B4, 0x005C00B8, 0x80008000},
        {0x00250033, 0x4FA870F0, 0x4410BCE0, 0x9C00C0F0, 0x4EC05280, 0x031B03E4, 0x02AF03A4, 0x02660372,
         0x025D031C, 0x018E025C, 0x0135022F, 0x00B701D2, 0x00B5018F, 0x008000B4, 0x0026004C, 0x80008000},
        {0x007F00B1, 0x4FA870F0, 0x4510BCE0, 0xB4C0BEF0, 0x4EC05280, 0x076B0904, 0x065F0824, 0x061607A2,
         0x05ED076C, 0x042E05EC, 0x0305050F, 0x02B70462, 0x0265042F, 0x01B20264, 0x00800100, 0x80008000},
        {0x00A900E3, 0x4FA86F60, 0x4510BCE0, 0xA680BEF0, 0x52C05680, 0x0B580DFB, 0x0A3C0D09, 0x09730BD9,
         0x08DA0B59, 0x05E908D9, 0x04B007EC, 0x03D206EF, 0x031D05EA, 0x0238031C, 0x00AA0154, 0x80008000},
        {0x013901A5, 0x50006000, 0xB8004C00, 0xC000BC00, 0x5C006000, 0x11BB15BA, 0x10BD14C2, 0x0DC111BC,
         0x0DC311C0, 0x09C10DC0, 0x07C10BC4, 0x06CD0A00, 0x05C109C2, 0x041A05C0, 0x013A0274, 0x80008000},
        {0x0231033D, 0x50007E00, 0xB000B400, 0xB0004C00, 0x54006000, 0x1A311ED6, 0x183B1D14, 0x16B21BC2,
         0x15EF1A32, 0x105515EE, 0x0F2D1334, 0x0C5D11F6, 0x0AE11056, 0x07A20AE0, 0x02320464, 0x80008000},
        {0x00030003, 0x7FFF7FFF, 0x00000000, 0x81000000, 0x00000000, 0x0FFD1FFD, 0x00091009, 0x00000000,
         0x00091009, 0x1FFF1FFF, 0x1FFE1FFE, 0x1FFE1FFE, 0x1FFE1FFE, 0x10041008, 0x00040008, 0x80008000},
        {0x00030003, 0x7FFF7FFF, 0x00000000, 0x00000000, 0x00000000, 0x0FFD1FFD, 0x00091009, 0x00000000,
         0x00091009, 0x1FFF1FFF, 0x1FFE1FFE, 0x1FFE1FFE, 0x1FFE1FFE, 0x10041008, 0x00040008, 0x80008000},
        {0x00130017, 0x4FA870F0, 0x4510BCE0, 0x8500BEF0, 0x54C05F80, 0x02AF0371, 0x01DF02E5, 0x01D702B0,
         0x026A0358, 0x011E01D6, 0x00B1012D, 0x0059011F, 0x00E301A0, 0x00400058, 0x00140028, 0x80008000},
    };

    void SoundEngine::setEffectAttr(uint32_t core, int32_t mode, int16_t depthL, int16_t depthR, int32_t delay, int32_t feedback)
    {
        core &= 1u;
        const bool clear = (static_cast<uint32_t>(mode) & 0x100u) != 0u;
        mode = static_cast<int32_t>(static_cast<uint32_t>(mode) & ~0x100u);
        if (mode < 0 || mode > 9)
            return;
        Reverb &reverb = m_reverb[core];
        if (reverb.buf.empty())
            reverb.buf.assign(0x20000u, 0);
        for (uint32_t i = 0; i < 16u; ++i)
        {
            reverb.coef[i * 2u] = static_cast<uint16_t>(kEffectPacked[mode][i]);
            reverb.coef[i * 2u + 1u] = static_cast<uint16_t>(kEffectPacked[mode][i] >> 16);
        }
        // echo/delay: libsd rewrites vWALL and the address words from the caller's delay and feedback
        if (mode == 7 || mode == 8)
        {
            const uint16_t scaledDelay = static_cast<uint16_t>((delay + 1) << 5);
            const uint16_t scaledFeedback = static_cast<uint16_t>(((feedback << 7) + feedback) << 1);
            uint16_t *p = reverb.coef.data();
            p[7] = scaledFeedback;
            p[10] = static_cast<uint16_t>((scaledDelay << 1) - p[0]);
            p[11] = static_cast<uint16_t>(scaledDelay - p[1]);
            p[16] = static_cast<uint16_t>(p[17] + scaledDelay);
            p[12] = static_cast<uint16_t>(p[13] + scaledDelay);
            p[27] = static_cast<uint16_t>(p[29] + scaledDelay);
            p[26] = static_cast<uint16_t>(p[28] + scaledDelay);
        }
        if (clear)
            std::fill(reverb.buf.begin(), reverb.buf.end(), int16_t{0});
        reverb.pos = 0;
        reverb.half = false;
        reverb.accL = reverb.accR = reverb.holdL = reverb.holdR = 0;
        reverb.enabled = mode != 0;
        setParam((0x0Bu << 8) | 0x80u | core, static_cast<uint16_t>(depthL));
        setParam((0x0Cu << 8) | 0x80u | core, static_cast<uint16_t>(depthR));
        setCoreAttr(2u | core, mode != 0 ? 1u : 0u);
        std::fprintf(stderr, "[libsd] effect core=%u mode=%d depth=%d/%d delay=%d feedback=%d vmixE=%06x/%06x\n", core,
                     mode, static_cast<int>(depthL), static_cast<int>(depthR), delay, feedback, m_cores[core].vmixEL,
                     m_cores[core].vmixER);
    }

    // One reverb sample at 24 kHz. `disp` is a sample offset from the moving pointer; negative wraps in the work area.
    void SoundEngine::processReverb(uint32_t core, int32_t inL, int32_t inR, int32_t &outL, int32_t &outR)
    {
        Reverb &reverb = m_reverb[core & 1u];
        if (!reverb.enabled || reverb.buf.empty())
        {
            outL = outR = 0;
            return;
        }
        constexpr uint32_t kMask = 0x1FFFFu;
        reverb.accL += inL;
        reverb.accR += inR;
        reverb.half = !reverb.half;
        if (reverb.half)
        {
            outL = reverb.holdL;
            outR = reverb.holdR;
            return;
        }
        const int32_t lin = reverb.accL >> 1;
        const int32_t rin = reverb.accR >> 1;
        reverb.accL = reverb.accR = 0;
        const auto at = [&](int32_t disp) -> int32_t
        {
            return reverb.buf[(reverb.pos + static_cast<uint32_t>(disp)) & kMask];
        };
        const auto put = [&](int32_t disp, int32_t value)
        {
            reverb.buf[(reverb.pos + static_cast<uint32_t>(disp)) & kMask] = clamp16(value);
        };
        const auto coef = [&](uint32_t index) -> int32_t { return static_cast<int16_t>(reverb.coef[index]); };
        const auto addr = [&](uint32_t index) -> int32_t { return static_cast<int32_t>(reverb.coef[index]); };
        const auto mul = [](int32_t sample, int32_t gain) -> int32_t
        {
            return static_cast<int32_t>((static_cast<int64_t>(sample) * gain) >> 15);
        };
        const int32_t wall = coef(7);
        const int32_t iir = coef(2);
        const int32_t inLeft = mul(lin, coef(30));
        const int32_t inRight = mul(rin, coef(31));
        const int32_t sameL = mul(iir, inLeft + mul(wall, at(addr(16))) - at(addr(10) - 1)) + at(addr(10) - 1);
        const int32_t sameR = mul(iir, inRight + mul(wall, at(addr(17))) - at(addr(11) - 1)) + at(addr(11) - 1);
        put(addr(10), sameL);
        put(addr(11), sameR);
        const int32_t diffL = mul(iir, inLeft + mul(wall, at(addr(25))) - at(addr(18) - 1)) + at(addr(18) - 1);
        const int32_t diffR = mul(iir, inRight + mul(wall, at(addr(24))) - at(addr(19) - 1)) + at(addr(19) - 1);
        put(addr(18), diffL);
        put(addr(19), diffR);
        int32_t left = mul(coef(3), at(addr(12))) + mul(coef(4), at(addr(14))) + mul(coef(5), at(addr(20))) + mul(coef(6), at(addr(22)));
        int32_t right = mul(coef(3), at(addr(13))) + mul(coef(4), at(addr(15))) + mul(coef(5), at(addr(21))) + mul(coef(6), at(addr(23)));
        const auto allpass = [&](int32_t &sample, int32_t dest, int32_t delay, int32_t gain)
        {
            const int32_t src = at(dest - delay);
            const int32_t next = sample - mul(gain, src);
            put(dest, src + mul(gain, next));
            sample = at(dest);
        };
        allpass(left, addr(26), addr(0), coef(8));
        allpass(right, addr(27), addr(0), coef(8));
        allpass(left, addr(28), addr(1), coef(9));
        allpass(right, addr(29), addr(1), coef(9));
        reverb.holdL = left;
        reverb.holdR = right;
        reverb.pos = (reverb.pos + 1u) & kMask;
        outL = reverb.holdL;
        outR = reverb.holdR;
    }

    // ---- sample RAM --------------------------------------------------------------------------------------------------

    void SoundEngine::writeSampleRam(uint32_t byteAddress, const uint8_t *source, size_t bytes)
    {
        uint32_t halfword = byteAddress >> 1u;
        for (size_t i = 0; i + 1u < bytes; i += 2u)
            m_ram[halfword++ & (RamHalfwords - 1u)] = static_cast<uint16_t>(source[i] | (source[i + 1u] << 8u));
    }

    void SoundEngine::readSampleRam(uint32_t byteAddress, uint8_t *destination, size_t bytes) const
    {
        uint32_t halfword = byteAddress >> 1u;
        for (size_t i = 0; i + 1u < bytes; i += 2u)
        {
            const uint16_t value = m_ram[halfword++ & (RamHalfwords - 1u)];
            destination[i] = static_cast<uint8_t>(value);
            destination[i + 1u] = static_cast<uint8_t>(value >> 8u);
        }
    }

    void SoundEngine::clearSampleRam(uint32_t byteAddress, size_t bytes)
    {
        uint32_t halfword = byteAddress >> 1u;
        for (size_t i = 0; i + 1u < bytes; i += 2u)
            m_ram[halfword++ & (RamHalfwords - 1u)] = 0u;
    }

    // ---- core input stream -------------------------------------------------------------------------------------------

    void SoundEngine::startStream(uint32_t core, const uint8_t *ring, uint32_t size, bool loop)
    {
        StreamRing &s = m_stream[core & 1u];
        s.loop = loop;
        s.base = ring;
        s.size = size & ~0x3FFu; // whole 1 KB L/R block pairs
        s.position = 0;
        s.active = ring != nullptr && s.size != 0u;
        m_streamHalf[core & 1u] = false;
    }

    void SoundEngine::stopStream(uint32_t core)
    {
        m_stream[core & 1u].active = false;
    }

    bool SoundEngine::takeStreamHalf(uint32_t core) noexcept
    {
        const bool half = m_streamHalf[core & 1u];
        m_streamHalf[core & 1u] = false;
        return half;
    }

    // One sample per channel from the ring: blocks of 256 left samples (512 bytes) then 256 right samples.
    void SoundEngine::streamSample(uint32_t core, int32_t &left, int32_t &right)
    {
        StreamRing &s = m_stream[core];
        if (!s.active)
            return;
        const uint32_t block = s.position & ~0x3FFu, index = (s.position & 0x3FFu) >> 1u; // sample 0..255 in the block
        // GOW_SND_STALE=1: a 1 KB block played again unchanged one lap later = the driver did not refill the ring in time
        // (an underrun: a whole stale lap is heard), with the time since the driver last polled the position
        static const bool staleDiag = std::getenv("GOW_SND_STALE") != nullptr;
        if (staleDiag && index == 0u)
        {
            uint32_t h = 2166136261u, nz = 0u;
            for (uint32_t i = 0; i < 0x400u; ++i) { h = (h ^ s.base[block + i]) * 16777619u; nz |= s.base[block + i]; }
            uint32_t &prev = m_blockHash[(block >> 10) & 63u];
            static unsigned count = 0;
            if (nz != 0u && prev == h)
                std::fprintf(stderr, "[snd-stale] #%u block=%u cycle=%llu since_poll_ms=%.2f max_advance_ms=%.2f\n", ++count, block >> 10,
                             static_cast<unsigned long long>(m_lastCycle), (m_lastCycle - m_lastPollCycle) / (CyclesPerSample * 48.0),
                             m_maxAdvance / (CyclesPerSample * 48.0)), m_maxAdvance = 0;
            prev = h;
        }
        int16_t l, r;
        std::memcpy(&l, s.base + block + index * 2u, 2u);
        std::memcpy(&r, s.base + block + 0x200u + index * 2u, 2u);
        const Core &c = m_cores[core];
        // BVOL (sound-data input volume) is a plain 16-bit signed level (0x7FFF = 1.0), not the voice volume format
        // with fixed/sweep modes; 989snd sets 0x7FFE for the movie stream.
        const auto inputVolume = [](uint16_t v) { return static_cast<int32_t>(static_cast<int16_t>(v)) >> 1; };
        if ((c.mmix & 0x0080u) != 0u) // MMIX sound-input bits: the driver enables them while its stream runs
            left += (l * inputVolume(c.bvolL)) >> 14;
        if ((c.mmix & 0x0040u) != 0u)
            right += (r * inputVolume(c.bvolR)) >> 14;
        const uint32_t before = s.position;
        s.position = block + ((index + 1u) * 2u);
        if (index + 1u == 256u)
            s.position = block + 0x400u;
        if (s.position >= s.size)
        {
            s.position = 0;
            if (!s.loop)
                s.active = false;
        }
        const uint32_t half = s.size / 2u;
        if ((before < half && s.position >= half) || s.position == 0)
            m_streamHalf[core] = true;
    }

    // ---- voices ------------------------------------------------------------------------------------------------------

    void SoundEngine::keyOn(Core &c, uint32_t mask)
    {
        for (uint32_t index = 0; index < VoicesPerCore; ++index)
        {
            if ((mask & (1u << index)) == 0u)
                continue;
            Voice &v = c.voices[index];
            Voice fresh{};
            fresh.volL = v.volL;
            fresh.volR = v.volR;
            fresh.pitch = v.pitch;
            fresh.adsr1 = v.adsr1;
            fresh.adsr2 = v.adsr2;
            fresh.ssa = v.ssa;
            fresh.active = true;
            fresh.nax = v.ssa & ~7u;
            fresh.lsa = v.ssa & ~7u;
            fresh.playBlock = v.ssa & ~7u; // NAX reads SSA | 1 until the first block is decoded (PCSX2 Start())
            fresh.phase = Voice::Phase::Attack;
            v = fresh;
            c.endx &= ~(1u << index);
        }
    }

    void SoundEngine::keyOff(Core &c, uint32_t mask)
    {
        for (uint32_t index = 0; index < VoicesPerCore; ++index)
        {
            Voice &v = c.voices[index];
            if ((mask & (1u << index)) != 0u && v.active)
            {
                v.phase = Voice::Phase::Release;
                v.envelopeCounter = 0;
            }
        }
    }

    void SoundEngine::decodeBlock(Core &c, uint32_t index)
    {
        Voice &v = c.voices[index];
        // end of the previous block: with the end flag jump to LSA; without "repeat" the voice stops
        if ((v.blockFlags & 1u) != 0u)
        {
            c.endx |= 1u << index;
            v.nax = v.lsa;
            if ((v.blockFlags & 2u) == 0u)
            {
                v.active = false;
                v.phase = Voice::Phase::Off;
                v.envelope = 0;
            }
        }
        v.blockFlags = 0;
        if (!v.active)
            return;

        const uint32_t block = v.nax & (RamHalfwords - 1u);
        const uint16_t header = m_ram[block];
        uint32_t shift = header & 0xFu;
        if (shift > 12u)
            shift = 9u;
        const uint32_t filter = std::min<uint32_t>((header >> 4u) & 7u, 4u);
        v.blockFlags = static_cast<uint8_t>(header >> 8u);
        if ((v.blockFlags & 4u) != 0u && !v.lsaFromDriver)
            v.lsa = block;

        for (uint32_t word = 1; word < 8u; ++word)
        {
            const uint16_t packed = m_ram[(block + word) & (RamHalfwords - 1u)];
            for (uint32_t nibble = 0; nibble < 4u; ++nibble)
            {
                const int32_t raw = static_cast<int16_t>(((packed >> (nibble * 4u)) & 0xFu) << 12u) >> shift;
                const int32_t predicted = (v.hist1 * kAdpcmFilters[filter][0] + v.hist2 * kAdpcmFilters[filter][1] + 32) >> 6;
                const int16_t sample = clamp16(raw + predicted);
                v.hist2 = v.hist1;
                v.hist1 = sample;
                v.decoded[(word - 1u) * 4u + nibble] = sample;
            }
        }
        v.decodedIndex = 0;
        v.nax = (block + 8u) & (RamHalfwords - 1u);
        v.playBlock = block;
        v.playStarted = true;
        // SPU2 IRQ when the block just decoded contains IRQA and the core's IRQ enable (attr bit 6) is on.
        // advanceTo stops on the latch; iop_emulator runs sceSdSetSpu2IntrHandler before the next sample.
        if ((c.attr & 0x0040u) != 0u && c.irqa >= block && c.irqa < block + 8u)
        {
            const uint32_t core = static_cast<uint32_t>(&c - m_cores.data());
            if (core < 2u)
                m_irqLatched[core] = true;
            if (soundTrace())
            {
                TraceState &state = traceState();
                ++state.irqHits;
                if (state.irqHits <= 8u)
                    std::fprintf(stderr, "[snd-trace] irq-cross core=%u voice=%u irqa=%x block=%x\n", core, index,
                                 c.irqa, block);
            }
        }
    }

    void SoundEngine::stepEnvelope(Voice &v)
    {
        uint32_t rate = 0;
        bool exponential = false;
        bool increase = false;
        switch (v.phase)
        {
        case Voice::Phase::Attack:
            rate = (v.adsr1 >> 8u) & 0x7Fu;
            exponential = (v.adsr1 & 0x8000u) != 0u;
            increase = true;
            break;
        case Voice::Phase::Decay:
            rate = ((v.adsr1 >> 4u) & 0xFu) * 4u;
            exponential = true;
            break;
        case Voice::Phase::Sustain:
            rate = (v.adsr2 >> 6u) & 0x7Fu;
            exponential = (v.adsr2 & 0x8000u) != 0u;
            increase = (v.adsr2 & 0x4000u) == 0u;
            break;
        case Voice::Phase::Release:
            rate = (v.adsr2 & 0x1Fu) * 4u;
            exponential = (v.adsr2 & 0x20u) != 0u;
            break;
        case Voice::Phase::Off:
            return;
        }

        // psx-spx "SPU ADSR": cycles = 1 << max(0, shift - 11); step = value << max(0, 11 - shift)
        const int32_t shiftValue = static_cast<int32_t>(rate >> 2u);
        const int32_t stepValue = static_cast<int32_t>(rate & 3u);
        int32_t step = increase ? 7 - stepValue : -8 + stepValue;
        uint32_t cycles = 1u << static_cast<uint32_t>(std::max(0, shiftValue - 11));
        step <<= std::max(0, 11 - shiftValue);
        if (exponential && increase && v.envelope > 0x6000)
            cycles *= 4u;
        if (exponential && !increase)
            step = step * v.envelope / 0x8000;

        if (++v.envelopeCounter >= cycles)
        {
            v.envelopeCounter = 0;
            v.envelope = std::clamp(v.envelope + step, 0, 0x7FFF);
        }

        switch (v.phase)
        {
        case Voice::Phase::Attack:
            if (v.envelope >= 0x7FFF)
            {
                v.phase = Voice::Phase::Decay;
                v.envelopeCounter = 0;
            }
            break;
        case Voice::Phase::Decay:
            if (v.envelope <= static_cast<int32_t>(((v.adsr1 & 0xFu) + 1u) * 0x800u))
            {
                v.phase = Voice::Phase::Sustain;
                v.envelopeCounter = 0;
            }
            break;
        case Voice::Phase::Release:
            if (v.envelope <= 0)
            {
                v.envelope = 0;
                v.phase = Voice::Phase::Off;
                v.active = false;
            }
            break;
        default:
            break;
        }
    }

    int32_t SoundEngine::stepVoice(Core &c, uint32_t index)
    {
        Voice &v = c.voices[index];
        if (!v.active)
            return 0;
        // pitch 4.12 (0x1000 = 48 kHz), linear interpolation between the last two samples (the hardware uses a
        // 4-point gaussian)
        v.pitchCounter += std::min<uint32_t>(v.pitch, 0x3FFFu);
        while (v.pitchCounter >= 0x1000u && v.active)
        {
            v.pitchCounter -= 0x1000u;
            if (v.decodedIndex >= 28u)
                decodeBlock(c, index);
            if (!v.active)
                break;
            v.prev[1] = v.prev[0];
            v.prev[0] = v.decoded[v.decodedIndex++];
        }
        if (!v.active)
            return 0;
        const int32_t fraction = static_cast<int32_t>(v.pitchCounter & 0xFFFu);
        const int32_t sample = v.prev[1] + (((v.prev[0] - v.prev[1]) * fraction) >> 12);
        stepEnvelope(v);
        return clamp16((sample * v.envelope) >> 15);
    }

    void SoundEngine::step()
    {
        int32_t coreOut[2][2] = {};
        for (uint32_t core = 0; core < 2u; ++core)
        {
            Core &c = m_cores[core];
            int32_t left = 0, right = 0, sendL = 0, sendR = 0;
            for (uint32_t index = 0; index < VoicesPerCore; ++index)
            {
                const int32_t sample = stepVoice(c, index);
                if (sample == 0)
                    continue;
                const int32_t scaledL = (sample * volumeOf(c.voices[index].volL)) >> 14;
                const int32_t scaledR = (sample * volumeOf(c.voices[index].volR)) >> 14;
                if ((c.vmixL & (1u << index)) != 0u)
                    left += scaledL;
                if ((c.vmixR & (1u << index)) != 0u)
                    right += scaledR;
                if ((c.vmixEL & (1u << index)) != 0u)
                    sendL += scaledL;
                if ((c.vmixER & (1u << index)) != 0u)
                    sendR += scaledR;
            }
            streamSample(core, left, right);
            // PCSX2 MixCore clamp_mixes the dry (and wet) sum to s16 before the master multiply.
            // The old int32 (sum * master) >> 14 wraps once four full-scale voices are in the sum
            // (4 * 32767 * 0x3FFF > 2^31-1) and the wrap lands on the negative rail.
            left = clamp16(left);
            right = clamp16(right);
            sendL = clamp16(sendL);
            sendR = clamp16(sendR);
            if ((c.attr & 0x0080u) != 0u)
            {
                int32_t wetL = 0, wetR = 0;
                processReverb(core, sendL, sendR, wetL, wetR);
                left = clamp16(left + static_cast<int32_t>((static_cast<int64_t>(wetL) * volumeOf(c.evolL)) >> 14));
                right = clamp16(right + static_cast<int32_t>((static_cast<int64_t>(wetR) * volumeOf(c.evolR)) >> 14));
            }
            coreOut[core][0] = left;
            coreOut[core][1] = right;
        }
        // core 0 through its master volume, added to core 1's dry output, all through core 1's master volume
        // (simplified MMIX routing). Same order as PCSX2 spu2Mix: clamp, master0, add core 1, clamp, master1.
        std::lock_guard<std::mutex> lock(m_outputMutex);
        const int32_t master[2][2] = {{volumeOf(m_cores[0].mvolL), volumeOf(m_cores[0].mvolR)},
                                      {volumeOf(m_cores[1].mvolL), volumeOf(m_cores[1].mvolR)}};
        for (uint32_t side = 0; side < 2u; ++side)
        {
            const int32_t core0 = static_cast<int32_t>((static_cast<int64_t>(coreOut[0][side]) * master[0][side]) >> 14);
            const int64_t summed = static_cast<int64_t>(core0) + coreOut[1][side];
            const int32_t sum = static_cast<int32_t>(std::clamp(summed, static_cast<int64_t>(-32768), static_cast<int64_t>(32767)));
            m_output.push_back(clamp16(static_cast<int32_t>((static_cast<int64_t>(sum) * master[1][side]) >> 14)));
        }
        // GOW_SND_VOICEDUMP=1: while a core input stream (a movie) plays, once per second for its first 10 seconds, every
        // voice that is still sounding (active, pitch and envelope non-zero) with its volumes and sample addresses
        static const bool voiceDump = std::getenv("GOW_SND_VOICEDUMP") != nullptr;
        if (voiceDump)
        {
            static uint64_t streamSamples = 0;
            const bool streaming = m_stream[0].active || m_stream[1].active;
            streamSamples = streaming ? streamSamples + 1u : 0u;
            if (streaming && streamSamples % 48000u == 1u && streamSamples < 48000u * 10u)
            {
                std::fprintf(stderr, "[snd-voices] t=%llus mvol0=%04x/%04x mvol1=%04x/%04x\n",
                             static_cast<unsigned long long>(streamSamples / 48000u), m_cores[0].mvolL, m_cores[0].mvolR,
                             m_cores[1].mvolL, m_cores[1].mvolR);
                for (uint32_t core = 0; core < 2u; ++core)
                    for (uint32_t index = 0; index < VoicesPerCore; ++index)
                    {
                        const Voice &v = m_cores[core].voices[index];
                        if (v.active && v.pitch != 0u && v.envelope != 0)
                            std::fprintf(stderr, "[snd-voices]   core=%u voice=%u pitch=%04x vol=%04x/%04x env=%d ssa=%x lsa=%x nax=%x\n",
                                         core, index, v.pitch, v.volL, v.volR, v.envelope, v.ssa * 2u, v.lsa * 2u, v.nax * 2u);
                    }
            }
        }
        // GOW_AUDIO_RAW=<file>: the mixed samples as produced (s16le stereo 48 kHz), before the output's rate adaptation,
        // so a recording can be compared sample by sample with its source (tools/movies.py)
        static std::FILE *raw = [] { const char *path = std::getenv("GOW_AUDIO_RAW"); return path ? std::fopen(path, "wb") : nullptr; }();
        if (raw)
        {
            static std::vector<int16_t> pending;
            pending.insert(pending.end(), m_output.end() - 2, m_output.end());
            if (pending.size() >= 8192u)
            {
                std::fwrite(pending.data(), sizeof(int16_t), pending.size(), raw);
                std::fflush(raw);
                pending.clear();
            }
        }
        if (m_output.size() > kMaxBufferedFrames * 2u)
        {
            const size_t erase = m_output.size() / 2u & ~size_t{1};
            if (soundTrace())
                traceState().dropped += erase / 2u;
            m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(erase));
        }
        if (soundTrace())
        {
            TraceState &state = traceState();
            if (!state.started)
            {
                state.t0 = std::chrono::steady_clock::now();
                state.started = true;
            }
            ++state.stepped;
            const int absL = std::abs(static_cast<int>(m_output[m_output.size() - 2u]));
            const int absR = std::abs(static_cast<int>(m_output[m_output.size() - 1u]));
            state.peak = std::max(state.peak, std::max(absL, absR));
            if ((state.stepped % 48000u) == 0u)
            {
                int active = 0;
                for (const Core &core : m_cores)
                    for (const Voice &voice : core.voices)
                        if (voice.active)
                            ++active;
                const bool edge = state.lastActive != active && (active == 0 || state.lastActive == 0);
                state.lastActive = active;
                if (edge || (state.stepped % (48000u * 5u)) == 0u)
                {
                    std::fprintf(stderr,
                                 "[snd-trace] mix=%.0f wall=%.1f active=%d mvol=%04x/%04x vmix=%06x/%06x endx=%06x/%06x "
                                 "irqa=%x/%x attr=%04x/%04x irqHits=%llu dropped=%llu maxAdv=%llu peak=%d\n",
                                 traceMixSeconds(), traceWallSeconds(), active, m_cores[0].mvolL, m_cores[0].mvolR,
                                 m_cores[0].vmixL, m_cores[1].vmixL, m_cores[0].endx, m_cores[1].endx, m_cores[0].irqa,
                                 m_cores[1].irqa, m_cores[0].attr, m_cores[1].attr,
                                 static_cast<unsigned long long>(state.irqHits),
                                 static_cast<unsigned long long>(state.dropped),
                                 static_cast<unsigned long long>(state.maxAdvance), state.peak);
                    state.peak = 0;
                }
            }
        }
    }

    void SoundEngine::advanceTo(uint64_t iopCycle)
    {
        if (iopCycle <= m_lastCycle)
            return;
        const uint64_t samples = (iopCycle - m_lastCycle) / CyclesPerSample;
        m_maxAdvance = std::max<uint64_t>(m_maxAdvance, iopCycle - m_lastCycle);
        if (soundTrace())
            traceState().maxAdvance = std::max(traceState().maxAdvance, samples);
        uint64_t done = 0;
        while (done < samples)
        {
            step();
            ++done;
            // One IRQ per stop. libsd clears bit 6; serviceSpu2Irq runs the woken thread before more samples.
            if (m_irqLatched[0] || m_irqLatched[1])
                break;
        }
        m_lastCycle += done * CyclesPerSample;
    }

    uint32_t SoundEngine::takeSpu2Irq() noexcept
    {
        uint32_t bits = 0;
        for (uint32_t core = 0; core < 2u; ++core)
        {
            if (!m_irqLatched[core])
                continue;
            m_irqLatched[core] = false;
            // Sony libsd Spu2Interrupt: ATTR &= ~0x0040 before the user handler. The driver sets the bit
            // again with sceSdSetCoreAttr(SD_CORE_IRQ_ENABLE, 1). Leaving it set skipped that re-arm.
            m_cores[core].attr = static_cast<uint16_t>(m_cores[core].attr & ~0x0040u);
            bits |= 1u << core;
        }
        return bits;
    }

    size_t SoundEngine::pendingFrames() const
    {
        std::lock_guard<std::mutex> lock(m_outputMutex);
        return m_output.size() / 2u;
    }

    // The device asks for maxFrames; the engine produces at the guest's IOP clock, which can run slightly faster than the
    // device (the title screen and menus run the guest clock ~3.7 % ahead of the wall clock). Above the latency target the
    // buffer is played back up to 5 % faster with linear interpolation instead of dropping blocks of samples (an audible
    // crackle); at or below the target the samples are copied as they are. Only a buffer far above the target (after a
    // stall) is cut back.
    size_t SoundEngine::drainOutput(int16_t *stereo, size_t maxFrames, size_t maxLatencyFrames)
    {
        std::lock_guard<std::mutex> lock(m_outputMutex);
        size_t available = m_output.size() / 2u;
        size_t dropped = 0;
        double rate = 1.0;
        if (maxLatencyFrames != SIZE_MAX)
        {
            if (available > maxFrames + 4u * maxLatencyFrames)
            {
                dropped = available - maxFrames - maxLatencyFrames;
                m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(dropped * 2u));
                available -= dropped;
                m_resamplePhase = 0.0;
            }
            const double over = (static_cast<double>(available) - static_cast<double>(maxFrames + maxLatencyFrames)) /
                                static_cast<double>(maxLatencyFrames);
            rate = 1.0 + std::clamp(over * 0.1, 0.0, 0.05);
        }
        size_t frames = 0;
        double position = m_resamplePhase;
        if (rate == 1.0 && position == 0.0)
        {
            frames = std::min(maxFrames, available);
            std::copy_n(m_output.begin(), frames * 2u, stereo);
            position = static_cast<double>(frames);
        }
        else
        {
            for (; frames < maxFrames; ++frames, position += rate)
            {
                const size_t index = static_cast<size_t>(position);
                if (index + 1u >= available)
                    break;
                const double fraction = position - static_cast<double>(index);
                for (size_t side = 0; side < 2u; ++side)
                {
                    const double a = m_output[index * 2u + side], b = m_output[(index + 1u) * 2u + side];
                    stereo[frames * 2u + side] = static_cast<int16_t>(std::lround(a + (b - a) * fraction));
                }
            }
        }
        const size_t consumed = std::min(available, static_cast<size_t>(position));
        m_resamplePhase = position - static_cast<double>(consumed);
        m_output.erase(m_output.begin(), m_output.begin() + static_cast<std::ptrdiff_t>(consumed * 2u));
        // GOW_AUDIO_STATS=1: every ~5 s, frames requested by the device, frames missing (underrun -> silence), frames
        // dropped after a stall, the highest playback rate and the buffered level
        static const bool stats = std::getenv("GOW_AUDIO_STATS") != nullptr;
        if (stats)
        {
            static size_t requested = 0, missing = 0, droppedSum = 0, calls = 0, minLevel = SIZE_MAX, maxLevel = 0;
            static double maxRate = 1.0;
            requested += maxFrames;
            missing += maxFrames - frames;
            droppedSum += dropped;
            maxRate = std::max(maxRate, rate);
            minLevel = std::min(minLevel, available);
            maxLevel = std::max(maxLevel, available);
            ++calls;
            if (requested >= 240000u)
            {
                std::fprintf(stderr, "[audio-stats] requested=%zu missing=%zu dropped=%zu calls=%zu rate<=%.4f level=%zu-%zu\n",
                             requested, missing, droppedSum, calls, maxRate, minLevel, maxLevel);
                requested = missing = droppedSum = calls = maxLevel = 0;
                minLevel = SIZE_MAX;
                maxRate = 1.0;
            }
        }
        return frames;
    }
}
