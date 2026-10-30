#include "ps2x/iop/iop_subsystem.h"

#include "../src/emulator/core/iop_kernel.h"
#include "../src/emulator/core/iop_memory.h"
#include "../src/emulator/iop_emulator.h"
#include "../src/emulator/iop_emulator_const.h"

#include <map>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    using namespace ps2x::iop;

    class TestHost final : public IopHost
    {
    public:
        explicit TestHost(size_t guestBytes = 4096u)
            : guest(guestBytes, 0u)
        {
        }

        bool readGuest(uint32_t address, void *destination, size_t size) const override
        {
            if ((!destination && size != 0u) || address > guest.size() || size > guest.size() - address)
                return false;
            if (size != 0u)
                std::memcpy(destination, guest.data() + address, size);
            return true;
        }

        bool writeGuest(uint32_t address, const void *source, size_t size) override
        {
            if ((!source && size != 0u) || address > guest.size() || size > guest.size() - address)
                return false;
            if (size != 0u)
                std::memcpy(guest.data() + address, source, size);
            return true;
        }

        bool zeroGuest(uint32_t address, size_t size) override
        {
            if (address > guest.size() || size > guest.size() - address)
                return false;
            if (size != 0u)
                std::memset(guest.data() + address, 0, size);
            return true;
        }

        bool normalizeGuestAddress(uint32_t address, uint32_t &normalized) const override
        {
            normalized = address;
            return address <= guest.size();
        }

        uint32_t allocateIopHandle(IopHandleKind) override { return 1u; }
        uint32_t allocateGuest(uint32_t, uint32_t) override { return 0u; }
        void freeGuest(uint32_t) override {}
        void audioCommand(uint32_t, uint32_t, GuestBuffer, GuestBuffer) override {}
        std::string hostPath(HostPathKind kind) const override
        {
            return kind == HostPathKind::CdRoot ? cdRoot : std::string{};
        }
        std::string translateGuestPath(std::string_view path) const override { return std::string(path); }
        uint64_t openHostFile(std::string_view) override { return 0u; }
        bool hostFileSize(uint64_t, uint64_t &) const override { return false; }
        bool readHostFile(uint64_t, uint64_t, void *, size_t, size_t &) override { return false; }
        void closeHostFile(uint64_t) override {}
        int32_t memoryCard(const MemoryCardRequest &) override { return 0; }
        bool hasGuestFunction(uint32_t) const override { return false; }
        bool invokeGuestFunction(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t *) override { return false; }
        void log(LogLevel, std::string_view message) override { logs.emplace_back(message); }

        std::vector<uint8_t> guest;
        std::vector<std::string> logs;
        std::string cdRoot;
    };

#pragma pack(push, 1)
    struct ElfHeader
    {
        uint8_t ident[16];
        uint16_t type;
        uint16_t machine;
        uint32_t version;
        uint32_t entry;
        uint32_t phoff;
        uint32_t shoff;
        uint32_t flags;
        uint16_t ehsize;
        uint16_t phentsize;
        uint16_t phnum;
        uint16_t shentsize;
        uint16_t shnum;
        uint16_t shstrndx;
    };

    struct ProgramHeader
    {
        uint32_t type;
        uint32_t offset;
        uint32_t vaddr;
        uint32_t paddr;
        uint32_t filesz;
        uint32_t memsz;
        uint32_t flags;
        uint32_t align;
    };

    struct SectionHeader
    {
        uint32_t name;
        uint32_t type;
        uint32_t flags;
        uint32_t address;
        uint32_t offset;
        uint32_t size;
        uint32_t link;
        uint32_t info;
        uint32_t alignment;
        uint32_t entrySize;
    };

    struct Relocation
    {
        uint32_t offset;
        uint32_t info;
    };
#pragma pack(pop)

    static_assert(sizeof(ElfHeader) == 52u);
    static_assert(sizeof(ProgramHeader) == 32u);
    static_assert(sizeof(SectionHeader) == 40u);
    static_assert(sizeof(Relocation) == 8u);

    void writeMinimalIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu;
        header.ident[1] = 'E';
        header.ident[2] = 'L';
        header.ident[3] = 'F';
        header.ident[4] = 1u;
        header.ident[5] = 1u;
        header.ident[6] = 1u;
        header.type = 2u;
        header.machine = 8u;
        header.version = 1u;
        header.entry = loadAddress;
        header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader);
        header.phentsize = sizeof(ProgramHeader);
        header.phnum = 1u;

        // Branch is deliberately included so the test checks that its delay slot runs.
        const uint32_t code[] = {
            0x24080001u, // addiu t0, zero, 1
            0x11080002u, // beq   t0, t0, +2
            0x24020007u, // addiu v0, zero, 7   (delay slot)
            0x24020063u, // addiu v0, zero, 99  (must be skipped)
            0x03E00008u, // jr    ra
            0x00000000u, // nop
        };

        ProgramHeader program{};
        program.type = 1u;
        program.offset = codeOffset;
        program.vaddr = loadAddress;
        program.paddr = loadAddress;
        program.filesz = sizeof(code);
        program.memsz = sizeof(code);
        program.flags = 5u;
        program.align = 4u;

        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, code, sizeof(code));
    }

    void writeRpcServerIrx(TestHost &host,
                           uint32_t address,
                           uint32_t sid = 0xF00DCAFEu)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t importTableOffset = 0x60u;
        constexpr uint32_t importStubAddress = loadAddress + importTableOffset + 20u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu;
        header.ident[1] = 'E';
        header.ident[2] = 'L';
        header.ident[3] = 'F';
        header.ident[4] = 1u;
        header.ident[5] = 1u;
        header.ident[6] = 1u;
        header.type = 2u;
        header.machine = 8u;
        header.version = 1u;
        header.entry = loadAddress;
        header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader);
        header.phentsize = sizeof(ProgramHeader);
        header.phnum = 1u;

        const uint32_t code[] = {
            0x27BDFFE0u, // addiu sp, sp, -0x20
            0xAFBF001Cu, // sw    ra, 0x1c(sp)
            0x3C040001u, // lui   a0, 1
            0x34840200u, // ori   a0, a0, 0x200 (server data)
            0x3C050000u | ((sid >> 16u) & 0xFFFFu), // lui a1, SID upper
            0x34A50000u | (sid & 0xFFFFu),          // ori a1, a1, SID lower
            0x3C060001u, // lui   a2, 1
            0x34C60300u, // ori   a2, a2, 0x300 (server function)
            0x3C070001u, // lui   a3, 1
            0x34E70400u, // ori   a3, a3, 0x400 (server buffer)
            0xAFA00010u, // sw    zero, 0x10(sp)
            0xAFA00014u, // sw    zero, 0x14(sp)
            0xAFA00018u, // sw    zero, 0x18(sp)
            0x0C000000u | ((importStubAddress >> 2u) & 0x03FFFFFFu), // jal sceSifRegisterRpc
            0x00000000u, // nop
            0x8FBF001Cu, // lw    ra, 0x1c(sp)
            0x00001021u, // addu  v0, zero, zero
            0x27BD0020u, // addiu sp, sp, 0x20
            0x03E00008u, // jr    ra
            0x00000000u, // nop
        };
        const uint32_t importTable[] = {
            0x41E00000u, // IRX import magic
            0x00000000u,
            0x00000101u,
            0x63666973u, // "sifc"
            0x0000646Du, // "md"
            0x03E00008u, // jr ra
            0x24000011u, // addiu zero, zero, 17 (sceSifRegisterRpc)
            0x00000000u,
            0x00000000u,
        };

        ProgramHeader program{};
        program.type = 1u;
        program.offset = codeOffset;
        program.vaddr = loadAddress;
        program.paddr = loadAddress;
        program.filesz = 0xA0u;
        program.memsz = 0x500u;
        program.flags = 7u;
        program.align = 4u;

        std::fill(host.guest.begin() + address, host.guest.end(), 0u);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, code, sizeof(code));
        std::memcpy(host.guest.data() + address + codeOffset + importTableOffset,
                    importTable, sizeof(importTable));
    }

    void writeRelocatableRpcServerIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t entryAddress = 0x20u;
        constexpr uint32_t importTableAddress = 0x80u;
        constexpr uint32_t importStubAddress = importTableAddress + 20u;
        constexpr uint32_t handlerAddress = 0xC0u;
        constexpr uint32_t gpAddress = 0xD0u;
        constexpr uint32_t iopModFileOffset = 0xE0u;
        constexpr uint32_t relocationFileOffset = 0x200u;
        constexpr uint32_t sectionTableOffset = 0x240u;
        constexpr uint32_t rpcSid = 0xA11CE001u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu;
        header.ident[1] = 'E';
        header.ident[2] = 'L';
        header.ident[3] = 'F';
        header.ident[4] = 1u;
        header.ident[5] = 1u;
        header.ident[6] = 1u;
        header.type = 0xFF80u; // ET_SCE_IOPRELEXEC
        header.machine = 8u;
        header.version = 1u;
        header.entry = entryAddress;
        header.phoff = sizeof(ElfHeader);
        header.shoff = sectionTableOffset;
        header.ehsize = sizeof(ElfHeader);
        header.phentsize = sizeof(ProgramHeader);
        header.phnum = 2u;
        header.shentsize = sizeof(SectionHeader);
        header.shnum = 3u;

        const uint32_t code[] = {
            0x27BDFFE0u, // addiu sp, sp, -0x20
            0xAFBF001Cu, // sw    ra, 0x1c(sp)
            0x00002021u, // addu  a0, zero, zero
            0x3C05A11Cu, // lui   a1, 0xa11c
            0x34A5E001u, // ori   a1, a1, 0xe001
            0x3C060000u, // lui   a2, 0 (HI16 handler address)
            0x24C600C0u, // addiu a2, a2, 0xc0 (LO16 handler address)
            0x00003821u, // addu  a3, zero, zero
            0xAFA00010u, // sw    zero, 0x10(sp)
            0xAFA00014u, // sw    zero, 0x14(sp)
            0xAFA00018u, // sw    zero, 0x18(sp)
            0x0C000000u | ((importStubAddress >> 2u) & 0x03FFFFFFu), // jal sceSifRegisterRpc
            0x00000000u, // nop
            0x8FBF001Cu, // lw    ra, 0x1c(sp)
            0x00001021u, // addu  v0, zero, zero
            0x27BD0020u, // addiu sp, sp, 0x20
            0x03E00008u, // jr    ra
            0x00000000u, // nop
        };
        const uint32_t importTable[] = {
            0x41E00000u,
            0x00000000u,
            0x00000101u,
            0x63666973u, // "sifc"
            0x0000646Du, // "md"
            0x03E00008u,
            0x24000011u, // sceSifRegisterRpc
            0x00000000u,
        };
        const uint32_t handler[] = {
            0x03E00008u, // jr ra
            0x03801021u, // addu v0, gp, zero
        };
        constexpr uint32_t gpData = 0x47505250u; // "GPRP"
        const uint32_t iopModuleHeader[] = {
            0u,
            entryAddress,
            gpAddress,
        };

        ProgramHeader iopModule{};
        iopModule.type = 0x70000080u; // PT_SCE_IOPMOD
        iopModule.offset = iopModFileOffset;
        iopModule.filesz = sizeof(iopModuleHeader);
        iopModule.memsz = sizeof(iopModuleHeader);
        iopModule.align = 4u;

        ProgramHeader program{};
        program.type = 1u;
        program.offset = codeOffset;
        program.vaddr = 0u;
        program.paddr = 0u;
        program.filesz = gpAddress + sizeof(gpData);
        program.memsz = 0x500u;
        program.flags = 7u;
        program.align = 4u;

        SectionHeader text{};
        text.type = 1u; // SHT_PROGBITS
        text.flags = 0x6u; // SHF_ALLOC | SHF_EXECINSTR
        text.address = entryAddress;
        text.offset = codeOffset + entryAddress;
        text.size = 0xB0u;
        text.alignment = 4u;

        SectionHeader relocations{};
        relocations.type = 9u; // SHT_REL
        relocations.offset = relocationFileOffset;
        relocations.size = 3u * sizeof(Relocation);
        relocations.info = 1u; // target .text
        relocations.alignment = 4u;
        relocations.entrySize = sizeof(Relocation);

        const Relocation relocationData[] = {
            {entryAddress + 5u * sizeof(uint32_t), 5u},  // symbol 0, R_MIPS_HI16
            {entryAddress + 6u * sizeof(uint32_t), 6u},  // symbol 0, R_MIPS_LO16
            {entryAddress + 11u * sizeof(uint32_t), 4u}, // symbol 0, R_MIPS_26
        };

        std::fill(host.guest.begin() + address, host.guest.end(), 0u);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &iopModule, sizeof(iopModule));
        std::memcpy(host.guest.data() + address + sizeof(header) + sizeof(iopModule), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + iopModFileOffset,
                    iopModuleHeader, sizeof(iopModuleHeader));
        std::memcpy(host.guest.data() + address + codeOffset + entryAddress, code, sizeof(code));
        std::memcpy(host.guest.data() + address + codeOffset + importTableAddress,
                    importTable, sizeof(importTable));
        std::memcpy(host.guest.data() + address + codeOffset + handlerAddress,
                    handler, sizeof(handler));
        std::memcpy(host.guest.data() + address + codeOffset + gpAddress,
                    &gpData, sizeof(gpData));
        std::memcpy(host.guest.data() + address + relocationFileOffset,
                    relocationData, sizeof(relocationData));
        std::memcpy(host.guest.data() + address + sectionTableOffset + sizeof(SectionHeader),
                    &text, sizeof(text));
        std::memcpy(host.guest.data() + address + sectionTableOffset + 2u * sizeof(SectionHeader),
                    &relocations, sizeof(relocations));
    }

    void writeExportProviderIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t targetAddress = loadAddress + 0x60u;
        constexpr uint32_t exportTableAddress = loadAddress + 0x80u;
        constexpr uint32_t importTableAddress = loadAddress + 0xC0u;
        constexpr uint32_t importStubAddress = importTableAddress + 20u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0xF0u; program.memsz = 0xF0u; program.flags = 7u; program.align = 4u;

        const uint32_t code[] = {
            0x27BDFFE0u, 0xAFBF001Cu,
            0x3C040001u, 0x34840080u,
            0x0C000000u | ((importStubAddress >> 2u) & 0x03FFFFFFu), 0x00000000u,
            0x8FBF001Cu, 0x00001021u,
            0x27BD0020u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t target[] = {0x03E00008u, 0x24020042u};
        const uint32_t exportTable[] = {
            0x41C00000u, 0u, 0x00000101u,
            0x6C747374u, 0x00006269u, // "tstlib"
            loadAddress, loadAddress, loadAddress, targetAddress, 0u,
        };
        const uint32_t importTable[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x64616F6Cu, 0x65726F63u, // "loadcore"
            0x03E00008u, 0x24000006u, 0u, 0u,
        };

        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, code, sizeof(code));
        std::memcpy(host.guest.data() + address + codeOffset + 0x60u, target, sizeof(target));
        std::memcpy(host.guest.data() + address + codeOffset + 0x80u, exportTable, sizeof(exportTable));
        std::memcpy(host.guest.data() + address + codeOffset + 0xC0u, importTable, sizeof(importTable));
    }

    void writeExportConsumerIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00011000u;
        constexpr uint32_t importTableAddress = loadAddress + 0x40u;
        constexpr uint32_t importStubAddress = importTableAddress + 20u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x80u; program.memsz = 0x80u; program.flags = 7u; program.align = 4u;

        const uint32_t code[] = {
            0x27BDFFF0u, 0xAFBF000Cu,
            0x0C000000u | ((importStubAddress >> 2u) & 0x03FFFFFFu), 0x00000000u,
            0x8FBF000Cu, 0x27BD0010u,
            0x03E00008u, 0x00000000u,
        };
        const uint32_t importTable[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x6C747374u, 0x00006269u, // "tstlib"
            0x03E00008u, 0x24000003u, 0u, 0u,
        };

        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, code, sizeof(code));
        std::memcpy(host.guest.data() + address + codeOffset + 0x40u, importTable, sizeof(importTable));
    }

    void writeVblankSchedulingIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t highThreadAddress = loadAddress + 0x100u;
        constexpr uint32_t lowThreadAddress = loadAddress + 0x140u;
        constexpr uint32_t thbaseTableAddress = loadAddress + 0x180u;
        constexpr uint32_t createThreadStub = thbaseTableAddress + 20u;
        constexpr uint32_t startThreadStub = createThreadStub + 8u;
        constexpr uint32_t vblankTableAddress = loadAddress + 0x1C0u;
        constexpr uint32_t waitVblankEndStub = vblankTableAddress + 20u;
        constexpr uint32_t highThreadDescriptor = loadAddress + 0x300u;
        constexpr uint32_t lowThreadDescriptor = loadAddress + 0x320u;
        constexpr uint32_t lowThreadMarker = loadAddress + 0x400u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x500u; program.memsz = 0x500u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const auto jump = [](uint32_t target) { return 0x08000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const uint32_t entry[] = {
            0x27BDFFE0u, 0xAFBF001Cu,
            0x3C040001u, 0x34840300u, jal(createThreadStub), 0x00000000u,
            0x00408021u, // move s0, v0
            0x3C040001u, 0x34840320u, jal(createThreadStub), 0x00000000u,
            0x00408821u, // move s1, v0
            0x02002021u, 0x00002821u, jal(startThreadStub), 0x00000000u,
            0x02202021u, 0x00002821u, jal(startThreadStub), 0x00000000u,
            0x8FBF001Cu, 0x00001021u, 0x27BD0020u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t highThread[] = {
            jal(waitVblankEndStub), 0x00000000u,
            jump(highThreadAddress), 0x00000000u,
        };
        const uint32_t lowThread[] = {
            0x3C080001u, 0x35080400u,
            0x24090001u, 0xAD090000u,
            0x03E00008u, 0x00000000u,
        };
        const uint32_t thbaseImports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x61626874u, 0x00006573u, // "thbase"
            0x03E00008u, 0x24000004u, // CreateThread
            0x03E00008u, 0x24000006u, // StartThread
            0u, 0u,
        };
        const uint32_t vblankImports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x616C6276u, 0x00006B6Eu, // "vblank"
            0x03E00008u, 0x24000005u, // WaitVblankEnd
            0u, 0u,
        };
        const uint32_t highDescriptor[] = {0u, 0u, highThreadAddress, 0x400u, 10u};
        const uint32_t lowDescriptor[] = {0u, 0u, lowThreadAddress, 0x400u, 20u};

        std::vector<uint8_t> segment(program.filesz, 0u);
        const auto put = [&](uint32_t offset, const void *data, size_t size)
        {
            std::memcpy(segment.data() + offset, data, size);
        };
        put(0u, entry, sizeof(entry));
        put(0x100u, highThread, sizeof(highThread));
        put(0x140u, lowThread, sizeof(lowThread));
        put(0x180u, thbaseImports, sizeof(thbaseImports));
        put(0x1C0u, vblankImports, sizeof(vblankImports));
        put(0x300u, highDescriptor, sizeof(highDescriptor));
        put(0x320u, lowDescriptor, sizeof(lowDescriptor));

        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    void writeSifDmaIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t importTableAddress = loadAddress + 0x100u;
        constexpr uint32_t setDmaStub = importTableAddress + 20u;
        constexpr uint32_t dmaStatStub = setDmaStub + 8u;
        constexpr uint32_t descriptorAddress = loadAddress + 0x180u;
        constexpr uint32_t payloadAddress = loadAddress + 0x1A0u;
        constexpr uint32_t eeDestination = 0xC00u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x200u; program.memsz = 0x200u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const uint32_t entry[] = {
            0x27BDFFF0u, 0xAFBF000Cu,
            0x3C040001u, 0x34840180u, 0x24050001u,
            jal(setDmaStub), 0x00000000u,
            0x00402021u, // move a0, v0
            jal(dmaStatStub), 0x00000000u,
            0x8FBF000Cu, 0x27BD0010u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t imports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x6D666973u, 0x00006E61u, // "sifman"
            0x03E00008u, 0x24000007u, // sceSifSetDma
            0x03E00008u, 0x24000008u, // sceSifDmaStat
            0u, 0u,
        };
        const uint32_t descriptor[] = {
            payloadAddress, eeDestination, sizeof(uint32_t), 0u,
        };
        constexpr uint32_t payload = 0x53494621u; // "SIF!"

        std::vector<uint8_t> segment(program.filesz, 0u);
        std::memcpy(segment.data(), entry, sizeof(entry));
        std::memcpy(segment.data() + 0x100u, imports, sizeof(imports));
        std::memcpy(segment.data() + 0x180u, descriptor, sizeof(descriptor));
        std::memcpy(segment.data() + 0x1A0u, &payload, sizeof(payload));
        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    void writeMcmanRegistrationIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t initAddress = loadAddress + 0x100u;
        constexpr uint32_t deinitAddress = loadAddress + 0x120u;
        constexpr uint32_t callbackAddress = loadAddress + 0x140u;
        constexpr uint32_t deviceAddress = loadAddress + 0x200u;
        constexpr uint32_t operationsAddress = loadAddress + 0x220u;
        constexpr uint32_t nameAddress = loadAddress + 0x280u;
        constexpr uint32_t markerAddress = loadAddress + 0x290u;
        constexpr uint32_t secrmanTableAddress = loadAddress + 0x300u;
        constexpr uint32_t secrCommandStub = secrmanTableAddress + 20u;
        constexpr uint32_t secrDeviceIdStub = secrCommandStub + 8u;
        constexpr uint32_t modloadTableAddress = loadAddress + 0x340u;
        constexpr uint32_t setKelfCallbackStub = modloadTableAddress + 20u;
        constexpr uint32_t iomanTableAddress = loadAddress + 0x380u;
        constexpr uint32_t deleteMissingDriverStub = iomanTableAddress + 20u;
        constexpr uint32_t addDriverStub = deleteMissingDriverStub + 8u;
        constexpr uint32_t deleteDriverStub = addDriverStub + 8u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x400u; program.memsz = 0x400u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const auto jump = [](uint32_t target) { return 0x08000000u | ((target >> 2u) & 0x03FFFFFFu); };
        constexpr uint32_t epilogueAddress = loadAddress + 35u * sizeof(uint32_t);
        const uint32_t entry[] = {
            0x27BDFFE0u, 0xAFBF001Cu,
            0x3C040001u, 0x34840140u, jal(secrCommandStub), 0x00000000u,
            0x3C040001u, 0x34840140u, jal(secrDeviceIdStub), 0x00000000u,
            0x3C040001u, 0x34840140u, jal(setKelfCallbackStub), 0x00000000u,
            0x3C040001u, 0x34840280u, jal(deleteMissingDriverStub), 0x00000000u,
            0x3C040001u, 0x34840200u, jal(addDriverStub), 0x00000000u,
            0x1440000Bu, 0x00000000u, // bnez v0, failure
            0x3C040001u, 0x34840280u, jal(deleteDriverStub), 0x00000000u,
            0x14400005u, 0x00000000u, // bnez v0, failure
            0x3C080001u, 0x8D020290u, jump(epilogueAddress), 0x00000000u,
            0x2402FFFFu, // failure: return -1
            0x8FBF001Cu, 0x27BD0020u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t init[] = {
            0x3C080001u, 0x35080290u, 0x24090001u, 0xAD090000u,
            0x03E00008u, 0x00001021u,
        };
        const uint32_t deinit[] = {
            0x3C080001u, 0x35080290u, 0x8D090000u, 0x00000000u,
            0x25290001u, 0xAD090000u, 0x03E00008u, 0x00001021u,
        };
        const uint32_t callback[] = {0x03E00008u, 0x00001021u};
        const uint32_t device[] = {nameAddress, 0u, 0u, 0u, operationsAddress};
        uint32_t operations[17]{};
        operations[0] = initAddress;
        operations[1] = deinitAddress;
        const char name[] = "mc";
        const uint32_t secrmanImports[] = {
            0x41E00000u, 0u, 0x00000104u,
            0x72636573u, 0x006E616Du, // "secrman"
            0x03E00008u, 0x24000004u,
            0x03E00008u, 0x24000005u,
            0u, 0u,
        };
        const uint32_t modloadImports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x6C646F6Du, 0x0064616Fu, // "modload"
            0x03E00008u, 0x2400000Du,
            0u, 0u,
        };
        const uint32_t iomanImports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x616D6F69u, 0x0000006Eu, // "ioman"
            0x03E00008u, 0x24000015u, // DelDrv
            0x03E00008u, 0x24000014u, // AddDrv
            0x03E00008u, 0x24000015u, // DelDrv
            0u, 0u,
        };

        std::vector<uint8_t> segment(program.filesz, 0u);
        const auto put = [&](uint32_t offset, const void *data, size_t size)
        {
            std::memcpy(segment.data() + offset, data, size);
        };
        put(0x000u, entry, sizeof(entry));
        put(0x100u, init, sizeof(init));
        put(0x120u, deinit, sizeof(deinit));
        put(0x140u, callback, sizeof(callback));
        put(0x200u, device, sizeof(device));
        put(0x220u, operations, sizeof(operations));
        put(0x280u, name, sizeof(name));
        put(0x300u, secrmanImports, sizeof(secrmanImports));
        put(0x340u, modloadImports, sizeof(modloadImports));
        put(0x380u, iomanImports, sizeof(iomanImports));

        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    void writeCdvdLifecycleIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t importTableAddress = loadAddress + 0x100u;
        constexpr uint32_t initStub = importTableAddress + 20u;
        constexpr uint32_t callbackStub = initStub + 8u;
        constexpr uint32_t epilogueAddress = loadAddress + 26u * sizeof(uint32_t);

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x140u; program.memsz = 0x140u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const auto jump = [](uint32_t target) { return 0x08000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const uint32_t entry[] = {
            0x27BDFFF0u, 0xAFBF000Cu,
            0x00002021u, jal(initStub), 0x00000000u,
            0x24080001u, 0x14480012u, 0x00000000u,
            0x3C041234u, 0x34845678u, jal(callbackStub), 0x00000000u,
            0x1440000Cu, 0x00000000u,
            0x3C0489ABu, 0x3484CDEFu, jal(callbackStub), 0x00000000u,
            0x3C081234u, 0x35085678u, 0x14480004u, 0x00000000u,
            0x00001021u, jump(epilogueAddress), 0x00000000u,
            0x2402FFFFu,
            0x8FBF000Cu, 0x27BD0010u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t imports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x64766463u, 0x006E616Du, // "cdvdman"
            0x03E00008u, 0x24000004u, // sceCdInit
            0x03E00008u, 0x24000025u, // sceCdCallback
            0u, 0u,
        };

        std::vector<uint8_t> segment(program.filesz, 0u);
        std::memcpy(segment.data(), entry, sizeof(entry));
        std::memcpy(segment.data() + 0x100u, imports, sizeof(imports));
        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    void writeCdvdPvdReadIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t readBuffer = loadAddress + 0x800u;
        constexpr uint32_t importTableAddress = loadAddress + 0x100u;
        constexpr uint32_t callbackStub = importTableAddress + 20u;
        constexpr uint32_t readStub = callbackStub + 8u;
        constexpr uint32_t callbackFunction = loadAddress + 0x180u;
        constexpr uint32_t epilogueAddress = loadAddress + 26u * sizeof(uint32_t);

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x200u; program.memsz = 0x1000u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const auto jump = [](uint32_t target) { return 0x08000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const uint32_t entry[] = {
            0x27BDFFF0u, 0xAFBF000Cu,
            0x3C040001u, 0x34840180u, jal(callbackStub), 0x00000000u,
            0x24040010u, 0x24050001u,
            0x3C060001u, 0x34C60800u, jal(readStub), 0x00000000u,
            0x1040000Cu, 0x00000000u,
            0x90C80000u, 0x24090001u, 0x15090008u, 0x00000000u,
            0x90C80001u, 0x24090043u, 0x15090004u, 0x00000000u,
            0x00001021u, jump(epilogueAddress), 0x00000000u,
            0x2402FFFFu,
            0x8FBF000Cu, 0x27BD0010u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t imports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x64766463u, 0x006E616Du, // "cdvdman"
            0x03E00008u, 0x24000025u, // sceCdCallback
            0x03E00008u, 0x24000006u, // sceCdRead
            0u, 0u,
        };
        const uint32_t callback[] = {
            0x3C080001u, 0x350801C0u, 0xAD040000u,
            0x03E00008u, 0x00000000u,
        };

        std::vector<uint8_t> segment(program.filesz, 0u);
        std::memcpy(segment.data(), entry, sizeof(entry));
        std::memcpy(segment.data() + 0x100u, imports, sizeof(imports));
        std::memcpy(segment.data() + (callbackFunction - loadAddress), callback, sizeof(callback));
        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    void writeCdvdSeekIrx(TestHost &host, uint32_t address)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t importTableAddress = loadAddress + 0x100u;
        constexpr uint32_t initStub = importTableAddress + 20u;
        constexpr uint32_t callbackStub = initStub + 8u;
        constexpr uint32_t seekStub = callbackStub + 8u;
        constexpr uint32_t callbackFunction = loadAddress + 0x180u;

        ElfHeader header{};
        header.ident[0] = 0x7Fu; header.ident[1] = 'E'; header.ident[2] = 'L'; header.ident[3] = 'F';
        header.ident[4] = 1u; header.ident[5] = 1u; header.ident[6] = 1u;
        header.type = 2u; header.machine = 8u; header.version = 1u;
        header.entry = loadAddress; header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader); header.phentsize = sizeof(ProgramHeader); header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u; program.offset = codeOffset; program.vaddr = loadAddress; program.paddr = loadAddress;
        program.filesz = 0x200u; program.memsz = 0x200u; program.flags = 7u; program.align = 4u;

        const auto jal = [](uint32_t target) { return 0x0C000000u | ((target >> 2u) & 0x03FFFFFFu); };
        const uint32_t entry[] = {
            0x27BDFFF0u, 0xAFBF000Cu,
            0x00002021u, jal(initStub), 0x00000000u,
            0x3C040001u, 0x34840180u, jal(callbackStub), 0x00000000u,
            0x24041234u, jal(seekStub), 0x00000000u,
            0x8FBF000Cu, 0x27BD0010u, 0x03E00008u, 0x00000000u,
        };
        const uint32_t imports[] = {
            0x41E00000u, 0u, 0x00000101u,
            0x64766463u, 0x006E616Du, // "cdvdman"
            0x03E00008u, 0x24000004u, // sceCdInit
            0x03E00008u, 0x24000025u, // sceCdCallback
            0x03E00008u, 0x24000007u, // sceCdSeek
            0u, 0u,
        };
        const uint32_t callback[] = {
            0x3C080001u, 0x350801C0u, 0xAD040000u,
            0x03E00008u, 0x00001021u,
        };

        std::vector<uint8_t> segment(program.filesz, 0u);
        std::memcpy(segment.data(), entry, sizeof(entry));
        std::memcpy(segment.data() + 0x100u, imports, sizeof(imports));
        std::memcpy(segment.data() + (callbackFunction - loadAddress), callback, sizeof(callback));
        std::memset(host.guest.data() + address, 0, codeOffset + program.filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
    }

    bool expect(bool value, const char *message)
    {
        if (!value)
            std::cerr << "FAIL: " << message << '\n';
        return value;
    }

    // Scheduling decisions locked to the pre-queue scan: lowest priority number, then lowest id.
    // Delay promotion is every due thread before the pick. nextWakeCycle is min(fallback, delay wakes)
    // and does not promote. Yield does not rotate. Semaphore / clear-event ties follow map id order.
    bool testIopScheduleOrderAndWakeCycles()
    {
        using namespace ps2x::iop::detail;

        struct Fix
        {
            IopMemory memory;
            IopKernel kernel;
            uint32_t desc = 0x2000u;

            Fix()
                : kernel(memory)
            {
                kernel.reset();
            }

            int create(uint32_t priority, uint32_t entry = 0x10000u)
            {
                IopCpuState cpu{};
                memory.write32(desc + 0u, 0u);
                memory.write32(desc + 4u, 0u);
                memory.write32(desc + 8u, entry);
                memory.write32(desc + 12u, 0x100u);
                memory.write32(desc + 16u, priority);
                cpu.gpr[4] = desc;
                desc += 0x20u;
                if (!kernel.dispatchThreadImport(4u, cpu, 0u))
                    return -1;
                return static_cast<int>(cpu.gpr[2]);
            }

            bool start(int id)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                return kernel.dispatchThreadImport(6u, cpu, 0u) && cpu.gpr[2] == 0u;
            }

            bool threadOp(uint16_t ordinal, int id)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                return kernel.dispatchThreadImport(ordinal, cpu, 0u);
            }

            bool setPri(int id, uint32_t priority)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                cpu.gpr[5] = priority;
                return kernel.dispatchThreadImport(14u, cpu, 0u);
            }

            int semaphore(int current, int maximum)
            {
                IopCpuState cpu{};
                memory.write32(desc + 0u, 0u);
                memory.write32(desc + 4u, 0u);
                memory.write32(desc + 8u, static_cast<uint32_t>(current));
                memory.write32(desc + 12u, static_cast<uint32_t>(maximum));
                cpu.gpr[4] = desc;
                desc += 0x20u;
                if (!kernel.dispatchSemaphoreImport(4u, cpu))
                    return -1;
                return static_cast<int>(cpu.gpr[2]);
            }

            bool signal(int id)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                return kernel.dispatchSemaphoreImport(6u, cpu);
            }

            bool waitSem(IopThread &thread, int id)
            {
                thread.cpu.gpr[4] = static_cast<uint32_t>(id);
                return kernel.dispatchSemaphoreImport(8u, thread.cpu);
            }

            int eventFlag()
            {
                IopCpuState cpu{};
                memory.write32(desc + 0u, 0u);
                memory.write32(desc + 4u, 0u);
                memory.write32(desc + 8u, 0u);
                cpu.gpr[4] = desc;
                desc += 0x20u;
                if (!kernel.dispatchEventImport(4u, cpu))
                    return -1;
                return static_cast<int>(cpu.gpr[2]);
            }

            bool waitEvent(IopThread &thread, int id, uint32_t bits, uint32_t mode)
            {
                thread.cpu.gpr[4] = static_cast<uint32_t>(id);
                thread.cpu.gpr[5] = bits;
                thread.cpu.gpr[6] = mode;
                thread.cpu.gpr[7] = 0u;
                return kernel.dispatchEventImport(10u, thread.cpu);
            }

            bool setEvent(int id, uint32_t bits)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                cpu.gpr[5] = bits;
                return kernel.dispatchEventImport(6u, cpu);
            }

            void end(IopThread &thread) { kernel.endTimeslice(thread, kThreadReturnSentinel); }

            bool beginIs(uint64_t cycle, int expectId, const char *what, IopThread *&out)
            {
                out = kernel.beginNextReady(cycle);
                const int got = out != nullptr ? out->id : 0;
                if (got == expectId)
                    return true;
                std::cerr << "FAIL: schedule " << what << " begin got " << got << " expected " << expectId << '\n';
                return false;
            }

            bool wakeIs(uint64_t fallback, uint64_t expectWake, const char *what)
            {
                const uint64_t first = kernel.nextWakeCycle(fallback);
                const uint64_t second = kernel.nextWakeCycle(fallback);
                if (first == second && first == expectWake)
                    return true;
                std::cerr << "FAIL: schedule " << what << " wake " << first << " again " << second
                          << " expected " << expectWake << '\n';
                return false;
            }

            bool statusIs(int id, uint32_t expectStatus, const char *what)
            {
                IopCpuState cpu{};
                cpu.gpr[4] = static_cast<uint32_t>(id);
                cpu.gpr[5] = 0x9000u;
                if (!kernel.dispatchThreadImport(22u, cpu, 0u))
                    return false;
                const uint32_t status = memory.read32(0x9008u);
                if (status == expectStatus)
                    return true;
                std::cerr << "FAIL: schedule " << what << " status " << status << " expected " << expectStatus << '\n';
                return false;
            }
        };

        IopThread *thread = nullptr;
        {
            Fix fx;
            if (fx.create(30u) != 1 || fx.create(10u) != 2 || fx.create(10u) != 3 || fx.create(20u) != 4 || fx.create(1u) != 5)
                return expect(false, "schedule create ids");
            for (const int id : {1, 2, 3, 4, 5})
                if (!fx.start(id))
                    return expect(false, "schedule start");
            if (!fx.wakeIs(1000000000ull, 1000000000ull, "pri-none") ||
                !fx.beginIs(0u, 5, "pri-best", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 5, "pri-norotate", thread))
                return false;
            if (!fx.kernel.dispatchThreadImport(16u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 5, "pri-after-yield", thread) || !fx.setPri(5, 40u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 2, "pri-changed", thread) || !fx.setPri(3, 1u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 3, "pri-ready-requeue", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(20u) != 1 || fx.create(5u) != 2 || fx.create(30u) != 3 || fx.create(50u) != 4)
                return expect(false, "alarm ids");
            for (const int id : {1, 2, 3, 4})
                if (!fx.start(id))
                    return false;
            if (!fx.beginIs(0u, 2, "al-run", thread))
                return false;
            fx.kernel.delayCurrentUntil(1000u, thread->cpu);
            fx.end(*thread);
            if (!fx.beginIs(0u, 1, "al-next", thread))
                return false;
            fx.kernel.delayCurrentUntil(10u, thread->cpu);
            fx.end(*thread);
            if (!fx.beginIs(0u, 3, "al-third", thread))
                return false;
            fx.kernel.delayCurrentUntil(10u, thread->cpu);
            fx.end(*thread);
            if (!fx.wakeIs(5u, 5u, "al-fallback-wins") || !fx.wakeIs(50u, 10u, "al-min") ||
                !fx.beginIs(9u, 4, "al-before", thread))
                return false;
            fx.end(*thread);
            if (!fx.wakeIs(100u, 10u, "al-due-still-queued") || !fx.beginIs(10u, 1, "al-promote-best", thread) ||
                !fx.wakeIs(1000000000ull, 1000u, "al-next-alarm"))
                return false;
            if (!fx.kernel.dispatchThreadImport(24u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(15u, 3, "al-sibling-promoted", thread))
                return false;
            if (!fx.kernel.dispatchThreadImport(24u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(16u, 4, "al-only-low", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(1000u, 2, "al-late", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(10u) != 1 || fx.create(10u) != 2 || fx.create(40u) != 3)
                return false;
            for (const int id : {1, 2, 3})
                if (!fx.start(id))
                    return false;
            if (!fx.beginIs(0u, 1, "tie-a", thread))
                return false;
            fx.kernel.delayCurrentUntil(500u, thread->cpu);
            fx.end(*thread);
            if (!fx.beginIs(0u, 2, "tie-b", thread))
                return false;
            fx.kernel.delayCurrentUntil(500u, thread->cpu);
            fx.end(*thread);
            if (!fx.beginIs(0u, 3, "tie-c", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(499u, 3, "tie-early", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(500u, 1, "tie-same-cycle-low-id", thread))
                return false;
            if (!fx.threadOp(29u, 1))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(500u, 2, "tie-after-suspend", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(1u) != 1 || fx.create(2u) != 2)
                return false;
            if (!fx.start(1) || !fx.start(2) || !fx.beginIs(0u, 1, "sl-a", thread))
                return false;
            if (!fx.kernel.dispatchThreadImport(24u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 2, "sl-b", thread))
                return false;
            fx.end(*thread);
            if (!fx.threadOp(25u, 1) || !fx.beginIs(0u, 1, "sl-wake", thread))
                return false;
            if (!fx.threadOp(25u, 1) || !fx.kernel.dispatchThreadImport(24u, thread->cpu, 0u))
                return false;
            if (!fx.statusIs(1, 0x01u, "sl-swallow"))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 1, "sl-swallow-stays", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(4u) != 1 || fx.create(4u) != 2 || fx.create(8u) != 3)
                return false;
            for (const int id : {1, 2, 3})
                if (!fx.start(id))
                    return false;
            const int sid = fx.semaphore(0, 1);
            if (sid <= 0 || !fx.threadOp(29u, 1) || !fx.beginIs(0u, 2, "sem-b-first", thread))
                return false;
            if (!fx.waitSem(*thread, sid))
                return false;
            fx.end(*thread);
            if (!fx.threadOp(31u, 1) || !fx.beginIs(0u, 1, "sem-a-second", thread))
                return false;
            if (!fx.waitSem(*thread, sid))
                return false;
            fx.end(*thread);
            if (!fx.signal(sid) || !fx.beginIs(0u, 1, "sem-wake-low-id", thread))
                return false;
            fx.end(*thread);
            if (!fx.signal(sid) || !fx.beginIs(0u, 1, "sem-still-lower-id", thread))
                return false;
            if (!fx.threadOp(29u, 1))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 2, "sem-fifo-would-differ", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(10u) != 1 || fx.create(10u) != 2 || !fx.start(1) || !fx.start(2))
                return false;
            if (!fx.beginIs(0u, 1, "pd", thread))
                return false;
            fx.kernel.delayCurrentUntil(100u, thread->cpu);
            fx.end(*thread);
            if (!fx.setPri(1, 1u) || !fx.beginIs(99u, 2, "pd-before", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(100u, 1, "pd-promoted-pri", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(1u) != 1 || fx.create(2u) != 2 || fx.create(3u) != 3)
                return false;
            for (const int id : {1, 2, 3})
                if (!fx.start(id))
                    return false;
            if (!fx.beginIs(0u, 1, "rd", thread))
                return false;
            fx.kernel.delayCurrentUntil(80u, thread->cpu);
            fx.end(*thread);
            if (!fx.threadOp(18u, 1) || !fx.wakeIs(5000u, 5000u, "rd-released") ||
                !fx.beginIs(0u, 1, "rd-back", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 1, "rd-hold", thread) || !fx.threadOp(5u, 2))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 1, "rd-deleted", thread) || !fx.threadOp(29u, 1))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 3, "rd-survivor", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            const int victim = fx.create(1u, 0x31000u);
            const int other = fx.create(5u, 0x32000u);
            if (victim != 1 || other != 2 || !fx.start(1) || !fx.start(2) || !fx.beginIs(0u, 1, "term", thread))
                return false;
            fx.kernel.terminateThreadsInRange(0x31000u, 0x100u);
            fx.end(*thread);
            if (fx.kernel.threadCount() != 1u || !fx.beginIs(0u, 2, "term-other", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(10u) != 1 || fx.create(10u) != 2 || !fx.start(1) || !fx.start(2))
                return false;
            const int eid = fx.eventFlag();
            if (eid <= 0 || !fx.threadOp(29u, 1) || !fx.beginIs(0u, 2, "ev-b-first", thread))
                return false;
            if (!fx.waitEvent(*thread, eid, 1u, 0x10u))
                return false;
            fx.end(*thread);
            if (!fx.threadOp(31u, 1) || !fx.beginIs(0u, 1, "ev-a-second", thread))
                return false;
            if (!fx.waitEvent(*thread, eid, 1u, 0x10u))
                return false;
            fx.end(*thread);
            if (!fx.setEvent(eid, 1u) || !fx.beginIs(0u, 1, "ev-clear-low-id", thread))
                return false;
            if (!fx.threadOp(29u, 1))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 0, "ev-other-stays", thread))
                return false;
        }
        {
            Fix fx;
            if (fx.create(1u) != 1 || fx.create(2u) != 2 || !fx.start(1) || !fx.start(2))
                return false;
            if (!fx.beginIs(0u, 1, "usec", thread))
                return false;
            thread->cpu.gpr[4] = 1000u;
            if (!fx.kernel.dispatchThreadImport(33u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            const uint64_t delayCycles = (1000ull * kIopClockHz + 999999ull) / 1000000ull;
            const uint64_t wake = std::max<uint64_t>(delayCycles, 1u);
            if (!fx.wakeIs(1000000000ull, wake, "usec-alarm") || !fx.beginIs(wake - 1u, 2, "usec-before", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(wake, 1, "usec-due", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            if (fx.create(5u) != 1 || fx.create(5u) != 2 || fx.create(15u) != 3)
                return false;
            for (const int id : {1, 2, 3})
                if (!fx.start(id))
                    return false;
            if (!fx.beginIs(0u, 1, "mix", thread))
                return false;
            fx.kernel.delayCurrentUntil(100u, thread->cpu);
            fx.end(*thread);
            const int sid = fx.semaphore(0, 1);
            if (sid <= 0 || !fx.beginIs(0u, 2, "mix-b", thread) || !fx.waitSem(*thread, sid))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, 3, "mix-c", thread) || !fx.kernel.dispatchThreadImport(24u, thread->cpu, 0u))
                return false;
            fx.end(*thread);
            if (!fx.wakeIs(1000u, 100u, "mix-alarm") || !fx.beginIs(100u, 1, "mix-due", thread) || !fx.signal(sid))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(100u, 1, "mix-low-id", thread) || !fx.threadOp(29u, 1))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(100u, 2, "mix-signalled", thread))
                return false;
            fx.end(*thread);
        }
        {
            Fix fx;
            int best = 0;
            uint32_t bestPri = 999u;
            int second = 0;
            uint32_t secondPri = 999u;
            for (int i = 0; i < 48; ++i)
            {
                const uint32_t pri = static_cast<uint32_t>((i * 17) % 126 + 1);
                const int id = fx.create(pri, 0x20000u + static_cast<uint32_t>(i) * 4u);
                if (id <= 0 || !fx.start(id))
                    return expect(false, "schedule 48 create");
                if (best == 0 || pri < bestPri || (pri == bestPri && id < best))
                {
                    if (best != 0 && (second == 0 || bestPri < secondPri || (bestPri == secondPri && best < second)))
                    {
                        second = best;
                        secondPri = bestPri;
                    }
                    best = id;
                    bestPri = pri;
                }
                else if (second == 0 || pri < secondPri || (pri == secondPri && id < second))
                {
                    second = id;
                    secondPri = pri;
                }
            }
            if (!fx.beginIs(0u, best, "many", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(0u, best, "many-again", thread))
                return false;
            fx.kernel.delayCurrentUntil(1000u, thread->cpu);
            fx.end(*thread);
            if (!fx.beginIs(0u, second, "many-second", thread))
                return false;
            fx.end(*thread);
            if (!fx.beginIs(1000u, best, "many-alarm", thread))
                return false;
            fx.end(*thread);
            const auto started = std::chrono::steady_clock::now();
            for (int i = 0; i < 50000; ++i)
            {
                IopThread *next = fx.kernel.beginNextReady(0u);
                if (next == nullptr || next->id != best)
                    return expect(false, "schedule bench picked a different thread");
                fx.kernel.endTimeslice(*next, kThreadReturnSentinel);
            }
            const auto readyUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
            std::cout << "iop sched bench ready-pick " << readyUs << " us ("
                      << (readyUs * 1000 / 50000) << " ns/call) 48 threads x 50000\n";
        }
        {
            Fix fx;
            for (int i = 0; i < 32; ++i)
            {
                const int id = fx.create(static_cast<uint32_t>((i * 3) % 60 + 1));
                if (id <= 0 || !fx.start(id))
                    return false;
            }
            int armed = 0;
            for (;;)
            {
                IopThread *next = fx.kernel.beginNextReady(0u);
                if (next == nullptr)
                    break;
                fx.kernel.delayCurrentUntil(500000ull + static_cast<uint64_t>(next->id), next->cpu);
                fx.end(*next);
                if (++armed > 32)
                    return expect(false, "schedule wake bench did not drain ready threads");
            }
            if (armed != 32 || !fx.wakeIs(~uint64_t{0}, 500001ull, "wake-min"))
                return false;
            const auto started = std::chrono::steady_clock::now();
            for (int i = 0; i < 50000; ++i)
                if (fx.kernel.nextWakeCycle(~uint64_t{0}) != 500001ull)
                    return expect(false, "schedule cached wake changed");
            const auto wakeUs = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - started)
                                    .count();
            std::cout << "iop sched bench next-wake " << wakeUs << " us ("
                      << (wakeUs * 1000 / 50000) << " ns/call) 32 delays x 50000\n";
        }
        return true;
    }

    // Two-pass MIPS so a missed branch cannot silently land in a delay slot.
    class MipsProg
    {
    public:
        explicit MipsProg(uint32_t base) : m_base(base) {}

        void label(const char *name) { m_labels[name] = m_base + static_cast<uint32_t>(m_words.size() * 4u); }
        void dw(uint32_t word) { m_words.push_back(word); }
        void nop() { dw(0u); }
        void jalAbs(uint32_t target) { dw(0x0C000000u | ((target >> 2u) & 0x03FFFFFFu)); }
        void jLabel(const char *name) { m_fix.push_back(Fix{m_words.size(), name, false, 0, 0, false}); dw(0u); }
        void bLabel(bool bne, int rs, int rt, const char *name)
        {
            m_fix.push_back(Fix{m_words.size(), name, true, rs, rt, bne});
            dw(0u);
        }

        [[nodiscard]] bool finish(std::string &error)
        {
            for (const Fix &fix : m_fix)
            {
                const auto it = m_labels.find(fix.label);
                if (it == m_labels.end())
                {
                    error = std::string("missing label ") + fix.label;
                    return false;
                }
                if (!fix.branch)
                {
                    m_words[fix.index] = 0x08000000u | (it->second >> 2u);
                    continue;
                }
                const uint32_t pc = m_base + static_cast<uint32_t>(fix.index * 4u);
                const int32_t offset = static_cast<int32_t>(it->second - (pc + 4u)) >> 2;
                if (offset < -32768 || offset > 32767)
                {
                    error = std::string("branch out of range ") + fix.label;
                    return false;
                }
                const uint32_t op = fix.bne ? 5u : 4u;
                m_words[fix.index] = (op << 26u) | (static_cast<uint32_t>(fix.rs) << 21u) |
                                     (static_cast<uint32_t>(fix.rt) << 16u) |
                                     (static_cast<uint32_t>(offset) & 0xFFFFu);
            }
            return true;
        }

        [[nodiscard]] const std::vector<uint32_t> &words() const { return m_words; }
        [[nodiscard]] uint32_t end() const { return m_base + static_cast<uint32_t>(m_words.size() * 4u); }

    private:
        struct Fix
        {
            size_t index;
            const char *label;
            bool branch;
            int rs;
            int rt;
            bool bne;
        };
        uint32_t m_base;
        std::vector<uint32_t> m_words;
        std::vector<Fix> m_fix;
        std::map<std::string, uint32_t> m_labels;
    };

    void putWords(std::vector<uint8_t> &segment, uint32_t offset, const std::vector<uint32_t> &words)
    {
        for (size_t i = 0; i < words.size(); ++i)
            std::memcpy(segment.data() + offset + i * 4u, &words[i], sizeof(uint32_t));
    }

    void putImport(std::vector<uint8_t> &segment, uint32_t offset, uint32_t name0, uint32_t name1,
                   const uint16_t *ordinals, size_t count)
    {
        std::vector<uint32_t> words;
        words.push_back(0x41E00000u);
        words.push_back(0u);
        words.push_back(0x00000101u);
        words.push_back(name0);
        words.push_back(name1);
        for (size_t i = 0; i < count; ++i)
        {
            words.push_back(0x03E00008u);
            words.push_back(0x24000000u | ordinals[i]);
        }
        words.push_back(0u);
        words.push_back(0u);
        putWords(segment, offset, words);
    }

    // Sleeper modes: 0 stamp, 1 Sleep, 2 WaitSema, 3 WaitEventFlag, 4 Suspend self, 5 Delay 1000 us.
    // RPC a0: 1 Wake, 2 Signal, 3 SetEvent, 4 Start, 5 Resume, 6 ChangePri, 7 arm timer, 8 SPU DMA, 9 ReleaseWait.
    bool writeIdleSkipIrx(TestHost &host, uint32_t address, uint32_t mode, bool startThread, std::string &error)
    {
        constexpr uint32_t codeOffset = 0x100u;
        constexpr uint32_t loadAddress = 0x00010000u;
        constexpr uint32_t sleeperAddress = 0x00010200u;
        constexpr uint32_t irqAddress = 0x00010380u;
        constexpr uint32_t timerAddress = 0x000103C0u;
        constexpr uint32_t rpcAddress = 0x00010420u;
        constexpr uint32_t thbase = 0x00010900u;
        constexpr uint32_t createStub = thbase + 20u;
        constexpr uint32_t startStub = createStub + 8u;
        constexpr uint32_t sleepStub = startStub + 8u;
        constexpr uint32_t wakeStub = sleepStub + 8u;
        constexpr uint32_t delayStub = wakeStub + 8u;
        constexpr uint32_t getTimeStub = delayStub + 8u;
        constexpr uint32_t getIdStub = getTimeStub + 8u;
        constexpr uint32_t suspendStub = getIdStub + 8u;
        constexpr uint32_t resumeStub = suspendStub + 8u;
        constexpr uint32_t releaseStub = resumeStub + 8u;
        constexpr uint32_t changePriStub = releaseStub + 8u;
        constexpr uint32_t thsemap = 0x00010980u;
        constexpr uint32_t semCreateStub = thsemap + 20u;
        constexpr uint32_t semSignalStub = semCreateStub + 8u;
        constexpr uint32_t semWaitStub = semSignalStub + 8u;
        constexpr uint32_t thevent = 0x00010A00u;
        constexpr uint32_t evCreateStub = thevent + 20u;
        constexpr uint32_t evSetStub = evCreateStub + 8u;
        constexpr uint32_t evWaitStub = evSetStub + 8u;
        constexpr uint32_t intrman = 0x00010A80u;
        constexpr uint32_t regIntrStub = intrman + 20u;
        constexpr uint32_t enaIntrStub = regIntrStub + 8u;
        constexpr uint32_t timrman = 0x00010B00u;
        constexpr uint32_t allocStub = timrman + 20u;
        constexpr uint32_t setupStub = allocStub + 8u;
        constexpr uint32_t setHandlerStub = setupStub + 8u;
        constexpr uint32_t startTimerStub = setHandlerStub + 8u;
        constexpr uint32_t sifcmd = 0x00010B80u;
        constexpr uint32_t regRpcStub = sifcmd + 20u;

        MipsProg entry(loadAddress);
        entry.dw(0x27BDFFE0u);
        entry.dw(0xAFBF001Cu);
        entry.dw(0xAFB00018u);
        entry.dw(0xAFB10014u);
        entry.dw(0xAFB20010u);
        entry.dw(0x3C040001u);
        entry.dw(0x34841120u);
        entry.jalAbs(semCreateStub);
        entry.nop();
        entry.dw(0x00408821u);
        entry.dw(0x3C040001u);
        entry.dw(0x34841140u);
        entry.jalAbs(evCreateStub);
        entry.nop();
        entry.dw(0x00409021u);
        entry.dw(0x3C040001u);
        entry.dw(0x34841100u);
        entry.jalAbs(createStub);
        entry.nop();
        entry.dw(0x00408021u);
        entry.dw(0x3C080001u);
        entry.dw(0xAD101160u);
        entry.dw(0xAD111164u);
        entry.dw(0xAD121168u);
        entry.dw(0x24040024u);
        entry.dw(0x00002821u);
        entry.dw(0x3C060001u);
        entry.dw(0x34C60380u);
        entry.dw(0x00003821u);
        entry.jalAbs(regIntrStub);
        entry.nop();
        entry.dw(0x24040024u);
        entry.jalAbs(enaIntrStub);
        entry.nop();
        entry.dw(0x3C040001u);
        entry.dw(0x348411A0u);
        entry.dw(0x3C051111u);
        entry.dw(0x34A50001u);
        entry.dw(0x3C060001u);
        entry.dw(0x34C60420u);
        entry.dw(0x3C070001u);
        entry.dw(0x34E711E0u);
        entry.dw(0xAFA00010u);
        entry.dw(0xAFA00014u);
        entry.dw(0xAFA00018u);
        entry.jalAbs(regRpcStub);
        entry.nop();
        if (startThread)
        {
            entry.dw(0x02002021u);
            entry.jalAbs(startStub);
            entry.nop();
        }
        entry.dw(0x8FBF001Cu);
        entry.dw(0x8FB00018u);
        entry.dw(0x8FB10014u);
        entry.dw(0x8FB20010u);
        entry.dw(0x27BD0020u);
        entry.dw(0x03E00008u);
        entry.dw(0x00001021u);

        // Saves ra so it returns to the thread sentinel (goes dormant) instead of looping on its own jal ra.
        MipsProg sleeper(sleeperAddress);
        sleeper.dw(0x27BDFFF0u);
        sleeper.dw(0xAFBF000Cu);
        sleeper.dw(0x3C080001u);
        sleeper.dw(0x8D0B116Cu);
        sleeper.nop(); // R3000 load delay: the branch would still see the old t3
        sleeper.bLabel(true, 11, 0, "not0");
        sleeper.nop();
        sleeper.jLabel("stamp");
        sleeper.nop();
        sleeper.label("not0");
        sleeper.dw(0x240C0001u);
        sleeper.bLabel(true, 11, 12, "not1");
        sleeper.nop();
        sleeper.jalAbs(sleepStub);
        sleeper.nop();
        sleeper.jLabel("stamp");
        sleeper.nop();
        sleeper.label("not1");
        sleeper.dw(0x240C0002u);
        sleeper.bLabel(true, 11, 12, "not2");
        sleeper.nop();
        sleeper.dw(0x8D041164u);
        sleeper.jalAbs(semWaitStub);
        sleeper.nop();
        sleeper.jLabel("stamp");
        sleeper.nop();
        sleeper.label("not2");
        sleeper.dw(0x240C0003u);
        sleeper.bLabel(true, 11, 12, "not3");
        sleeper.nop();
        sleeper.dw(0x8D041168u);
        sleeper.dw(0x24050001u);
        sleeper.dw(0x00003021u);
        sleeper.dw(0x00003821u);
        sleeper.jalAbs(evWaitStub);
        sleeper.nop();
        sleeper.jLabel("stamp");
        sleeper.nop();
        sleeper.label("not3");
        sleeper.dw(0x240C0004u);
        sleeper.bLabel(true, 11, 12, "not4");
        sleeper.nop();
        sleeper.jalAbs(getIdStub);
        sleeper.nop();
        sleeper.dw(0x00402021u);
        sleeper.jalAbs(suspendStub);
        sleeper.nop();
        sleeper.jLabel("stamp");
        sleeper.nop();
        sleeper.label("not4");
        sleeper.dw(0x240403E8u);
        sleeper.jalAbs(delayStub);
        sleeper.nop();
        sleeper.label("stamp");
        sleeper.dw(0x3C040001u);
        sleeper.dw(0x34841180u);
        sleeper.jalAbs(getTimeStub);
        sleeper.nop();
        sleeper.dw(0x3C080001u);
        sleeper.dw(0x24090001u);
        sleeper.dw(0xAD091188u);
        sleeper.dw(0x8FBF000Cu);
        sleeper.nop();
        sleeper.dw(0x27BD0010u);
        sleeper.dw(0x03E00008u);
        sleeper.nop();

        MipsProg irq(irqAddress);
        irq.dw(0x3C040001u);
        irq.dw(0x8C841160u);
        irq.jalAbs(wakeStub);
        irq.nop();
        irq.dw(0x03E00008u);
        irq.nop();

        MipsProg timerCb(timerAddress);
        timerCb.dw(0x3C040001u);
        timerCb.dw(0x8C841160u);
        timerCb.jalAbs(wakeStub);
        timerCb.nop();
        timerCb.dw(0x03E00008u);
        timerCb.dw(0x00001021u);

        MipsProg rpc(rpcAddress);
        const auto dispatch = [&](uint32_t command, const char *cmd, const char *skip)
        {
            rpc.dw(0x240C0000u | command);
            rpc.bLabel(true, 4, 12, skip);
            rpc.nop();
            rpc.jLabel(cmd);
            rpc.nop();
            rpc.label(skip);
        };
        rpc.dw(0x27BDFFE0u);
        rpc.dw(0xAFBF001Cu);
        rpc.dw(0xAFB00018u);
        rpc.dw(0x3C080001u);
        dispatch(1u, "cmd1", "sk1");
        dispatch(2u, "cmd2", "sk2");
        dispatch(3u, "cmd3", "sk3");
        dispatch(4u, "cmd4", "sk4");
        dispatch(5u, "cmd5", "sk5");
        dispatch(6u, "cmd6", "sk6");
        dispatch(7u, "cmd7", "sk7");
        dispatch(8u, "cmd8", "sk8");
        dispatch(9u, "cmd9", "sk9");
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd1");
        rpc.dw(0x8D041160u);
        rpc.jalAbs(wakeStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd2");
        rpc.dw(0x8D041164u);
        rpc.jalAbs(semSignalStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd3");
        rpc.dw(0x8D041168u);
        rpc.dw(0x24050001u);
        rpc.jalAbs(evSetStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd4");
        rpc.dw(0x8D041160u);
        rpc.jalAbs(startStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd5");
        rpc.dw(0x8D041160u);
        rpc.jalAbs(resumeStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd6");
        rpc.dw(0x8D041160u);
        rpc.dw(0x24050001u);
        rpc.jalAbs(changePriStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd7");
        rpc.dw(0x24040001u);
        rpc.dw(0x24050010u);
        rpc.dw(0x24060001u);
        rpc.jalAbs(allocStub);
        rpc.nop();
        rpc.dw(0x00408021u);
        rpc.dw(0x02002021u);
        rpc.dw(0x24050001u);
        rpc.dw(0x24060001u);
        rpc.dw(0x24070001u);
        rpc.jalAbs(setupStub);
        rpc.nop();
        rpc.dw(0x02002021u);
        rpc.dw(0x24050040u);
        rpc.dw(0x3C060001u);
        rpc.dw(0x34C603C0u);
        rpc.dw(0x00003821u);
        rpc.jalAbs(setHandlerStub);
        rpc.nop();
        rpc.dw(0x02002021u);
        rpc.jalAbs(startTimerStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd8");
        rpc.dw(0x3C081F80u);
        rpc.dw(0x350810C8u);
        rpc.dw(0x3C090100u);
        rpc.dw(0xAD090000u);
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("cmd9");
        rpc.dw(0x8D041160u);
        rpc.jalAbs(releaseStub);
        rpc.nop();
        rpc.jLabel("done");
        rpc.nop();
        rpc.label("done");
        rpc.dw(0x8FBF001Cu);
        rpc.dw(0x8FB00018u);
        rpc.dw(0x27BD0020u);
        rpc.dw(0x03E00008u);
        rpc.dw(0x00001021u);

        if (!entry.finish(error) || !sleeper.finish(error) || !irq.finish(error) || !timerCb.finish(error) ||
            !rpc.finish(error))
            return false;
        if (entry.end() > sleeperAddress || sleeper.end() > irqAddress || irq.end() > timerAddress ||
            timerCb.end() > rpcAddress || rpc.end() > thbase)
        {
            error = "idle-skip IRX blocks overlap";
            return false;
        }

        constexpr uint32_t filesz = 0x1300u;
        ElfHeader header{};
        header.ident[0] = 0x7Fu;
        header.ident[1] = 'E';
        header.ident[2] = 'L';
        header.ident[3] = 'F';
        header.ident[4] = 1u;
        header.ident[5] = 1u;
        header.ident[6] = 1u;
        header.type = 2u;
        header.machine = 8u;
        header.version = 1u;
        header.entry = loadAddress;
        header.phoff = sizeof(ElfHeader);
        header.ehsize = sizeof(ElfHeader);
        header.phentsize = sizeof(ProgramHeader);
        header.phnum = 1u;

        ProgramHeader program{};
        program.type = 1u;
        program.offset = codeOffset;
        program.vaddr = loadAddress;
        program.paddr = loadAddress;
        program.filesz = filesz;
        program.memsz = filesz;
        program.flags = 7u;
        program.align = 4u;

        std::vector<uint8_t> segment(filesz, 0u);
        putWords(segment, 0u, entry.words());
        putWords(segment, sleeperAddress - loadAddress, sleeper.words());
        putWords(segment, irqAddress - loadAddress, irq.words());
        putWords(segment, timerAddress - loadAddress, timerCb.words());
        putWords(segment, rpcAddress - loadAddress, rpc.words());
        const uint16_t thbaseOrdinals[] = {4u, 6u, 24u, 25u, 33u, 34u, 20u, 29u, 31u, 18u, 14u};
        const uint16_t semOrdinals[] = {4u, 6u, 8u};
        const uint16_t evOrdinals[] = {4u, 6u, 10u};
        const uint16_t intrOrdinals[] = {4u, 6u};
        const uint16_t timerOrdinals[] = {4u, 22u, 20u, 23u};
        const uint16_t sifOrdinals[] = {17u};
        putImport(segment, thbase - loadAddress, 0x61626874u, 0x00006573u, thbaseOrdinals, 11u);
        putImport(segment, thsemap - loadAddress, 0x65736874u, 0x0070616Du, semOrdinals, 3u);
        putImport(segment, thevent - loadAddress, 0x76656874u, 0x00746E65u, evOrdinals, 3u);
        putImport(segment, intrman - loadAddress, 0x72746E69u, 0x006E616Du, intrOrdinals, 2u);
        putImport(segment, timrman - loadAddress, 0x726D6974u, 0x006E616Du, timerOrdinals, 4u);
        putImport(segment, sifcmd - loadAddress, 0x63666973u, 0x0000646Du, sifOrdinals, 1u);
        const uint32_t threadDesc[] = {0u, 0u, sleeperAddress, 0x400u, 16u};
        const uint32_t semaDesc[] = {0u, 0u, 0u, 1u};
        const uint32_t eventDesc[] = {0u, 0u, 0u};
        putWords(segment, 0x1100u, std::vector<uint32_t>(threadDesc, threadDesc + 5));
        putWords(segment, 0x1120u, std::vector<uint32_t>(semaDesc, semaDesc + 4));
        putWords(segment, 0x1140u, std::vector<uint32_t>(eventDesc, eventDesc + 3));
        std::memcpy(segment.data() + 0x116Cu, &mode, sizeof(mode));

        if (address > host.guest.size() || codeOffset + filesz > host.guest.size() - address)
        {
            error = "guest buffer too small for idle-skip IRX";
            return false;
        }
        std::memset(host.guest.data() + address, 0, codeOffset + filesz);
        std::memcpy(host.guest.data() + address, &header, sizeof(header));
        std::memcpy(host.guest.data() + address + sizeof(header), &program, sizeof(program));
        std::memcpy(host.guest.data() + address + codeOffset, segment.data(), segment.size());
        return true;
    }

    struct IdleRun
    {
        uint64_t cycles = 0;
        uint64_t clock = 0;
        uint64_t skips = 0;
        uint32_t flag = 0;
    };

    void dumpIdleLogs(const TestHost &host)
    {
        for (const std::string &line : host.logs)
            std::cerr << "  " << line << '\n';
    }

    bool runIdleSkipCase(bool cache, uint32_t mode, bool startThread, uint32_t command, uint64_t eeAfter, bool probe,
                         bool priority, IdleRun &out, std::string &why, TestHost &host)
    {
        using namespace ps2x::iop::detail;
        std::string error;
        if (!writeIdleSkipIrx(host, 0x100u, mode, startThread, error))
        {
            why = error;
            return false;
        }
        IopEmulator iop(host);
        iop.setIdleSkipCache(cache);
        const ModuleLoadResult loaded = iop.loadModuleBuffer(0x100u, nullptr, 0u);
        if (!loaded.handled || loaded.startResult != 0)
        {
            why = "idle-skip IRX start " + std::to_string(loaded.startResult);
            dumpIdleLogs(host);
            return false;
        }
        if (probe)
        {
            // The first call runs the sleeper into its wait and seals the cache on its last idle step;
            // only a later call can skip, so the probe makes two.
            iop.runEeCycles(8ull * 500ull);
            iop.runEeCycles(8ull * 500ull);
            uint32_t earlyFlag = 0xFFFFFFFFu;
            if (!iop.readMemory(0x11188u, &earlyFlag, sizeof(earlyFlag)))
            {
                why = "probe flag read failed";
                return false;
            }
            if (earlyFlag != 0u)
            {
                why = "probe ran the sleeper before the wake";
                return false;
            }
            if (cache)
            {
                if (iop.testIdleSkipCount() < 1u)
                {
                    why = "probe did not skip kernel entry (cache never sealed)";
                    return false;
                }
            }
            else if (iop.testIdleSkipCount() != 0u)
            {
                why = "cache off counted an idle skip";
                return false;
            }
        }
        const auto rpc = [&](uint32_t function) -> bool
        {
            RpcRequest request;
            request.sid = 0x11110001u;
            request.function = function;
            const RpcResult result = iop.handleRpc(request);
            if (!result.handled)
            {
                why = "rpc " + std::to_string(function) + " was not handled";
                dumpIdleLogs(host);
                return false;
            }
            return true;
        };
        if (priority)
        {
            if (!rpc(4u))
                return false;
            iop.testSealIdleSkip(iop.cycles() + 1000000ull);
            if (!rpc(6u))
                return false;
            iop.runEeCycles(8ull * 2000ull);
        }
        else if (command != 0u)
        {
            if (!rpc(command))
                return false;
            iop.runEeCycles(eeAfter);
        }
        uint32_t flag = 0;
        uint32_t clockLo = 0;
        uint32_t clockHi = 0;
        if (!iop.readMemory(0x11188u, &flag, sizeof(flag)) ||
            !iop.readMemory(0x11180u, &clockLo, sizeof(clockLo)) ||
            !iop.readMemory(0x11184u, &clockHi, sizeof(clockHi)))
        {
            why = "stamp read failed";
            return false;
        }
        out.cycles = iop.cycles();
        out.clock = (static_cast<uint64_t>(clockHi) << 32u) | clockLo;
        out.skips = iop.testIdleSkipCount();
        out.flag = flag;
        return true;
    }

    // Each source must wake at the same cycle with the idle-skip cache on and off.
    // The probe fails if the cache never seals, so a missing invalidation cannot pass by accident.
    bool testIopIdleSkipInvalidation()
    {
        struct Case
        {
            const char *name;
            uint32_t mode;
            bool start;
            uint32_t command;
            uint64_t eeAfter;
            bool probe;
            bool priority;
        };
        const Case cases[] = {
            {"wake", 1u, true, 1u, 8ull * 2000ull, true, false},
            {"release", 1u, true, 9u, 8ull * 2000ull, true, false},
            {"signal", 2u, true, 2u, 8ull * 2000ull, true, false},
            {"event", 3u, true, 3u, 8ull * 2000ull, true, false},
            {"irq", 1u, true, 8u, 8ull * 2000ull, true, false},
            {"timer", 1u, true, 7u, 8ull * 2000ull, true, false},
            {"start", 0u, false, 4u, 8ull * 2000ull, true, false},
            {"resume", 4u, true, 5u, 8ull * 2000ull, true, false},
            {"priority", 0u, false, 0u, 0ull, true, true},
            {"delay", 5u, true, 0u, 0ull, false, false},
        };
        for (const Case &item : cases)
        {
            TestHost offHost(0x4000u);
            TestHost onHost(0x4000u);
            IdleRun off;
            IdleRun on;
            std::string why;
            if (!runIdleSkipCase(false, item.mode, item.start, item.command, item.eeAfter, item.probe, item.priority,
                                 off, why, offHost))
            {
                std::cerr << "FAIL: idle-skip " << item.name << " cache off: " << why << '\n';
                return false;
            }
            if (!runIdleSkipCase(true, item.mode, item.start, item.command, item.eeAfter, item.probe, item.priority, on,
                                 why, onHost))
            {
                std::cerr << "FAIL: idle-skip " << item.name << " cache on: " << why << '\n';
                return false;
            }
            if (off.flag != 1u || on.flag != 1u || off.clock == 0u || off.clock != on.clock || off.cycles != on.cycles)
            {
                std::cerr << "FAIL: idle-skip " << item.name
                          << " off flag=" << off.flag << " clock=" << off.clock << " cycles=" << off.cycles
                          << " skips=" << off.skips
                          << " on flag=" << on.flag << " clock=" << on.clock << " cycles=" << on.cycles
                          << " skips=" << on.skips << '\n';
                dumpIdleLogs(onHost);
                return false;
            }
        }
        std::cout << "iop idle-skip 10 sources matched cache on/off\n";
        return true;
    }
}

int main()
{
    if (!testIopScheduleOrderAndWakeCycles())
        return 1;
    if (!testIopIdleSkipInvalidation())
        return 1;

    TestHost host(0x20000u);
    IopSubsystem iop(host);

    writeMinimalIrx(host, 0x100u);
    const ModuleLoadResult result = iop.loadModuleBuffer(0x100u);
    const DebugSnapshot snapshot = iop.debugSnapshot();

    if (!expect(result.handled, "Emulator must claim IRX loads")) return 1;
    if (!expect(result.moduleId == 1, "First emulated IRX must get module id 1")) return 1;
    if (!expect(result.startResult == 7, "R3000A branch delay slot result mismatch")) return 1;
    if (!expect(snapshot.emulatorLoadedModules == 1u, "Loaded-module debug count mismatch")) return 1;
    if (!expect(snapshot.emulatorInstructions >= 5u, "Instruction counter did not advance")) return 1;

    int32_t stopResult = -1;
    if (!expect(iop.stopModule(result.moduleId, &stopResult), "Emulated module stop failed")) return 1;
    if (!expect(stopResult == 0, "Emulated module stop result mismatch")) return 1;
    if (!expect(iop.debugSnapshot().emulatorLoadedModules == 0u, "Module was not released")) return 1;

    constexpr uint32_t rpcSid = 0xF00DCAFEu;
    writeRpcServerIrx(host, 0x100u);
    const ModuleLoadResult rpcModule = iop.loadModuleBuffer(0x100u);
    if (!expect(rpcModule.handled && rpcModule.startResult == 0,
                "Synthetic RPC server IRX did not start")) return 1;
    if (!expect(iop.debugSnapshot().emulatorRpcServers == 1u,
                "Registered RPC server count mismatch")) return 1;
    if (!expect(iop.canBindRpc(rpcSid),
                "Emulator did not expose the SID registered by the IRX")) return 1;

    iop.reset();
    if (!expect(!iop.canBindRpc(rpcSid),
                "IOP reset did not remove the registered RPC server")) return 1;

    constexpr uint32_t lotrSoundSid = 0x00012345u;
    writeRpcServerIrx(host, 0x100u, lotrSoundSid);
    const ModuleLoadResult physicalSoundModule = iop.loadModuleBuffer(0x100u);
    if (!expect(physicalSoundModule.handled && physicalSoundModule.startResult == 0,
                "Synthetic physical sound RPC server did not start")) return 1;

    const uint32_t soundHandler[] = {
        0x3C020001u, // lui v0, 1
        0x34420400u, // ori v0, v0, 0x400 (registered server buffer)
        0x03E00008u, // jr ra
        0x00000000u, // nop
    };
    if (!expect(iop.writeMemory(0x00010300u, soundHandler, sizeof(soundHandler)),
                "Could not install the physical sound RPC handler")) return 1;
    constexpr uint32_t soundPayload = 0x1234ABCDu;
    std::memcpy(host.guest.data() + 0x800u, &soundPayload, sizeof(soundPayload));
    RpcRequest soundRequest{};
    soundRequest.sid = lotrSoundSid;
    soundRequest.send = {0x800u, sizeof(soundPayload)};
    soundRequest.receive = {0x900u, sizeof(soundPayload)};
    const uint64_t soundInstructionsBefore = iop.debugSnapshot().emulatorInstructions;
    const RpcResult soundResult = iop.handleRpc(soundRequest);
    uint32_t soundResponse = 0u;
    std::memcpy(&soundResponse, host.guest.data() + 0x900u, sizeof(soundResponse));
    if (!expect(soundResult.handled && soundResponse == soundPayload,
                "Physical sound RPC did not return its own payload")) return 1;
    if (!expect(iop.debugSnapshot().emulatorInstructions > soundInstructionsBefore,
                "Sound RPC did not execute the physical IOP handler")) return 1;

    constexpr uint32_t relocatableRpcSid = 0xA11CE001u;
    writeRelocatableRpcServerIrx(host, 0x100u);
    const ModuleLoadResult relocatableRpcModule = iop.loadModuleBuffer(0x100u);
    if (!expect(relocatableRpcModule.handled && relocatableRpcModule.startResult == 0,
                "Relocatable RPC server IRX did not start")) return 1;
    if (!expect(iop.canBindRpc(relocatableRpcSid),
                "R_MIPS_26 did not relocate a symbol-less IRX import call")) return 1;

    RpcRequest relocatableRequest{};
    relocatableRequest.sid = relocatableRpcSid;
    relocatableRequest.receive = {0x800u, sizeof(uint32_t)};
    const uint64_t rpcInstructionsBefore = iop.debugSnapshot().emulatorInstructions;
    const RpcResult relocatableRpcResult = iop.handleRpc(relocatableRequest);
    const uint64_t rpcInstructions = iop.debugSnapshot().emulatorInstructions - rpcInstructionsBefore;
    if (!expect(relocatableRpcResult.handled,
                "Relocatable IRX RPC handler was not dispatched")) return 1;
    if (!expect(rpcInstructions < 100u,
                "R_MIPS_HI16/LO16 did not relocate the IRX RPC handler")) return 1;
    uint32_t relocatedGpData = 0u;
    std::memcpy(&relocatedGpData, host.guest.data() + relocatableRequest.receive.address,
                sizeof(relocatedGpData));
    if (!expect(relocatedGpData == 0x47505250u,
                "Physical RPC callback did not inherit the registering module's GP")) return 1;

    iop.reset();
    writeExportProviderIrx(host, 0x100u);
    const ModuleLoadResult exportProvider = iop.loadModuleBuffer(0x100u);
    if (!expect(exportProvider.handled && exportProvider.startResult == 0,
                "Synthetic export provider did not register")) return 1;
    writeExportConsumerIrx(host, 0x500u);
    const ModuleLoadResult exportConsumer = iop.loadModuleBuffer(0x500u);
    if (!expect(exportConsumer.handled && exportConsumer.startResult == 0x42,
                "IRX export ordinal was shifted by implicit function slots")) return 1;

    iop.reset();
    constexpr uint32_t eeSource = 0x100u;
    constexpr uint32_t iopBuffer = 0x500u;
    constexpr uint32_t eeDestination = 0x900u;
    const uint32_t transferPayload = 0x53494621u; // "SIF!"
    std::memcpy(host.guest.data() + eeSource, &transferPayload, sizeof(transferPayload));
    if (!expect(iop.writeMemory(iopBuffer, &transferPayload, sizeof(transferPayload)),
                "Could not initialize physical IOP memory")) return 1;
    iop.onSifTransfer({
        SifTransferKind::SetDma,
        SifTransferPhase::AfterCopy,
        eeSource,
        iopBuffer,
        sizeof(transferPayload),
    });
    std::memset(host.guest.data() + iopBuffer, 0, sizeof(transferPayload));
    iop.onSifTransfer({
        SifTransferKind::GetOtherData,
        SifTransferPhase::BeforeCopy,
        iopBuffer,
        eeDestination,
        sizeof(transferPayload),
    });
    uint32_t stagedIopPayload = 0u;
    std::memcpy(&stagedIopPayload, host.guest.data() + iopBuffer, sizeof(stagedIopPayload));
    if (!expect(stagedIopPayload == 0u,
                "SIF notification aliased physical IOP memory into EE RAM")) return 1;
    uint32_t physicalIopPayload = 0u;
    if (!expect(iop.readMemory(iopBuffer, &physicalIopPayload, sizeof(physicalIopPayload)) &&
                    physicalIopPayload == transferPayload,
                "SIF notification corrupted physical IOP memory")) return 1;

    iop.reset();
    constexpr uint32_t sifDmaDestination = 0xC00u;
    writeSifDmaIrx(host, 0x100u);
    const ModuleLoadResult sifDmaModule = iop.loadModuleBuffer(0x100u);
    uint32_t sifDmaPayload = 0u;
    std::memcpy(&sifDmaPayload, host.guest.data() + sifDmaDestination,
                sizeof(sifDmaPayload));
    if (!expect(sifDmaModule.handled && sifDmaModule.startResult == -1,
                "sceSifDmaStat did not report the synchronous transfer as complete")) return 1;
    if (!expect(sifDmaPayload == 0x53494621u,
                "IOP sceSifSetDma did not copy the payload into EE memory")) return 1;

    iop.reset();
    writeVblankSchedulingIrx(host, 0x100u);
    const ModuleLoadResult vblankModule = iop.loadModuleBuffer(0x100u);
    if (!expect(vblankModule.handled && vblankModule.startResult == 0,
                "Synthetic VBlank scheduling IRX did not start")) return 1;
    iop.runEeCycles(8u * 4096u);
    iop.onSifTransfer({
        SifTransferKind::GetOtherData,
        SifTransferPhase::BeforeCopy,
        0x00010400u,
        0x1000u,
        sizeof(uint32_t),
    });
    uint32_t lowPriorityMarker = 0u;
    if (!expect(iop.readMemory(0x00010400u, &lowPriorityMarker, sizeof(lowPriorityMarker)),
                "Could not read the IOP scheduling marker")) return 1;
    if (!expect(lowPriorityMarker == 1u,
                "WaitVblankEnd returned immediately and starved a lower-priority IOP thread")) return 1;

    iop.reset();
    host.logs.clear();
    writeMcmanRegistrationIrx(host, 0x100u);
    const ModuleLoadResult mcmanRegistration = iop.loadModuleBuffer(0x100u);
    if (mcmanRegistration.startResult != 2)
    {
        std::cerr << "MCMAN synthetic start result: " << mcmanRegistration.startResult << '\n';
        for (const auto &message : host.logs)
            std::cerr << message << '\n';
    }
    if (!expect(mcmanRegistration.handled && mcmanRegistration.startResult == 2,
                "MCMAN callback registration or IOMAN AddDrv/DelDrv lifecycle failed")) return 1;
    const bool emittedUnhandledImport = std::any_of(
        host.logs.begin(), host.logs.end(),
        [](const std::string &message) { return message.find("unhandled import") != std::string::npos; });
    if (!expect(!emittedUnhandledImport,
                "MCMAN registration still emitted an unhandled IOP import")) return 1;

    iop.reset();
    host.logs.clear();
    writeCdvdLifecycleIrx(host, 0x100u);
    const ModuleLoadResult cdvdLifecycle = iop.loadModuleBuffer(0x100u);
    if (!expect(cdvdLifecycle.handled && cdvdLifecycle.startResult == 0,
                "CDVD init or callback replacement semantics failed")) return 1;
    const bool emittedUnhandledCdvdImport = std::any_of(
        host.logs.begin(), host.logs.end(),
        [](const std::string &message) { return message.find("unhandled import cdvdman") != std::string::npos; });
    if (!expect(!emittedUnhandledCdvdImport,
                "CDVD lifecycle still emitted an unhandled IOP import")) return 1;

    iop.reset();
    writeCdvdLifecycleIrx(host, 0x100u);
    const ModuleLoadResult cdvdAfterReset = iop.loadModuleBuffer(0x100u);
    if (!expect(cdvdAfterReset.handled && cdvdAfterReset.startResult == 0,
                "IOP reset did not clear the CDVD callback")) return 1;

    iop.reset();
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path virtualCdRoot =
        std::filesystem::temp_directory_path() /
        ("ps2x-iop-cdvd-" + std::to_string(uniqueSuffix));
    std::error_code cdRootError;
    std::filesystem::create_directory(virtualCdRoot, cdRootError);
    if (!expect(!cdRootError, "Could not create the virtual-CD test directory")) return 1;
    host.cdRoot = virtualCdRoot.string();
    writeCdvdPvdReadIrx(host, 0x100u);
    const ModuleLoadResult cdvdPvdRead = iop.loadModuleBuffer(0x100u);
    iop.runEeCycles(4096u);
    iop.onSifTransfer({
        SifTransferKind::GetOtherData,
        SifTransferPhase::BeforeCopy,
        0x000101C0u,
        0x1000u,
        sizeof(uint32_t),
    });
    uint32_t cdvdCallbackReason = 0u;
    if (!expect(iop.readMemory(0x000101C0u, &cdvdCallbackReason, sizeof(cdvdCallbackReason)),
                "Could not read the physical CDVD callback result")) return 1;
    host.cdRoot.clear();
    std::filesystem::remove(virtualCdRoot, cdRootError);
    if (!expect(cdvdPvdRead.handled && cdvdPvdRead.startResult == 0,
                "CDVD did not expose the extracted CD root as an ISO9660 PVD")) return 1;
    if (!expect(cdvdCallbackReason == 1u,
                "CDVD read completion did not invoke the registered guest callback")) return 1;

    iop.reset();
    host.logs.clear();
    writeCdvdSeekIrx(host, 0x100u);
    const ModuleLoadResult cdvdSeek = iop.loadModuleBuffer(0x100u);
    iop.runEeCycles(4096u);
    iop.onSifTransfer({
        SifTransferKind::GetOtherData,
        SifTransferPhase::BeforeCopy,
        0x000101C0u,
        0x1100u,
        sizeof(uint32_t),
    });
    uint32_t cdvdSeekCallbackReason = 0u;
    if (!expect(iop.readMemory(0x000101C0u, &cdvdSeekCallbackReason, sizeof(cdvdSeekCallbackReason)),
                "Could not read the physical CDVD callback result")) return 1;
    if (!expect(cdvdSeek.handled && cdvdSeek.startResult == 1,
                "sceCdSeek did not accept a valid LBN")) return 1;
    if (!expect(cdvdSeekCallbackReason == 4u,
                "sceCdSeek completion did not invoke the registered callback")) return 1;
    const bool emittedUnhandledSeek = std::any_of(
        host.logs.begin(), host.logs.end(),
        [](const std::string &message)
        { return message.find("unhandled import cdvdman:7") != std::string::npos; });
    if (!expect(!emittedUnhandledSeek,
                "sceCdSeek still emitted an unhandled IOP import")) return 1;

    std::cout << "ps2xIOP emulator smoke tests passed\n";
    return 0;
}
