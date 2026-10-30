#include "iop_ioman.h"

#include "../core/iop_cpu.h"
#include "../core/iop_memory.h"
#include "../services/iop_rpc.h"
#include "ps2x/iop/iop_host.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <sstream>
#include <vector>

namespace ps2x::iop::detail
{
    namespace
    {
        constexpr uint64_t kDataBinLbn = 1236u;
        constexpr uint64_t kDataBinSize = 2694412288u;
        constexpr uint32_t kMaxRead = 2u * 1024u * 1024u;

        std::string canonicalPath(std::string path)
        {
            for (char &character : path)
            {
                if (character == '\\')
                    character = '/';
                else
                    character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            }
            return path;
        }
    }

    IopIoman::IopIoman(IopHost &host, IopMemory &memory) noexcept
        : m_host(host), m_memory(memory)
    {
    }

    void IopIoman::reset()
    {
        if (m_image != 0u)
            m_host.closeHostFile(m_image);
        m_image = 0u;
        m_files.clear();
        m_devices.clear();
    }

    bool IopIoman::soundRoute() const
    {
        return !m_host.hostPath(HostPathKind::CdImage).empty();
    }

    bool IopIoman::ensureImage()
    {
        if (m_image != 0u)
            return true;
        const std::string path = m_host.hostPath(HostPathKind::CdImage);
        if (path.empty())
            return false;
        m_image = m_host.openHostFile(path);
        return m_image != 0u;
    }

    bool IopIoman::cdromOpen(IopCpuState &cpu)
    {
        const std::string name = m_memory.readString(cpu.gpr[4], 1024u);
        const std::string path = canonicalPath(name);
        if (path.rfind("CDROM", 0) != 0)
            return false;

        auto fail = [&](int32_t code)
        {
            cpu.gpr[2] = static_cast<uint32_t>(code);
            return true;
        };
        if (!ensureImage())
            return fail(-1);

        uint64_t imageSize = 0u;
        if (!m_host.hostFileSize(m_image, imageSize))
            return fail(-1);

        uint64_t base = 0u;
        uint64_t size = imageSize;
        const size_t dataBin = path.find("TEKKEN/DATA.BIN");
        if (dataBin != std::string::npos)
        {
            const size_t after = dataBin + std::char_traits<char>::length("TEKKEN/DATA.BIN");
            if (after < path.size() && path[after] != ';')
                return fail(-1);
            base = kDataBinLbn * 2048u;
            if (base >= imageSize)
                return fail(-1);
            size = std::min(imageSize - base, kDataBinSize);
        }
        else if (path.find("/TEKKEN/") != std::string::npos)
            return fail(-1);

        if (m_files.empty())
            m_files.emplace_back();
        size_t fd = 0u;
        for (size_t i = 1u; i < m_files.size(); ++i)
        {
            if (!m_files[i].open)
            {
                fd = i;
                break;
            }
        }
        if (fd == 0u)
        {
            if (m_files.size() >= 32u)
                return fail(-1);
            fd = m_files.size();
            m_files.emplace_back();
        }
        m_files[fd] = File{true, 0u, base, size};
        std::ostringstream out;
        out << "[ioman] open " << name << " fd=" << fd << " base=" << base << " size=" << size;
        m_host.log(LogLevel::Info, out.str());
        cpu.gpr[2] = static_cast<uint32_t>(fd);
        return true;
    }

    bool IopIoman::cdromClose(IopCpuState &cpu)
    {
        const uint32_t fd = cpu.gpr[4];
        if (fd == 0u || fd >= m_files.size() || !m_files[fd].open)
            return false;
        m_files[fd].open = false;
        cpu.gpr[2] = 0u;
        return true;
    }

    bool IopIoman::cdromRead(IopCpuState &cpu)
    {
        const uint32_t fd = cpu.gpr[4];
        if (fd == 0u || fd >= m_files.size() || !m_files[fd].open)
            return false;
        File &file = m_files[fd];
        uint32_t wanted = cpu.gpr[6];
        if (wanted > kMaxRead)
            wanted = kMaxRead;
        if (file.position >= file.size || wanted == 0u)
        {
            cpu.gpr[2] = 0u;
            return true;
        }
        const uint64_t remain = file.size - file.position;
        if (wanted > remain)
            wanted = static_cast<uint32_t>(remain);
        std::vector<uint8_t> bytes(wanted);
        size_t got = 0u;
        if (!m_host.readHostFile(m_image, file.base + file.position, bytes.data(), wanted, got) || got == 0u)
        {
            cpu.gpr[2] = static_cast<uint32_t>(-1);
            return true;
        }
        {
            static int reads = 0;
            if (reads < 4)
            {
                ++reads;
                std::ostringstream out;
                out << "[ioman] read fd=" << fd << " n=" << got << " pos=" << file.position;
                m_host.log(LogLevel::Info, out.str());
            }
        }
        if (!m_memory.writeRam(cpu.gpr[5], bytes.data(), got))
        {
            cpu.gpr[2] = static_cast<uint32_t>(-1);
            return true;
        }
        file.position += got;
        cpu.gpr[2] = static_cast<uint32_t>(got);
        return true;
    }

    bool IopIoman::cdromSeek(IopCpuState &cpu)
    {
        const uint32_t fd = cpu.gpr[4];
        if (fd == 0u || fd >= m_files.size() || !m_files[fd].open)
            return false;
        File &file = m_files[fd];
        const int32_t offset = static_cast<int32_t>(cpu.gpr[5]);
        const int32_t whence = static_cast<int32_t>(cpu.gpr[6]);
        int64_t next = static_cast<int64_t>(file.position);
        if (whence == 0)
            next = offset;
        else if (whence == 1)
            next += offset;
        else if (whence == 2)
            next = static_cast<int64_t>(file.size) + offset;
        else
        {
            cpu.gpr[2] = static_cast<uint32_t>(-1);
            return true;
        }
        if (next < 0)
            next = 0;
        if (static_cast<uint64_t>(next) > file.size)
            next = static_cast<int64_t>(file.size);
        file.position = static_cast<uint64_t>(next);
        cpu.gpr[2] = static_cast<uint32_t>(file.position);
        return true;
    }

    bool IopIoman::dispatchImport(uint16_t ordinal, IopCpuState &cpu, IopGuestExecutor &executor)
    {
        constexpr size_t kMaxDevices = 16u;
        const uint32_t a0 = cpu.gpr[4];
        const auto setV0 = [&](uint32_t value)
        {
            cpu.gpr[2] = value;
        };

        switch (ordinal)
        {
        case 4: // open
            return soundRoute() && cdromOpen(cpu);
        case 5: // close
            return soundRoute() && cdromClose(cpu);
        case 6: // read
            return soundRoute() && cdromRead(cpu);
        case 8: // lseek
            return soundRoute() && cdromSeek(cpu);
        case 20: // AddDrv
        {
            if (a0 == 0u || m_devices.size() >= kMaxDevices)
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            const uint32_t nameAddress = m_memory.read32(a0);
            const uint32_t operations = m_memory.read32(a0 + 16u);
            const std::string name = m_memory.readString(nameAddress, 64u);
            if (nameAddress == 0u || operations == 0u || name.empty())
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            m_devices.push_back({a0, cpu.gpr[28], name});
            const uint32_t init = m_memory.read32(operations);
            if (init != 0u)
            {
                const int32_t result = static_cast<int32_t>(
                    executor.executeGuestFunction(init, a0, 0u, 0u, 0u, cpu.gpr[28]));
                if (result < 0)
                {
                    m_devices.pop_back();
                    setV0(0xFFFFFFFFu);
                    return true;
                }
            }

            setV0(0u);
            return true;
        }
        case 21: // DelDrv
        {
            const std::string name = m_memory.readString(a0, 64u);
            const auto device = std::find_if(
                m_devices.begin(), m_devices.end(),
                [&](const Device &candidate)
                { return candidate.name == name; });
            if (device == m_devices.end())
            {
                setV0(0xFFFFFFFFu);
                return true;
            }

            const uint32_t operations = m_memory.read32(device->address + 16u);
            const uint32_t deinit = operations != 0u ? m_memory.read32(operations + 4u) : 0u;
            if (deinit != 0u)
                (void)executor.executeGuestFunction(deinit, device->address, 0u, 0u, 0u, device->gp);
            m_devices.erase(device);
            setV0(0u);
            return true;
        }
        default:
            return false;
        }
    }
}
