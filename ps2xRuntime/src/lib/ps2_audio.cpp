#include "runtime/ps2_audio.h"
#include "runtime/ps2_memory.h"
#include "ps2_host_backend.h"
#include "ps2x/iop/iop_host.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <vector>

namespace
{
    std::vector<uint8_t> buildWavFromPcm(const int16_t *pcm, size_t sampleCount, uint32_t sampleRate)
    {
        const uint32_t dataSize = static_cast<uint32_t>(sampleCount * 2);
        const uint32_t fileSize = 36 + dataSize;
        std::vector<uint8_t> wav(8 + fileSize);

        uint8_t *p = wav.data();
        p[0] = 'R';
        p[1] = 'I';
        p[2] = 'F';
        p[3] = 'F';
        p[4] = static_cast<uint8_t>(fileSize);
        p[5] = static_cast<uint8_t>(fileSize >> 8);
        p[6] = static_cast<uint8_t>(fileSize >> 16);
        p[7] = static_cast<uint8_t>(fileSize >> 24);
        p[8] = 'W';
        p[9] = 'A';
        p[10] = 'V';
        p[11] = 'E';
        p[12] = 'f';
        p[13] = 'm';
        p[14] = 't';
        p[15] = ' ';
        p[16] = 16;
        p[17] = 0;
        p[18] = 0;
        p[19] = 0;
        p[20] = 1;
        p[21] = 0;
        p[22] = 1;
        p[23] = 0;
        p[24] = static_cast<uint8_t>(sampleRate);
        p[25] = static_cast<uint8_t>(sampleRate >> 8);
        p[26] = static_cast<uint8_t>(sampleRate >> 16);
        p[27] = static_cast<uint8_t>(sampleRate >> 24);
        const uint32_t byteRate = sampleRate * 2;
        p[28] = static_cast<uint8_t>(byteRate);
        p[29] = static_cast<uint8_t>(byteRate >> 8);
        p[30] = static_cast<uint8_t>(byteRate >> 16);
        p[31] = static_cast<uint8_t>(byteRate >> 24);
        p[32] = 2;
        p[33] = 0;
        p[34] = 16;
        p[35] = 0;
        p[36] = 'd';
        p[37] = 'a';
        p[38] = 't';
        p[39] = 'a';
        p[40] = static_cast<uint8_t>(dataSize);
        p[41] = static_cast<uint8_t>(dataSize >> 8);
        p[42] = static_cast<uint8_t>(dataSize >> 16);
        p[43] = static_cast<uint8_t>(dataSize >> 24);
        std::memcpy(p + 44, pcm, dataSize);
        return wav;
    }
}

namespace ps2_vag
{
    bool decode(const uint8_t *data, uint32_t sizeBytes,
                std::vector<int16_t> &outPcm, uint32_t &outSampleRate);
}

struct PS2AudioBackend::Impl
{
    struct TrackedSound
    {
        Sound snd;
        uint32_t sampleKey;
    };
    std::vector<TrackedSound> activeSounds;
};

PS2AudioBackend::PS2AudioBackend() : m_impl(std::make_unique<Impl>())
{
}

PS2AudioBackend::~PS2AudioBackend()
{
    if (m_impl)
        stopAll();
}

void PS2AudioBackend::onVagTransfer(const uint8_t *rdram, uint32_t srcAddr, uint32_t sizeBytes)
{
    if (!rdram || sizeBytes < 48)
        return;

    const uint32_t physAddr = srcAddr & PS2_RAM_MASK;
    if (physAddr + sizeBytes > PS2_RAM_SIZE)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(rdram + physAddr, sizeBytes, pcm, sampleRate))
        return;

    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = std::move(sample);
    m_mostRecentSampleKey = physAddr;
}

void PS2AudioBackend::onVagTransferFromBuffer(const uint8_t *data, uint32_t sizeBytes, uint32_t keyAddr)
{
    if (!data || sizeBytes < 48)
        return;

    std::vector<int16_t> pcm;
    uint32_t sampleRate = 44100;
    if (!ps2_vag::decode(data, sizeBytes, pcm, sampleRate))
        return;

    const uint32_t physAddr = keyAddr & PS2_RAM_MASK;
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample sample;
    sample.pcm = std::move(pcm);
    sample.sampleRate = sampleRate;
    m_sampleBank[physAddr] = sample;
    m_mostRecentSampleKey = physAddr;
    m_loadOrderSamples.push_back(std::move(sample));
    m_loadOrderSampleKeys.push_back(physAddr);
    constexpr size_t kMaxLoadOrderSamples = 32;
    if (m_loadOrderSamples.size() > kMaxLoadOrderSamples)
    {
        m_loadOrderSamples.erase(m_loadOrderSamples.begin());
        m_loadOrderSampleKeys.erase(m_loadOrderSampleKeys.begin());
    }
}

namespace
{
    constexpr uint32_t LIBSD_CMD_SET_VOICE = 0x8010u;
}

void PS2AudioBackend::onSoundCommand(uint32_t sid, uint32_t rpcNum,
                                     const uint8_t *sendBuf, uint32_t sendSize,
                                     uint8_t *recvBuf, uint32_t recvSize)
{
    if (sid != 0x80000701u)
        return;

    if ((rpcNum == LIBSD_CMD_SET_VOICE || (rpcNum & 0xFF00u) == 0x8100u) &&
        sendBuf && sendSize >= 20)
    {
        uint32_t sampleAddr = 0;
        uint32_t voiceIndex = 0xFFFFFFFFu;
        for (int vo = 4; vo >= 0 && voiceIndex == 0xFFFFFFFFu; vo -= 4)
        {
            if (vo < static_cast<int>(sendSize))
            {
                uint32_t v = 0;
                std::memcpy(&v, sendBuf + vo, sizeof(v));
                if (v < 24u)
                    voiceIndex = v;
            }
        }

        constexpr uint32_t kMinPlausibleAddr = 0x1000u;
        for (int off = 12; off <= 24 && sampleAddr == 0; off += 4)
        {
            if (sendSize >= static_cast<uint32_t>(off + 4))
            {
                uint32_t cand = 0;
                std::memcpy(&cand, sendBuf + off, sizeof(cand));
                if (cand >= kMinPlausibleAddr && (cand <= PS2_RAM_MASK || (cand & ~PS2_RAM_MASK) == 0))
                    sampleAddr = cand;
            }
        }
        if (sampleAddr == 0)
            sampleAddr = m_mostRecentSampleKey;

        float pitch = 1.0f;
        if (sendSize >= 12)
        {
            uint16_t pitchHalf = 0;
            std::memcpy(&pitchHalf, sendBuf + 8, sizeof(pitchHalf));
            if (pitchHalf != 0)
                pitch = 4096.0f / static_cast<float>(pitchHalf);
        }
        play(sampleAddr, pitch, 1.0f, voiceIndex);
    }
}

void PS2AudioBackend::play(uint32_t sampleAddr, float pitch, float volume, uint32_t voiceIndex)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    DecodedSample *sampleToPlay = nullptr;
    uint32_t sampleKey = 0;

    auto it = m_sampleBank.find(sampleAddr & PS2_RAM_MASK);
    if (it != m_sampleBank.end())
    {
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    else if (voiceIndex != 0xFFFFFFFFu &&
             voiceIndex < m_loadOrderSamples.size() &&
             voiceIndex < m_loadOrderSampleKeys.size())
    {
        sampleToPlay = &m_loadOrderSamples[voiceIndex];
        sampleKey = m_loadOrderSampleKeys[voiceIndex];
    }
    else
    {
        it = m_sampleBank.find(m_mostRecentSampleKey);
        if (it == m_sampleBank.end())
            return;
        sampleToPlay = &it->second;
        sampleKey = it->first;
    }
    if (!sampleToPlay || sampleToPlay->pcm.empty())
        return;

    const bool isBgm = (sampleToPlay->pcm.size() > static_cast<size_t>(sampleToPlay->sampleRate * 5));
    playDecodedSample(sampleKey, *sampleToPlay, pitch, volume, isBgm);
}

void PS2AudioBackend::pruneFinishedSounds()
{
#if defined(PLATFORM_VITA)
    return;
#else
    auto &sounds = m_impl->activeSounds;
    auto it = sounds.begin();
    while (it != sounds.end())
    {
        if (!IsSoundPlaying(it->snd))
        {
            UnloadSound(it->snd);
            it = sounds.erase(it);
        }
        else
        {
            ++it;
        }
    }
#endif
}

void PS2AudioBackend::playDecodedSample(uint32_t sampleKey, DecodedSample &sample, float pitch, float volume,
                                        bool isBgm)
{
#if defined(PLATFORM_VITA)
    (void)sampleKey;
    (void)sample;
    (void)pitch;
    (void)volume;
    (void)isBgm;
    return;
#else
    if (!m_audioReady || sample.pcm.empty())
        return;

    pruneFinishedSounds();

    for (const auto &t : m_impl->activeSounds)
    {
        if (t.sampleKey == sampleKey && IsSoundPlaying(t.snd))
            return;
    }

    auto &sounds = m_impl->activeSounds;
    if (isBgm)
    {
        for (auto it = sounds.begin(); it != sounds.end();)
        {
            if (IsSoundPlaying(it->snd))
            {
                StopSound(it->snd);
                UnloadSound(it->snd);
                it = sounds.erase(it);
            }
            else
                ++it;
        }
    }

    constexpr int kMaxConcurrentSounds = 4;
    while (static_cast<int>(sounds.size()) >= kMaxConcurrentSounds)
    {
        StopSound(sounds.front().snd);
        UnloadSound(sounds.front().snd);
        sounds.erase(sounds.begin());
    }

    std::vector<uint8_t> wav = buildWavFromPcm(sample.pcm.data(), sample.pcm.size(), sample.sampleRate);
    Wave wave = LoadWaveFromMemory(".wav", wav.data(), static_cast<int>(wav.size()));
    if (wave.frameCount <= 0)
        return;
    Sound snd = LoadSoundFromWave(wave);
    UnloadWave(wave);
    SetSoundPitch(snd, pitch);
    SetSoundVolume(snd, volume);
    m_impl->activeSounds.push_back({snd, sampleKey});
    PlaySound(snd);
#endif
}

void PS2AudioBackend::stop(uint32_t voiceId)
{
    (void)voiceId;
}

void PS2AudioBackend::stopAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(PLATFORM_VITA)
    return;
#else
    for (auto &t : m_impl->activeSounds)
    {
        StopSound(t.snd);
        UnloadSound(t.snd);
    }
    m_impl->activeSounds.clear();
#endif
}

std::mutex g_movieMutex;
std::vector<int16_t> g_moviePcm;

namespace
{
#if !defined(PLATFORM_VITA)
    // Playback queue between the game thread (push) and raylib's audio thread (pull, stream callback).
    // UpdateAudioStream pads every write up to the stream's sub-buffer (device rate / 30 = 1600 frames at 48 kHz)
    // with silence, so the old 1024-frame writes played ~36 % silence in 12 ms gaps (choppy sound, unintelligible
    // movie voices) and fell behind until the queue dropped whole seconds. The device now pulls what it needs.
    // Never destroyed: the audio thread can still call in during static destruction.
    class DeviceQueue
    {
    public:
        static constexpr size_t kTargetFrames = 2048u; // ~43 ms buffered ahead of the device
        static constexpr size_t kMaxFrames = 9600u;    // 200 ms: after a stall, cut back to the target

        void push(const int16_t *pcm, size_t frames)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_read != 0u && m_read * 2u >= m_samples.size())
            {
                m_samples.erase(m_samples.begin(), m_samples.begin() + static_cast<std::ptrdiff_t>(m_read));
                m_read = 0u;
            }
            m_samples.insert(m_samples.end(), pcm, pcm + frames * 2u);
            const size_t level = levelLocked();
            if (level > kMaxFrames)
            {
                m_read += (level - kTargetFrames) * 2u;
                m_dropped += level - kTargetFrames;
                m_phase = 0.0;
            }
        }

        // Fills exactly `frames` frames. Around the target level the queue plays up to 5 % faster or slower (linear
        // interpolation), so a guest clock running ahead doesn't build latency and one running behind doesn't gap.
        // An empty queue plays silence and waits for the target level again instead of crackling at the edge.
        void pull(int16_t *out, size_t frames)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            const size_t level = levelLocked();
            if (!m_primed && level >= kTargetFrames)
                m_primed = true;
            size_t written = 0u;
            double rate = 1.0;
            if (m_primed)
            {
                const double over = (static_cast<double>(level) - static_cast<double>(kTargetFrames)) /
                                    static_cast<double>(kTargetFrames);
                rate = 1.0 + std::clamp(over * 0.1, -0.05, 0.05);
                const int16_t *in = m_samples.data() + m_read;
                double position = m_phase;
                for (; written < frames; ++written, position += rate)
                {
                    const size_t index = static_cast<size_t>(position);
                    if (index + 1u >= level)
                        break;
                    const double fraction = position - static_cast<double>(index);
                    for (size_t side = 0u; side < 2u; ++side)
                    {
                        const double a = in[index * 2u + side], b = in[(index + 1u) * 2u + side];
                        out[written * 2u + side] = static_cast<int16_t>(std::lround(a + (b - a) * fraction));
                    }
                }
                const size_t consumed = std::min(level, static_cast<size_t>(position));
                m_phase = position - static_cast<double>(consumed);
                m_read += consumed * 2u;
                if (written < frames)
                    m_primed = false;
            }
            if (written < frames)
            {
                std::memset(out + written * 2u, 0, (frames - written) * 4u);
                m_missing += frames - written;
            }
            logStats(frames, level, rate);
        }

    private:
        size_t levelLocked() const { return (m_samples.size() - m_read) / 2u; }

        // TEKKEN4_AUDIO_STATS=1: every ~5 s of device time, frames the device asked for, frames it got as silence
        // (underrun), frames cut after a stall, the queue level range and the playback rate range.
        void logStats(size_t frames, size_t level, double rate)
        {
            static const bool enabled = std::getenv("TEKKEN4_AUDIO_STATS") != nullptr;
            if (!enabled)
                return;
            m_statPulled += frames;
            m_statMinLevel = std::min(m_statMinLevel, level);
            m_statMaxLevel = std::max(m_statMaxLevel, level);
            m_statMinRate = std::min(m_statMinRate, rate);
            m_statMaxRate = std::max(m_statMaxRate, rate);
            if (m_statPulled < 240000u)
                return;
            std::fprintf(stderr, "[audio-device] pulled=%zu missing=%zu dropped=%zu level=%zu-%zu rate=%.3f-%.3f\n",
                         m_statPulled, m_missing, m_dropped, m_statMinLevel, m_statMaxLevel, m_statMinRate,
                         m_statMaxRate);
            m_statPulled = m_missing = m_dropped = m_statMaxLevel = 0u;
            m_statMinLevel = SIZE_MAX;
            m_statMinRate = m_statMaxRate = 1.0;
        }

        std::mutex m_mutex;
        std::vector<int16_t> m_samples;
        size_t m_read = 0u; // samples already played from the front of m_samples
        double m_phase = 0.0;
        bool m_primed = false;
        size_t m_missing = 0u, m_dropped = 0u;
        size_t m_statPulled = 0u, m_statMinLevel = SIZE_MAX, m_statMaxLevel = 0u;
        double m_statMinRate = 1.0, m_statMaxRate = 1.0;
    };

    DeviceQueue &deviceQueue()
    {
        static DeviceQueue *queue = new DeviceQueue();
        return *queue;
    }

    void onDeviceRequest(void *buffer, unsigned int frames)
    {
        deviceQueue().pull(static_cast<int16_t *>(buffer), frames);
    }
#endif

    // 48 kHz s16 stereo from the IOP mixer (ps2x::iop::setMixedAudioFn). The WAV is the check Claude can read;
    // the raylib stream plays it through DeviceQueue.
    class MixedOutput
    {
    public:
        void push(const int16_t *pcm, size_t frames)
        {
            if (!pcm || frames == 0u)
                return;
            std::lock_guard<std::mutex> lock(m_mutex);
            openWav();
            if (m_wav)
            {
                m_wav.write(reinterpret_cast<const char *>(pcm), static_cast<std::streamsize>(frames * 4u));
                m_frames += static_cast<uint64_t>(frames);
                m_sinceFlush += static_cast<uint32_t>(frames);
                if (m_sinceFlush >= 48000u)
                {
                    patchHeader();
                    m_wav.flush();
                    m_sinceFlush = 0u;
                }
            }
#if !defined(PLATFORM_VITA)
            pumpDevice(pcm, frames);
#endif
        }

        ~MixedOutput()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_wav)
            {
                patchHeader();
                m_wav.flush();
            }
        }

    private:
        static void writeLe32(uint8_t *p, uint32_t value)
        {
            p[0] = static_cast<uint8_t>(value);
            p[1] = static_cast<uint8_t>(value >> 8);
            p[2] = static_cast<uint8_t>(value >> 16);
            p[3] = static_cast<uint8_t>(value >> 24);
        }

        void openWav()
        {
            if (m_wavTried)
                return;
            m_wavTried = true;
            const char *path = std::getenv("TEKKEN4_SOUND_WAV");
            if (!path || path[0] == '\0')
                return;
            std::error_code ec;
            const std::filesystem::path file(path);
            if (file.has_parent_path())
                std::filesystem::create_directories(file.parent_path(), ec);
            m_wav.open(file, std::ios::binary | std::ios::trunc);
            if (!m_wav)
            {
                std::fprintf(stderr, "[tekken4] sound wav open failed %s\n", path);
                return;
            }
            uint8_t header[44] = {};
            m_wav.write(reinterpret_cast<const char *>(header), 44);
            patchHeader();
            m_wav.flush();
            std::fprintf(stderr, "[tekken4] sound wav %s\n", path);
        }

        void patchHeader()
        {
            if (!m_wav)
                return;
            const uint64_t dataBytes64 = m_frames * 4u;
            const uint32_t dataBytes = dataBytes64 > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<uint32_t>(dataBytes64);
            const uint32_t fileSize = 36u + dataBytes;
            uint8_t h[44] = {};
            h[0] = 'R'; h[1] = 'I'; h[2] = 'F'; h[3] = 'F';
            writeLe32(h + 4, fileSize);
            h[8] = 'W'; h[9] = 'A'; h[10] = 'V'; h[11] = 'E';
            h[12] = 'f'; h[13] = 'm'; h[14] = 't'; h[15] = ' ';
            writeLe32(h + 16, 16u);
            h[20] = 1; h[22] = 2;
            writeLe32(h + 24, 48000u);
            writeLe32(h + 28, 48000u * 4u);
            h[32] = 4; h[34] = 16;
            h[36] = 'd'; h[37] = 'a'; h[38] = 't'; h[39] = 'a';
            writeLe32(h + 40, dataBytes);
            const auto end = m_wav.tellp();
            m_wav.seekp(0);
            m_wav.write(reinterpret_cast<const char *>(h), 44);
            m_wav.seekp(end);
        }

#if !defined(PLATFORM_VITA)
        void pumpDevice(const int16_t *pcm, size_t frames)
        {
            if (!m_deviceGaveUp && !m_streamOn && IsAudioDeviceReady())
            {
                m_stream = LoadAudioStream(48000, 16, 2);
                if (m_stream.sampleRate == 48000)
                {
                    SetAudioStreamCallback(m_stream, &onDeviceRequest);
                    PlayAudioStream(m_stream);
                    m_streamOn = true;
                    std::fprintf(stderr, "[tekken4] sound device 48000 Hz stereo (pulled by the audio thread)\n");
                }
                else
                    m_deviceGaveUp = true;
            }
            if (m_streamOn)
                deviceQueue().push(pcm, frames);
        }

        AudioStream m_stream{};
        bool m_streamOn = false;
        bool m_deviceGaveUp = false;
#endif
        std::mutex m_mutex;
        std::ofstream m_wav;
        bool m_wavTried = false;
        uint64_t m_frames = 0;
        uint32_t m_sinceFlush = 0;
    };

    MixedOutput &mixedOutput()
    {
        static MixedOutput output;
        return output;
    }

    void onMixedAudio(const int16_t *pcm, size_t frames)
    {
        std::vector<int16_t> mixed;
        const int16_t *played = pcm;
        {
            std::lock_guard<std::mutex> lock(g_movieMutex);
            if (!g_moviePcm.empty() && pcm && frames != 0u)
            {
                mixed.assign(pcm, pcm + frames * 2u);
                const size_t have = g_moviePcm.size() / 2u;
                const size_t n = frames < have ? frames : have;
                for (size_t i = 0; i < n * 2u; ++i)
                {
                    const int32_t sum = static_cast<int32_t>(mixed[i]) + static_cast<int32_t>(g_moviePcm[i]);
                    mixed[i] = static_cast<int16_t>(sum > 32767 ? 32767 : sum < -32768 ? -32768 : sum);
                }
                g_moviePcm.erase(g_moviePcm.begin(), g_moviePcm.begin() + static_cast<std::ptrdiff_t>(n * 2u));
                played = mixed.data();
            }
        }
        mixedOutput().push(played, frames);
    }

    struct MixedAudioRegistration
    {
        MixedAudioRegistration()
        {
            ps2x::iop::setMixedAudioFn(&onMixedAudio);
        }
        ~MixedAudioRegistration()
        {
            ps2x::iop::setMixedAudioFn(nullptr);
        }
    };

    MixedAudioRegistration g_mixedAudioRegistration;
}

void ps2PushMoviePcm(const int16_t *stereoInterleaved, size_t frames)
{
    if (!stereoInterleaved || frames == 0u)
        return;
    std::lock_guard<std::mutex> lock(g_movieMutex);
    constexpr size_t kMaxFrames = 48000u * 8u;
    g_moviePcm.insert(g_moviePcm.end(), stereoInterleaved, stereoInterleaved + frames * 2u);
    if (g_moviePcm.size() / 2u > kMaxFrames)
        g_moviePcm.erase(g_moviePcm.begin(), g_moviePcm.end() - static_cast<std::ptrdiff_t>(kMaxFrames * 2u));
}

void ps2ClearMoviePcm()
{
    std::lock_guard<std::mutex> lock(g_movieMutex);
    g_moviePcm.clear();
}
