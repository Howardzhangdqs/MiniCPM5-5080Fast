// safetensors_min.hpp — 测试工具专用 minimal safetensors 读取器（mmap）。
// 仅服务 native/tests 下的 make_wpk / reference_forward；运行时库不使用。
// 手写 JSON 字段扫描（查找 "name" → dtype/shape/data_offsets），无第三方依赖。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace stmin {

struct Tensor {
    std::string name;
    uint64_t offset = 0; // 数据区内的字节偏移（相对 data base）
    uint64_t bytes = 0;
    uint32_t rows = 0, cols = 0; // rank<=2（权重全是 1D/2D）
};

class SafeTensors {
public:
    ~SafeTensors() {
        if (map_ != nullptr) munmap(map_, size_);
        if (fd_ >= 0) close(fd_);
    }

    bool open(const char* path) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) return false;
        struct stat st{};
        if (fstat(fd_, &st) != 0) return false;
        size_ = (size_t)st.st_size;
        map_ = (char*)mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            map_ = nullptr;
            return false;
        }
        // header: u64 N + N bytes JSON + data
        if (size_ < 8) return false;
        uint64_t n = 0;
        memcpy(&n, map_, 8);
        if (8 + n + 1 > size_) return false;
        header_ = std::string(map_ + 8, (size_t)n);
        data_base_ = map_ + 8 + n;
        if (!parse_header()) return false;
        return true;
    }

    // 返回 bf16 权重指针 [rows, cols]（row-major，文件原始布局）
    const uint16_t* bf16(const std::string& name) const {
        for (const auto& t : tensors_) {
            if (t.name == name) return (const uint16_t*)(data_base_ + t.offset);
        }
        return nullptr;
    }

    bool has(const std::string& name) const { return bf16(name) != nullptr; }
    uint64_t tensor_bytes(const std::string& name) const {
        for (const auto& t : tensors_)
            if (t.name == name) return t.bytes;
        return 0;
    }
    const std::vector<Tensor>& all() const { return tensors_; }

private:
    static bool find_u64(const std::string& h, size_t pos, const char* key,
                         uint64_t* out, size_t* end) {
        size_t k = h.find(key, pos);
        if (k == std::string::npos) return false;
        k += strlen(key);
        *out = strtoull(h.c_str() + k, nullptr, 10);
        if (end != nullptr) {
            size_t comma = h.find(',', k);
            *end = comma == std::string::npos ? h.size() : comma;
        }
        return true;
    }

    bool parse_header() {
        tensors_.clear();
        size_t pos = 0;
        while (true) {
            size_t nstart = header_.find("\"", pos);
            if (nstart == std::string::npos) break;
            size_t nend = header_.find("\"", nstart + 1);
            if (nend == std::string::npos) break;
            std::string name = header_.substr(nstart + 1, nend - nstart - 1);
            size_t obj_end = header_.find('}', nend);
            if (obj_end == std::string::npos) break;
            std::string obj = header_.substr(nend, obj_end - nend);
            if (obj.find("\"BF16\"") == std::string::npos &&
                obj.find("\"F32\"") == std::string::npos) {
                pos = obj_end + 1;
                continue;
            }
            Tensor t;
            t.name = name;
            // shape: [a, b] or [a]
            size_t s = obj.find("\"shape\":[");
            if (s == std::string::npos) {
                pos = obj_end + 1;
                continue;
            }
            uint64_t a = strtoull(obj.c_str() + s + 9, nullptr, 10);
            uint64_t b = 1;
            size_t comma = obj.find(',', s + 9);
            if (comma != std::string::npos && comma < obj_end)
                b = strtoull(obj.c_str() + comma + 1, nullptr, 10);
            t.rows = (uint32_t)a;
            t.cols = (uint32_t)b;
            uint64_t off0 = 0, off1 = 0;
            if (!find_u64(obj, 0, "\"data_offsets\":[", &off0, nullptr)) {
                pos = obj_end + 1;
                continue;
            }
            // 第二个偏移：从第一个偏移数字之后找下一个数字
            size_t k = obj.find("\"data_offsets\":[") + strlen("\"data_offsets\":[");
            size_t c2 = obj.find(',', k);
            off1 = strtoull(obj.c_str() + c2 + 1, nullptr, 10);
            t.offset = off0;
            t.bytes = off1 - off0;
            tensors_.push_back(t);
            pos = obj_end + 1;
        }
        return !tensors_.empty();
    }

    int fd_ = -1;
    char* map_ = nullptr;
    size_t size_ = 0;
    std::string header_;
    const char* data_base_ = nullptr;
    std::vector<Tensor> tensors_;
};

// bf16 ↔ fp32（round-to-nearest-even），CPU 侧。
inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1;
    uint32_t rounded = (u + 0x7fffu + lsb) >> 16;
    return (uint16_t)rounded;
}

} // namespace stmin
