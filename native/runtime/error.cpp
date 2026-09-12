// error.cpp — thread_local 错误缓冲实现 + mc_last_error ABI 导出。
#include "runtime/error.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "runtime/export.h"

namespace {

constexpr size_t kErrorBufBytes = 512;

// 函数级 thread_local：避免跨 TU 的 TLS 构造/析构顺序问题。
char* error_buffer() {
    static thread_local char buf[kErrorBufBytes] = "no error";
    return buf;
}

} // namespace

namespace mc {

void set_error(const char* fmt, ...) {
    char* buf = error_buffer();
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, kErrorBufBytes, fmt, ap);
    va_end(ap);
    buf[kErrorBufBytes - 1] = '\0'; // vsnprintf 保证 NUL 结尾，这里双保险
}

void clear_error() {
    strcpy(error_buffer(), "no error");
}

} // namespace mc

extern "C" MC_API const char* mc_last_error(void) {
    return error_buffer();
}
