// export.h — 只导出公共 ABI 符号（配合 CMake 的 hidden visibility preset）。
#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define MC_API __attribute__((visibility("default")))
#else
#define MC_API
#endif
