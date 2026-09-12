// error.h — 线程局部错误诊断（ABI 契约：mc_last_error 是 thread-local 诊断，
// 不是错误协议；错误协议是返回的 mc_status_t）。
//
// 约定：所有 ABI 入口在返回非 MC_OK 之前调用 mc::set_error() 写入英文
// 错误信息（便于日志检索）；成功路径入口调用 mc::clear_error() 清为
// "no error"。异常绝不允许跨越 ABI 边界（入口处 catch(...) 转 MC_E_INTERNAL）。
#pragma once

#include "minicpm_runtime.h"

namespace mc {

// printf 风格写入线程局部错误缓冲（上限 511 字节，超出安全截断）。
void set_error(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// 清为 "no error"。
void clear_error();

} // namespace mc
