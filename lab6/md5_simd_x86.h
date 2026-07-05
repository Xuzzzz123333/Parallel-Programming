#ifndef MD5_SIMD_X86_H
#define MD5_SIMD_X86_H

#include "md5_portable.h"
#include <string>

// 4-way x86 SIMD MD5 for short messages.  It computes four independent MD5
// messages in parallel using 128-bit SSE2 lanes.  Messages that do not fit in
// one MD5 block fall back to the portable scalar MD5 implementation.
void MD5HashSIMD4X86(
    const std::string &s0,
    const std::string &s1,
    const std::string &s2,
    const std::string &s3,
    bit32 states[4][4]
);

#endif
