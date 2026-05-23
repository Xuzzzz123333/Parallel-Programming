#include <iostream>
#include <string>
#include <cstring>
#include <arm_neon.h>

using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21

// ===== 串行版宏 =====
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

// ===== NEON 4路并行版宏 =====
#define F_NEON(x, y, z) vorrq_u32(vandq_u32((x), (y)), vandq_u32(vmvnq_u32(x), (z)))
#define G_NEON(x, y, z) vorrq_u32(vandq_u32((x), (z)), vandq_u32((y), vmvnq_u32(z)))
#define H_NEON(x, y, z) veorq_u32(veorq_u32((x), (y)), (z))
#define I_NEON(x, y, z) veorq_u32((y), vorrq_u32((x), vmvnq_u32(z)))

#define ROTATELEFT_NEON(num, n) \
  vorrq_u32(vshlq_n_u32((num), (n)), vshrq_n_u32((num), 32 - (n)))

#define FF_VEC(a, b, c, d, x, s, ac) { \
  (a) = vaddq_u32((a), vaddq_u32(F_NEON((b), (c), (d)), vaddq_u32((x), vdupq_n_u32(ac)))); \
  (a) = ROTATELEFT_NEON((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define GG_VEC(a, b, c, d, x, s, ac) { \
  (a) = vaddq_u32((a), vaddq_u32(G_NEON((b), (c), (d)), vaddq_u32((x), vdupq_n_u32(ac)))); \
  (a) = ROTATELEFT_NEON((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define HH_VEC(a, b, c, d, x, s, ac) { \
  (a) = vaddq_u32((a), vaddq_u32(H_NEON((b), (c), (d)), vaddq_u32((x), vdupq_n_u32(ac)))); \
  (a) = ROTATELEFT_NEON((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

#define II_VEC(a, b, c, d, x, s, ac) { \
  (a) = vaddq_u32((a), vaddq_u32(I_NEON((b), (c), (d)), vaddq_u32((x), vdupq_n_u32(ac)))); \
  (a) = ROTATELEFT_NEON((a), (s)); \
  (a) = vaddq_u32((a), (b)); \
}

// 串行版本
void MD5Hash(string input, bit32 *state);

// 4路NEON版本：一次同时处理4条消息
void MD5HashSIMD4(string inputs[4], bit32 states[4][4]);
void MD5HashSIMD4Ref(
    const string &s0,
    const string &s1,
    const string &s2,
    const string &s3,
    bit32 states[4][4]
);
extern "C" void MD5RoundSIMD4_asm(
    const unsigned int packed_words[16][4],
    unsigned int out_abcd[4][4]
);

void MD5HashSIMD4AsmRef(
    const string &s0,
    const string &s1,
    const string &s2,
    const string &s3,
    bit32 states[4][4]
);