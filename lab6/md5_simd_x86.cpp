#include "md5_simd_x86.h"

#include <emmintrin.h>
#include <cstring>

static inline bit32 LoadWordLE_X86(const Byte *p)
{
    return ((bit32)p[0]) |
           ((bit32)p[1] << 8) |
           ((bit32)p[2] << 16) |
           ((bit32)p[3] << 24);
}

static inline bit32 ByteSwap32_X86(bit32 value)
{
    return ((value & 0x000000ffU) << 24) |
           ((value & 0x0000ff00U) << 8)  |
           ((value & 0x00ff0000U) >> 8)  |
           ((value & 0xff000000U) >> 24);
}

static inline __m128i VCONST(bit32 x)
{
    return _mm_set1_epi32((int)x);
}

static inline __m128i VADD(__m128i a, __m128i b)
{
    return _mm_add_epi32(a, b);
}

static inline __m128i VNOT(__m128i x)
{
    return _mm_xor_si128(x, _mm_set1_epi32(-1));
}

static inline __m128i VF(__m128i x, __m128i y, __m128i z)
{
    return _mm_or_si128(_mm_and_si128(x, y), _mm_and_si128(VNOT(x), z));
}

static inline __m128i VG(__m128i x, __m128i y, __m128i z)
{
    return _mm_or_si128(_mm_and_si128(x, z), _mm_and_si128(y, VNOT(z)));
}

static inline __m128i VH(__m128i x, __m128i y, __m128i z)
{
    return _mm_xor_si128(_mm_xor_si128(x, y), z);
}

static inline __m128i VI(__m128i x, __m128i y, __m128i z)
{
    return _mm_xor_si128(y, _mm_or_si128(x, VNOT(z)));
}

static inline __m128i VROTL(__m128i x, int n)
{
    return _mm_or_si128(_mm_slli_epi32(x, n), _mm_srli_epi32(x, 32 - n));
}

#define FFV(a,b,c,d,x,s,ac) do { \
    (a) = VADD((a), VADD(VF((b),(c),(d)), VADD((x), VCONST((bit32)(ac))))); \
    (a) = VROTL((a), (s)); \
    (a) = VADD((a), (b)); \
} while (0)

#define GGV(a,b,c,d,x,s,ac) do { \
    (a) = VADD((a), VADD(VG((b),(c),(d)), VADD((x), VCONST((bit32)(ac))))); \
    (a) = VROTL((a), (s)); \
    (a) = VADD((a), (b)); \
} while (0)

#define HHV(a,b,c,d,x,s,ac) do { \
    (a) = VADD((a), VADD(VH((b),(c),(d)), VADD((x), VCONST((bit32)(ac))))); \
    (a) = VROTL((a), (s)); \
    (a) = VADD((a), (b)); \
} while (0)

#define IIV(a,b,c,d,x,s,ac) do { \
    (a) = VADD((a), VADD(VI((b),(c),(d)), VADD((x), VCONST((bit32)(ac))))); \
    (a) = VROTL((a), (s)); \
    (a) = VADD((a), (b)); \
} while (0)

static void BuildSingleBlock(const std::string &s, Byte block[64])
{
    std::memset(block, 0, 64);
    std::memcpy(block, s.data(), s.size());
    block[s.size()] = 0x80;
    unsigned long long bit_len = (unsigned long long)s.size() * 8ULL;
    for (int i = 0; i < 8; ++i)
    {
        block[56 + i] = (Byte)((bit_len >> (8 * i)) & 0xff);
    }
}

void MD5HashSIMD4X86(
    const std::string &s0,
    const std::string &s1,
    const std::string &s2,
    const std::string &s3,
    bit32 states[4][4]
)
{
    const std::string *inputs[4] = {&s0, &s1, &s2, &s3};

    // One-block fast path.  Longer messages fall back to the scalar routine.
    for (int lane = 0; lane < 4; ++lane)
    {
        if (inputs[lane]->size() >= 56)
        {
            for (int i = 0; i < 4; ++i)
            {
                MD5Hash(*inputs[i], states[i]);
            }
            return;
        }
    }

    Byte blocks[4][64];
    for (int lane = 0; lane < 4; ++lane)
    {
        BuildSingleBlock(*inputs[lane], blocks[lane]);
    }

    __m128i x[16];
    for (int j = 0; j < 16; ++j)
    {
        bit32 lane_word[4];
        for (int lane = 0; lane < 4; ++lane)
        {
            lane_word[lane] = LoadWordLE_X86(blocks[lane] + 4 * j);
        }
        // _mm_set_epi32 argument order is lane3,lane2,lane1,lane0.
        x[j] = _mm_set_epi32((int)lane_word[3], (int)lane_word[2],
                             (int)lane_word[1], (int)lane_word[0]);
    }

    __m128i a0 = VCONST(0x67452301U);
    __m128i b0 = VCONST(0xefcdab89U);
    __m128i c0 = VCONST(0x98badcfeU);
    __m128i d0 = VCONST(0x10325476U);

    __m128i a = a0, b = b0, c = c0, d = d0;

    FFV(a,b,c,d,x[ 0],s11,0xd76aa478U); FFV(d,a,b,c,x[ 1],s12,0xe8c7b756U); FFV(c,d,a,b,x[ 2],s13,0x242070dbU); FFV(b,c,d,a,x[ 3],s14,0xc1bdceeeU);
    FFV(a,b,c,d,x[ 4],s11,0xf57c0fafU); FFV(d,a,b,c,x[ 5],s12,0x4787c62aU); FFV(c,d,a,b,x[ 6],s13,0xa8304613U); FFV(b,c,d,a,x[ 7],s14,0xfd469501U);
    FFV(a,b,c,d,x[ 8],s11,0x698098d8U); FFV(d,a,b,c,x[ 9],s12,0x8b44f7afU); FFV(c,d,a,b,x[10],s13,0xffff5bb1U); FFV(b,c,d,a,x[11],s14,0x895cd7beU);
    FFV(a,b,c,d,x[12],s11,0x6b901122U); FFV(d,a,b,c,x[13],s12,0xfd987193U); FFV(c,d,a,b,x[14],s13,0xa679438eU); FFV(b,c,d,a,x[15],s14,0x49b40821U);

    GGV(a,b,c,d,x[ 1],s21,0xf61e2562U); GGV(d,a,b,c,x[ 6],s22,0xc040b340U); GGV(c,d,a,b,x[11],s23,0x265e5a51U); GGV(b,c,d,a,x[ 0],s24,0xe9b6c7aaU);
    GGV(a,b,c,d,x[ 5],s21,0xd62f105dU); GGV(d,a,b,c,x[10],s22,0x02441453U); GGV(c,d,a,b,x[15],s23,0xd8a1e681U); GGV(b,c,d,a,x[ 4],s24,0xe7d3fbc8U);
    GGV(a,b,c,d,x[ 9],s21,0x21e1cde6U); GGV(d,a,b,c,x[14],s22,0xc33707d6U); GGV(c,d,a,b,x[ 3],s23,0xf4d50d87U); GGV(b,c,d,a,x[ 8],s24,0x455a14edU);
    GGV(a,b,c,d,x[13],s21,0xa9e3e905U); GGV(d,a,b,c,x[ 2],s22,0xfcefa3f8U); GGV(c,d,a,b,x[ 7],s23,0x676f02d9U); GGV(b,c,d,a,x[12],s24,0x8d2a4c8aU);

    HHV(a,b,c,d,x[ 5],s31,0xfffa3942U); HHV(d,a,b,c,x[ 8],s32,0x8771f681U); HHV(c,d,a,b,x[11],s33,0x6d9d6122U); HHV(b,c,d,a,x[14],s34,0xfde5380cU);
    HHV(a,b,c,d,x[ 1],s31,0xa4beea44U); HHV(d,a,b,c,x[ 4],s32,0x4bdecfa9U); HHV(c,d,a,b,x[ 7],s33,0xf6bb4b60U); HHV(b,c,d,a,x[10],s34,0xbebfbc70U);
    HHV(a,b,c,d,x[13],s31,0x289b7ec6U); HHV(d,a,b,c,x[ 0],s32,0xeaa127faU); HHV(c,d,a,b,x[ 3],s33,0xd4ef3085U); HHV(b,c,d,a,x[ 6],s34,0x04881d05U);
    HHV(a,b,c,d,x[ 9],s31,0xd9d4d039U); HHV(d,a,b,c,x[12],s32,0xe6db99e5U); HHV(c,d,a,b,x[15],s33,0x1fa27cf8U); HHV(b,c,d,a,x[ 2],s34,0xc4ac5665U);

    IIV(a,b,c,d,x[ 0],s41,0xf4292244U); IIV(d,a,b,c,x[ 7],s42,0x432aff97U); IIV(c,d,a,b,x[14],s43,0xab9423a7U); IIV(b,c,d,a,x[ 5],s44,0xfc93a039U);
    IIV(a,b,c,d,x[12],s41,0x655b59c3U); IIV(d,a,b,c,x[ 3],s42,0x8f0ccc92U); IIV(c,d,a,b,x[10],s43,0xffeff47dU); IIV(b,c,d,a,x[ 1],s44,0x85845dd1U);
    IIV(a,b,c,d,x[ 8],s41,0x6fa87e4fU); IIV(d,a,b,c,x[15],s42,0xfe2ce6e0U); IIV(c,d,a,b,x[ 6],s43,0xa3014314U); IIV(b,c,d,a,x[13],s44,0x4e0811a1U);
    IIV(a,b,c,d,x[ 4],s41,0xf7537e82U); IIV(d,a,b,c,x[11],s42,0xbd3af235U); IIV(c,d,a,b,x[ 2],s43,0x2ad7d2bbU); IIV(b,c,d,a,x[ 9],s44,0xeb86d391U);

    a = VADD(a0, a);
    b = VADD(b0, b);
    c = VADD(c0, c);
    d = VADD(d0, d);

    alignas(16) bit32 aa[4], bb[4], cc[4], dd[4];
    _mm_storeu_si128((__m128i*)aa, a);
    _mm_storeu_si128((__m128i*)bb, b);
    _mm_storeu_si128((__m128i*)cc, c);
    _mm_storeu_si128((__m128i*)dd, d);

    for (int lane = 0; lane < 4; ++lane)
    {
        states[lane][0] = ByteSwap32_X86(aa[lane]);
        states[lane][1] = ByteSwap32_X86(bb[lane]);
        states[lane][2] = ByteSwap32_X86(cc[lane]);
        states[lane][3] = ByteSwap32_X86(dd[lane]);
    }
}
