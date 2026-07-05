#include "md5_portable.h"
#include <cassert>

static inline bit32 LoadWordLE(const Byte *p)
{
    return ((bit32)p[0]) |
           ((bit32)p[1] << 8) |
           ((bit32)p[2] << 16) |
           ((bit32)p[3] << 24);
}

static inline bit32 ByteSwap32(bit32 value)
{
    return ((value & 0x000000ff) << 24) |
           ((value & 0x0000ff00) << 8) |
           ((value & 0x00ff0000) >> 8) |
           ((value & 0xff000000) >> 24);
}

static Byte *StringProcessPortable(const string &input, int *n_byte)
{
    int length = (int)input.length();
    int bitLength = length * 8;

    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
    {
        paddingBits = 512 - (paddingBits - 448);
    }
    else if (paddingBits < 448)
    {
        paddingBits = 448 - paddingBits;
    }
    else
    {
        paddingBits = 512;
    }

    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;
    Byte *paddedMessage = new Byte[paddedLength];

    memcpy(paddedMessage, input.data(), length);
    paddedMessage[length] = 0x80;
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);

    uint64_t bitLen64 = (uint64_t)length * 8;
    for (int i = 0; i < 8; ++i)
    {
        paddedMessage[length + paddingBytes + i] = (Byte)((bitLen64 >> (i * 8)) & 0xff);
    }

    *n_byte = paddedLength;
    return paddedMessage;
}

void MD5Hash(const string &input, bit32 *state)
{
    int messageLength = 0;
    Byte *paddedMessage = StringProcessPortable(input, &messageLength);
    int n_blocks = messageLength / 64;

    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    for (int block = 0; block < n_blocks; block += 1)
    {
        bit32 x[16];
        const Byte *base = paddedMessage + block * 64;
        for (int i = 0; i < 16; ++i)
        {
            x[i] = LoadWordLE(base + 4 * i);
        }

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        FF(a, b, c, d, x[0], s11, 0xd76aa478); FF(d, a, b, c, x[1], s12, 0xe8c7b756); FF(c, d, a, b, x[2], s13, 0x242070db); FF(b, c, d, a, x[3], s14, 0xc1bdceee);
        FF(a, b, c, d, x[4], s11, 0xf57c0faf); FF(d, a, b, c, x[5], s12, 0x4787c62a); FF(c, d, a, b, x[6], s13, 0xa8304613); FF(b, c, d, a, x[7], s14, 0xfd469501);
        FF(a, b, c, d, x[8], s11, 0x698098d8); FF(d, a, b, c, x[9], s12, 0x8b44f7af); FF(c, d, a, b, x[10], s13, 0xffff5bb1); FF(b, c, d, a, x[11], s14, 0x895cd7be);
        FF(a, b, c, d, x[12], s11, 0x6b901122); FF(d, a, b, c, x[13], s12, 0xfd987193); FF(c, d, a, b, x[14], s13, 0xa679438e); FF(b, c, d, a, x[15], s14, 0x49b40821);

        GG(a, b, c, d, x[1], s21, 0xf61e2562); GG(d, a, b, c, x[6], s22, 0xc040b340); GG(c, d, a, b, x[11], s23, 0x265e5a51); GG(b, c, d, a, x[0], s24, 0xe9b6c7aa);
        GG(a, b, c, d, x[5], s21, 0xd62f105d); GG(d, a, b, c, x[10], s22, 0x02441453); GG(c, d, a, b, x[15], s23, 0xd8a1e681); GG(b, c, d, a, x[4], s24, 0xe7d3fbc8);
        GG(a, b, c, d, x[9], s21, 0x21e1cde6); GG(d, a, b, c, x[14], s22, 0xc33707d6); GG(c, d, a, b, x[3], s23, 0xf4d50d87); GG(b, c, d, a, x[8], s24, 0x455a14ed);
        GG(a, b, c, d, x[13], s21, 0xa9e3e905); GG(d, a, b, c, x[2], s22, 0xfcefa3f8); GG(c, d, a, b, x[7], s23, 0x676f02d9); GG(b, c, d, a, x[12], s24, 0x8d2a4c8a);

        HH(a, b, c, d, x[5], s31, 0xfffa3942); HH(d, a, b, c, x[8], s32, 0x8771f681); HH(c, d, a, b, x[11], s33, 0x6d9d6122); HH(b, c, d, a, x[14], s34, 0xfde5380c);
        HH(a, b, c, d, x[1], s31, 0xa4beea44); HH(d, a, b, c, x[4], s32, 0x4bdecfa9); HH(c, d, a, b, x[7], s33, 0xf6bb4b60); HH(b, c, d, a, x[10], s34, 0xbebfbc70);
        HH(a, b, c, d, x[13], s31, 0x289b7ec6); HH(d, a, b, c, x[0], s32, 0xeaa127fa); HH(c, d, a, b, x[3], s33, 0xd4ef3085); HH(b, c, d, a, x[6], s34, 0x04881d05);
        HH(a, b, c, d, x[9], s31, 0xd9d4d039); HH(d, a, b, c, x[12], s32, 0xe6db99e5); HH(c, d, a, b, x[15], s33, 0x1fa27cf8); HH(b, c, d, a, x[2], s34, 0xc4ac5665);

        II(a, b, c, d, x[0], s41, 0xf4292244); II(d, a, b, c, x[7], s42, 0x432aff97); II(c, d, a, b, x[14], s43, 0xab9423a7); II(b, c, d, a, x[5], s44, 0xfc93a039);
        II(a, b, c, d, x[12], s41, 0x655b59c3); II(d, a, b, c, x[3], s42, 0x8f0ccc92); II(c, d, a, b, x[10], s43, 0xffeff47d); II(b, c, d, a, x[1], s44, 0x85845dd1);
        II(a, b, c, d, x[8], s41, 0x6fa87e4f); II(d, a, b, c, x[15], s42, 0xfe2ce6e0); II(c, d, a, b, x[6], s43, 0xa3014314); II(b, c, d, a, x[13], s44, 0x4e0811a1);
        II(a, b, c, d, x[4], s41, 0xf7537e82); II(d, a, b, c, x[11], s42, 0xbd3af235); II(c, d, a, b, x[2], s43, 0x2ad7d2bb); II(b, c, d, a, x[9], s44, 0xeb86d391);

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
    }

    for (int i = 0; i < 4; i++)
    {
        state[i] = ByteSwap32(state[i]);
    }

    delete[] paddedMessage;
}
