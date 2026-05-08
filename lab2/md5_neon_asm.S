.text
.align 4
.global MD5RoundSIMD4_asm
.type MD5RoundSIMD4_asm, %function

// void MD5RoundSIMD4_asm(const uint32_t packed_words[16][4],
//                         uint32_t out_abcd[4][4])
// x0 = packed_words
// x1 = out_abcd
//
// output layout:
// out_abcd[0] = A lanes
// out_abcd[1] = B lanes
// out_abcd[2] = C lanes
// out_abcd[3] = D lanes

.macro LOADC imm
    ldr     w9, =\imm
    dup     v6.4s, w9
.endm

.macro ROL A, S
    shl     v5.4s, \A\().4s, #\S
    ushr    \A\().4s, \A\().4s, #(32-\S)
    orr     \A\().16b, \A\().16b, v5.16b
.endm

// F(b,c,d) = (b & c) | (~b & d)
.macro FF A, B, C, D, X, S, AC
    mov     v4.16b, \B\().16b
    bsl     v4.16b, \C\().16b, \D\().16b
    add     \A\().4s, \A\().4s, v4.4s
    add     \A\().4s, \A\().4s, \X\().4s
    LOADC   \AC
    add     \A\().4s, \A\().4s, v6.4s
    ROL     \A, \S
    add     \A\().4s, \A\().4s, \B\().4s
.endm

// G(b,c,d) = (b & d) | (c & ~d)
.macro GG A, B, C, D, X, S, AC
    mov     v4.16b, \D\().16b
    bsl     v4.16b, \B\().16b, \C\().16b
    add     \A\().4s, \A\().4s, v4.4s
    add     \A\().4s, \A\().4s, \X\().4s
    LOADC   \AC
    add     \A\().4s, \A\().4s, v6.4s
    ROL     \A, \S
    add     \A\().4s, \A\().4s, \B\().4s
.endm

// H(b,c,d) = b ^ c ^ d
.macro HH A, B, C, D, X, S, AC
    eor     v4.16b, \B\().16b, \C\().16b
    eor     v4.16b, v4.16b, \D\().16b
    add     \A\().4s, \A\().4s, v4.4s
    add     \A\().4s, \A\().4s, \X\().4s
    LOADC   \AC
    add     \A\().4s, \A\().4s, v6.4s
    ROL     \A, \S
    add     \A\().4s, \A\().4s, \B\().4s
.endm

// I(b,c,d) = c ^ (b | ~d)
.macro II A, B, C, D, X, S, AC
    not     v4.16b, \D\().16b
    orr     v4.16b, v4.16b, \B\().16b
    eor     v4.16b, v4.16b, \C\().16b
    add     \A\().4s, \A\().4s, v4.4s
    add     \A\().4s, \A\().4s, \X\().4s
    LOADC   \AC
    add     \A\().4s, \A\().4s, v6.4s
    ROL     \A, \S
    add     \A\().4s, \A\().4s, \B\().4s
.endm

MD5RoundSIMD4_asm:
    // load x[0..15]
    ld1     {v16.4s, v17.4s, v18.4s, v19.4s}, [x0], #64
    ld1     {v20.4s, v21.4s, v22.4s, v23.4s}, [x0], #64
    ld1     {v24.4s, v25.4s, v26.4s, v27.4s}, [x0], #64
    ld1     {v28.4s, v29.4s, v30.4s, v31.4s}, [x0]

    // a,b,c,d initial values
    ldr     w9, =0x67452301
    dup     v0.4s, w9
    ldr     w9, =0xefcdab89
    dup     v1.4s, w9
    ldr     w9, =0x98badcfe
    dup     v2.4s, w9
    ldr     w9, =0x10325476
    dup     v3.4s, w9

    // Round 1
    FF v0, v1, v2, v3, v16, 7,  0xd76aa478
    FF v3, v0, v1, v2, v17, 12, 0xe8c7b756
    FF v2, v3, v0, v1, v18, 17, 0x242070db
    FF v1, v2, v3, v0, v19, 22, 0xc1bdceee
    FF v0, v1, v2, v3, v20, 7,  0xf57c0faf
    FF v3, v0, v1, v2, v21, 12, 0x4787c62a
    FF v2, v3, v0, v1, v22, 17, 0xa8304613
    FF v1, v2, v3, v0, v23, 22, 0xfd469501
    FF v0, v1, v2, v3, v24, 7,  0x698098d8
    FF v3, v0, v1, v2, v25, 12, 0x8b44f7af
    FF v2, v3, v0, v1, v26, 17, 0xffff5bb1
    FF v1, v2, v3, v0, v27, 22, 0x895cd7be
    FF v0, v1, v2, v3, v28, 7,  0x6b901122
    FF v3, v0, v1, v2, v29, 12, 0xfd987193
    FF v2, v3, v0, v1, v30, 17, 0xa679438e
    FF v1, v2, v3, v0, v31, 22, 0x49b40821

    // Round 2
    GG v0, v1, v2, v3, v17, 5,  0xf61e2562
    GG v3, v0, v1, v2, v22, 9,  0xc040b340
    GG v2, v3, v0, v1, v27, 14, 0x265e5a51
    GG v1, v2, v3, v0, v16, 20, 0xe9b6c7aa
    GG v0, v1, v2, v3, v21, 5,  0xd62f105d
    GG v3, v0, v1, v2, v26, 9,  0x02441453
    GG v2, v3, v0, v1, v31, 14, 0xd8a1e681
    GG v1, v2, v3, v0, v20, 20, 0xe7d3fbc8
    GG v0, v1, v2, v3, v25, 5,  0x21e1cde6
    GG v3, v0, v1, v2, v30, 9,  0xc33707d6
    GG v2, v3, v0, v1, v19, 14, 0xf4d50d87
    GG v1, v2, v3, v0, v24, 20, 0x455a14ed
    GG v0, v1, v2, v3, v29, 5,  0xa9e3e905
    GG v3, v0, v1, v2, v18, 9,  0xfcefa3f8
    GG v2, v3, v0, v1, v23, 14, 0x676f02d9
    GG v1, v2, v3, v0, v28, 20, 0x8d2a4c8a

    // Round 3
    HH v0, v1, v2, v3, v21, 4,  0xfffa3942
    HH v3, v0, v1, v2, v24, 11, 0x8771f681
    HH v2, v3, v0, v1, v27, 16, 0x6d9d6122
    HH v1, v2, v3, v0, v30, 23, 0xfde5380c
    HH v0, v1, v2, v3, v17, 4,  0xa4beea44
    HH v3, v0, v1, v2, v20, 11, 0x4bdecfa9
    HH v2, v3, v0, v1, v23, 16, 0xf6bb4b60
    HH v1, v2, v3, v0, v26, 23, 0xbebfbc70
    HH v0, v1, v2, v3, v29, 4,  0x289b7ec6
    HH v3, v0, v1, v2, v16, 11, 0xeaa127fa
    HH v2, v3, v0, v1, v19, 16, 0xd4ef3085
    HH v1, v2, v3, v0, v22, 23, 0x04881d05
    HH v0, v1, v2, v3, v25, 4,  0xd9d4d039
    HH v3, v0, v1, v2, v28, 11, 0xe6db99e5
    HH v2, v3, v0, v1, v31, 16, 0x1fa27cf8
    HH v1, v2, v3, v0, v18, 23, 0xc4ac5665

    // Round 4
    II v0, v1, v2, v3, v16, 6,  0xf4292244
    II v3, v0, v1, v2, v23, 10, 0x432aff97
    II v2, v3, v0, v1, v30, 15, 0xab9423a7
    II v1, v2, v3, v0, v21, 21, 0xfc93a039
    II v0, v1, v2, v3, v28, 6,  0x655b59c3
    II v3, v0, v1, v2, v19, 10, 0x8f0ccc92
    II v2, v3, v0, v1, v26, 15, 0xffeff47d
    II v1, v2, v3, v0, v17, 21, 0x85845dd1
    II v0, v1, v2, v3, v24, 6,  0x6fa87e4f
    II v3, v0, v1, v2, v31, 10, 0xfe2ce6e0
    II v2, v3, v0, v1, v22, 15, 0xa3014314
    II v1, v2, v3, v0, v29, 21, 0x4e0811a1
    II v0, v1, v2, v3, v20, 6,  0xf7537e82
    II v3, v0, v1, v2, v27, 10, 0xbd3af235
    II v2, v3, v0, v1, v18, 15, 0x2ad7d2bb
    II v1, v2, v3, v0, v25, 21, 0xeb86d391

    // add initial values back
    ldr     w9, =0x67452301
    dup     v4.4s, w9
    add     v0.4s, v0.4s, v4.4s

    ldr     w9, =0xefcdab89
    dup     v4.4s, w9
    add     v1.4s, v1.4s, v4.4s

    ldr     w9, =0x98badcfe
    dup     v4.4s, w9
    add     v2.4s, v2.4s, v4.4s

    ldr     w9, =0x10325476
    dup     v4.4s, w9
    add     v3.4s, v3.4s, v4.4s

    st1     {v0.4s, v1.4s, v2.4s, v3.4s}, [x1]

    ret

.ltorg
.size MD5RoundSIMD4_asm, .-MD5RoundSIMD4_asm