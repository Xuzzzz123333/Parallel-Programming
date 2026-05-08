

## 1. 文件说明

```text
lab2/
├── PCFG.h              # PCFG 口令生成相关数据结构
├── train.cpp           # PCFG 模型训练
├── guessing.cpp        # PCFG 口令生成
├── main.cpp            # 完整口令猜测流程：Serial / SIMD Ref
├── main_asm.cpp        # 完整口令猜测流程：Serial / ASM SIMD
├── md5.h               # MD5 串行、NEON SIMD、ASM 接口声明
├── md5.cpp             # MD5 串行版、初版 SIMD、SIMD Ref、ASM 包装函数
├── md5_neon_asm.S      # 手写 AArch64 NEON 汇编核心
├── correctness.cpp     # 初版 SIMD 正确性测试
└── hash_bench.cpp      # hash-only 微基准测试
````

本实验中涉及四类实现：

| 版本           | 对应函数 / 文件                                                | 说明                                 |
| ------------ | -------------------------------------------------------- | ---------------------------------- |
| Serial       | `MD5Hash`                                                | 串行 baseline                        |
| Initial SIMD | `MD5HashSIMD4`                                           | 初版 4 路 NEON SIMD                   |
| SIMD Ref     | `MD5HashSIMD4Ref` / `main.cpp`                           | 加入 short-message fast path 和引用传参优化 |
| ASM SIMD     | `MD5HashSIMD4AsmRef` / `main_asm.cpp` / `md5_neon_asm.S` | 手写 AArch64 NEON 汇编核心               |

## 2. 编译环境

实验在 AArch64 / ARMv8 平台上进行，需要支持 ARM NEON。

推荐编译器：

```bash
g++
```

推荐基础编译参数：

```bash
-g -march=native
```

可测试不同优化等级：

```bash
-O0
-O1
-O2
```

完整流程依赖训练集路径：

```cpp
/guessdata/Rockyou-singleLined-full.txt
```

如果本地或服务器上没有该路径，需要修改 `main.cpp` 或 `main_asm.cpp` 中的训练集路径。

## 3. 版本切换说明

本实验中不同版本的测试入口不完全相同：

| 版本           | 测试方式                                                       |
| ------------ | ---------------------------------------------------------- |
| Serial       | 使用 `main.cpp`，运行时传入 `serial` 参数                            |
| Initial SIMD | 需要在 `main.cpp` 中将 SIMD 分支改为 `MD5HashSIMD4(inputs, states)` |
| SIMD Ref     | 使用 `main.cpp`，SIMD 分支调用 `MD5HashSIMD4Ref(...)`             |
| ASM SIMD     | 使用 `main_asm.cpp`，SIMD 分支调用 `MD5HashSIMD4AsmRef(...)`      |

因此，如果需要复现实验报告中的初版 SIMD 和 SIMD Ref 完整流程结果，需要确认 `main.cpp` 中 SIMD 分支调用的是对应函数。

### 3.1 Initial SIMD 在 `main.cpp` 中的调用方式

初版 SIMD 应使用：

```cpp
string inputs[4] = {
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3]
};

bit32 states[4][4];
MD5HashSIMD4(inputs, states);
```

### 3.2 SIMD Ref 在 `main.cpp` 中的调用方式

SIMD Ref 应使用：

```cpp
bit32 states[4][4];

MD5HashSIMD4Ref(
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3],
    states
);
```

### 3.3 ASM SIMD 的调用方式

ASM SIMD 不需要修改 `main.cpp`，而是直接使用 `main_asm.cpp`：

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm -O2 -g -march=native
./main_asm simd
```

## 4. 完整流程输出说明

完整流程会执行 PCFG 训练、口令生成和 MD5 哈希。程序会输出：

```text
Guess time:
Hash time:
Train time:
```

其中：

* `Guess time`：口令生成时间；
* `Hash time`：MD5 哈希计算时间；
* `Train time`：PCFG 模型训练时间。

本实验主要关注 `Hash time`，因为 SIMD 优化作用于 MD5 哈希阶段。

## 5. 测试版本一：Serial 串行版本

`main.cpp` 支持命令行参数 `serial`，用于运行串行 MD5 版本。

### 5.1 编译

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native
```

### 5.2 运行

```bash
./main_ref serial
```

### 5.3 测试不同优化等级

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O0 -O0 -g -march=native
./main_ref_O0 serial

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O1 -O1 -g -march=native
./main_ref_O1 serial

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O2 -O2 -g -march=native
./main_ref_O2 serial
```

## 6. 测试版本二：Initial SIMD 初版 SIMD

初版 SIMD 函数为：

```cpp
MD5HashSIMD4(string inputs[4], bit32 states[4][4])
```

它是最基础的 4 路 NEON SIMD 实现：每次从候选口令列表中取出 4 条口令，构造 `string inputs[4]`，然后调用 `MD5HashSIMD4(inputs, states)` 进行并行 MD5 计算。

### 6.1 初版 SIMD 正确性测试

`correctness.cpp` 会对 4 条典型口令分别计算串行 MD5 和 SIMD MD5，并比较输出。

编译：

```bash
g++ correctness.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o correctness -O2 -g -march=native
```

运行：

```bash
./correctness
```

如果正确，会输出：

```text
MD5HashSIMD4 correctness test passed!
```

这说明初版 SIMD 计算结果与串行 MD5 一致。

### 6.2 在完整流程中测试 Initial SIMD

当前仓库中默认的 `main.cpp` 通常用于测试 SIMD Ref 版本。如果需要在完整口令猜测流程中测试初版 SIMD，需要临时修改 `main.cpp` 中的 SIMD 分支。

将 SIMD 分支中的调用：

```cpp
bit32 states[4][4];

MD5HashSIMD4Ref(
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3],
    states
);
```

改为初版 SIMD 调用方式：

```cpp
string inputs[4] = {
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3]
};

bit32 states[4][4];
MD5HashSIMD4(inputs, states);
```

然后重新编译：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_initial -O2 -g -march=native
```

运行：

```bash
./main_initial simd
```

测试不同优化等级：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_initial_O0 -O0 -g -march=native
./main_initial_O0 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_initial_O1 -O1 -g -march=native
./main_initial_O1 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_initial_O2 -O2 -g -march=native
./main_initial_O2 simd
```

如果使用课程脚本，则需要将该版本编译为名为 `main` 的可执行文件：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
sh test.sh 1 1
```

> 注意：测试完 Initial SIMD 后，如果还要测试 SIMD Ref，需要把 `main.cpp` 中的 SIMD 分支改回 `MD5HashSIMD4Ref(...)`。

## 7. 测试版本三：SIMD Ref 优化版

SIMD Ref 版本对应函数为：

```cpp
MD5HashSIMD4Ref(
    const string &s0,
    const string &s1,
    const string &s2,
    const string &s3,
    bit32 states[4][4]
)
```

它是在初版 SIMD 的基础上进一步优化得到的版本，主要改动包括：

1. 使用 short-message fast path，针对长度小于 56 字节的短口令直接构造单个 512-bit block，减少 `StringProcess` 和动态内存分配开销；
2. 使用 `const string&` 引用传参，避免每 4 条口令额外构造 `string inputs[4]`；
3. 保持 4 路 NEON SIMD 核心计算逻辑不变，但减少外围数据组织成本。

### 7.1 `main.cpp` 中的调用方式

测试 SIMD Ref 时，`main.cpp` 的 SIMD 分支应调用：

```cpp
bit32 states[4][4];

MD5HashSIMD4Ref(
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3],
    states
);
```

如果之前为了测试 Initial SIMD，把这里改成了：

```cpp
string inputs[4] = {
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3]
};

bit32 states[4][4];
MD5HashSIMD4(inputs, states);
```

则需要改回 `MD5HashSIMD4Ref(...)` 后再编译测试 SIMD Ref。

### 7.2 编译

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native
```

### 7.3 运行

```bash
./main_ref simd
```

也可以不加参数运行，因为 `main.cpp` 默认走 SIMD Ref：

```bash
./main_ref
```

### 7.4 测试不同优化等级

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O0 -O0 -g -march=native
./main_ref_O0 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O1 -O1 -g -march=native
./main_ref_O1 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O2 -O2 -g -march=native
./main_ref_O2 simd
```

如果使用课程脚本提交 SIMD Ref 版本：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
sh test.sh 1 1
```

## 8. 测试版本四：ASM SIMD 手写汇编版

`main_asm.cpp` 的 `simd` 模式调用 `MD5HashSIMD4AsmRef`，核心 64 步 MD5 round 由 `md5_neon_asm.S` 中的手写 AArch64 NEON 汇编实现。

### 8.1 编译

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm -O2 -g -march=native
```

### 8.2 运行

```bash
./main_asm simd
```

### 8.3 测试不同优化等级

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O0 -O0 -g -march=native
./main_asm_O0 simd

g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O1 -O1 -g -march=native
./main_asm_O1 simd

g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O2 -O2 -g -march=native
./main_asm_O2 simd
```

如果使用课程脚本提交 ASM SIMD 版本：

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
sh test.sh 1 1
```

## 9. hash-only 微基准测试

`hash_bench.cpp` 用于单独测试 MD5 哈希模块，不包含 PCFG 训练和口令生成阶段。它适合用于分析哈希函数本身的吞吐率和 checksum 一致性。

### 9.1 编译

```bash
g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench -O2 -g -march=native
```

### 9.2 串行测试

```bash
./hash_bench serial 1000000
```

### 9.3 SIMD 测试

```bash
./hash_bench simd 1000000
```

输出示例：

```text
mode = serial
N = 1000000
hash time = ...
throughput = ...
checksum = ...

mode = simd
N = 1000000
hash time = ...
throughput = ...
checksum = ...
```

如果 serial 和 simd 的 checksum 一致，说明批量哈希结果正确。

### 9.4 测试不同优化等级

```bash
g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench_O0 -O0 -g -march=native
./hash_bench_O0 serial 1000000
./hash_bench_O0 simd 1000000

g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench_O1 -O1 -g -march=native
./hash_bench_O1 serial 1000000
./hash_bench_O1 simd 1000000

g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench_O2 -O2 -g -march=native
./hash_bench_O2 serial 1000000
./hash_bench_O2 simd 1000000
```

## 10. 使用课程脚本提交完整流程

课程环境中如果使用官方 `test.sh/qsub.sh`，通常需要先将目标版本编译为名为 `main` 的可执行文件，然后通过脚本提交。

注意：`test.sh` 只会运行当前目录下名为 `main` 的可执行文件。因此测试不同版本时，需要先把对应版本编译为 `main`。

### 10.1 提交 SIMD Ref 版本

确认 `main.cpp` 中 SIMD 分支调用的是：

```cpp
MD5HashSIMD4Ref(...)
```

然后编译为 `main`：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
```

提交：

```bash
sh test.sh 1 1
```

### 10.2 提交 Initial SIMD 版本

先临时修改 `main.cpp`，将 SIMD 分支改为：

```cpp
string inputs[4] = {
    guess_vec[idx],
    guess_vec[idx + 1],
    guess_vec[idx + 2],
    guess_vec[idx + 3]
};

bit32 states[4][4];
MD5HashSIMD4(inputs, states);
```

然后编译为 `main`：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
```

提交：

```bash
sh test.sh 1 1
```

测试完后，如需继续测试 SIMD Ref，需要把 `main.cpp` 改回 `MD5HashSIMD4Ref(...)`。

### 10.3 提交 ASM SIMD 版本

编译 `main_asm.cpp` 为 `main`：

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
```

提交：

```bash
sh test.sh 1 1
```

### 10.4 查看输出

任务完成后查看日志：

```bash
cat test.o
```

重点查看：

```text
MD5Hash test passed!
Guess time:
Hash time:
Train time:
```

## 11. 常用命令汇总

### 11.1 Serial / SIMD Ref 完整流程

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native

./main_ref serial
./main_ref simd
```

### 11.2 Initial SIMD 正确性测试

```bash
g++ correctness.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o correctness -O2 -g -march=native
./correctness
```

### 11.3 ASM SIMD 完整流程

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm -O2 -g -march=native

./main_asm simd
```

### 11.4 hash-only 微基准

```bash
g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench -O2 -g -march=native

./hash_bench serial 1000000
./hash_bench simd 1000000
```

### 11.5 反汇编检查 NEON 指令

```bash
objdump -Cd --demangle main_ref > main_ref.asm
grep -n "MD5HashSIMD4Ref" main_ref.asm
grep -n "v.*4s" main_ref.asm | head
```

常见 NEON 指令包括：

```text
add v?.4s
shl v?.4s
ushr v?.4s
orr v?.16b
and v?.16b
eor v?.16b
```

## 12. 注意事项

1. 完整流程依赖数据集路径：

   ```cpp
   /guessdata/Rockyou-singleLined-full.txt
   ```

   如果本地没有该数据集，需要修改 `main.cpp` / `main_asm.cpp` 中的训练集路径。

2. `md5.cpp` 中包含 ASM 包装函数，因此建议所有编译命令都带上：

   ```bash
   md5_neon_asm.S
   ```

   否则可能出现链接错误：

   ```text
   undefined reference to `MD5RoundSIMD4_asm'
   ```

3. `main.cpp` 用于 Serial / SIMD Ref 对比；`main_asm.cpp` 用于 ASM SIMD 对比。

4. `correctness.cpp` 用于验证初版 SIMD，即 `MD5HashSIMD4`。

5. `hash_bench.cpp` 用于模块级性能分析，不包含 PCFG 训练和口令生成。

6. 如果使用课程集群脚本，请确保最终要测试的版本被编译为名为 `main` 的可执行文件。

7. Initial SIMD 和 SIMD Ref 都依赖 `main.cpp` 中 SIMD 分支的具体调用方式。复现实验时需要先确认当前 `main.cpp` 调用的是 `MD5HashSIMD4(...)` 还是 `MD5HashSIMD4Ref(...)`。

