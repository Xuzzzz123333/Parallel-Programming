
本文件夹为并行程序设计第二次实验代码，主题为 **SIMD 编程优化口令猜测中的 MD5 哈希阶段**。实验基于 PCFG 口令生成框架，在候选口令生成后，对 MD5 哈希计算部分进行串行、ARM NEON SIMD、优化 SIMD Ref 和手写 AArch64 NEON 汇编版本的对比。

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
| Initial SIMD | `MD5HashSIMD4`                                           | 初版 4 路 NEON SIMD，用于基础要求和正确性验证      |
| SIMD Ref     | `MD5HashSIMD4Ref` / `main.cpp`                           | 加入 short-message fast path 和引用传参优化 |
| ASM SIMD     | `MD5HashSIMD4AsmRef` / `main_asm.cpp` / `md5_neon_asm.S` | 手写 AArch64 NEON 汇编核心               |

## 2. 编译环境

实验在 AArch64 / ARMv8 平台上进行，需要支持 ARM NEON。

推荐编译器：

```bash
g++
```

推荐编译参数：

```bash
-g -march=native
```

可测试不同优化等级：

```bash
-O0
-O1
-O2
```

## 3. 完整流程编译与运行

完整流程会执行 PCFG 训练、口令生成和 MD5 哈希。程序会输出：

```text
Guess time:
Hash time:
Train time:
```

其中 `Hash time` 是本实验主要关注的指标。

### 3.1 Serial 版本

`main.cpp` 支持命令行参数 `serial`，用于运行串行 MD5 版本。

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native
./main_ref serial
```

如需测试不同优化等级：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O0 -O0 -g -march=native
./main_ref_O0 serial

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O1 -O1 -g -march=native
./main_ref_O1 serial

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O2 -O2 -g -march=native
./main_ref_O2 serial
```

### 3.2 SIMD Ref 版本

`main.cpp` 的 `simd` 模式调用 `MD5HashSIMD4Ref`，即优化后的 C++ NEON intrinsics 版本。

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native
./main_ref simd
```

也可以不加参数运行，因为 `main.cpp` 默认走 SIMD Ref：

```bash
./main_ref
```

测试不同优化等级：

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O0 -O0 -g -march=native
./main_ref_O0 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O1 -O1 -g -march=native
./main_ref_O1 simd

g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref_O2 -O2 -g -march=native
./main_ref_O2 simd
```

### 3.3 ASM SIMD 版本

`main_asm.cpp` 的 `simd` 模式调用 `MD5HashSIMD4AsmRef`，核心 64 步 MD5 round 由 `md5_neon_asm.S` 中的手写 AArch64 NEON 汇编实现。

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm -O2 -g -march=native
./main_asm simd
```

测试不同优化等级：

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O0 -O0 -g -march=native
./main_asm_O0 simd

g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O1 -O1 -g -march=native
./main_asm_O1 simd

g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm_O2 -O2 -g -march=native
./main_asm_O2 simd
```

## 4. 初版 SIMD 正确性测试

初版 SIMD 函数为：

```cpp
MD5HashSIMD4(string inputs[4], bit32 states[4][4])
```

它在 `correctness.cpp` 中进行测试。该测试会对 4 条典型口令分别计算串行 MD5 和 SIMD MD5，并比较输出。

编译：

```bash
g++ correctness.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o correctness -O2 -g -march=native
```

运行：

```bash
./correctness
```

若正确，会输出：

```text
MD5HashSIMD4 correctness test passed!
```

说明初版 SIMD 计算结果与串行 MD5 一致。

## 5. hash-only 微基准测试

`hash_bench.cpp` 用于单独测试 MD5 哈希模块，不包含 PCFG 训练和口令生成阶段。它适合用于分析哈希函数本身的吞吐率和 checksum 一致性。

当前 `hash_bench.cpp` 的 `simd` 模式调用 ASM SIMD 版本。

编译：

```bash
g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench -O2 -g -march=native
```

串行测试：

```bash
./hash_bench serial 1000000
```

SIMD 测试：

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

## 6. 使用课程脚本提交完整流程

课程环境中如果使用官方 `test.sh/qsub.sh`，通常需要先将目标版本编译为名为 `main` 的可执行文件，然后通过脚本提交。

### 6.1 提交 SIMD Ref 版本

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
sh test.sh 1 1
```

此时 `main.cpp` 默认运行 SIMD Ref 版本。

### 6.2 提交 ASM SIMD 版本

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main -O2 -g -march=native
sh test.sh 1 1
```

### 6.3 查看输出

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

## 7. 常用测试命令汇总

### Serial / SIMD Ref 完整流程

```bash
g++ main.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_ref -O2 -g -march=native

./main_ref serial
./main_ref simd
```

### ASM SIMD 完整流程

```bash
g++ main_asm.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o main_asm -O2 -g -march=native

./main_asm simd
```

### 初版 SIMD 正确性

```bash
g++ correctness.cpp train.cpp guessing.cpp md5.cpp md5_neon_asm.S -o correctness -O2 -g -march=native
./correctness
```

### hash-only 微基准

```bash
g++ hash_bench.cpp md5.cpp md5_neon_asm.S -o hash_bench -O2 -g -march=native

./hash_bench serial 1000000
./hash_bench simd 1000000
```

## 8. 注意事项

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

4. `correctness.cpp` 用于验证初版 SIMD，即 `MD5HashSIMD4`；`hash_bench.cpp` 用于模块级性能分析，不包含 PCFG 训练和口令生成。

5. 如果使用课程集群脚本，请确保最终要测试的版本被编译为名为 `main` 的可执行文件。

