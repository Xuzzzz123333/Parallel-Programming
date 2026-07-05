# Windows + MS-MPI + CUDA 编译说明

## 1. 进入目录

在 `x64 Native Tools Command Prompt for VS 2022` 中执行：

```cmd
cd /d D:\并行程序设计1\guess\final_hybrid_guess
```

## 2. 检查环境

```cmd
where nvcc
where cl
where mpiexec
dir "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
```

如果 `mpiexec` 找不到，但 `mpiexec.exe` 存在，可以临时加入 PATH：

```cmd
set "PATH=C:\Program Files\Microsoft MPI\Bin;%PATH%"
```

## 3. 编译

```cmd
build_windows.bat
```

`build_windows.bat` 会先编译 CUDA 文件：

```cmd
nvcc -O2 -std=c++17 -Xcompiler "/utf-8" -c guessing_gpu.cu -o guessing_gpu.obj
```

再用 MSVC 链接 C++、OpenMP、MS-MPI 和 CUDA runtime：

```cmd
cl /EHsc /O2 /std:c++17 /openmp /utf-8 ^
  /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
  correctness_guess_hybrid.cpp ^
  train_serial.cpp train_mpi.cpp md5_portable.cpp md5_simd_x86.cpp guessing_gpu.obj ^
  /link ^
  /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib ^
  /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib ^
  /OUT:hybrid_guess.exe
```

如果你的 CUDA 版本不是 12.8，请把 `build_windows.bat` 里的 CUDA 路径改成你的实际版本。

## 4. 运行示例

```cmd
mpiexec -n 1 hybrid_guess.exe 100000 2147483647 1 ..\guessdata\Rockyou-singleLined-full.txt 100000 serial serial_train
mpiexec -n 2 hybrid_guess.exe 100000 2147483647 4 ..\guessdata\Rockyou-singleLined-full.txt 100000 mpi_omp_simd mpi_train
mpiexec -n 2 hybrid_guess.exe 100000 1000 4 ..\guessdata\Rockyou-singleLined-full.txt 100000 hybrid mpi_train
```
