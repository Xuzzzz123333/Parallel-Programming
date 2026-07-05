@echo off
setlocal

nvcc -O2 -std=c++17 -Xcompiler "/utf-8" -c guessing_gpu.cu -o guessing_gpu.obj
if errorlevel 1 exit /b 1

cl /EHsc /O2 /std:c++17 /openmp /utf-8 ^
  /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
  correctness_guess_hybrid.cpp ^
  train_serial.cpp train_thread.cpp train_mpi.cpp md5_portable.cpp md5_simd_x86.cpp guessing_gpu.obj ^
  /link ^
  /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib ^
  /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64" cudart.lib ^
  /OUT:hybrid_guess.exe

endlocal
