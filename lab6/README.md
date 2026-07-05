# final_hybrid_guess_v9_indexed_runtime

This version keeps the v6/v7 fast path and adds a **runtime model lookup index**.

## Added exploration

Previous runtime functions repeatedly used `model::FindPT`, `FindLetter`, `FindDigit`, and `FindSymbol`, which linearly scan model vectors. This affects:

- priority queue initialization;
- PT work preparation during online guessing;
- probability recomputation when successor PTs are inserted.

v9 builds hash indexes after training:

- `PTKey -> PT id`
- `letter length -> segment id`
- `digit length -> segment id`
- `symbol length -> segment id`

Then runtime lookup uses average O(1) hash-table lookup instead of repeated linear scans.

## Recommended command

```cmd
build_windows.bat
mpiexec -n 1 hybrid_guess.exe 10000000 2147483647 8 ..\guessdata\Rockyou-singleLined-full.txt 1000000 omp_simd thread_train
```

Compare against v6:

- v6 best: `Total PCFG time = 6.549762s`, `Priority init time = 1.065517s`, `Online wall time = 0.407045s`.
- v9 should be judged mainly by `Priority init time`, `Online wall time`, `Total PCFG time`, `Generated`, and `Cracked`.

`fused_omp_simd thread_train` is still available if you want to combine runtime indexing with Generate-Hash Fusion.
