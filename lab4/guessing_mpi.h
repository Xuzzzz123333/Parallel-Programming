#ifndef GUESSING_MPI_H
#define GUESSING_MPI_H

#include "PCFG.h"
#include <mpi.h>
#include <string>
#include <unordered_set>

using namespace std;

struct MPILocalResult
{
    long long generated;
    long long cracked;
    double generate_time;
    double hash_time;
    double compute_time;
};

// rank 0 将当前 PT 广播给所有进程
void BcastPT(PT &pt, int rank);

// 基础 MPI 版本：所有 rank 对同一个 PT 的最后一个 segment 分段生成并哈希
MPILocalResult GenerateAndHashPTMPI(
    model &m,
    const PT &pt,
    const unordered_set<string> &test_set,
    int rank,
    int world_size
);

// 进阶 MPI 版本：只处理一个 PT 中指定的 value_idx 区间
MPILocalResult GenerateAndHashPTRange(
    model &m,
    const PT &pt,
    const unordered_set<string> &test_set,
    int start_idx,
    int end_idx
);

#endif