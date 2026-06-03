#ifndef TRAIN_MPI_H
#define TRAIN_MPI_H

#include "PCFG.h"
#include <mpi.h>
#include <string>

using namespace std;

// MPI 多进程训练：每个 rank 训练部分数据，rank 0 合并并广播全局模型
void train_mpi(model &m, const string &path, int rank, int world_size);

#endif