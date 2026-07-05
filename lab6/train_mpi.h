#ifndef TRAIN_MPI_H
#define TRAIN_MPI_H

#include "PCFG.h"
#include <string>

void train_mpi_model(model &m, const std::string &path, int rank, int world_size);

#endif
