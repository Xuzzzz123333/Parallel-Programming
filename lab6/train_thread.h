#ifndef TRAIN_THREAD_H
#define TRAIN_THREAD_H

#include "PCFG.h"
#include <string>

void train_thread_model(model &m, const std::string &path, int thread_count);

#endif
