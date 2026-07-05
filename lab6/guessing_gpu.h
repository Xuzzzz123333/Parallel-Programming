#ifndef GUESSING_GPU_H
#define GUESSING_GPU_H

#include <string>
#include <vector>

struct GPUGuessResult
{
    std::vector<std::string> guesses;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_gpu_ms = 0.0;
};

struct GPUBatchWork
{
    std::string prefix;
    const std::vector<std::string> *values = nullptr;
};

struct GPUBatchResult
{
    std::vector<std::string> guesses;
    std::vector<int> pt_offsets;
    std::vector<int> pt_counts;
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_gpu_ms = 0.0;
};

// Basic version: generate candidates for one PT:
// candidate[i] = prefix + values[i].
GPUGuessResult GenerateCandidatesGPU(
    const std::string &prefix,
    const std::vector<std::string> &values,
    int max_password_len = 64,
    int block_size = 256
);

// Advanced requirement 1: batch multiple PTs in one GPU launch.
// All works are still generated on GPU; this function does not implement
// CPU/GPU threshold scheduling or CPU/GPU overlap.
GPUBatchResult GenerateCandidatesBatchGPU(
    const std::vector<GPUBatchWork> &works,
    int max_password_len = 64,
    int block_size = 256
);


struct GPUBatchAsyncJob
{
    std::vector<int> pt_offsets;
    std::vector<int> pt_counts;
    std::vector<std::string> guesses;

    int total_candidates = 0;
    int max_password_len = 64;

    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_gpu_ms = 0.0;

    // Opaque CUDA resources. They are managed by Start/Finish functions.
    void *internal = nullptr;
};

// Advanced requirement 2: asynchronous GPU batch generation.
// Start schedules GPU generation and returns after H2D submission/launch;
// Finish waits for completion, reconstructs strings, and releases resources.
GPUBatchAsyncJob StartCandidatesBatchGPUAsync(
    const std::vector<GPUBatchWork> &works,
    int max_password_len = 64,
    int block_size = 256
);

GPUBatchResult FinishCandidatesBatchGPUAsync(GPUBatchAsyncJob &job);

#endif
