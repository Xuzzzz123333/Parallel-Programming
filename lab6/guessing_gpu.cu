#include "guessing_gpu.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess)                                               \
        {                                                                       \
            std::ostringstream oss__;                                           \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                  << " : " << cudaGetErrorString(err__);                       \
            throw std::runtime_error(oss__.str());                              \
        }                                                                       \
    } while (0)

__global__ void GenerateCandidatesKernel(
    const char *prefix,
    int prefix_len,
    const char *value_chars,
    const int *value_offsets,
    const int *value_lens,
    int n_values,
    char *out_chars,
    int *out_lens,
    int max_password_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values)
    {
        return;
    }

    int value_offset = value_offsets[idx];
    int value_len = value_lens[idx];
    int total_len = prefix_len + value_len;

    if (total_len >= max_password_len)
    {
        out_lens[idx] = 0;
        return;
    }

    char *dst = out_chars + (long long)idx * max_password_len;
    for (int i = 0; i < prefix_len; i += 1)
    {
        dst[i] = prefix[i];
    }
    for (int i = 0; i < value_len; i += 1)
    {
        dst[prefix_len + i] = value_chars[value_offset + i];
    }
    dst[total_len] = '\0';
    out_lens[idx] = total_len;
}

__global__ void GenerateCandidatesBatchKernel(
    const char *prefix_chars,
    const int *candidate_prefix_offsets,
    const int *candidate_prefix_lens,
    const char *value_chars,
    const int *value_offsets,
    const int *value_lens,
    int total_candidates,
    char *out_chars,
    int *out_lens,
    int max_password_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_candidates)
    {
        return;
    }

    int prefix_offset = candidate_prefix_offsets[idx];
    int prefix_len = candidate_prefix_lens[idx];
    int value_offset = value_offsets[idx];
    int value_len = value_lens[idx];
    int total_len = prefix_len + value_len;

    if (total_len >= max_password_len)
    {
        out_lens[idx] = 0;
        return;
    }

    char *dst = out_chars + (long long)idx * max_password_len;
    const char *prefix_ptr = prefix_chars + prefix_offset;
    const char *value_ptr = value_chars + value_offset;

    for (int i = 0; i < prefix_len; i += 1)
    {
        dst[i] = prefix_ptr[i];
    }
    for (int i = 0; i < value_len; i += 1)
    {
        dst[prefix_len + i] = value_ptr[i];
    }
    dst[total_len] = '\0';
    out_lens[idx] = total_len;
}

static double ElapsedMs(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return (double)ms;
}

GPUGuessResult GenerateCandidatesGPU(
    const std::string &prefix,
    const std::vector<std::string> &values,
    int max_password_len,
    int block_size)
{
    GPUGuessResult result;
    if (values.empty())
    {
        return result;
    }
    if ((int)prefix.size() >= max_password_len)
    {
        throw std::runtime_error("prefix length must be smaller than max_password_len");
    }

    const int n_values = (int)values.size();
    std::vector<int> h_offsets(n_values);
    std::vector<int> h_lens(n_values);
    std::vector<char> h_value_chars;
    h_value_chars.reserve(values.size() * 8);

    for (int i = 0; i < n_values; i += 1)
    {
        h_offsets[i] = (int)h_value_chars.size();
        h_lens[i] = (int)values[i].size();
        h_value_chars.insert(h_value_chars.end(), values[i].begin(), values[i].end());
    }

    const size_t prefix_bytes = prefix.empty() ? 1 : prefix.size();
    const size_t value_chars_bytes = h_value_chars.empty() ? 1 : h_value_chars.size() * sizeof(char);
    const size_t offsets_bytes = h_offsets.size() * sizeof(int);
    const size_t lens_bytes = h_lens.size() * sizeof(int);
    const size_t out_chars_bytes = (size_t)n_values * (size_t)max_password_len * sizeof(char);
    const size_t out_lens_bytes = (size_t)n_values * sizeof(int);

    char *d_prefix = nullptr;
    char *d_value_chars = nullptr;
    int *d_offsets = nullptr;
    int *d_lens = nullptr;
    char *d_out_chars = nullptr;
    int *d_out_lens = nullptr;

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    CUDA_CHECK(cudaEventCreate(&t3));

    CUDA_CHECK(cudaMalloc(&d_prefix, prefix_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_chars, value_chars_bytes));
    CUDA_CHECK(cudaMalloc(&d_offsets, offsets_bytes));
    CUDA_CHECK(cudaMalloc(&d_lens, lens_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_chars, out_chars_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_lens, out_lens_bytes));


    CUDA_CHECK(cudaEventRecord(t0));
    if (!prefix.empty())
    {
        CUDA_CHECK(cudaMemcpy(d_prefix, prefix.data(), prefix.size(), cudaMemcpyHostToDevice));
    }
    if (!h_value_chars.empty())
    {
        CUDA_CHECK(cudaMemcpy(d_value_chars, h_value_chars.data(), value_chars_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens, h_lens.data(), lens_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    int grid_size = (n_values + block_size - 1) / block_size;
    GenerateCandidatesKernel<<<grid_size, block_size>>>(
        d_prefix, (int)prefix.size(), d_value_chars, d_offsets, d_lens,
        n_values, d_out_chars, d_out_lens, max_password_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaEventSynchronize(t2));


    std::vector<char> h_out_chars(out_chars_bytes);
    std::vector<int> h_out_lens(n_values);
    CUDA_CHECK(cudaMemcpy(h_out_chars.data(), d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_lens.data(), d_out_lens, out_lens_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));


    result.h2d_ms = ElapsedMs(t0, t1);
    result.kernel_ms = ElapsedMs(t1, t2);
    result.d2h_ms = ElapsedMs(t2, t3);
    result.total_gpu_ms = ElapsedMs(t0, t3);

    result.guesses.reserve(n_values);
    for (int i = 0; i < n_values; i += 1)
    {
        int len = h_out_lens[i];
        if (len <= 0)
        {
            result.guesses.emplace_back();
        }
        else
        {
            const char *ptr = h_out_chars.data() + (long long)i * max_password_len;
            result.guesses.emplace_back(ptr, ptr + len);
        }
    }

    CUDA_CHECK(cudaFree(d_prefix));
    CUDA_CHECK(cudaFree(d_value_chars));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lens));
    CUDA_CHECK(cudaFree(d_out_chars));
    CUDA_CHECK(cudaFree(d_out_lens));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2));
    CUDA_CHECK(cudaEventDestroy(t3));
    return result;
}

GPUBatchResult GenerateCandidatesBatchGPU(
    const std::vector<GPUBatchWork> &works,
    int max_password_len,
    int block_size)
{
    GPUBatchResult result;
    if (works.empty())
    {
        return result;
    }

    const int pt_count = (int)works.size();
    result.pt_offsets.assign(pt_count, 0);
    result.pt_counts.assign(pt_count, 0);

    std::vector<char> h_prefix_chars;
    std::vector<char> h_value_chars;
    std::vector<int> h_candidate_prefix_offsets;
    std::vector<int> h_candidate_prefix_lens;
    std::vector<int> h_value_offsets;
    std::vector<int> h_value_lens;

    int total_candidates = 0;
    for (int pt = 0; pt < pt_count; pt += 1)
    {
        const std::string &prefix = works[pt].prefix;
        const std::vector<std::string> *values = works[pt].values;
        if (values == nullptr || values->empty())
        {
            continue;
        }
        if ((int)prefix.size() >= max_password_len)
        {
            throw std::runtime_error("prefix length must be smaller than max_password_len");
        }

        int prefix_offset = (int)h_prefix_chars.size();
        int prefix_len = (int)prefix.size();
        h_prefix_chars.insert(h_prefix_chars.end(), prefix.begin(), prefix.end());

        result.pt_offsets[pt] = total_candidates;
        result.pt_counts[pt] = (int)values->size();

        for (const std::string &v : *values)
        {
            h_candidate_prefix_offsets.push_back(prefix_offset);
            h_candidate_prefix_lens.push_back(prefix_len);
            h_value_offsets.push_back((int)h_value_chars.size());
            h_value_lens.push_back((int)v.size());
            h_value_chars.insert(h_value_chars.end(), v.begin(), v.end());
            total_candidates += 1;
        }
    }

    if (total_candidates == 0)
    {
        return result;
    }


    const size_t prefix_chars_bytes = h_prefix_chars.empty() ? 1 : h_prefix_chars.size() * sizeof(char);
    const size_t candidate_prefix_offsets_bytes = h_candidate_prefix_offsets.size() * sizeof(int);
    const size_t candidate_prefix_lens_bytes = h_candidate_prefix_lens.size() * sizeof(int);
    const size_t value_chars_bytes = h_value_chars.empty() ? 1 : h_value_chars.size() * sizeof(char);
    const size_t value_offsets_bytes = h_value_offsets.size() * sizeof(int);
    const size_t value_lens_bytes = h_value_lens.size() * sizeof(int);
    const size_t out_chars_bytes = (size_t)total_candidates * (size_t)max_password_len * sizeof(char);
    const size_t out_lens_bytes = (size_t)total_candidates * sizeof(int);

    char *d_prefix_chars = nullptr;
    int *d_candidate_prefix_offsets = nullptr;
    int *d_candidate_prefix_lens = nullptr;
    char *d_value_chars = nullptr;
    int *d_value_offsets = nullptr;
    int *d_value_lens = nullptr;
    char *d_out_chars = nullptr;
    int *d_out_lens = nullptr;

    cudaEvent_t t0, t1, t2, t3;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));
    CUDA_CHECK(cudaEventCreate(&t3));


    CUDA_CHECK(cudaMalloc(&d_prefix_chars, prefix_chars_bytes));
    CUDA_CHECK(cudaMalloc(&d_candidate_prefix_offsets, candidate_prefix_offsets_bytes));
    CUDA_CHECK(cudaMalloc(&d_candidate_prefix_lens, candidate_prefix_lens_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_chars, value_chars_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_offsets, value_offsets_bytes));
    CUDA_CHECK(cudaMalloc(&d_value_lens, value_lens_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_chars, out_chars_bytes));
    CUDA_CHECK(cudaMalloc(&d_out_lens, out_lens_bytes));


    CUDA_CHECK(cudaEventRecord(t0));
    if (!h_prefix_chars.empty())
    {
        CUDA_CHECK(cudaMemcpy(d_prefix_chars, h_prefix_chars.data(), prefix_chars_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_candidate_prefix_offsets, h_candidate_prefix_offsets.data(), candidate_prefix_offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidate_prefix_lens, h_candidate_prefix_lens.data(), candidate_prefix_lens_bytes, cudaMemcpyHostToDevice));
    if (!h_value_chars.empty())
    {
        CUDA_CHECK(cudaMemcpy(d_value_chars, h_value_chars.data(), value_chars_bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_value_offsets, h_value_offsets.data(), value_offsets_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_value_lens, h_value_lens.data(), value_lens_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));


    int grid_size = (total_candidates + block_size - 1) / block_size;
    GenerateCandidatesBatchKernel<<<grid_size, block_size>>>(
        d_prefix_chars, d_candidate_prefix_offsets, d_candidate_prefix_lens,
        d_value_chars, d_value_offsets, d_value_lens,
        total_candidates, d_out_chars, d_out_lens, max_password_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaEventSynchronize(t2));


    std::vector<char> h_out_chars(out_chars_bytes);
    std::vector<int> h_out_lens(total_candidates);
    CUDA_CHECK(cudaMemcpy(h_out_chars.data(), d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_lens.data(), d_out_lens, out_lens_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t3));
    CUDA_CHECK(cudaEventSynchronize(t3));


    result.h2d_ms = ElapsedMs(t0, t1);
    result.kernel_ms = ElapsedMs(t1, t2);
    result.d2h_ms = ElapsedMs(t2, t3);
    result.total_gpu_ms = ElapsedMs(t0, t3);

    result.guesses.reserve(total_candidates);
    for (int i = 0; i < total_candidates; i += 1)
    {
        int len = h_out_lens[i];
        if (len <= 0)
        {
            result.guesses.emplace_back();
        }
        else
        {
            const char *ptr = h_out_chars.data() + (long long)i * max_password_len;
            result.guesses.emplace_back(ptr, ptr + len);
        }
    }


    CUDA_CHECK(cudaFree(d_prefix_chars));
    CUDA_CHECK(cudaFree(d_candidate_prefix_offsets));
    CUDA_CHECK(cudaFree(d_candidate_prefix_lens));
    CUDA_CHECK(cudaFree(d_value_chars));
    CUDA_CHECK(cudaFree(d_value_offsets));
    CUDA_CHECK(cudaFree(d_value_lens));
    CUDA_CHECK(cudaFree(d_out_chars));
    CUDA_CHECK(cudaFree(d_out_lens));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2));
    CUDA_CHECK(cudaEventDestroy(t3));

    return result;
}

struct AsyncBatchInternal
{
    char *d_prefix_chars = nullptr;
    int *d_candidate_prefix_offsets = nullptr;
    int *d_candidate_prefix_lens = nullptr;
    char *d_value_chars = nullptr;
    int *d_value_offsets = nullptr;
    int *d_value_lens = nullptr;
    char *d_out_chars = nullptr;
    int *d_out_lens = nullptr;

    char *h_out_chars = nullptr;
    int *h_out_lens = nullptr;

    size_t out_chars_bytes = 0;
    size_t out_lens_bytes = 0;

    cudaStream_t stream = nullptr;
    cudaEvent_t t0 = nullptr;
    cudaEvent_t t1 = nullptr;
    cudaEvent_t t2 = nullptr;
    cudaEvent_t t3 = nullptr;
};

static void FreeAsyncBatchInternal(AsyncBatchInternal *r)
{
    if (r == nullptr)
    {
        return;
    }
    if (r->d_prefix_chars) cudaFree(r->d_prefix_chars);
    if (r->d_candidate_prefix_offsets) cudaFree(r->d_candidate_prefix_offsets);
    if (r->d_candidate_prefix_lens) cudaFree(r->d_candidate_prefix_lens);
    if (r->d_value_chars) cudaFree(r->d_value_chars);
    if (r->d_value_offsets) cudaFree(r->d_value_offsets);
    if (r->d_value_lens) cudaFree(r->d_value_lens);
    if (r->d_out_chars) cudaFree(r->d_out_chars);
    if (r->d_out_lens) cudaFree(r->d_out_lens);
    if (r->h_out_chars) cudaFreeHost(r->h_out_chars);
    if (r->h_out_lens) cudaFreeHost(r->h_out_lens);
    if (r->t0) cudaEventDestroy(r->t0);
    if (r->t1) cudaEventDestroy(r->t1);
    if (r->t2) cudaEventDestroy(r->t2);
    if (r->t3) cudaEventDestroy(r->t3);
    if (r->stream) cudaStreamDestroy(r->stream);
    delete r;
}

GPUBatchAsyncJob StartCandidatesBatchGPUAsync(
    const std::vector<GPUBatchWork> &works,
    int max_password_len,
    int block_size)
{
    GPUBatchAsyncJob job;
    if (works.empty())
    {
        return job;
    }

    const int pt_count = (int)works.size();
    job.pt_offsets.assign(pt_count, 0);
    job.pt_counts.assign(pt_count, 0);
    job.max_password_len = max_password_len;

    std::vector<char> h_prefix_chars;
    std::vector<char> h_value_chars;
    std::vector<int> h_candidate_prefix_offsets;
    std::vector<int> h_candidate_prefix_lens;
    std::vector<int> h_value_offsets;
    std::vector<int> h_value_lens;

    int total_candidates = 0;
    for (int pt = 0; pt < pt_count; pt += 1)
    {
        const std::string &prefix = works[pt].prefix;
        const std::vector<std::string> *values = works[pt].values;
        if (values == nullptr || values->empty())
        {
            continue;
        }
        if ((int)prefix.size() >= max_password_len)
        {
            throw std::runtime_error("prefix length must be smaller than max_password_len");
        }

        int prefix_offset = (int)h_prefix_chars.size();
        int prefix_len = (int)prefix.size();
        h_prefix_chars.insert(h_prefix_chars.end(), prefix.begin(), prefix.end());

        job.pt_offsets[pt] = total_candidates;
        job.pt_counts[pt] = (int)values->size();

        for (const std::string &v : *values)
        {
            h_candidate_prefix_offsets.push_back(prefix_offset);
            h_candidate_prefix_lens.push_back(prefix_len);
            h_value_offsets.push_back((int)h_value_chars.size());
            h_value_lens.push_back((int)v.size());
            h_value_chars.insert(h_value_chars.end(), v.begin(), v.end());
            total_candidates += 1;
        }
    }

    job.total_candidates = total_candidates;
    if (total_candidates == 0)
    {
        return job;
    }

    const size_t prefix_chars_bytes = h_prefix_chars.empty() ? 1 : h_prefix_chars.size() * sizeof(char);
    const size_t candidate_prefix_offsets_bytes = h_candidate_prefix_offsets.size() * sizeof(int);
    const size_t candidate_prefix_lens_bytes = h_candidate_prefix_lens.size() * sizeof(int);
    const size_t value_chars_bytes = h_value_chars.empty() ? 1 : h_value_chars.size() * sizeof(char);
    const size_t value_offsets_bytes = h_value_offsets.size() * sizeof(int);
    const size_t value_lens_bytes = h_value_lens.size() * sizeof(int);
    const size_t out_chars_bytes = (size_t)total_candidates * (size_t)max_password_len * sizeof(char);
    const size_t out_lens_bytes = (size_t)total_candidates * sizeof(int);

    AsyncBatchInternal *r = new AsyncBatchInternal();
    r->out_chars_bytes = out_chars_bytes;
    r->out_lens_bytes = out_lens_bytes;
    job.internal = r;

    try
    {
        CUDA_CHECK(cudaStreamCreate(&r->stream));
        CUDA_CHECK(cudaEventCreate(&r->t0));
        CUDA_CHECK(cudaEventCreate(&r->t1));
        CUDA_CHECK(cudaEventCreate(&r->t2));
        CUDA_CHECK(cudaEventCreate(&r->t3));

        CUDA_CHECK(cudaMalloc(&r->d_prefix_chars, prefix_chars_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_candidate_prefix_offsets, candidate_prefix_offsets_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_candidate_prefix_lens, candidate_prefix_lens_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_value_chars, value_chars_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_value_offsets, value_offsets_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_value_lens, value_lens_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_out_chars, out_chars_bytes));
        CUDA_CHECK(cudaMalloc(&r->d_out_lens, out_lens_bytes));
        CUDA_CHECK(cudaMallocHost(&r->h_out_chars, out_chars_bytes));
        CUDA_CHECK(cudaMallocHost(&r->h_out_lens, out_lens_bytes));

        CUDA_CHECK(cudaEventRecord(r->t0, r->stream));
        if (!h_prefix_chars.empty())
        {
            CUDA_CHECK(cudaMemcpyAsync(r->d_prefix_chars, h_prefix_chars.data(), prefix_chars_bytes, cudaMemcpyHostToDevice, r->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(r->d_candidate_prefix_offsets, h_candidate_prefix_offsets.data(), candidate_prefix_offsets_bytes, cudaMemcpyHostToDevice, r->stream));
        CUDA_CHECK(cudaMemcpyAsync(r->d_candidate_prefix_lens, h_candidate_prefix_lens.data(), candidate_prefix_lens_bytes, cudaMemcpyHostToDevice, r->stream));
        if (!h_value_chars.empty())
        {
            CUDA_CHECK(cudaMemcpyAsync(r->d_value_chars, h_value_chars.data(), value_chars_bytes, cudaMemcpyHostToDevice, r->stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(r->d_value_offsets, h_value_offsets.data(), value_offsets_bytes, cudaMemcpyHostToDevice, r->stream));
        CUDA_CHECK(cudaMemcpyAsync(r->d_value_lens, h_value_lens.data(), value_lens_bytes, cudaMemcpyHostToDevice, r->stream));
        CUDA_CHECK(cudaEventRecord(r->t1, r->stream));

        // The input packing vectors are local to this function. Synchronize H2D before they go out of scope.
        CUDA_CHECK(cudaEventSynchronize(r->t1));

        int grid_size = (total_candidates + block_size - 1) / block_size;
        GenerateCandidatesBatchKernel<<<grid_size, block_size, 0, r->stream>>>(
            r->d_prefix_chars, r->d_candidate_prefix_offsets, r->d_candidate_prefix_lens,
            r->d_value_chars, r->d_value_offsets, r->d_value_lens,
            total_candidates, r->d_out_chars, r->d_out_lens, max_password_len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(r->t2, r->stream));
        CUDA_CHECK(cudaMemcpyAsync(r->h_out_chars, r->d_out_chars, out_chars_bytes, cudaMemcpyDeviceToHost, r->stream));
        CUDA_CHECK(cudaMemcpyAsync(r->h_out_lens, r->d_out_lens, out_lens_bytes, cudaMemcpyDeviceToHost, r->stream));
        CUDA_CHECK(cudaEventRecord(r->t3, r->stream));
    }
    catch (...)
    {
        FreeAsyncBatchInternal(r);
        job.internal = nullptr;
        throw;
    }

    return job;
}

GPUBatchResult FinishCandidatesBatchGPUAsync(GPUBatchAsyncJob &job)
{
    GPUBatchResult result;
    result.pt_offsets = job.pt_offsets;
    result.pt_counts = job.pt_counts;

    AsyncBatchInternal *r = (AsyncBatchInternal *)job.internal;
    if (r == nullptr || job.total_candidates == 0)
    {
        return result;
    }

    CUDA_CHECK(cudaEventSynchronize(r->t3));

    result.h2d_ms = ElapsedMs(r->t0, r->t1);
    result.kernel_ms = ElapsedMs(r->t1, r->t2);
    result.d2h_ms = ElapsedMs(r->t2, r->t3);
    result.total_gpu_ms = ElapsedMs(r->t0, r->t3);

    result.guesses.reserve(job.total_candidates);
    for (int i = 0; i < job.total_candidates; i += 1)
    {
        int len = r->h_out_lens[i];
        if (len <= 0)
        {
            result.guesses.emplace_back();
        }
        else
        {
            const char *ptr = r->h_out_chars + (long long)i * job.max_password_len;
            result.guesses.emplace_back(ptr, ptr + len);
        }
    }

    FreeAsyncBatchInternal(r);
    job.internal = nullptr;
    job.guesses.clear();
    job.total_candidates = 0;
    return result;
}
