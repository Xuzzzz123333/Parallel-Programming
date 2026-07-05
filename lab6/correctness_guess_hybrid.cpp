#include "PCFG.h"
#include "guessing_gpu.h"
#include "md5_portable.h"
#include "md5_simd_x86.h"
#include "train_mpi.h"
#include "train_thread.h"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

#ifndef HYBRID_PT_BATCH_SIZE
#define HYBRID_PT_BATCH_SIZE 4
#endif

struct RuntimeConfig
{
    string mode = "hybrid";
    int omp_threads = 1;
    bool use_openmp = false;
    bool use_simd = true;
    bool use_cuda = true;
    bool use_fusion = false;
};

struct HybridStats
{
    long long generated = 0;
    long long cracked = 0;
    long long cpu_tasks = 0;
    long long gpu_tasks = 0;
    double compute_sec = 0.0;
    double generate_sec = 0.0;
    double hash_check_sec = 0.0;
    double gpu_h2d_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_d2h_ms = 0.0;
    double gpu_total_ms = 0.0;
};

// Runtime index for the trained model.  The original model lookup functions
// FindPT/FindLetter/FindDigit/FindSymbol linearly scan vectors.  That cost shows
// up in priority initialization and in online PT expansion.  This index keeps
// the trained model unchanged, but replaces repeated structural lookup with
// hash-table lookup in the runtime path.
struct ModelLookupIndex
{
    unordered_map<string, int> pt_index;
    unordered_map<int, int> letters_index;
    unordered_map<int, int> digits_index;
    unordered_map<int, int> symbols_index;
};

static string MakePTKeyRuntime(const PT &pt)
{
    string key;
    key.reserve(pt.content.size() * 8);
    for (const segment &seg : pt.content)
    {
        key += to_string(seg.type);
        key += ':';
        key += to_string(seg.length);
        key += '|';
    }
    return key;
}

static ModelLookupIndex BuildModelLookupIndex(const model &m)
{
    ModelLookupIndex idx;
    idx.pt_index.reserve(m.preterminals.size() * 2 + 1);
    idx.letters_index.reserve(m.letters.size() * 2 + 1);
    idx.digits_index.reserve(m.digits.size() * 2 + 1);
    idx.symbols_index.reserve(m.symbols.size() * 2 + 1);

    for (int i = 0; i < (int)m.preterminals.size(); ++i)
    {
        idx.pt_index.emplace(MakePTKeyRuntime(m.preterminals[i]), i);
    }
    for (int i = 0; i < (int)m.letters.size(); ++i) idx.letters_index.emplace(m.letters[i].length, i);
    for (int i = 0; i < (int)m.digits.size(); ++i) idx.digits_index.emplace(m.digits[i].length, i);
    for (int i = 0; i < (int)m.symbols.size(); ++i) idx.symbols_index.emplace(m.symbols[i].length, i);

    return idx;
}

static int FindPTIndexed(const ModelLookupIndex &idx, const PT &pt)
{
    auto it = idx.pt_index.find(MakePTKeyRuntime(pt));
    return it == idx.pt_index.end() ? -1 : it->second;
}

static int FindSegmentIndexed(const ModelLookupIndex &idx, const segment &seg)
{
    const unordered_map<int, int> *mp = nullptr;
    if (seg.type == 1) mp = &idx.letters_index;
    else if (seg.type == 2) mp = &idx.digits_index;
    else if (seg.type == 3) mp = &idx.symbols_index;
    else return -1;

    auto it = mp->find(seg.length);
    return it == mp->end() ? -1 : it->second;
}

static void CalProbLocal(model &m, const ModelLookupIndex &idx, PT &pt)
{
    pt.prob = pt.preterm_prob;
    int seg_pos = 0;

    for (int value_pos : pt.curr_indices)
    {
        if (seg_pos >= (int)pt.content.size()) break;
        const segment &seg = pt.content[seg_pos];
        int sid = FindSegmentIndexed(idx, seg);

        if (sid >= 0)
        {
            if (seg.type == 1 && value_pos >= 0 && value_pos < (int)m.letters[sid].ordered_freqs.size())
            {
                pt.prob *= m.letters[sid].ordered_freqs[value_pos];
                pt.prob /= m.letters[sid].total_freq;
            }
            else if (seg.type == 2 && value_pos >= 0 && value_pos < (int)m.digits[sid].ordered_freqs.size())
            {
                pt.prob *= m.digits[sid].ordered_freqs[value_pos];
                pt.prob /= m.digits[sid].total_freq;
            }
            else if (seg.type == 3 && value_pos >= 0 && value_pos < (int)m.symbols[sid].ordered_freqs.size())
            {
                pt.prob *= m.symbols[sid].ordered_freqs[value_pos];
                pt.prob /= m.symbols[sid].total_freq;
            }
        }
        seg_pos += 1;
    }
}

static const vector<string> *GetSegmentValues(model &m, const ModelLookupIndex &idx, const segment &seg)
{
    int id = FindSegmentIndexed(idx, seg);
    if (id < 0) return nullptr;

    if (seg.type == 1) return &m.letters[id].ordered_values;
    if (seg.type == 2) return &m.digits[id].ordered_values;
    if (seg.type == 3) return &m.symbols[id].ordered_values;
    return nullptr;
}

static priority_queue<PT, vector<PT>, PTCompare> InitPriorityLocal(model &m, const ModelLookupIndex &idx)
{
    priority_queue<PT, vector<PT>, PTCompare> pq;

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();
        pt.curr_indices.clear();
        pt.pivot = 0;

        bool ok = true;
        for (segment seg : pt.content)
        {
            const vector<string> *values = GetSegmentValues(m, idx, seg);
            if (values == nullptr || values->empty())
            {
                ok = false;
                break;
            }
            pt.max_indices.emplace_back((int)values->size());
            pt.curr_indices.emplace_back(0);
        }
        if (!ok) continue;

        int pt_id = FindPTIndexed(idx, pt);
        if (pt_id >= 0 && m.total_preterm > 0)
        {
            pt.preterm_prob = float(m.preterm_freq[pt_id]) / m.total_preterm;
        }
        else
        {
            pt.preterm_prob = 0.0f;
        }

        CalProbLocal(m, idx, pt);
        pq.push(pt);
    }

    return pq;
}

static vector<PT> NewPTsLocal(const PT &input)
{
    vector<PT> res;
    if (input.content.size() <= 1) return res;

    PT pt = input;
    int init_pivot = pt.pivot;

    for (int i = pt.pivot; i < (int)pt.curr_indices.size() - 1; i += 1)
    {
        pt.curr_indices[i] += 1;
        if (i < (int)pt.max_indices.size() && pt.curr_indices[i] < pt.max_indices[i])
        {
            pt.pivot = i;
            res.emplace_back(pt);
        }
        pt.curr_indices[i] -= 1;
    }

    pt.pivot = init_pivot;
    return res;
}

static bool BuildPTWork(model &m, const ModelLookupIndex &idx, const PT &pt, string &prefix, const vector<string> *&values)
{
    prefix.clear();
    values = nullptr;

    if (pt.content.empty()) return false;

    if (pt.content.size() > 1)
    {
        int seg_idx = 0;
        for (int value_pos : pt.curr_indices)
        {
            if (seg_idx == (int)pt.content.size() - 1) break;
            const vector<string> *seg_values = GetSegmentValues(m, idx, pt.content[seg_idx]);
            if (seg_values == nullptr || value_pos < 0 || value_pos >= (int)seg_values->size()) return false;
            prefix += (*seg_values)[value_pos];
            seg_idx += 1;
        }
    }

    int last_idx = (int)pt.content.size() - 1;
    values = GetSegmentValues(m, idx, pt.content[last_idx]);
    return values != nullptr && !values->empty();
}

static string MakeGuess(const string &prefix, const vector<string> &values, int value_idx, bool has_prefix)
{
    const string &val = values[value_idx];
    if (!has_prefix) return val;
    string result;
    result.reserve(prefix.size() + val.size());
    result.append(prefix);
    result.append(val);
    return result;
}

static unordered_set<string> LoadTestSet(const string &path, int limit)
{
    unordered_set<string> test_set;
    test_set.reserve((size_t)limit * 2);
    ifstream fin(path);
    if (!fin.is_open())
    {
        cerr << "Failed to open test file: " << path << endl;
        return test_set;
    }

    string pw;
    int count = 0;
    while (fin >> pw)
    {
        test_set.insert(pw);
        count += 1;
        if (count >= limit) break;
    }
    return test_set;
}

static long long HashCheckVector(
    const vector<string> &guesses,
    const unordered_set<string> &test_set,
    const RuntimeConfig &cfg)
{
    long long cracked = 0;

    if (cfg.use_simd)
    {
        const long long groups = (long long)guesses.size() / 4;

        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (long long g = 0; g < groups; ++g)
            {
                long long i = g * 4;
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 1]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 2]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 3]) != test_set.end()) cracked += 1;

                bit32 states[4][4];
                MD5HashSIMD4X86(
                    guesses[(size_t)i],
                    guesses[(size_t)i + 1],
                    guesses[(size_t)i + 2],
                    guesses[(size_t)i + 3],
                    states
                );
            }
        }
        else
        {
            for (long long g = 0; g < groups; ++g)
            {
                long long i = g * 4;
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 1]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 2]) != test_set.end()) cracked += 1;
                if (test_set.find(guesses[(size_t)i + 3]) != test_set.end()) cracked += 1;

                bit32 states[4][4];
                MD5HashSIMD4X86(
                    guesses[(size_t)i],
                    guesses[(size_t)i + 1],
                    guesses[(size_t)i + 2],
                    guesses[(size_t)i + 3],
                    states
                );
            }
        }

        const long long tail_start = groups * 4;
        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (long long i = tail_start; i < (long long)guesses.size(); ++i)
            {
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(guesses[(size_t)i], state);
            }
        }
        else
        {
            for (long long i = tail_start; i < (long long)guesses.size(); ++i)
            {
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(guesses[(size_t)i], state);
            }
        }
    }
    else
    {
        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (long long i = 0; i < (long long)guesses.size(); ++i)
            {
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(guesses[(size_t)i], state);
            }
        }
        else
        {
            for (long long i = 0; i < (long long)guesses.size(); ++i)
            {
                if (test_set.find(guesses[(size_t)i]) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(guesses[(size_t)i], state);
            }
        }
    }

    return cracked;
}


static HybridStats ProcessRangeCPUFused(
    const string &prefix,
    const vector<string> &values,
    int start_idx,
    int end_idx,
    bool has_prefix,
    const unordered_set<string> &test_set,
    const RuntimeConfig &cfg)
{
    HybridStats stats;
    if (start_idx < 0) start_idx = 0;
    if (end_idx > (int)values.size()) end_idx = (int)values.size();
    if (start_idx >= end_idx) return stats;


    stats.cpu_tasks = 1;
    stats.generated = end_idx - start_idx;
    const int count = end_idx - start_idx;

    double t0 = MPI_Wtime();
    long long cracked = 0;

    // Generate-Hash Fusion:
    // Do not materialize a vector<string> for the whole PT range.  Each worker
    // constructs a few local candidate strings, immediately checks them, and
    // immediately runs MD5/SIMD MD5.  This removes the intermediate candidate
    // container and the second pass over it.
    if (cfg.use_simd)
    {
        const int groups = count / 4;
        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (int g = 0; g < groups; ++g)
            {
                int base = start_idx + g * 4;
                string s0 = MakeGuess(prefix, values, base, has_prefix);
                string s1 = MakeGuess(prefix, values, base + 1, has_prefix);
                string s2 = MakeGuess(prefix, values, base + 2, has_prefix);
                string s3 = MakeGuess(prefix, values, base + 3, has_prefix);

                if (test_set.find(s0) != test_set.end()) cracked += 1;
                if (test_set.find(s1) != test_set.end()) cracked += 1;
                if (test_set.find(s2) != test_set.end()) cracked += 1;
                if (test_set.find(s3) != test_set.end()) cracked += 1;

                bit32 states[4][4];
                MD5HashSIMD4X86(s0, s1, s2, s3, states);
            }
        }
        else
        {
            for (int g = 0; g < groups; ++g)
            {
                int base = start_idx + g * 4;
                string s0 = MakeGuess(prefix, values, base, has_prefix);
                string s1 = MakeGuess(prefix, values, base + 1, has_prefix);
                string s2 = MakeGuess(prefix, values, base + 2, has_prefix);
                string s3 = MakeGuess(prefix, values, base + 3, has_prefix);

                if (test_set.find(s0) != test_set.end()) cracked += 1;
                if (test_set.find(s1) != test_set.end()) cracked += 1;
                if (test_set.find(s2) != test_set.end()) cracked += 1;
                if (test_set.find(s3) != test_set.end()) cracked += 1;

                bit32 states[4][4];
                MD5HashSIMD4X86(s0, s1, s2, s3, states);
            }
        }

        const int tail_start = start_idx + groups * 4;
        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (int i = tail_start; i < end_idx; ++i)
            {
                string s0 = MakeGuess(prefix, values, i, has_prefix);
                if (test_set.find(s0) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(s0, state);
            }
        }
        else
        {
            for (int i = tail_start; i < end_idx; ++i)
            {
                string s0 = MakeGuess(prefix, values, i, has_prefix);
                if (test_set.find(s0) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(s0, state);
            }
        }
    }
    else
    {
        if (cfg.use_openmp)
        {
#pragma omp parallel for schedule(static) reduction(+:cracked)
            for (int i = start_idx; i < end_idx; ++i)
            {
                string s0 = MakeGuess(prefix, values, i, has_prefix);
                if (test_set.find(s0) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(s0, state);
            }
        }
        else
        {
            for (int i = start_idx; i < end_idx; ++i)
            {
                string s0 = MakeGuess(prefix, values, i, has_prefix);
                if (test_set.find(s0) != test_set.end()) cracked += 1;
                bit32 state[4];
                MD5Hash(s0, state);
            }
        }
    }

    double t1 = MPI_Wtime();
    stats.cracked = cracked;
    stats.compute_sec = t1 - t0;
    // In fusion mode generation and hash/check happen in the same loop.  We
    // charge the fused loop to hash_check_sec to avoid pretending that there is
    // still a separate materialization phase.
    stats.generate_sec = 0.0;
    stats.hash_check_sec = t1 - t0;
    return stats;
}

static HybridStats ProcessRangeCPU(
    const string &prefix,
    const vector<string> &values,
    int start_idx,
    int end_idx,
    bool has_prefix,
    const unordered_set<string> &test_set,
    const RuntimeConfig &cfg)
{
    HybridStats stats;
    if (start_idx < 0) start_idx = 0;
    if (end_idx > (int)values.size()) end_idx = (int)values.size();
    if (start_idx >= end_idx) return stats;

    if (cfg.use_fusion)
    {
        return ProcessRangeCPUFused(prefix, values, start_idx, end_idx, has_prefix, test_set, cfg);
    }

    stats.cpu_tasks = 1;
    stats.generated = end_idx - start_idx;
    const int count = end_idx - start_idx;

    double t0 = MPI_Wtime();
    double t_gen0 = MPI_Wtime();

    vector<string> guesses((size_t)count);
    if (cfg.use_openmp)
    {
#pragma omp parallel for schedule(static)
        for (int off = 0; off < count; ++off)
        {
            guesses[(size_t)off] = MakeGuess(prefix, values, start_idx + off, has_prefix);
        }
    }
    else
    {
        for (int off = 0; off < count; ++off)
        {
            guesses[(size_t)off] = MakeGuess(prefix, values, start_idx + off, has_prefix);
        }
    }

    double t_gen1 = MPI_Wtime();
    double t_hash0 = MPI_Wtime();
    stats.cracked = HashCheckVector(guesses, test_set, cfg);
    double t1 = MPI_Wtime();

    stats.generate_sec = t_gen1 - t_gen0;
    stats.hash_check_sec = t1 - t_hash0;
    stats.compute_sec = t1 - t0;
    return stats;
}

static HybridStats ProcessRangeGPUThenCPUHash(
    const string &prefix,
    const vector<string> &values,
    int start_idx,
    int end_idx,
    const unordered_set<string> &test_set,
    const RuntimeConfig &cfg)
{
    HybridStats stats;
    if (start_idx < 0) start_idx = 0;
    if (end_idx > (int)values.size()) end_idx = (int)values.size();
    if (start_idx >= end_idx) return stats;

    stats.gpu_tasks = 1;
    stats.generated = end_idx - start_idx;

    double t0 = MPI_Wtime();
    double t_gen0 = MPI_Wtime();

    // Simple hybrid version: copy this rank's range into a sub-vector and
    // reuse the existing CUDA candidate generation function. This keeps the
    // kernel interface stable and makes CPU/GPU timing easy to compare.
    vector<string> sub_values;
    sub_values.reserve((size_t)(end_idx - start_idx));
    for (int i = start_idx; i < end_idx; ++i)
    {
        sub_values.emplace_back(values[(size_t)i]);
    }

    GPUGuessResult gpu_result = GenerateCandidatesGPU(prefix, sub_values);
    double t_gen1 = MPI_Wtime();

    stats.gpu_h2d_ms += gpu_result.h2d_ms;
    stats.gpu_kernel_ms += gpu_result.kernel_ms;
    stats.gpu_d2h_ms += gpu_result.d2h_ms;
    stats.gpu_total_ms += gpu_result.total_gpu_ms;

    double t_hash0 = MPI_Wtime();
    stats.cracked = HashCheckVector(gpu_result.guesses, test_set, cfg);
    double t1 = MPI_Wtime();

    stats.generate_sec = t_gen1 - t_gen0;
    stats.hash_check_sec = t1 - t_hash0;
    stats.compute_sec = t1 - t0;
    return stats;
}

static void PackPTToVector(const PT &pt, vector<int> &data)
{
    data.emplace_back((int)pt.content.size());
    for (const segment &seg : pt.content)
    {
        data.emplace_back(seg.type);
        data.emplace_back(seg.length);
    }

    data.emplace_back(pt.pivot);

    data.emplace_back((int)pt.curr_indices.size());
    for (int x : pt.curr_indices) data.emplace_back(x);

    data.emplace_back((int)pt.max_indices.size());
    for (int x : pt.max_indices) data.emplace_back(x);
}

static PT UnpackPTFromVector(const vector<int> &data, int &pos, float preterm_prob, float prob)
{
    PT pt;

    int content_size = data[pos++];
    for (int i = 0; i < content_size; ++i)
    {
        int type = data[pos++];
        int length = data[pos++];
        pt.content.emplace_back(segment(type, length));
    }

    pt.pivot = data[pos++];

    int curr_size = data[pos++];
    for (int i = 0; i < curr_size; ++i) pt.curr_indices.emplace_back(data[pos++]);

    int max_size = data[pos++];
    for (int i = 0; i < max_size; ++i) pt.max_indices.emplace_back(data[pos++]);

    pt.preterm_prob = preterm_prob;
    pt.prob = prob;
    return pt;
}

static void BcastPTBatch(vector<PT> &batch_pts, int rank)
{
    vector<int> packed_ints;
    vector<float> packed_probs;
    int int_size = 0;
    int prob_size = 0;

    if (rank == 0)
    {
        packed_ints.emplace_back((int)batch_pts.size());
        for (const PT &pt : batch_pts)
        {
            PackPTToVector(pt, packed_ints);
            packed_probs.emplace_back(pt.preterm_prob);
            packed_probs.emplace_back(pt.prob);
        }
        int_size = (int)packed_ints.size();
        prob_size = (int)packed_probs.size();
    }

    MPI_Bcast(&int_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&prob_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        packed_ints.resize(int_size);
        packed_probs.resize(prob_size);
    }

    MPI_Bcast(packed_ints.data(), int_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(packed_probs.data(), prob_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        int pos = 0;
        int count = packed_ints[pos++];
        batch_pts.clear();
        batch_pts.reserve(count);
        for (int i = 0; i < count; ++i)
        {
            float preterm_prob = packed_probs[2 * i];
            float prob = packed_probs[2 * i + 1];
            batch_pts.emplace_back(UnpackPTFromVector(packed_ints, pos, preterm_prob, prob));
        }
    }
}

static void AddStats(HybridStats &dst, const HybridStats &src)
{
    dst.generated += src.generated;
    dst.cracked += src.cracked;
    dst.cpu_tasks += src.cpu_tasks;
    dst.gpu_tasks += src.gpu_tasks;
    dst.compute_sec += src.compute_sec;
    dst.generate_sec += src.generate_sec;
    dst.hash_check_sec += src.hash_check_sec;
    dst.gpu_h2d_ms += src.gpu_h2d_ms;
    dst.gpu_kernel_ms += src.gpu_kernel_ms;
    dst.gpu_d2h_ms += src.gpu_d2h_ms;
    dst.gpu_total_ms += src.gpu_total_ms;
}

static RuntimeConfig BuildConfig(const string &mode, int requested_threads)
{
    RuntimeConfig cfg;
    cfg.mode = mode;
    cfg.omp_threads = max(1, requested_threads);

    if (mode == "serial")
    {
        cfg.omp_threads = 1;
        cfg.use_openmp = false;
        cfg.use_simd = false;
        cfg.use_cuda = false;
    }
    else if (mode == "simd")
    {
        cfg.omp_threads = 1;
        cfg.use_openmp = false;
        cfg.use_simd = true;
        cfg.use_cuda = false;
    }
    else if (mode == "omp_simd")
    {
        cfg.use_openmp = (cfg.omp_threads > 1);
        cfg.use_simd = true;
        cfg.use_cuda = false;
        cfg.use_fusion = false;
    }
    else if (mode == "fused_omp_simd")
    {
        cfg.use_openmp = (cfg.omp_threads > 1);
        cfg.use_simd = true;
        cfg.use_cuda = false;
        cfg.use_fusion = true;
    }
    else if (mode == "mpi_omp_simd")
    {
        cfg.use_openmp = (cfg.omp_threads > 1);
        cfg.use_simd = true;
        cfg.use_cuda = false;
    }
    else if (mode == "hybrid")
    {
        cfg.use_openmp = (cfg.omp_threads > 1);
        cfg.use_simd = true;
        cfg.use_cuda = true;
    }
    else
    {
        // Unknown mode: keep the full hybrid configuration but print the mode
        // name so the user can notice the typo in output.
        cfg.use_openmp = (cfg.omp_threads > 1);
        cfg.use_simd = true;
        cfg.use_cuda = true;
    }

    return cfg;
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const long long generate_limit = (argc >= 2) ? atoll(argv[1]) : 10000000LL;
    const int gpu_threshold = (argc >= 3) ? atoi(argv[2]) : 150000;
    const int requested_omp_threads = (argc >= 4) ? atoi(argv[3]) : 8;
    const string data_path = (argc >= 5) ? argv[4] : string("guessdata/Rockyou-singleLined-full.txt");
    const int test_limit = (argc >= 6) ? atoi(argv[5]) : 1000000;
    const string mode = (argc >= 7) ? argv[6] : string("hybrid");
    const string requested_train_mode = (argc >= 8) ? argv[7] : string("auto");

    RuntimeConfig cfg = BuildConfig(mode, requested_omp_threads);
    omp_set_num_threads(cfg.omp_threads);

    string train_mode = requested_train_mode;
    if (train_mode == "auto")
    {
        // Fastest practical default on a single Windows workstation: keep one MPI rank
        // and use local threaded training for the expensive Train phase.  MPI training
        // can still be requested explicitly with mpi_train.
        if (world_size == 1 && (cfg.mode == "simd" || cfg.mode == "omp_simd" || cfg.mode == "fused_omp_simd" || cfg.mode == "serial"))
        {
            train_mode = "thread_train";
        }
        else if ((cfg.mode == "mpi_omp_simd" || cfg.mode == "hybrid") && world_size > 1)
        {
            train_mode = "mpi_train";
        }
        else
        {
            train_mode = "serial_train";
        }
    }

    if (rank == 0)
    {
        cout << "Hybrid PCFG password guessing" << endl;
        cout << "Mode:" << cfg.mode << endl;
        cout << "MPI ranks:" << world_size << endl;
        cout << "OpenMP enabled:" << (cfg.use_openmp ? "YES" : "NO") << endl;
        cout << "OpenMP threads per rank:" << cfg.omp_threads << endl;
        cout << "SIMD enabled:" << (cfg.use_simd ? "YES" : "NO") << endl;
        cout << "CUDA enabled:" << (cfg.use_cuda ? "YES" : "NO") << endl;
        cout << "Generate-Hash fusion:" << (cfg.use_fusion ? "YES" : "NO") << endl;
        cout << "Runtime model lookup index:YES" << endl;
        cout << "Train mode:" << train_mode << endl;
        cout << "GPU threshold:" << gpu_threshold << endl;
        cout << "Generate limit:" << generate_limit << endl;
        cout << "Data path:" << data_path << endl;
    }

    model m;
    double train_start = MPI_Wtime();

    if (train_mode == "mpi_train")
    {
        train_mpi_model(m, data_path, rank, world_size);
    }
    else if (train_mode == "thread_train")
    {
        // Threaded training is intended for the single-rank fast workstation path.
        // If the user launches multiple ranks, every rank will build the same model;
        // keep non-root output muted to avoid log spam.
        std::ostringstream muted_train_output;
        std::streambuf *old_cout_buf = nullptr;
        if (rank != 0)
        {
            old_cout_buf = cout.rdbuf(muted_train_output.rdbuf());
        }
        train_thread_model(m, data_path, cfg.omp_threads);
        m.order();
        if (rank != 0 && old_cout_buf != nullptr)
        {
            cout.rdbuf(old_cout_buf);
        }
    }
    else
    {
        // serial_train keeps the original single-model training path.  Non-root
        // ranks are muted if the user intentionally runs this mode with -n > 1.
        std::ostringstream muted_train_output;
        std::streambuf *old_cout_buf = nullptr;
        if (rank != 0)
        {
            old_cout_buf = cout.rdbuf(muted_train_output.rdbuf());
        }
        m.train(data_path);
        m.order();
        if (rank != 0 && old_cout_buf != nullptr)
        {
            cout.rdbuf(old_cout_buf);
        }
    }

    double train_end = MPI_Wtime();
    double local_train_sec = train_end - train_start;
    double train_sec = 0.0;
    MPI_Reduce(&local_train_sec, &train_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double test_load_start = MPI_Wtime();
    unordered_set<string> test_set = LoadTestSet(data_path, test_limit);
    double test_load_end = MPI_Wtime();
    double local_test_load_sec = test_load_end - test_load_start;
    double test_load_sec = 0.0;
    MPI_Reduce(&local_test_load_sec, &test_load_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    priority_queue<PT, vector<PT>, PTCompare> pq;
    ModelLookupIndex lookup_index;
    double priority_start = MPI_Wtime();
    lookup_index = BuildModelLookupIndex(m);
    if (rank == 0)
    {
        pq = InitPriorityLocal(m, lookup_index);
        cout << "Priority queue initialized with runtime lookup index." << endl;
    }
    double priority_end = MPI_Wtime();
    double local_priority_sec = priority_end - priority_start;
    double priority_init_sec = 0.0;
    MPI_Reduce(&local_priority_sec, &priority_init_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double online_start = MPI_Wtime();

    long long history = 0;
    long long cracked_total = 0;
    long long next_report = 1000000;

    HybridStats local_all;

    while (true)
    {
        int active_count = 0;
        vector<PT> batch_pts;

        if (rank == 0)
        {
            while (!pq.empty() && history < generate_limit && active_count < HYBRID_PT_BATCH_SIZE)
            {
                PT curr = pq.top();
                pq.pop();
                batch_pts.emplace_back(curr);
                active_count += 1;
            }
        }

        MPI_Bcast(&active_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (active_count == 0) break;

        BcastPTBatch(batch_pts, rank);

        HybridStats local_batch;
        for (int b = 0; b < active_count; ++b)
        {
            string prefix;
            const vector<string> *values = nullptr;
            if (!BuildPTWork(m, lookup_index, batch_pts[b], prefix, values)) continue;

            const int last_idx = (int)batch_pts[b].content.size() - 1;
            int n = batch_pts[b].max_indices[last_idx];
            if (n < 0) n = 0;
            if (n > (int)values->size()) n = (int)values->size();

            int start_idx = (long long)n * rank / world_size;
            int end_idx = (long long)n * (rank + 1) / world_size;
            bool has_prefix = (batch_pts[b].content.size() > 1);

            HybridStats one;
            const int local_count = end_idx - start_idx;
            const bool use_gpu = (cfg.use_cuda && rank == 0 && local_count >= gpu_threshold && gpu_threshold > 0);

            try
            {
                if (use_gpu)
                {
                    one = ProcessRangeGPUThenCPUHash(prefix, *values, start_idx, end_idx, test_set, cfg);
                }
                else
                {
                    one = ProcessRangeCPU(prefix, *values, start_idx, end_idx, has_prefix, test_set, cfg);
                }
            }
            catch (const exception &e)
            {
                // If CUDA fails on rank 0, fall back to CPU so the hybrid program
                // can still finish and produce comparable results.
                if (rank == 0)
                {
                    cerr << "GPU path failed, falling back to CPU: " << e.what() << endl;
                }
                one = ProcessRangeCPU(prefix, *values, start_idx, end_idx, has_prefix, test_set, cfg);
            }

            AddStats(local_batch, one);
        }

        AddStats(local_all, local_batch);

        long long global_generated = 0;
        long long global_cracked = 0;
        MPI_Reduce(&local_batch.generated, &global_generated, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_batch.cracked, &global_cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            history += global_generated;
            cracked_total += global_cracked;

            while (history >= next_report)
            {
                cout << "Guesses processed: " << next_report << endl;
                next_report += 1000000;
            }

            for (PT &pt : batch_pts)
            {
                vector<PT> next_pts = NewPTsLocal(pt);
                for (PT next : next_pts)
                {
                    CalProbLocal(m, lookup_index, next);
                    pq.push(next);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double online_end = MPI_Wtime();
    double local_online_sec = online_end - online_start;

    HybridStats global_sum;
    double online_sec = 0.0;
    double compute_sec_max = 0.0;
    double generate_sec_max = 0.0;
    double hash_check_sec_max = 0.0;

    MPI_Reduce(&local_online_sec, &online_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.compute_sec, &compute_sec_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.generate_sec, &generate_sec_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.hash_check_sec, &hash_check_sec_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_all.generated, &global_sum.generated, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.cracked, &global_sum.cracked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.cpu_tasks, &global_sum.cpu_tasks, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.gpu_tasks, &global_sum.gpu_tasks, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_all.gpu_h2d_ms, &global_sum.gpu_h2d_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.gpu_kernel_ms, &global_sum.gpu_kernel_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.gpu_d2h_ms, &global_sum.gpu_d2h_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_all.gpu_total_ms, &global_sum.gpu_total_ms, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << fixed << setprecision(6);
        cout << "Generated:" << history << endl;
        cout << "Cracked:" << cracked_total << endl;
        cout << "Train time:" << train_sec << "seconds" << endl;
        cout << "Test-set load time:" << test_load_sec << "seconds" << endl;
        cout << "Priority init time:" << priority_init_sec << "seconds" << endl;
        cout << "Online wall time:" << online_sec << "seconds" << endl;
        cout << "Total PCFG time:" << (train_sec + priority_init_sec + online_sec) << "seconds" << endl;
        cout << "End-to-end measured time:" << (train_sec + test_load_sec + priority_init_sec + online_sec) << "seconds" << endl;
        cout << "Rank compute max time:" << compute_sec_max << "seconds" << endl;
        cout << "Candidate generation max time:" << generate_sec_max << "seconds" << endl;
        cout << "Hash/check max time:" << hash_check_sec_max << "seconds" << endl;
        cout << "MPI/control/wait time:" << max(0.0, online_sec - compute_sec_max) << "seconds" << endl;
        cout << "CPU task count:" << global_sum.cpu_tasks << endl;
        cout << "GPU task count:" << global_sum.gpu_tasks << endl;
        cout << "GPU H2D time:" << global_sum.gpu_h2d_ms / 1000.0 << "seconds" << endl;
        cout << "GPU kernel time:" << global_sum.gpu_kernel_ms / 1000.0 << "seconds" << endl;
        cout << "GPU D2H time:" << global_sum.gpu_d2h_ms / 1000.0 << "seconds" << endl;
        cout << "GPU event total time:" << global_sum.gpu_total_ms / 1000.0 << "seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
