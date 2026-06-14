#include "PCFG.h"
#include "guessing_gpu.h"
#include "md5_portable.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

struct BatchStats
{
    double train_sec = 0.0;
    double generation_wall_sec = 0.0;
    double hash_sec = 0.0;
    double priority_sec = 0.0;

    double gpu_h2d_ms = 0.0;
    double gpu_kernel_ms = 0.0;
    double gpu_d2h_ms = 0.0;
    double gpu_total_ms = 0.0;

    long long processed_pt_count = 0;
    long long gpu_batch_count = 0;
    long long generated = 0;
    long long cracked = 0;
};

static double SecondsBetween(system_clock::time_point a, system_clock::time_point b)
{
    return duration_cast<microseconds>(b - a).count() / 1000000.0;
}

static void CalProbLocal(model &m, PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (index >= (int)pt.content.size())
        {
            break;
        }

        if (pt.content[index].type == 1)
        {
            int sid = m.FindLetter(pt.content[index]);
            if (sid >= 0 && idx >= 0 && idx < (int)m.letters[sid].ordered_freqs.size())
            {
                pt.prob *= m.letters[sid].ordered_freqs[idx];
                pt.prob /= m.letters[sid].total_freq;
            }
        }
        else if (pt.content[index].type == 2)
        {
            int sid = m.FindDigit(pt.content[index]);
            if (sid >= 0 && idx >= 0 && idx < (int)m.digits[sid].ordered_freqs.size())
            {
                pt.prob *= m.digits[sid].ordered_freqs[idx];
                pt.prob /= m.digits[sid].total_freq;
            }
        }
        else if (pt.content[index].type == 3)
        {
            int sid = m.FindSymbol(pt.content[index]);
            if (sid >= 0 && idx >= 0 && idx < (int)m.symbols[sid].ordered_freqs.size())
            {
                pt.prob *= m.symbols[sid].ordered_freqs[idx];
                pt.prob /= m.symbols[sid].total_freq;
            }
        }

        index += 1;
    }
}

static const vector<string> *GetSegmentValues(model &m, const segment &seg)
{
    if (seg.type == 1)
    {
        int id = m.FindLetter(seg);
        return id >= 0 ? &m.letters[id].ordered_values : nullptr;
    }
    if (seg.type == 2)
    {
        int id = m.FindDigit(seg);
        return id >= 0 ? &m.digits[id].ordered_values : nullptr;
    }
    if (seg.type == 3)
    {
        int id = m.FindSymbol(seg);
        return id >= 0 ? &m.symbols[id].ordered_values : nullptr;
    }
    return nullptr;
}

static priority_queue<PT, vector<PT>, PTCompare> InitPriority(model &m)
{
    priority_queue<PT, vector<PT>, PTCompare> pq;

    for (PT pt : m.ordered_pts)
    {
        pt.max_indices.clear();
        pt.curr_indices.clear();
        pt.pivot = 0;

        for (segment seg : pt.content)
        {
            const vector<string> *values = GetSegmentValues(m, seg);
            int value_count = values ? (int)values->size() : 0;
            pt.max_indices.emplace_back(value_count);
            pt.curr_indices.emplace_back(0);
        }

        int pt_id = m.FindPT(pt);
        if (pt_id >= 0 && m.total_preterm > 0)
        {
            pt.preterm_prob = float(m.preterm_freq[pt_id]) / m.total_preterm;
        }

        CalProbLocal(m, pt);
        pq.push(pt);
    }

    return pq;
}

static vector<PT> NewPTsLocal(const PT &input)
{
    vector<PT> res;
    if (input.content.size() == 1)
    {
        return res;
    }

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

static bool BuildPTWork(model &m, const PT &pt, string &prefix, const vector<string> *&values)
{
    prefix.clear();
    values = nullptr;

    if (pt.content.empty())
    {
        return false;
    }

    if (pt.content.size() > 1)
    {
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (seg_idx == (int)pt.content.size() - 1)
            {
                break;
            }

            const vector<string> *seg_values = GetSegmentValues(m, pt.content[seg_idx]);
            if (seg_values == nullptr || idx < 0 || idx >= (int)seg_values->size())
            {
                return false;
            }

            prefix += (*seg_values)[idx];
            seg_idx += 1;
        }
    }

    int last_idx = (int)pt.content.size() - 1;
    values = GetSegmentValues(m, pt.content[last_idx]);
    return values != nullptr && !values->empty();
}

static unordered_set<string> LoadTestSet(const string &path, int limit)
{
    unordered_set<string> test_set;
    ifstream fin(path);
    string pw;
    int count = 0;

    if (!fin.is_open())
    {
        cerr << "Failed to open test file: " << path << endl;
        return test_set;
    }

    while (fin >> pw)
    {
        test_set.insert(pw);
        count += 1;
        if (count >= limit)
        {
            break;
        }
    }

    return test_set;
}

static bool VerifyBatchSamples(
    const vector<GPUBatchWork> &works,
    const GPUBatchResult &batch_result,
    int sample_per_pt)
{
    for (int pt = 0; pt < (int)works.size(); pt += 1)
    {
        if (works[pt].values == nullptr || works[pt].values->empty())
        {
            continue;
        }
        int offset = batch_result.pt_offsets[pt];
        int count = batch_result.pt_counts[pt];
        int n = min(sample_per_pt, count);
        for (int i = 0; i < n; i += 1)
        {
            string expected = works[pt].prefix + (*(works[pt].values))[i];
            const string &actual = batch_result.guesses[offset + i];
            if (expected != actual)
            {
                cerr << "Mismatch at PT " << pt << ", sample " << i << endl;
                cerr << "expected: " << expected << endl;
                cerr << "actual  : " << actual << endl;
                return false;
            }
        }
    }
    return true;
}

static volatile bit32 g_md5_sink = 0;

struct HashLocalResult
{
    long long cracked = 0;
    bit32 sink = 0;
};

static void HashAndCheck(
    const vector<string> &guesses,
    const unordered_set<string> &test_set,
    BatchStats &stats)
{
    auto start_hash = system_clock::now();

    bit32 state[4];
    for (const string &pw : guesses)
    {
        if (test_set.find(pw) != test_set.end())
        {
            stats.cracked += 1;
        }

        MD5Hash(pw, state);
        g_md5_sink ^= state[0];
    }

    auto end_hash = system_clock::now();
    stats.hash_sec += SecondsBetween(start_hash, end_hash);
}


static vector<string> HashAndCheckReturn(
    const vector<string> &guesses,
    const unordered_set<string> &test_set,
    BatchStats &stats,
    double *hash_time_sec = nullptr)
{
    auto start_hash = system_clock::now();

    bit32 state[4];
    for (const string &pw : guesses)
    {
        if (test_set.find(pw) != test_set.end())
        {
            stats.cracked += 1;
        }

        MD5Hash(pw, state);
        g_md5_sink ^= state[0];
    }

    auto end_hash = system_clock::now();
    double sec = SecondsBetween(start_hash, end_hash);
    stats.hash_sec += sec;
    if (hash_time_sec != nullptr)
    {
        *hash_time_sec = sec;
    }
    return vector<string>();
}

static void HashAndCheckParallel(
    const vector<string> &guesses,
    const unordered_set<string> &test_set,
    BatchStats &stats,
    int cpu_threads,
    double *hash_time_sec = nullptr)
{
    auto start_hash = system_clock::now();

    if (cpu_threads <= 1 || guesses.size() < 4096)
    {
        HashAndCheckReturn(guesses, test_set, stats, hash_time_sec);
        return;
    }

    int n_threads = cpu_threads;
    int n = (int)guesses.size();
    if (n_threads > n)
    {
        n_threads = n;
    }

    vector<HashLocalResult> locals(n_threads);
    vector<thread> workers;
    workers.reserve(n_threads);

    for (int t = 0; t < n_threads; t += 1)
    {
        int begin = (long long)n * t / n_threads;
        int end = (long long)n * (t + 1) / n_threads;

        workers.emplace_back([begin, end, t, &guesses, &test_set, &locals]() {
            bit32 state[4];
            long long local_cracked = 0;
            bit32 local_sink = 0;

            for (int i = begin; i < end; i += 1)
            {
                const string &pw = guesses[i];
                if (test_set.find(pw) != test_set.end())
                {
                    local_cracked += 1;
                }

                MD5Hash(pw, state);
                local_sink ^= state[0];
            }

            locals[t].cracked = local_cracked;
            locals[t].sink = local_sink;
        });
    }

    for (thread &worker : workers)
    {
        worker.join();
    }

    long long cracked_sum = 0;
    bit32 sink_sum = 0;
    for (const HashLocalResult &r : locals)
    {
        cracked_sum += r.cracked;
        sink_sum ^= r.sink;
    }

    stats.cracked += cracked_sum;
    g_md5_sink ^= sink_sum;

    auto end_hash = system_clock::now();
    double sec = SecondsBetween(start_hash, end_hash);
    stats.hash_sec += sec;
    if (hash_time_sec != nullptr)
    {
        *hash_time_sec = sec;
    }
}

int main(int argc, char **argv)
{
    string data_path = "guessdata/Rockyou-singleLined-full.txt";
    long long generate_limit = 10000000;
    int batch_pt_count = 4;
    long long max_batch_candidates = 1000000;
    int cpu_threads = (int)thread::hardware_concurrency();
    if (cpu_threads <= 0)
    {
        cpu_threads = 4;
    }

    if (argc >= 2)
    {
        generate_limit = atoll(argv[1]);
    }
    if (argc >= 3)
    {
        batch_pt_count = atoi(argv[2]);
        if (batch_pt_count <= 0)
        {
            batch_pt_count = 1;
        }
    }
    if (argc >= 4)
    {
        max_batch_candidates = atoll(argv[3]);
        if (max_batch_candidates <= 0)
        {
            max_batch_candidates = 0;
        }
    }
    if (argc >= 5)
    {
        cpu_threads = atoi(argv[4]);
        if (cpu_threads <= 0)
        {
            cpu_threads = 1;
        }
    }
    if (argc >= 6)
    {
        data_path = argv[5];
    }

    cout << "=== Advanced 2 Plus: CUDA Batch Generation with CPU/GPU Overlap and CPU Parallel Hash ===" << endl;
    cout << "Train/test data: " << data_path << endl;
    cout << "Generate limit: " << generate_limit << endl;
    cout << "Advanced policy 2: while current GPU batch is in flight, CPU hashes/checks the previous batch." << endl;
    cout << "Batch PT count: " << batch_pt_count << endl;
    cout << "Max batch candidates: " << max_batch_candidates << endl;
    cout << "No CPU/GPU threshold scheduling." << endl;
    cout << "CPU hash/check worker threads: " << cpu_threads << endl;

    BatchStats stats;
    model m;

    auto start_train = system_clock::now();
    m.train(data_path);
    m.order();
    auto end_train = system_clock::now();
    stats.train_sec = SecondsBetween(start_train, end_train);

    unordered_set<string> test_set = LoadTestSet(data_path, 1000000);
    cout << "Test set loaded: " << test_set.size() << endl;

    auto pq = InitPriority(m);
    cout << "Initial priority queue size: " << pq.size() << endl;

    long long next_report = 1000000;
    bool sample_checked = false;

    vector<string> previous_guesses;
    bool has_previous = false;
    long long batches_started = 0;
    double cpu_hash_during_gpu_sec = 0.0;
    double estimated_overlap_sec = 0.0;
    double gpu_submit_finish_wall_sec = 0.0;

    auto start_total = system_clock::now();

    while (!pq.empty() && stats.generated < generate_limit)
    {
        vector<PT> batch_pts;
        vector<string> batch_prefixes;
        vector<GPUBatchWork> works;
        long long batch_candidates = 0;

        auto start_pop = system_clock::now();
        while (!pq.empty() && (int)batch_pts.size() < batch_pt_count)
        {
            PT curr_pt = pq.top();

            string prefix;
            const vector<string> *values = nullptr;
            bool ok = BuildPTWork(m, curr_pt, prefix, values);
            long long value_count = (ok && values != nullptr) ? (long long)values->size() : 0;

            if (!batch_pts.empty() && max_batch_candidates > 0 && batch_candidates + value_count > max_batch_candidates)
            {
                break;
            }

            pq.pop();
            batch_pts.emplace_back(curr_pt);
            batch_prefixes.emplace_back(prefix);

            GPUBatchWork work;
            work.prefix = batch_prefixes.back();
            work.values = ok ? values : nullptr;
            works.emplace_back(work);

            batch_candidates += value_count;
            stats.processed_pt_count += 1;
        }
        auto end_pop = system_clock::now();
        stats.priority_sec += SecondsBetween(start_pop, end_pop);

        if (batch_pts.empty())
        {
            break;
        }

        if (!sample_checked)
        {
            cout << "First overlap batch prepared PT count: " << works.size() << endl;
            cout << "First overlap batch planned candidate count: " << batch_candidates << endl;
        }

        auto start_gen = system_clock::now();
        GPUBatchAsyncJob job;
        try
        {
            job = StartCandidatesBatchGPUAsync(works, 64, 256);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "[ERROR] StartCandidatesBatchGPUAsync failed: " << ex.what() << std::endl;
            return 1;
        }

        // CPU work during the current GPU batch: hash/check the previous batch.
        double current_overlap_hash = 0.0;
        if (has_previous && !previous_guesses.empty())
        {
            HashAndCheckParallel(previous_guesses, test_set, stats, cpu_threads, &current_overlap_hash);
            cpu_hash_during_gpu_sec += current_overlap_hash;
            previous_guesses.clear();
            has_previous = false;
        }

        GPUBatchResult batch_result;
        try
        {
            batch_result = FinishCandidatesBatchGPUAsync(job);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "[ERROR] FinishCandidatesBatchGPUAsync failed: " << ex.what() << std::endl;
            return 1;
        }
        auto end_gen = system_clock::now();

        gpu_submit_finish_wall_sec += SecondsBetween(start_gen, end_gen);
        stats.generation_wall_sec += SecondsBetween(start_gen, end_gen);
        stats.gpu_h2d_ms += batch_result.h2d_ms;
        stats.gpu_kernel_ms += batch_result.kernel_ms;
        stats.gpu_d2h_ms += batch_result.d2h_ms;
        stats.gpu_total_ms += batch_result.total_gpu_ms;
        stats.gpu_batch_count += 1;
        batches_started += 1;

        double async_gpu_sec = (batch_result.kernel_ms + batch_result.d2h_ms) / 1000.0;
        if (current_overlap_hash > 0.0)
        {
            estimated_overlap_sec += std::min(current_overlap_hash, async_gpu_sec);
        }

        if (!sample_checked)
        {
            bool pass = VerifyBatchSamples(works, batch_result, 3);
            cout << "First overlap GPU batch sample check: " << (pass ? "PASSED" : "FAILED") << endl;
            cout << "First overlap batch PT count: " << works.size() << endl;
            cout << "First overlap batch candidate count: " << batch_result.guesses.size() << endl;
            if (!batch_result.guesses.empty())
            {
                cout << "First guess sample: " << batch_result.guesses[0] << endl;
            }
            sample_checked = true;
            if (!pass)
            {
                return 1;
            }
        }

        // Current guesses will be hashed while the next GPU batch is running.
        if (!batch_result.guesses.empty())
        {
            stats.generated += (long long)batch_result.guesses.size();
            previous_guesses = std::move(batch_result.guesses);
            has_previous = true;
        }

        auto start_newpts = system_clock::now();
        for (const PT &old_pt : batch_pts)
        {
            vector<PT> new_pts = NewPTsLocal(old_pt);
            for (PT &pt : new_pts)
            {
                CalProbLocal(m, pt);
                pq.push(pt);
            }
        }
        auto end_newpts = system_clock::now();
        stats.priority_sec += SecondsBetween(start_newpts, end_newpts);

        while (stats.generated >= next_report)
        {
            cout << "Generated: " << stats.generated << endl;
            next_report += 1000000;
        }
    }

    // Hash/check the last batch after no further GPU work is available.
    double tail_hash_sec = 0.0;
    if (has_previous && !previous_guesses.empty())
    {
        HashAndCheckParallel(previous_guesses, test_set, stats, cpu_threads, &tail_hash_sec);
        previous_guesses.clear();
        has_previous = false;
    }

    auto end_total = system_clock::now();
    double total_sec = SecondsBetween(start_total, end_total);

    cout << fixed << setprecision(6);
    cout << endl;
    cout << "=== Result ===" << endl;
    cout << "Generated:" << stats.generated << endl;
    cout << "Cracked:" << stats.cracked << endl;
    cout << "Train time:" << stats.train_sec << "seconds" << endl;
    cout << "Generation submit/finish wall time:" << stats.generation_wall_sec << "seconds" << endl;
    cout << "Hash/check time:" << stats.hash_sec << "seconds" << endl;
    cout << "Hash/check time during GPU in-flight:" << cpu_hash_during_gpu_sec << "seconds" << endl;
    cout << "Tail hash/check time without overlap:" << tail_hash_sec << "seconds" << endl;
    cout << "Estimated overlap saved time:" << estimated_overlap_sec << "seconds" << endl;
    cout << "Priority/control time:" << stats.priority_sec << "seconds" << endl;
    cout << "Total online time:" << total_sec << "seconds" << endl;
    cout << "Processed PT count:" << stats.processed_pt_count << endl;
    cout << "GPU batch count:" << stats.gpu_batch_count << endl;
    cout << "Batch PT count setting:" << batch_pt_count << endl;
    cout << "Max batch candidates setting:" << max_batch_candidates << endl;
    cout << "CPU hash/check worker threads:" << cpu_threads << endl;
    cout << "GPU H2D time:" << stats.gpu_h2d_ms / 1000.0 << "seconds" << endl;
    cout << "GPU kernel time:" << stats.gpu_kernel_ms / 1000.0 << "seconds" << endl;
    cout << "GPU D2H time:" << stats.gpu_d2h_ms / 1000.0 << "seconds" << endl;
    cout << "GPU event total time:" << stats.gpu_total_ms / 1000.0 << "seconds" << endl;
    cout << "MD5 sink:" << g_md5_sink << endl;

    return 0;
}
