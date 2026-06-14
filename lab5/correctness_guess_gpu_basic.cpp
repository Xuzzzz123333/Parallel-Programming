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
#include <unordered_set>
#include <vector>

using namespace std;
using namespace chrono;

struct BasicStats
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

static bool VerifySamples(
    const string &prefix,
    const vector<string> &values,
    const vector<string> &gpu_guesses,
    int sample_count)
{
    int n = min(sample_count, (int)min(values.size(), gpu_guesses.size()));
    for (int i = 0; i < n; i += 1)
    {
        string expected = prefix + values[i];
        if (expected != gpu_guesses[i])
        {
            cerr << "Mismatch at sample " << i << endl;
            cerr << "expected: " << expected << endl;
            cerr << "actual  : " << gpu_guesses[i] << endl;
            return false;
        }
    }
    return true;
}

static volatile bit32 g_md5_sink = 0;

static void HashAndCheck(
    const vector<string> &guesses,
    const unordered_set<string> &test_set,
    BasicStats &stats)
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

int main(int argc, char **argv)
{
    string data_path = "guessdata/Rockyou-singleLined-full.txt";
    long long generate_limit = 10000000;

    if (argc >= 2)
    {
        generate_limit = atoll(argv[1]);
    }
    if (argc >= 3)
    {
        data_path = argv[2];
    }

    cout << "=== Basic Full PCFG + CUDA Generation Experiment ===" << endl;
    cout << "Train/test data: " << data_path << endl;
    cout << "Generate limit: " << generate_limit << endl;
    cout << "Basic policy: each time pop exactly one PT; its inner value loop is generated on GPU." << endl;
    cout << "No multi-PT batch, no CPU/GPU overlap, no CPU/GPU threshold scheduling." << endl;

    BasicStats stats;
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

    auto start_total = system_clock::now();

    while (!pq.empty())
    {
        auto start_pop = system_clock::now();
        PT curr_pt = pq.top();
        pq.pop();
        auto end_pop = system_clock::now();
        stats.priority_sec += SecondsBetween(start_pop, end_pop);

        string prefix;
        const vector<string> *values = nullptr;
        vector<string> guesses;

        auto start_gen = system_clock::now();
        bool ok = BuildPTWork(m, curr_pt, prefix, values);

        if (ok)
        {
            GPUGuessResult gpu_result = GenerateCandidatesGPU(prefix, *values, 64, 256);
            guesses = std::move(gpu_result.guesses);

            stats.gpu_h2d_ms += gpu_result.h2d_ms;
            stats.gpu_kernel_ms += gpu_result.kernel_ms;
            stats.gpu_d2h_ms += gpu_result.d2h_ms;
            stats.gpu_total_ms += gpu_result.total_gpu_ms;
            stats.processed_pt_count += 1;

            if (!sample_checked)
            {
                bool pass = VerifySamples(prefix, *values, guesses, 5);
                cout << "First GPU PT sample check: " << (pass ? "PASSED" : "FAILED") << endl;
                cout << "First GPU PT value count: " << values->size() << endl;
                if (!guesses.empty())
                {
                    cout << "First guess sample: " << guesses[0] << endl;
                }
                sample_checked = true;
                if (!pass)
                {
                    return 1;
                }
            }
        }

        auto end_gen = system_clock::now();
        stats.generation_wall_sec += SecondsBetween(start_gen, end_gen);

        if (!guesses.empty())
        {
            stats.generated += (long long)guesses.size();
            HashAndCheck(guesses, test_set, stats);
        }

        auto start_newpts = system_clock::now();
        vector<PT> new_pts = NewPTsLocal(curr_pt);
        for (PT &pt : new_pts)
        {
            CalProbLocal(m, pt);
            pq.push(pt);
        }
        auto end_newpts = system_clock::now();
        stats.priority_sec += SecondsBetween(start_newpts, end_newpts);

        while (stats.generated >= next_report)
        {
            cout << "Generated: " << stats.generated << endl;
            next_report += 1000000;
        }

        if (stats.generated >= generate_limit)
        {
            break;
        }
    }

    auto end_total = system_clock::now();
    double total_sec = SecondsBetween(start_total, end_total);

    cout << fixed << setprecision(6);
    cout << endl;
    cout << "=== Result ===" << endl;
    cout << "Generated:" << stats.generated << endl;
    cout << "Cracked:" << stats.cracked << endl;
    cout << "Train time:" << stats.train_sec << "seconds" << endl;
    cout << "Generation wall time:" << stats.generation_wall_sec << "seconds" << endl;
    cout << "Hash/check time:" << stats.hash_sec << "seconds" << endl;
    cout << "Priority/control time:" << stats.priority_sec << "seconds" << endl;
    cout << "Total online time:" << total_sec << "seconds" << endl;
    cout << "Processed PT count:" << stats.processed_pt_count << endl;
    cout << "GPU H2D time:" << stats.gpu_h2d_ms / 1000.0 << "seconds" << endl;
    cout << "GPU kernel time:" << stats.gpu_kernel_ms / 1000.0 << "seconds" << endl;
    cout << "GPU D2H time:" << stats.gpu_d2h_ms / 1000.0 << "seconds" << endl;
    cout << "GPU event total time:" << stats.gpu_total_ms / 1000.0 << "seconds" << endl;
    cout << "MD5 sink:" << g_md5_sink << endl;

    return 0;
}
