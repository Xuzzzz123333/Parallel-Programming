#include "train_thread.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

using namespace std;

#ifndef TRAIN_LIMIT
#define TRAIN_LIMIT 3000000
#endif

static void clear_model_for_thread_train(model &m)
{
    m.preterm_id = -1;
    m.letters_id = -1;
    m.digits_id = -1;
    m.symbols_id = -1;
    m.total_preterm = 0;

    m.preterminals.clear();
    m.preterm_freq.clear();
    m.letters.clear();
    m.letters_freq.clear();
    m.digits.clear();
    m.digits_freq.clear();
    m.symbols.clear();
    m.symbols_freq.clear();
    m.ordered_pts.clear();
}

static string make_pt_key(const PT &pt)
{
    // Canonical structural key for a PT, e.g. L6D2S1 -> "1:6|2:2|3:1|".
    // This replaces repeated model::FindPT linear scans during local-model merge.
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

static void merge_segment_vector_indexed_thread(
    vector<segment> &dst_vec,
    unordered_map<int, int> &dst_freq,
    unordered_map<int, int> &length_index,
    const vector<segment> &src_vec,
    const unordered_map<int, int> &src_freq,
    int type)
{
    for (int i = 0; i < (int)src_vec.size(); i += 1)
    {
        const segment &src = src_vec[i];

        int id = -1;
        auto id_it = length_index.find(src.length);
        if (id_it == length_index.end())
        {
            segment new_seg(type, src.length);
            id = (int)dst_vec.size();
            dst_vec.emplace_back(new_seg);
            dst_freq[id] = 0;
            length_index[src.length] = id;
        }
        else
        {
            id = id_it->second;
        }

        auto freq_it = src_freq.find(i);
        if (freq_it != src_freq.end())
        {
            dst_freq[id] += freq_it->second;
        }

        for (const auto &kv : src.values)
        {
            const string &value = kv.first;
            int src_value_id = kv.second;
            int freq = src.freqs.at(src_value_id);

            auto value_it = dst_vec[id].values.find(value);
            if (value_it == dst_vec[id].values.end())
            {
                int dst_value_id = (int)dst_vec[id].values.size();
                dst_vec[id].values[value] = dst_value_id;
                dst_vec[id].freqs[dst_value_id] = freq;
            }
            else
            {
                int dst_value_id = value_it->second;
                dst_vec[id].freqs[dst_value_id] += freq;
            }
        }
    }
}

struct MergeIndex
{
    unordered_map<string, int> pt_index;
    unordered_map<int, int> letters_index;
    unordered_map<int, int> digits_index;
    unordered_map<int, int> symbols_index;
};

static void reserve_merge_index(MergeIndex &idx, int thread_count)
{
    // The number of PT/segment structures is small compared with value dictionaries,
    // but reserving avoids repeated rehash during multi-model merging.
    idx.pt_index.reserve(32768);
    idx.letters_index.reserve(64);
    idx.digits_index.reserve(64);
    idx.symbols_index.reserve(64);
}

static void merge_model_indexed_thread(model &dst, const model &src, MergeIndex &idx)
{
    dst.total_preterm += src.total_preterm;

    for (int i = 0; i < (int)src.preterminals.size(); i += 1)
    {
        const PT &pt = src.preterminals[i];
        string key = make_pt_key(pt);

        auto id_it = idx.pt_index.find(key);
        if (id_it == idx.pt_index.end())
        {
            int new_id = (int)dst.preterminals.size();
            dst.preterminals.emplace_back(pt);
            dst.preterm_freq[new_id] = src.preterm_freq.at(i);
            idx.pt_index.emplace(std::move(key), new_id);
        }
        else
        {
            dst.preterm_freq[id_it->second] += src.preterm_freq.at(i);
        }
    }

    merge_segment_vector_indexed_thread(
        dst.letters,
        dst.letters_freq,
        idx.letters_index,
        src.letters,
        src.letters_freq,
        1
    );

    merge_segment_vector_indexed_thread(
        dst.digits,
        dst.digits_freq,
        idx.digits_index,
        src.digits,
        src.digits_freq,
        2
    );

    merge_segment_vector_indexed_thread(
        dst.symbols,
        dst.symbols_freq,
        idx.symbols_index,
        src.symbols,
        src.symbols_freq,
        3
    );

    dst.preterm_id = (int)dst.preterminals.size() - 1;
    dst.letters_id = (int)dst.letters.size() - 1;
    dst.digits_id = (int)dst.digits.size() - 1;
    dst.symbols_id = (int)dst.symbols.size() - 1;
}

void train_thread_model(model &m, const string &path, int thread_count)
{
    cout << "Training..." << endl;
    cout << "Training mode: thread local-model training + hash-indexed merge" << endl;
    cout << "Training phase 1: reading passwords..." << endl;

    vector<string> passwords;
    passwords.reserve(3100000);

    ifstream train_set(path);
    if (!train_set.is_open())
    {
        cerr << "Failed to open training file: " << path << endl;
        clear_model_for_thread_train(m);
        return;
    }

    string pw;
    int line_id = 0;
    while (train_set >> pw)
    {
        if (line_id > TRAIN_LIMIT)
        {
            break;
        }
        passwords.emplace_back(pw);
        line_id += 1;
    }

    cout << "Training passwords loaded: " << passwords.size() << endl;

    if (thread_count < 1) thread_count = 1;
    if (thread_count > (int)passwords.size()) thread_count = (int)passwords.size();

    cout << "Training phase 1: parallel parsing passwords with "
         << thread_count << " threads..." << endl;

    vector<model> local_models(thread_count);
    vector<thread> workers;
    workers.reserve(thread_count);

    int n = (int)passwords.size();
    for (int t = 0; t < thread_count; t += 1)
    {
        int start = (long long)n * t / thread_count;
        int end = (long long)n * (t + 1) / thread_count;

        workers.emplace_back([&, t, start, end]() {
            clear_model_for_thread_train(local_models[t]);
            for (int i = start; i < end; i += 1)
            {
                local_models[t].parse(passwords[i]);
            }
        });
    }

    for (thread &worker : workers)
    {
        worker.join();
    }

    cout << "Training phase 1: hash-indexed merging local models..." << endl;
    clear_model_for_thread_train(m);
    MergeIndex merge_index;
    reserve_merge_index(merge_index, thread_count);
    for (int t = 0; t < thread_count; t += 1)
    {
        merge_model_indexed_thread(m, local_models[t], merge_index);
    }

    cout << "Training phase 1 finished." << endl;
}
