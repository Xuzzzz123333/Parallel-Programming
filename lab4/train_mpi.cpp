#include "train_mpi.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <iostream>

using namespace std;

static void append_int(string &buf, int x)
{
    const char *p = reinterpret_cast<const char *>(&x);
    buf.append(p, sizeof(int));
}

static void append_string(string &buf, const string &s)
{
    append_int(buf, (int)s.size());
    buf.append(s.data(), s.size());
}

static int read_int(const string &buf, size_t &pos)
{
    int x;
    memcpy(&x, buf.data() + pos, sizeof(int));
    pos += sizeof(int);
    return x;
}

static string read_string(const string &buf, size_t &pos)
{
    int len = read_int(buf, pos);
    string s(buf.data() + pos, len);
    pos += len;
    return s;
}

static void clear_model(model &m)
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

static void serialize_pt(string &buf, const PT &pt, int freq)
{
    append_int(buf, (int)pt.content.size());

    for (const segment &seg : pt.content)
    {
        append_int(buf, seg.type);
        append_int(buf, seg.length);
    }

    append_int(buf, freq);
}

static PT deserialize_pt(const string &buf, size_t &pos, int &freq)
{
    PT pt;

    int content_size = read_int(buf, pos);

    for (int i = 0; i < content_size; i += 1)
    {
        int type = read_int(buf, pos);
        int length = read_int(buf, pos);

        segment seg(type, length);
        pt.content.emplace_back(seg);
    }

    for (int i = 0; i < (int)pt.content.size(); i += 1)
    {
        pt.curr_indices.emplace_back(0);
    }

    freq = read_int(buf, pos);

    return pt;
}

static void serialize_segment(string &buf, const segment &seg, int seg_freq)
{
    append_int(buf, seg.type);
    append_int(buf, seg.length);
    append_int(buf, seg_freq);

    append_int(buf, (int)seg.values.size());

    for (auto kv : seg.values)
    {
        const string &value = kv.first;
        int value_id = kv.second;
        int freq = seg.freqs.at(value_id);

        append_string(buf, value);
        append_int(buf, freq);
    }
}

static segment deserialize_segment(const string &buf, size_t &pos, int &seg_freq)
{
    int type = read_int(buf, pos);
    int length = read_int(buf, pos);
    seg_freq = read_int(buf, pos);

    segment seg(type, length);

    int value_count = read_int(buf, pos);

    for (int i = 0; i < value_count; i += 1)
    {
        string value = read_string(buf, pos);
        int freq = read_int(buf, pos);

        int value_id = (int)seg.values.size();
        seg.values[value] = value_id;
        seg.freqs[value_id] = freq;
    }

    return seg;
}

static string serialize_model(const model &m)
{
    string buf;

    append_int(buf, m.total_preterm);

    append_int(buf, (int)m.preterminals.size());
    for (int i = 0; i < (int)m.preterminals.size(); i += 1)
    {
        int freq = 0;
        auto it = m.preterm_freq.find(i);
        if (it != m.preterm_freq.end())
        {
            freq = it->second;
        }

        serialize_pt(buf, m.preterminals[i], freq);
    }

    append_int(buf, (int)m.letters.size());
    for (int i = 0; i < (int)m.letters.size(); i += 1)
    {
        int freq = 0;
        auto it = m.letters_freq.find(i);
        if (it != m.letters_freq.end())
        {
            freq = it->second;
        }

        serialize_segment(buf, m.letters[i], freq);
    }

    append_int(buf, (int)m.digits.size());
    for (int i = 0; i < (int)m.digits.size(); i += 1)
    {
        int freq = 0;
        auto it = m.digits_freq.find(i);
        if (it != m.digits_freq.end())
        {
            freq = it->second;
        }

        serialize_segment(buf, m.digits[i], freq);
    }

    append_int(buf, (int)m.symbols.size());
    for (int i = 0; i < (int)m.symbols.size(); i += 1)
    {
        int freq = 0;
        auto it = m.symbols_freq.find(i);
        if (it != m.symbols_freq.end())
        {
            freq = it->second;
        }

        serialize_segment(buf, m.symbols[i], freq);
    }

    return buf;
}

static model deserialize_model(const string &buf)
{
    model m;
    clear_model(m);

    size_t pos = 0;

    m.total_preterm = read_int(buf, pos);

    int pt_count = read_int(buf, pos);
    for (int i = 0; i < pt_count; i += 1)
    {
        int freq = 0;
        PT pt = deserialize_pt(buf, pos, freq);

        m.preterminals.emplace_back(pt);
        m.preterm_freq[i] = freq;
    }

    int letter_count = read_int(buf, pos);
    for (int i = 0; i < letter_count; i += 1)
    {
        int seg_freq = 0;
        segment seg = deserialize_segment(buf, pos, seg_freq);

        m.letters.emplace_back(seg);
        m.letters_freq[i] = seg_freq;
    }

    int digit_count = read_int(buf, pos);
    for (int i = 0; i < digit_count; i += 1)
    {
        int seg_freq = 0;
        segment seg = deserialize_segment(buf, pos, seg_freq);

        m.digits.emplace_back(seg);
        m.digits_freq[i] = seg_freq;
    }

    int symbol_count = read_int(buf, pos);
    for (int i = 0; i < symbol_count; i += 1)
    {
        int seg_freq = 0;
        segment seg = deserialize_segment(buf, pos, seg_freq);

        m.symbols.emplace_back(seg);
        m.symbols_freq[i] = seg_freq;
    }

    m.preterm_id = (int)m.preterminals.size() - 1;
    m.letters_id = (int)m.letters.size() - 1;
    m.digits_id = (int)m.digits.size() - 1;
    m.symbols_id = (int)m.symbols.size() - 1;

    return m;
}

static void merge_segment_vector(
    vector<segment> &dst_vec,
    unordered_map<int, int> &dst_freq,
    const vector<segment> &src_vec,
    const unordered_map<int, int> &src_freq,
    int type
)
{
    for (int i = 0; i < (int)src_vec.size(); i += 1)
    {
        const segment &src = src_vec[i];

        int id = -1;

        for (int j = 0; j < (int)dst_vec.size(); j += 1)
        {
            if (dst_vec[j].length == src.length)
            {
                id = j;
                break;
            }
        }

        if (id == -1)
        {
            segment new_seg(type, src.length);
            id = (int)dst_vec.size();
            dst_vec.emplace_back(new_seg);
            dst_freq[id] = 0;
        }

        auto freq_it = src_freq.find(i);
        if (freq_it != src_freq.end())
        {
            dst_freq[id] += freq_it->second;
        }

        for (auto kv : src.values)
        {
            const string &value = kv.first;
            int src_value_id = kv.second;
            int freq = src.freqs.at(src_value_id);

            if (dst_vec[id].values.find(value) == dst_vec[id].values.end())
            {
                int dst_value_id = (int)dst_vec[id].values.size();
                dst_vec[id].values[value] = dst_value_id;
                dst_vec[id].freqs[dst_value_id] = freq;
            }
            else
            {
                int dst_value_id = dst_vec[id].values[value];
                dst_vec[id].freqs[dst_value_id] += freq;
            }
        }
    }
}

static void merge_model(model &dst, const model &src)
{
    dst.total_preterm += src.total_preterm;

    for (int i = 0; i < (int)src.preterminals.size(); i += 1)
    {
        PT pt = src.preterminals[i];

        int id = dst.FindPT(pt);

        if (id == -1)
        {
            int new_id = (int)dst.preterminals.size();
            dst.preterminals.emplace_back(pt);
            dst.preterm_freq[new_id] = src.preterm_freq.at(i);
        }
        else
        {
            dst.preterm_freq[id] += src.preterm_freq.at(i);
        }
    }

    merge_segment_vector(
        dst.letters,
        dst.letters_freq,
        src.letters,
        src.letters_freq,
        1
    );

    merge_segment_vector(
        dst.digits,
        dst.digits_freq,
        src.digits,
        src.digits_freq,
        2
    );

    merge_segment_vector(
        dst.symbols,
        dst.symbols_freq,
        src.symbols,
        src.symbols_freq,
        3
    );

    dst.preterm_id = (int)dst.preterminals.size() - 1;
    dst.letters_id = (int)dst.letters.size() - 1;
    dst.digits_id = (int)dst.digits.size() - 1;
    dst.symbols_id = (int)dst.symbols.size() - 1;
}

static void train_local_by_rank(model &local_model, const string &path, int rank, int world_size)
{
    clear_model(local_model);

    ifstream train_set(path);

    string pw;
    int line_id = 0;
    int local_count = 0;

    while (train_set >> pw)
    {
        if (line_id > 3000000)
        {
            break;
        }

        if (line_id % world_size == rank)
        {
            local_model.parse(pw);
            local_count += 1;
        }

        line_id += 1;
    }

   #ifdef MPI_VERIFY
    vector<int> all_counts;

    if (rank == 0)
    {
        all_counts.resize(world_size);
    }

    MPI_Gather(
        &local_count,
        1,
        MPI_INT,
        rank == 0 ? all_counts.data() : nullptr,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        int total_count = 0;

        cout << "[VERIFY][TRAIN] local training rows by rank: ";

        for (int r = 0; r < world_size; r += 1)
        {
            cout << "rank" << r << "=" << all_counts[r] << " ";
            total_count += all_counts[r];
        }

        cout << endl;
        cout << "[VERIFY][TRAIN] total assigned training rows = "
             << total_count << endl;
        cout << "[VERIFY][TRAIN] partition rule: line_id % world_size == rank"
             << endl;
    }
#else
    if (rank == 0)
    {
        cout << "MPI training finished." << endl;
    }
#endif
}

void train_mpi(model &m, const string &path, int rank, int world_size)
{
    clear_model(m);

    model local_model;
    train_local_by_rank(local_model, path, rank, world_size);

    string local_buf = serialize_model(local_model);
    int local_size = (int)local_buf.size();

    vector<int> recv_sizes;
    vector<int> displs;

    if (rank == 0)
    {
        recv_sizes.resize(world_size);
    }

    MPI_Gather(
        &local_size,
        1,
        MPI_INT,
        rank == 0 ? recv_sizes.data() : nullptr,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    string all_buf;

    if (rank == 0)
    {
        displs.resize(world_size);

        int total_size = 0;
        for (int i = 0; i < world_size; i += 1)
        {
            displs[i] = total_size;
            total_size += recv_sizes[i];
        }

        all_buf.resize(total_size);
    }

    MPI_Gatherv(
        local_buf.data(),
        local_size,
        MPI_CHAR,
        rank == 0 ? &all_buf[0] : nullptr,
        rank == 0 ? recv_sizes.data() : nullptr,
        rank == 0 ? displs.data() : nullptr,
        MPI_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        clear_model(m);

        for (int i = 0; i < world_size; i += 1)
        {
            string one_buf(
                all_buf.data() + displs[i],
                recv_sizes[i]
            );

            model one_model = deserialize_model(one_buf);
            merge_model(m, one_model);
        }
    }

    string global_buf;

    if (rank == 0)
    {
        global_buf = serialize_model(m);
    }

    int global_size = 0;

    if (rank == 0)
    {
        global_size = (int)global_buf.size();
    }

    MPI_Bcast(&global_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        global_buf.resize(global_size);
    }

    MPI_Bcast(&global_buf[0], global_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        m = deserialize_model(global_buf);
    }

    m.order();
}