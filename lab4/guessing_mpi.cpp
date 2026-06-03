#include "guessing_mpi.h"
#include "md5.h"

#include <vector>
#include <string>
#include <iostream>

using namespace std;

static vector<int> PackPTInt(const PT &pt)
{
    vector<int> data;

    data.emplace_back((int)pt.content.size());

    for (const segment &seg : pt.content)
    {
        data.emplace_back(seg.type);
        data.emplace_back(seg.length);
    }

    data.emplace_back(pt.pivot);

    data.emplace_back((int)pt.curr_indices.size());
    for (int x : pt.curr_indices)
    {
        data.emplace_back(x);
    }

    data.emplace_back((int)pt.max_indices.size());
    for (int x : pt.max_indices)
    {
        data.emplace_back(x);
    }

    return data;
}

static PT UnpackPTInt(const vector<int> &data, float preterm_prob, float prob)
{
    PT pt;

    int pos = 0;

    int content_size = data[pos++];

    for (int i = 0; i < content_size; i += 1)
    {
        int type = data[pos++];
        int length = data[pos++];

        segment seg(type, length);
        pt.content.emplace_back(seg);
    }

    pt.pivot = data[pos++];

    int curr_size = data[pos++];
    for (int i = 0; i < curr_size; i += 1)
    {
        pt.curr_indices.emplace_back(data[pos++]);
    }

    int max_size = data[pos++];
    for (int i = 0; i < max_size; i += 1)
    {
        pt.max_indices.emplace_back(data[pos++]);
    }

    pt.preterm_prob = preterm_prob;
    pt.prob = prob;

    return pt;
}

void BcastPT(PT &pt, int rank)
{
    vector<int> data;
    float prob_data[2] = {0.0f, 0.0f};

    int data_size = 0;

    if (rank == 0)
    {
        data = PackPTInt(pt);
        data_size = (int)data.size();

        prob_data[0] = pt.preterm_prob;
        prob_data[1] = pt.prob;
    }

    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        data.resize(data_size);
    }

    MPI_Bcast(data.data(), data_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(prob_data, 2, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        pt = UnpackPTInt(data, prob_data[0], prob_data[1]);
    }
}

static segment *FindSegmentInModel(model &m, const segment &seg)
{
    if (seg.type == 1)
    {
        return &m.letters[m.FindLetter(seg)];
    }

    if (seg.type == 2)
    {
        return &m.digits[m.FindDigit(seg)];
    }

    if (seg.type == 3)
    {
        return &m.symbols[m.FindSymbol(seg)];
    }

    return nullptr;
}

static string BuildPrefix(model &m, const PT &pt)
{
    string prefix;

    int seg_idx = 0;

    for (int idx : pt.curr_indices)
    {
        if (seg_idx == (int)pt.content.size() - 1)
        {
            break;
        }

        const segment &seg = pt.content[seg_idx];

        if (seg.type == 1)
        {
            prefix += m.letters[m.FindLetter(seg)].ordered_values[idx];
        }
        else if (seg.type == 2)
        {
            prefix += m.digits[m.FindDigit(seg)].ordered_values[idx];
        }
        else if (seg.type == 3)
        {
            prefix += m.symbols[m.FindSymbol(seg)].ordered_values[idx];
        }

        seg_idx += 1;
    }

    return prefix;
}

static string MakeGuess(
    const string &prefix,
    const vector<string> &values,
    int value_idx,
    bool has_prefix
)
{
    const string &val = values[value_idx];

    if (!has_prefix)
    {
        return val;
    }

    string result;
    result.reserve(prefix.size() + val.size());
    result.append(prefix);
    result.append(val);

    return result;
}
static void VerifyValueSplit(
    int n,
    int start,
    int end,
    int rank,
    int world_size
)
{
#ifdef MPI_VERIFY
    static int verify_round = 0;

    if (verify_round < 3)
    {
        int local_count = end - start;

        vector<int> starts;
        vector<int> ends;
        vector<int> counts;

        if (rank == 0)
        {
            starts.resize(world_size);
            ends.resize(world_size);
            counts.resize(world_size);
        }

        MPI_Gather(
            &start,
            1,
            MPI_INT,
            rank == 0 ? starts.data() : nullptr,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        MPI_Gather(
            &end,
            1,
            MPI_INT,
            rank == 0 ? ends.data() : nullptr,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        MPI_Gather(
            &local_count,
            1,
            MPI_INT,
            rank == 0 ? counts.data() : nullptr,
            1,
            MPI_INT,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            int sum_count = 0;
            bool continuous = true;

            for (int r = 0; r < world_size; r += 1)
            {
                sum_count += counts[r];

                if (r > 0 && starts[r] != ends[r - 1])
                {
                    continuous = false;
                }
            }

            bool cover_all = false;

            if (!starts.empty() && !ends.empty())
            {
                cover_all = (starts[0] == 0 && ends[world_size - 1] == n);
            }

            cout << "[VERIFY][GUESS] PT round " << verify_round
                 << ", total values = " << n << endl;

            for (int r = 0; r < world_size; r += 1)
            {
                cout << "  rank" << r
                     << " handles value_idx ["
                     << starts[r] << ", " << ends[r]
                     << "), count = " << counts[r] << endl;
            }

            cout << "[VERIFY][GUESS] sum local counts = "
                 << sum_count << endl;

            cout << "[VERIFY][GUESS] no-overlap continuous ranges = "
                 << (continuous ? "YES" : "NO") << endl;

            cout << "[VERIFY][GUESS] cover whole value range = "
                 << (cover_all ? "YES" : "NO") << endl;
        }
    }

    verify_round += 1;
#endif
}
MPILocalResult GenerateAndHashPTMPI(
    model &m,
    const PT &pt,
    const unordered_set<string> &test_set,
    int rank,
    int world_size
)
{
    MPILocalResult res;
    res.generated = 0;
    res.cracked = 0;
    res.generate_time = 0.0;
    res.hash_time = 0.0;
    res.compute_time = 0.0;

    double start_compute = MPI_Wtime();

    if (pt.content.empty())
    {
        return res;
    }

    bool has_prefix = (pt.content.size() > 1);
    string prefix;

    if (has_prefix)
    {
        prefix = BuildPrefix(m, pt);
    }

    int last_idx = (int)pt.content.size() - 1;

    segment *last_seg = FindSegmentInModel(m, pt.content[last_idx]);

    if (last_seg == nullptr)
    {
        return res;
    }

    const vector<string> &values = last_seg->ordered_values;

    int n = pt.max_indices[last_idx];

    int start = n * rank / world_size;
    int end = n * (rank + 1) / world_size;

    res.generated = end - start;
      VerifyValueSplit(n, start, end, rank, world_size);
    int i = start;

    for (; i + 3 < end; i += 4)
    {
        double start_gen = MPI_Wtime();

        string pw0 = MakeGuess(prefix, values, i, has_prefix);
        string pw1 = MakeGuess(prefix, values, i + 1, has_prefix);
        string pw2 = MakeGuess(prefix, values, i + 2, has_prefix);
        string pw3 = MakeGuess(prefix, values, i + 3, has_prefix);

        double end_gen = MPI_Wtime();
        res.generate_time += end_gen - start_gen;

        double start_hash = MPI_Wtime();

        if (test_set.find(pw0) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw1) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw2) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw3) != test_set.end())
        {
            res.cracked += 1;
        }

        bit32 states[4][4];
        MD5HashSIMD4Ref(pw0, pw1, pw2, pw3, states);

        double end_hash = MPI_Wtime();
        res.hash_time += end_hash - start_hash;
    }

    for (; i < end; i += 1)
    {
        double start_gen = MPI_Wtime();

        string pw = MakeGuess(prefix, values, i, has_prefix);

        double end_gen = MPI_Wtime();
        res.generate_time += end_gen - start_gen;

        double start_hash = MPI_Wtime();

        if (test_set.find(pw) != test_set.end())
        {
            res.cracked += 1;
        }

        bit32 state[4];
        MD5Hash(pw, state);

        double end_hash = MPI_Wtime();
        res.hash_time += end_hash - start_hash;
    }

    double end_compute = MPI_Wtime();

    res.compute_time = end_compute - start_compute;

    return res;
}
MPILocalResult GenerateAndHashPTRange(
    model &m,
    const PT &pt,
    const unordered_set<string> &test_set,
    int start_idx,
    int end_idx
)
{
    MPILocalResult res;
    res.generated = 0;
    res.cracked = 0;
    res.generate_time = 0.0;
    res.hash_time = 0.0;
    res.compute_time = 0.0;

    double start_compute = MPI_Wtime();

    if (pt.content.empty())
    {
        return res;
    }

    bool has_prefix = (pt.content.size() > 1);
    string prefix;

    if (has_prefix)
    {
        prefix = BuildPrefix(m, pt);
    }

    int last_idx = (int)pt.content.size() - 1;

    segment *last_seg = FindSegmentInModel(m, pt.content[last_idx]);

    if (last_seg == nullptr)
    {
        return res;
    }

    const vector<string> &values = last_seg->ordered_values;

    int n = pt.max_indices[last_idx];

    if (start_idx < 0)
    {
        start_idx = 0;
    }

    if (end_idx > n)
    {
        end_idx = n;
    }

    if (start_idx >= end_idx)
    {
        return res;
    }

    res.generated = end_idx - start_idx;

    int i = start_idx;

    for (; i + 3 < end_idx; i += 4)
    {
        double start_gen = MPI_Wtime();

        string pw0 = MakeGuess(prefix, values, i, has_prefix);
        string pw1 = MakeGuess(prefix, values, i + 1, has_prefix);
        string pw2 = MakeGuess(prefix, values, i + 2, has_prefix);
        string pw3 = MakeGuess(prefix, values, i + 3, has_prefix);

        double end_gen = MPI_Wtime();
        res.generate_time += end_gen - start_gen;

        double start_hash = MPI_Wtime();

        if (test_set.find(pw0) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw1) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw2) != test_set.end())
        {
            res.cracked += 1;
        }

        if (test_set.find(pw3) != test_set.end())
        {
            res.cracked += 1;
        }

        bit32 states[4][4];
        MD5HashSIMD4Ref(pw0, pw1, pw2, pw3, states);

        double end_hash = MPI_Wtime();
        res.hash_time += end_hash - start_hash;
    }

    for (; i < end_idx; i += 1)
    {
        double start_gen = MPI_Wtime();

        string pw = MakeGuess(prefix, values, i, has_prefix);

        double end_gen = MPI_Wtime();
        res.generate_time += end_gen - start_gen;

        double start_hash = MPI_Wtime();

        if (test_set.find(pw) != test_set.end())
        {
            res.cracked += 1;
        }

        bit32 state[4];
        MD5Hash(pw, state);

        double end_hash = MPI_Wtime();
        res.hash_time += end_hash - start_hash;
    }

    double end_compute = MPI_Wtime();

    res.compute_time = end_compute - start_compute;

    return res;
}