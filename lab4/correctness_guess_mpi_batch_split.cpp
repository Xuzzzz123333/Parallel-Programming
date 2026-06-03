#include "PCFG.h"
#include "guessing_mpi.h"
#include "train_mpi.h"
#include "mpi.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

#ifndef MPI_PT_BATCH_SIZE
#define MPI_PT_BATCH_SIZE 4
#endif
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
    for (int x : pt.curr_indices)
    {
        data.emplace_back(x);
    }

    data.emplace_back((int)pt.max_indices.size());
    for (int x : pt.max_indices)
    {
        data.emplace_back(x);
    }
}

static PT UnpackPTFromVector(const vector<int> &data, int &pos, float preterm_prob, float prob)
{
    PT pt;

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

        for (int i = 0; i < count; i += 1)
        {
            float preterm_prob = packed_probs[2 * i];
            float prob = packed_probs[2 * i + 1];

            PT pt = UnpackPTFromVector(
                packed_ints,
                pos,
                preterm_prob,
                prob
            );

            batch_pts.emplace_back(pt);
        }
    }
}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank != 0)
    {
        cout.setstate(std::ios_base::failbit);
    }

    PriorityQueue q;

    double train_start = MPI_Wtime();

    train_mpi(
        q.m,
        "/guessdata/Rockyou-singleLined-full.txt",
        rank,
        world_size
    );

    double train_end = MPI_Wtime();

    double local_train_time = train_end - train_start;
    double train_time = 0.0;

    MPI_Reduce(
        &local_train_time,
        &train_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    unordered_set<string> test_set;
    ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");

    int test_count = 0;
    string pw;

    while (test_data >> pw)
    {
        test_count += 1;
        test_set.insert(pw);

        if (test_count >= 1000000)
        {
            break;
        }
    }

    if (rank == 0)
    {
        cout.clear();

        q.init();

        cout << "MPI world size:" << world_size << endl;
        cout << "Advanced mode: batch PTs + intra-PT value split." << endl;
        cout << "Batch size:" << MPI_PT_BATCH_SIZE << endl;
        cout << "here" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    const long long generate_n = 10000000;

    long long history = 0;
    long long cracked = 0;
    long long next_report = 1000000;

    double local_generate_sum = 0.0;
    double local_hash_sum = 0.0;
    double local_compute_sum = 0.0;

    double loop_start = MPI_Wtime();

    while (true)
    {
        int active_count = 0;
        vector<PT> batch_pts;

        if (rank == 0)
        {
            while (!q.priority.empty() &&
                   history < generate_n &&
                   active_count < MPI_PT_BATCH_SIZE)
            {
                PT curr_pt = q.priority.top();
                q.priority.pop();

                batch_pts.emplace_back(curr_pt);
                active_count += 1;
            }
        }

        MPI_Bcast(&active_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (active_count == 0)
        {
            break;
        }

        BcastPTBatch(batch_pts, rank);

        MPILocalResult local_total;
        local_total.generated = 0;
        local_total.cracked = 0;
        local_total.generate_time = 0.0;
        local_total.hash_time = 0.0;
        local_total.compute_time = 0.0;

        for (int i = 0; i < active_count; i += 1)
        {
            MPILocalResult one = GenerateAndHashPTMPI(
                q.m,
                batch_pts[i],
                test_set,
                rank,
                world_size
            );

            local_total.generated += one.generated;
            local_total.cracked += one.cracked;
            local_total.generate_time += one.generate_time;
            local_total.hash_time += one.hash_time;
            local_total.compute_time += one.compute_time;
        }

        local_generate_sum += local_total.generate_time;
        local_hash_sum += local_total.hash_time;
        local_compute_sum += local_total.compute_time;

        long long global_generated = 0;
        long long global_cracked = 0;

        MPI_Reduce(
            &local_total.generated,
            &global_generated,
            1,
            MPI_LONG_LONG,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_total.cracked,
            &global_cracked,
            1,
            MPI_LONG_LONG,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        if (rank == 0)
        {
            history += global_generated;
            cracked += global_cracked;

            while (history >= next_report)
            {
                cout << "Guesses processed: " << next_report << endl;
                next_report += 1000000;
            }

            for (PT &pt : batch_pts)
            {
                vector<PT> new_pts = pt.NewPTs();

                for (PT new_pt : new_pts)
                {
                    q.CalProb(new_pt);
                    q.priority.push(new_pt);
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double loop_end = MPI_Wtime();

    double local_total_time = loop_end - loop_start;

    double total_time = 0.0;
    double generate_time = 0.0;
    double hash_time = 0.0;
    double compute_time = 0.0;

    MPI_Reduce(
        &local_total_time,
        &total_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_generate_sum,
        &generate_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_hash_sum,
        &hash_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    MPI_Reduce(
        &local_compute_sum,
        &compute_time,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0)
    {
        double guess_time = total_time - hash_time;

        if (guess_time < 0.0)
        {
            guess_time = 0.0;
        }

        double comm_control_time = total_time - compute_time;

        if (comm_control_time < 0.0)
        {
            comm_control_time = 0.0;
        }

        cout << "Guess time:" << guess_time << "seconds" << endl;
        cout << "Hash time:" << hash_time << "seconds" << endl;
        cout << "Train time:" << train_time << "seconds" << endl;
        cout << "Cracked:" << cracked << endl;
        cout << "Generated:" << history << endl;

        cout << "MPI total Guess+Hash time:" << total_time << "seconds" << endl;
        cout << "MPI generate only time:" << generate_time << "seconds" << endl;
        cout << "MPI compute time:" << compute_time << "seconds" << endl;
        cout << "MPI comm/control time:" << comm_control_time << "seconds" << endl;
    }

    MPI_Finalize();

    return 0;
}