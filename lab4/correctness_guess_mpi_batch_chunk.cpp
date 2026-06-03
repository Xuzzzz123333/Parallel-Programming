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
#include <algorithm>

using namespace std;

#ifndef MPI_PT_CHUNK_SIZE
#define MPI_PT_CHUNK_SIZE 20000
#endif

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
        cout << "Advanced mode: PT-level batch parallel generation with value_idx chunks." << endl;
        cout << "Chunk size:" << MPI_PT_CHUNK_SIZE << endl;
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
            int batch_limit = world_size;

            while (!q.priority.empty() &&
                   history < generate_n &&
                   active_count < batch_limit)
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

        if (rank != 0)
        {
            batch_pts.resize(active_count);
        }

        for (int i = 0; i < active_count; i += 1)
        {
            PT tmp_pt;

            if (rank == 0)
            {
                tmp_pt = batch_pts[i];
            }

            BcastPT(tmp_pt, rank);

            if (rank != 0)
            {
                batch_pts[i] = tmp_pt;
            }
        }

        MPILocalResult local_total;
        local_total.generated = 0;
        local_total.cracked = 0;
        local_total.generate_time = 0.0;
        local_total.hash_time = 0.0;
        local_total.compute_time = 0.0;

        int task_id = 0;

        for (int pt_idx = 0; pt_idx < active_count; pt_idx += 1)
        {
            const PT &pt = batch_pts[pt_idx];

            if (pt.content.empty())
            {
                continue;
            }

            int last_idx = (int)pt.content.size() - 1;
            int n = pt.max_indices[last_idx];

            for (int start_idx = 0; start_idx < n; start_idx += MPI_PT_CHUNK_SIZE)
            {
                int end_idx = min(start_idx + MPI_PT_CHUNK_SIZE, n);

                if (task_id % world_size == rank)
                {
                    MPILocalResult one = GenerateAndHashPTRange(
                        q.m,
                        pt,
                        test_set,
                        start_idx,
                        end_idx
                    );

                    local_total.generated += one.generated;
                    local_total.cracked += one.cracked;
                    local_total.generate_time += one.generate_time;
                    local_total.hash_time += one.hash_time;
                    local_total.compute_time += one.compute_time;
                }

                task_id += 1;
            }
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