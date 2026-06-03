#include "PCFG.h"
#include "guessing_mpi.h"
#include "mpi.h"
#include "train_mpi.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int world_size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 非 0 号进程关闭 cout，避免多个进程交替输出训练进度
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
        int active = 0;
        PT curr_pt;

        if (rank == 0)
        {
            if (!q.priority.empty() && history < generate_n)
            {
                active = 1;

                curr_pt = q.priority.top();
                q.priority.pop();
            }
            else
            {
                active = 0;
            }
        }

        MPI_Bcast(&active, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!active)
        {
            break;
        }

        BcastPT(curr_pt, rank);

        MPILocalResult local_result = GenerateAndHashPTMPI(
            q.m,
            curr_pt,
            test_set,
            rank,
            world_size
        );

       local_generate_sum += local_result.generate_time;
local_hash_sum += local_result.hash_time;
local_compute_sum += local_result.compute_time;

        long long global_generated = 0;
        long long global_cracked = 0;

        MPI_Reduce(
            &local_result.generated,
            &global_generated,
            1,
            MPI_LONG_LONG,
            MPI_SUM,
            0,
            MPI_COMM_WORLD
        );

        MPI_Reduce(
            &local_result.cracked,
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

            vector<PT> new_pts = curr_pt.NewPTs();

            for (PT pt : new_pts)
            {
                q.CalProb(pt);
                q.priority.push(pt);
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
    double comm_control_time = total_time - compute_time;

    if (comm_control_time < 0.0)
    {
        comm_control_time = 0.0;
    }

   double guess_time = total_time - hash_time;

if (guess_time < 0.0)
{
    guess_time = 0.0;
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