#include "PCFG.h"
#include "guessing_mpi.h"
#include "train_mpi.h"
#include "mpi.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <deque>
#include <algorithm>

using namespace std;

#ifndef PIPE_TASK_CHUNK_SIZE
#define PIPE_TASK_CHUNK_SIZE 50000
#endif

static const int TAG_TASK_SIZE = 200;
static const int TAG_TASK_DATA = 201;
static const int TAG_TASK_PROB = 202;
static const int TAG_RESULT_COUNT = 203;
static const int TAG_RESULT_TIME = 204;

struct RangeTask
{
    PT pt;
    int start_idx;
    int end_idx;
};

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

static vector<int> PackRangeTask(const RangeTask &task, float probs[2])
{
    vector<int> data;

    data.emplace_back(task.start_idx);
    data.emplace_back(task.end_idx);

    PackPTToVector(task.pt, data);

    probs[0] = task.pt.preterm_prob;
    probs[1] = task.pt.prob;

    return data;
}

static RangeTask UnpackRangeTask(const vector<int> &data, const float probs[2])
{
    RangeTask task;

    int pos = 0;

    task.start_idx = data[pos++];
    task.end_idx = data[pos++];

    task.pt = UnpackPTFromVector(
        data,
        pos,
        probs[0],
        probs[1]
    );

    return task;
}

static void SendRangeTask(int worker_rank, const RangeTask &task)
{
    float probs[2] = {0.0f, 0.0f};
    vector<int> data = PackRangeTask(task, probs);

    int data_size = (int)data.size();

    MPI_Send(
        &data_size,
        1,
        MPI_INT,
        worker_rank,
        TAG_TASK_SIZE,
        MPI_COMM_WORLD
    );

    MPI_Send(
        data.data(),
        data_size,
        MPI_INT,
        worker_rank,
        TAG_TASK_DATA,
        MPI_COMM_WORLD
    );

    MPI_Send(
        probs,
        2,
        MPI_FLOAT,
        worker_rank,
        TAG_TASK_PROB,
        MPI_COMM_WORLD
    );
}

static void SendStop(int worker_rank)
{
    int data_size = 0;

    MPI_Send(
        &data_size,
        1,
        MPI_INT,
        worker_rank,
        TAG_TASK_SIZE,
        MPI_COMM_WORLD
    );
}

static bool ReceiveRangeTask(RangeTask &task)
{
    int data_size = 0;

    MPI_Recv(
        &data_size,
        1,
        MPI_INT,
        0,
        TAG_TASK_SIZE,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    if (data_size == 0)
    {
        return false;
    }

    vector<int> data(data_size);
    float probs[2] = {0.0f, 0.0f};

    MPI_Recv(
        data.data(),
        data_size,
        MPI_INT,
        0,
        TAG_TASK_DATA,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    MPI_Recv(
        probs,
        2,
        MPI_FLOAT,
        0,
        TAG_TASK_PROB,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE
    );

    task = UnpackRangeTask(data, probs);

    return true;
}

static void WorkerLoop(
    model &m,
    const unordered_set<string> &test_set
)
{
    while (true)
    {
        RangeTask task;

        bool active = ReceiveRangeTask(task);

        if (!active)
        {
            break;
        }

        MPILocalResult res = GenerateAndHashPTRange(
            m,
            task.pt,
            test_set,
            task.start_idx,
            task.end_idx
        );

        long long result_count[2];
        result_count[0] = res.generated;
        result_count[1] = res.cracked;

        double result_time[3];
        result_time[0] = res.generate_time;
        result_time[1] = res.hash_time;
        result_time[2] = res.compute_time;

        MPI_Send(
            result_count,
            2,
            MPI_LONG_LONG,
            0,
            TAG_RESULT_COUNT,
            MPI_COMM_WORLD
        );

        MPI_Send(
            result_time,
            3,
            MPI_DOUBLE,
            0,
            TAG_RESULT_TIME,
            MPI_COMM_WORLD
        );
    }
}

static void PushTasksFromNextPT(
    PriorityQueue &q,
    deque<RangeTask> &task_queue,
    long long &scheduled,
    long long generate_limit
)
{
    if (q.priority.empty() || scheduled >= generate_limit)
    {
        return;
    }

    PT pt = q.priority.top();
    q.priority.pop();

    if (!pt.content.empty())
    {
        int last_idx = (int)pt.content.size() - 1;
        int n = pt.max_indices[last_idx];

        for (int start_idx = 0; start_idx < n; start_idx += PIPE_TASK_CHUNK_SIZE)
        {
            int end_idx = min(start_idx + PIPE_TASK_CHUNK_SIZE, n);

            RangeTask task;
            task.pt = pt;
            task.start_idx = start_idx;
            task.end_idx = end_idx;

            task_queue.emplace_back(task);
        }

        scheduled += n;
    }

    vector<PT> new_pts = pt.NewPTs();

    for (PT new_pt : new_pts)
    {
        q.CalProb(new_pt);
        q.priority.push(new_pt);
    }
}

static bool GetNextTask(
    PriorityQueue &q,
    deque<RangeTask> &task_queue,
    long long &scheduled,
    long long generate_limit,
    RangeTask &task
)
{
    while (task_queue.empty() &&
           !q.priority.empty() &&
           scheduled < generate_limit)
    {
        PushTasksFromNextPT(
            q,
            task_queue,
            scheduled,
            generate_limit
        );
    }

    if (task_queue.empty())
    {
        return false;
    }

    task = task_queue.front();
    task_queue.pop_front();

    return true;
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

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_size < 2)
    {
        if (rank == 0)
        {
            cout.clear();
            cout << "Task pipeline mode requires at least 2 MPI processes." << endl;
        }

        MPI_Finalize();
        return 0;
    }

    if (rank != 0)
    {
        WorkerLoop(
            q.m,
            test_set
        );

        MPI_Finalize();
        return 0;
    }

    cout.clear();

    q.init();

    cout << "MPI world size:" << world_size << endl;
    cout << "Advanced mode: range-task generation-hash pipeline." << endl;
    cout << "Task chunk size:" << PIPE_TASK_CHUNK_SIZE << endl;
    cout << "here" << endl;

    const long long generate_n = 10000000;

    long long scheduled = 0;
    long long generated_done = 0;
    long long cracked = 0;
    long long next_report = 1000000;

    int worker_count = world_size - 1;
    int outstanding = 0;

    deque<RangeTask> task_queue;

    vector<double> worker_generate_time(world_size, 0.0);
    vector<double> worker_hash_time(world_size, 0.0);
    vector<double> worker_compute_time(world_size, 0.0);

    double loop_start = MPI_Wtime();

    auto send_next_task = [&](int worker_rank) -> bool
    {
        RangeTask task;

        bool ok = GetNextTask(
            q,
            task_queue,
            scheduled,
            generate_n,
            task
        );

        if (!ok)
        {
            return false;
        }

        SendRangeTask(worker_rank, task);

        outstanding += 1;

        return true;
    };

    for (int worker = 1; worker < world_size; worker += 1)
    {
        bool ok = send_next_task(worker);

        if (!ok)
        {
            break;
        }
    }

    while (outstanding > 0)
    {
        long long result_count[2];
        double result_time[3];

        MPI_Status status;

        MPI_Recv(
            result_count,
            2,
            MPI_LONG_LONG,
            MPI_ANY_SOURCE,
            TAG_RESULT_COUNT,
            MPI_COMM_WORLD,
            &status
        );

        int worker_rank = status.MPI_SOURCE;

        MPI_Recv(
            result_time,
            3,
            MPI_DOUBLE,
            worker_rank,
            TAG_RESULT_TIME,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        outstanding -= 1;

        generated_done += result_count[0];
        cracked += result_count[1];

        worker_generate_time[worker_rank] += result_time[0];
        worker_hash_time[worker_rank] += result_time[1];
        worker_compute_time[worker_rank] += result_time[2];

        while (generated_done >= next_report)
        {
            cout << "Guesses processed: " << next_report << endl;
            next_report += 1000000;
        }

        bool ok = send_next_task(worker_rank);

        if (!ok)
        {
            continue;
        }
    }

    for (int worker = 1; worker < world_size; worker += 1)
    {
        SendStop(worker);
    }

    double loop_end = MPI_Wtime();

    double total_time = loop_end - loop_start;

    double generate_time = 0.0;
    double hash_time = 0.0;
    double compute_time = 0.0;

    for (int worker = 1; worker < world_size; worker += 1)
    {
        if (worker_generate_time[worker] > generate_time)
        {
            generate_time = worker_generate_time[worker];
        }

        if (worker_hash_time[worker] > hash_time)
        {
            hash_time = worker_hash_time[worker];
        }

        if (worker_compute_time[worker] > compute_time)
        {
            compute_time = worker_compute_time[worker];
        }
    }

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
    cout << "Generated:" << generated_done << endl;

    cout << "MPI total Guess+Hash time:" << total_time << "seconds" << endl;
    cout << "Pipeline task generate only time:" << generate_time << "seconds" << endl;
    cout << "Pipeline task compute time:" << compute_time << "seconds" << endl;
    cout << "Pipeline task comm/control time:" << comm_control_time << "seconds" << endl;

    MPI_Finalize();

    return 0;
}