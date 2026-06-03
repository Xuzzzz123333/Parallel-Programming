#include "PCFG.h"
#include "train_mpi.h"
#include "md5.h"
#include "mpi.h"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <algorithm>

using namespace std;

#ifndef PIPE_BATCH_SIZE
#define PIPE_BATCH_SIZE 100000
#endif

static const int TAG_SIZE = 100;
static const int TAG_DATA = 101;
static const int TAG_RESULT_COUNT = 102;
static const int TAG_RESULT_TIME = 103;

static void AppendInt(string &buf, int x)
{
    const char *p = reinterpret_cast<const char *>(&x);
    buf.append(p, sizeof(int));
}

static int ReadInt(const string &buf, size_t &pos)
{
    int x = 0;
    memcpy(&x, buf.data() + pos, sizeof(int));
    pos += sizeof(int);
    return x;
}

static void AppendString(string &buf, const string &s)
{
    AppendInt(buf, (int)s.size());
    buf.append(s.data(), s.size());
}

static string ReadString(const string &buf, size_t &pos)
{
    int len = ReadInt(buf, pos);
    string s(buf.data() + pos, len);
    pos += len;
    return s;
}

static string MaterializeGuess(const GuessRef &ref)
{
    const string &val = (*(ref.values))[ref.value_idx];

    if (!ref.has_prefix)
    {
        return val;
    }

    const string &prefix = *(ref.prefix);

    string result;
    result.reserve(prefix.size() + val.size());
    result.append(prefix);
    result.append(val);

    return result;
}

static bool GenerateNextGuessBatch(
    PriorityQueue &q,
    long long already_sent,
    long long generate_limit,
    string &buf,
    long long &batch_count
)
{
    buf.clear();
    batch_count = 0;

    q.guess_refs.clear();
    q.prefix_storage.clear();

    while (!q.priority.empty() &&
           (long long)q.guess_refs.size() < PIPE_BATCH_SIZE &&
           already_sent + (long long)q.guess_refs.size() < generate_limit)
    {
        q.PopNext();
    }

    if (q.guess_refs.empty())
    {
        return false;
    }

    batch_count = (long long)q.guess_refs.size();

    AppendInt(buf, (int)batch_count);

    for (size_t i = 0; i < q.guess_refs.size(); i += 1)
    {
        string guess = MaterializeGuess(q.guess_refs[i]);
        AppendString(buf, guess);
    }

    q.guess_refs.clear();
    q.prefix_storage.clear();

    return true;
}

static long long HashGuessBuffer(
    const string &buf,
    const unordered_set<string> &test_set,
    long long &generated
)
{
    size_t pos = 0;

    int count = ReadInt(buf, pos);
    generated = count;

    vector<string> guesses;
    guesses.reserve(count);

    for (int i = 0; i < count; i += 1)
    {
        guesses.emplace_back(ReadString(buf, pos));
    }

    long long cracked = 0;

    size_t idx = 0;

    for (; idx + 3 < guesses.size(); idx += 4)
    {
        const string &pw0 = guesses[idx];
        const string &pw1 = guesses[idx + 1];
        const string &pw2 = guesses[idx + 2];
        const string &pw3 = guesses[idx + 3];

        if (test_set.find(pw0) != test_set.end())
        {
            cracked += 1;
        }

        if (test_set.find(pw1) != test_set.end())
        {
            cracked += 1;
        }

        if (test_set.find(pw2) != test_set.end())
        {
            cracked += 1;
        }

        if (test_set.find(pw3) != test_set.end())
        {
            cracked += 1;
        }

        bit32 states[4][4];
        MD5HashSIMD4Ref(pw0, pw1, pw2, pw3, states);
    }

    for (; idx < guesses.size(); idx += 1)
    {
        const string &pw = guesses[idx];

        if (test_set.find(pw) != test_set.end())
        {
            cracked += 1;
        }

        bit32 state[4];
        MD5Hash(pw, state);
    }

    return cracked;
}

static void WorkerLoop(const unordered_set<string> &test_set)
{
    while (true)
    {
        int buf_size = 0;

        MPI_Recv(
            &buf_size,
            1,
            MPI_INT,
            0,
            TAG_SIZE,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        if (buf_size == 0)
        {
            break;
        }

        string buf;
        buf.resize(buf_size);

        MPI_Recv(
            &buf[0],
            buf_size,
            MPI_CHAR,
            0,
            TAG_DATA,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        double hash_start = MPI_Wtime();

        long long generated = 0;
        long long cracked = HashGuessBuffer(buf, test_set, generated);

        double hash_end = MPI_Wtime();

        long long result_count[2];
        result_count[0] = generated;
        result_count[1] = cracked;

        double hash_time = hash_end - hash_start;

        MPI_Send(
            result_count,
            2,
            MPI_LONG_LONG,
            0,
            TAG_RESULT_COUNT,
            MPI_COMM_WORLD
        );

        MPI_Send(
            &hash_time,
            1,
            MPI_DOUBLE,
            0,
            TAG_RESULT_TIME,
            MPI_COMM_WORLD
        );
    }
}

static void SendTaskToWorker(int worker_rank, const string &buf)
{
    int buf_size = (int)buf.size();

    MPI_Send(
        &buf_size,
        1,
        MPI_INT,
        worker_rank,
        TAG_SIZE,
        MPI_COMM_WORLD
    );

    MPI_Send(
        buf.data(),
        buf_size,
        MPI_CHAR,
        worker_rank,
        TAG_DATA,
        MPI_COMM_WORLD
    );
}

static void SendStopToWorker(int worker_rank)
{
    int stop_size = 0;

    MPI_Send(
        &stop_size,
        1,
        MPI_INT,
        worker_rank,
        TAG_SIZE,
        MPI_COMM_WORLD
    );
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
            cout << "Pipeline mode requires at least 2 MPI processes." << endl;
        }

        MPI_Finalize();
        return 0;
    }

    if (rank != 0)
    {
        WorkerLoop(test_set);
        MPI_Finalize();
        return 0;
    }

    cout.clear();

    q.init();

    cout << "MPI world size:" << world_size << endl;
    cout << "Advanced mode: generation-hash pipeline." << endl;
    cout << "Pipeline batch size:" << PIPE_BATCH_SIZE << endl;
    cout << "here" << endl;

    const long long generate_n = 10000000;

    long long sent = 0;
    long long generated_done = 0;
    long long cracked = 0;
    long long next_report = 1000000;

    int worker_count = world_size - 1;
    int next_worker = 1;
    int outstanding = 0;

    vector<double> worker_hash_time(world_size, 0.0);

    double generator_time = 0.0;

    double loop_start = MPI_Wtime();

    auto send_next_batch = [&](int worker_rank) -> bool
    {
        string buf;
        long long batch_count = 0;

        double gen_start = MPI_Wtime();

        bool ok = GenerateNextGuessBatch(
            q,
            sent,
            generate_n,
            buf,
            batch_count
        );

        double gen_end = MPI_Wtime();

        generator_time += gen_end - gen_start;

        if (!ok)
        {
            return false;
        }

        SendTaskToWorker(worker_rank, buf);

        sent += batch_count;
        outstanding += 1;

        return true;
    };

    for (int i = 0; i < worker_count; i += 1)
    {
        if (sent >= generate_n)
        {
            break;
        }

        bool ok = send_next_batch(next_worker);

        if (!ok)
        {
            break;
        }

        next_worker += 1;

        if (next_worker >= world_size)
        {
            next_worker = 1;
        }
    }

    while (outstanding > 0)
    {
        long long result_count[2];

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

        double worker_time = 0.0;

        MPI_Recv(
            &worker_time,
            1,
            MPI_DOUBLE,
            worker_rank,
            TAG_RESULT_TIME,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );

        outstanding -= 1;

        generated_done += result_count[0];
        cracked += result_count[1];
        worker_hash_time[worker_rank] += worker_time;

        while (generated_done >= next_report)
        {
            cout << "Guesses processed: " << next_report << endl;
            next_report += 1000000;
        }

        if (sent < generate_n)
        {
            bool ok = send_next_batch(worker_rank);

            if (!ok)
            {
                continue;
            }
        }
    }

    for (int worker = 1; worker < world_size; worker += 1)
    {
        SendStopToWorker(worker);
    }


    double loop_end = MPI_Wtime();

    double total_time = loop_end - loop_start;

    double hash_time = 0.0;

    for (int worker = 1; worker < world_size; worker += 1)
    {
        if (worker_hash_time[worker] > hash_time)
        {
            hash_time = worker_hash_time[worker];
        }
    }

    double guess_time = total_time - hash_time;

    if (guess_time < 0.0)
    {
        guess_time = 0.0;
    }

    double overlap_saved = generator_time + hash_time - total_time;

    if (overlap_saved < 0.0)
    {
        overlap_saved = 0.0;
    }

    cout << "Guess time:" << guess_time << "seconds" << endl;
    cout << "Hash time:" << hash_time << "seconds" << endl;
    cout << "Train time:" << train_time << "seconds" << endl;
    cout << "Cracked:" << cracked << endl;
    cout << "Generated:" << generated_done << endl;

    cout << "MPI total Guess+Hash time:" << total_time << "seconds" << endl;
    cout << "Pipeline generator time:" << generator_time << "seconds" << endl;
    cout << "Pipeline overlap saved time:" << overlap_saved << "seconds" << endl;

    MPI_Finalize();

    return 0;
}