#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>

using namespace std;
using namespace chrono;

extern double time_generate_only;
extern double time_priority_only;

int main()
{
    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;

    PriorityQueue q;

    auto start_train = system_clock::now();

    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();

    auto end_train = system_clock::now();
    auto duration_train = duration_cast<microseconds>(end_train - start_train);

    time_train = double(duration_train.count()) *
                 microseconds::period::num /
                 microseconds::period::den;

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

    int cracked = 0;

    q.init();

    q.guess_refs.reserve(1200000);

    cout << "here" << endl;

    size_t curr_num = 0;
    size_t history = 0;

    auto start = system_clock::now();

    auto materialize_guess = [&](const GuessRef &ref) -> string
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
    };

    while (!q.priority.empty())
    {
        q.PopNext();

        q.total_guesses = q.guess_refs.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            const size_t generate_n = 10000000;

            if (history + q.total_guesses > generate_n)
            {
                auto end = system_clock::now();
                auto duration = duration_cast<microseconds>(end - start);

                time_guess = double(duration.count()) *
                             microseconds::period::num /
                             microseconds::period::den;

                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                cout << "Cracked:" << cracked << endl;
                cout << "Generate only time:" << time_generate_only << "seconds" << endl;
                cout << "Priority/control time:" << time_priority_only << "seconds" << endl;

                break;
            }
        }

        if (curr_num > 1000000)
        {
            auto start_hash = system_clock::now();

            size_t idx = 0;

            for (; idx + 3 < q.guess_refs.size(); idx += 4)
            {
                string pw0 = materialize_guess(q.guess_refs[idx]);
                string pw1 = materialize_guess(q.guess_refs[idx + 1]);
                string pw2 = materialize_guess(q.guess_refs[idx + 2]);
                string pw3 = materialize_guess(q.guess_refs[idx + 3]);

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

            for (; idx < q.guess_refs.size(); idx += 1)
            {
                string pw = materialize_guess(q.guess_refs[idx]);

                if (test_set.find(pw) != test_set.end())
                {
                    cracked += 1;
                }

                bit32 state[4];
                MD5Hash(pw, state);
            }

            auto end_hash = system_clock::now();
            auto duration = duration_cast<microseconds>(end_hash - start_hash);

            time_hash += double(duration.count()) *
                         microseconds::period::num /
                         microseconds::period::den;

            history += curr_num;
            curr_num = 0;

            q.guess_refs.clear();
            q.prefix_storage.clear();
        }
    }

    return 0;
}