#include "md5.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;
using namespace chrono;

static vector<string> generate_passwords(size_t n)
{
    vector<string> passwords;
    passwords.reserve(n);

    for (size_t i = 0; i < n; i++)
    {
        // 保持为短口令，确保padding后大多数都是单个512-bit block
        passwords.push_back("pw" + to_string(i) + "A");
    }

    return passwords;
}

static unsigned long long run_serial(const vector<string> &passwords)
{
    unsigned long long checksum = 0;

    for (const string &pw : passwords)
    {
        bit32 state[4];
        MD5Hash(pw, state);

        checksum ^= state[0];
        checksum += state[1];
        checksum ^= state[2];
        checksum += state[3];
    }

    return checksum;
}

static unsigned long long run_simd4(const vector<string> &passwords)
{
    unsigned long long checksum = 0;
    size_t n = passwords.size();

    size_t i = 0;
    for (; i + 3 < n; i += 4)
    {
        bit32 states[4][4];

MD5HashSIMD4AsmRef(
    passwords[i],
    passwords[i + 1],
    passwords[i + 2],
    passwords[i + 3],
    states
);

        for (int lane = 0; lane < 4; lane++)
        {
            checksum ^= states[lane][0];
            checksum += states[lane][1];
            checksum ^= states[lane][2];
            checksum += states[lane][3];
        }
    }

    // 尾部不足4条时回退到串行
    for (; i < n; i++)
    {
        bit32 state[4];
        MD5Hash(passwords[i], state);

        checksum ^= state[0];
        checksum += state[1];
        checksum ^= state[2];
        checksum += state[3];
    }

    return checksum;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " serial|simd N" << endl;
        return 1;
    }

    string mode = argv[1];
    size_t n = strtoull(argv[2], nullptr, 10);

    vector<string> passwords = generate_passwords(n);

    auto start = system_clock::now();

    unsigned long long checksum = 0;
    if (mode == "serial")
    {
        checksum = run_serial(passwords);
    }
    else if (mode == "simd")
    {
        checksum = run_simd4(passwords);
    }
    else
    {
        cerr << "Unknown mode: " << mode << endl;
        return 1;
    }

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    double elapsed = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    cout << "mode = " << mode << endl;
    cout << "N = " << n << endl;
    cout << "hash time = " << elapsed << " seconds" << endl;
    cout << "throughput = " << (double)n / elapsed << " hashes/sec" << endl;
    cout << "checksum = " << checksum << endl;

    return 0;
}