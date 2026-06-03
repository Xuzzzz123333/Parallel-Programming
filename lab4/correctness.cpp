#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <sstream>

using namespace std;
using namespace chrono;

// 编译指令如下：
// g++ correctness.cpp train.cpp guessing.cpp md5.cpp -o test_correct -O1 -march=native

static string StateToHex(bit32 state[4])
{
    stringstream ss;
    for (int i = 0; i < 4; i++)
    {
        ss << setw(8) << setfill('0') << hex << state[i];
    }
    return ss.str();
}

int main()
{
    // 先选4条短口令，保证第一版SIMD一定会走4路NEON逻辑
    string inputs[4] = {
        "123456",
        "password",
        "12345678",
        "qwerty"
    };

    // 串行结果
    bit32 serial_states[4][4];
    for (int i = 0; i < 4; i++)
    {
        MD5Hash(inputs[i], serial_states[i]);
    }

    // SIMD结果
    bit32 simd_states[4][4];
    MD5HashSIMD4(inputs, simd_states);

    // 对比输出
    bool ok = true;
    for (int i = 0; i < 4; i++)
    {
        string serial_hex = StateToHex(serial_states[i]);
        string simd_hex = StateToHex(simd_states[i]);

        cout << "input[" << i << "] = " << inputs[i] << endl;
        cout << "serial = " << serial_hex << endl;
        cout << "simd   = " << simd_hex << endl;

        if (serial_hex != simd_hex)
        {
            ok = false;
            cout << "Mismatch at input[" << i << "]!" << endl;
        }

        cout << endl;
    }

    if (ok)
    {
        cout << "MD5HashSIMD4 correctness test passed!" << endl;
    }
    else
    {
        cout << "MD5HashSIMD4 correctness test failed!" << endl;
        return 1;
    }

    return 0;
}