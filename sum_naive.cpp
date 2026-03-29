#include <iostream>
#include <sys/time.h>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 3) {
        cerr << "usage: " << argv[0] << " n repeat" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    int repeat = atoi(argv[2]);

    long long *a = new long long[n];
    for (int i = 0; i < n; i++) {
        a[i] = i;
    }

    long long sum = 0;
    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    for (int r = 0; r < repeat; r++) {
        sum = 0;
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }
    }

    gettimeofday(&tv_end, NULL);

    double elapsed =
        (tv_end.tv_sec - tv_start.tv_sec) +
        (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;

    cout << "total time = " << elapsed << endl;
    cout << "avg time = " << elapsed / repeat << endl;
    cout << "sum = " << sum << endl;

    delete[] a;
    return 0;
}
