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

    int *a = new int[n];
    int **A = new int*[n];

    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        A[i] = new int[n];
        for (int j = 0; j < n; j++)
        {
            A[i][j] = i + j;
        }
    }

    long long *sum = new long long[n];

    struct timeval tv_start, tv_end;
    gettimeofday(&tv_start, NULL);

    for (int r = 0; r < repeat; r++)
    {
        for (int i = 0; i < n; i++)
        {
            sum[i] = 0;
            for (int j = 0; j < n; j++)
            {
                sum[i] += 1LL * A[j][i] * a[j];
            }
        }
    }

    gettimeofday(&tv_end, NULL);

    double elapsed =
        (tv_end.tv_sec - tv_start.tv_sec) +
        (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;

    long long checksum = 0;
    for (int i = 0; i < n; i++)
    {
        checksum += sum[i];
    }

    cout << "total time = " << elapsed << endl;
    cout << "avg time = " << elapsed / repeat << endl;
    cout << "checksum = " << checksum << endl;

    delete[] a;
    delete[] sum;
    for (int i = 0; i < n; i++)
    {
        delete[] A[i];
    }
    delete[] A;

    return 0;
}
