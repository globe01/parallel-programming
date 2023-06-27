#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>

using namespace std;
using namespace cl::sycl;

const int N = 250; // 问题规模250,500,1000
float M[N][N];

// 测试用例生成
void m_reset(queue& q)
{
    buffer<float, 2> buffer_M(M, range<2>(N, N));

    q.submit([&](handler& h) {
        auto accessor_M = buffer_M.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(N, N), [=](id<2> idx) {
            int i = idx[0];
            int j = idx[1];

            accessor_M[i][j] = 0;

            if (i == j)
                accessor_M[i][j] = 1.0f;
            else if (j > i)
                accessor_M[i][j] = rand();
            });

        h.parallel_for(range<1>(N), [=](id<1> idx) {
            for (int j = idx[0] + 1; j < N; j++)
                accessor_M[idx[0]][j] += accessor_M[idx[0]][idx[0]];
            });
        });
}

// 串行算法
void Serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            M[k][j] *= 1.0f / M[k][k];
        M[k][k] = 1.0f;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                M[i][j] -= M[i][k] * M[k][j];
            M[i][k] = 0;
        }
    }
}

// 并行算法
void Parallel(queue& q)
{
    buffer<float, 2> buffer_M(M, range<2>(N, N));

    q.submit([&](handler& h) {
        auto accessor_M = buffer_M.get_access<access::mode::read_write>(h);

        h.parallel_for(range<1>(N), [=](id<1> idx) {
            int k = idx[0];

            for (int j = k + 1; j < N; j++)
                accessor_M[k][j] *= 1.0f / accessor_M[k][k];

            accessor_M[k][k] = 1.0f;

            for (int i = k + 1; i < N; i++)
            {
                for (int j = k + 1; j < N; j++)
                    accessor_M[i][j] -= accessor_M[i][k] * accessor_M[k][j];
                accessor_M[i][k] = 0;
            }
            });
        });
}

int main()
{
    queue q;

    m_reset(q);

    // 测试串行算法的执行时间
    auto startSerial = chrono::steady_clock::now();

    Serial();

    auto endSerial = chrono::steady_clock::now();
    auto durationSerial = chrono::duration_cast<chrono::milliseconds>(endSerial - startSerial);

    cout << "Serial execution time: " << durationSerial.count() << " milliseconds" << endl;

    // 测试并行算法的执行时间
    auto startParallel = chrono::steady_clock::now();

    Parallel(q);

    auto endParallel = chrono::steady_clock::now();
    auto durationParallel = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

    cout << "Parallel execution time: " << durationParallel.count() << " milliseconds" << endl;

    return 0;
}
