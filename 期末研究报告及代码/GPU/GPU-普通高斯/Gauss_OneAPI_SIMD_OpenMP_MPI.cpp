#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace cl::sycl;

const int N = 500; // 问题规模500,1000,1500
float M[N][N];

// 测试用例生成
void m_reset()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            M[i][j] = 0;

            if (i == j)
                M[i][j] = 1.0f;
            else if (j > i)
                M[i][j] = rand();
        }
    }
}

// 并行算法
void Parallel(queue& q, int start, int end)
{
    buffer<float, 2> buffer_M(M, range<2>(N, N));

    q.submit([&](handler& h) {
        auto accessor_M = buffer_M.get_access<access::mode::read_write>(h);

        h.parallel_for(range<1>(end - start), [=](id<1> idx) {
            int k = start + idx[0];

            float inv_kk = 1.0f / accessor_M[k][k];

            for (int j = k + 1; j < N; j++)
                accessor_M[k][j] *= inv_kk;

            accessor_M[k][k] = 1.0f;

            for (int i = k + 1; i < N; i++)
            {
                float mik = accessor_M[i][k];
                for (int j = k + 1; j < N; j++)
                    accessor_M[i][j] -= mik * accessor_M[k][j];
                accessor_M[i][k] = 0;
            }
            });
        });
}

int main(int argc, char* argv[])
{
    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(nullptr)); // 初始化随机数种子

    m_reset();

    // 创建计算节点通信组
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm);

    // 获取计算节点ID和总数
    int node_id, num_nodes;
    MPI_Comm_rank(comm, &node_id);
    MPI_Comm_size(comm, &num_nodes);

    // 计算每个节点需要处理的块大小
    int block_size = N / num_nodes;
    int start = node_id * block_size;
    int end = start + block_size;

    // 创建队列并初始化SYCL设备
    queue q{ property::queue::in_order() };
    device dev = q.get_device();
    auto dev_type = dev.get_info<info::device::device_type>();

    // 测试串行算法的执行时间
    if (rank == 0)
    {
        auto startSerial = chrono::steady_clock::now();

        for (int k = 0; k < N; k++)
        {
            float inv_kk = 1.0f / M[k][k];
            for (int j = k + 1; j < N; j++)
                M[k][j] *= inv_kk;
            M[k][k] = 1.0f;

            for (int i = k + 1; i < N; i++)
            {
                float mik = M[i][k];
                for (int j = k + 1; j < N; j++)
                    M[i][j] -= mik * M[k][j];
                M[i][k] = 0;
            }
        }

        auto endSerial = chrono::steady_clock::now();
        auto durationSerial = chrono::duration_cast<chrono::milliseconds>(endSerial - startSerial);

        cout << "Serial execution time: " << durationSerial.count() << " milliseconds" << endl;
    }

    // 等待所有进程完成串行计算
    MPI_Barrier(MPI_COMM_WORLD);

    // 如果是SYCL设备，则使用SYCL并行算法
    if (dev_type == info::device_type::gpu || dev_type == info::device_type::accelerator)
    {
        auto startParallel = chrono::steady_clock::now();

        Parallel(q, start, end);

        auto endParallel = chrono::steady_clock::now();
        auto durationParallel = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

        // 汇总并打印并行执行时间
        int totalDuration;
        MPI_Reduce(&durationParallel, &totalDuration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            cout << "Parallel execution time: " << totalDuration << " milliseconds" << endl;
    }
    else // 如果是CPU设备，则使用OpenMP并行算法
    {
        auto startParallel = chrono::steady_clock::now();

#pragma omp parallel
        {
            Parallel(q, start, end);
        }

        auto endParallel = chrono::steady_clock::now();
        auto durationParallel = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

        // 汇总并打印并行执行时间
        int totalDuration;
        MPI_Reduce(&durationParallel, &totalDuration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            cout << "Parallel execution time: " << totalDuration << " milliseconds" << endl;
    }

    // 汇总结果
    MPI_Gather(&M[start][0], block_size * N, MPI_FLOAT, &M[0][0], block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
