#include <iostream>
#include <CL/sycl.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <mpi.h>

using namespace std;
using namespace cl::sycl;

const int N = 500; // �����ģ500,1000,1500
float M[N][N];

// ������������
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

// �����㷨
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

    srand(time(nullptr)); // ��ʼ�����������

    m_reset();

    // ��������ڵ�ͨ����
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm);

    // ��ȡ����ڵ�ID������
    int node_id, num_nodes;
    MPI_Comm_rank(comm, &node_id);
    MPI_Comm_size(comm, &num_nodes);

    // ����ÿ���ڵ���Ҫ����Ŀ��С
    int block_size = N / num_nodes;
    int start = node_id * block_size;
    int end = start + block_size;

    // �������в���ʼ��SYCL�豸
    queue q{ property::queue::in_order() };
    device dev = q.get_device();
    auto dev_type = dev.get_info<info::device::device_type>();

    // ���Դ����㷨��ִ��ʱ��
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

    // �ȴ����н�����ɴ��м���
    MPI_Barrier(MPI_COMM_WORLD);

    // �����SYCL�豸����ʹ��SYCL�����㷨
    if (dev_type == info::device_type::gpu || dev_type == info::device_type::accelerator)
    {
        auto startParallel = chrono::steady_clock::now();

        Parallel(q, start, end);

        auto endParallel = chrono::steady_clock::now();
        auto durationParallel = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

        // ���ܲ���ӡ����ִ��ʱ��
        int totalDuration;
        MPI_Reduce(&durationParallel, &totalDuration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            cout << "Parallel execution time: " << totalDuration << " milliseconds" << endl;
    }
    else // �����CPU�豸����ʹ��OpenMP�����㷨
    {
        auto startParallel = chrono::steady_clock::now();

#pragma omp parallel
        {
            Parallel(q, start, end);
        }

        auto endParallel = chrono::steady_clock::now();
        auto durationParallel = chrono::duration_cast<chrono::milliseconds>(endParallel - startParallel);

        // ���ܲ���ӡ����ִ��ʱ��
        int totalDuration;
        MPI_Reduce(&durationParallel, &totalDuration, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0)
            cout << "Parallel execution time: " << totalDuration << " milliseconds" << endl;
    }

    // ���ܽ��
    MPI_Gather(&M[start][0], block_size * N, MPI_FLOAT, &M[0][0], block_size * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
