//块划分 + 结合SIMD OpenMP

#include <iostream>
#include <mpi.h> // mpi头文件
#include <sys/time.h>
//#include <xmmintrin.h> //SSE
//#include <emmintrin.h> //SSE2
//#include <pmmintrin.h> //SSE3
//#include <tmmintrin.h> //SSSE3
//#include <smmintrin.h> //SSE4.1
//#include <nmmintrin.h> //SSSE4.2
//#include <immintrin.h> //AVX、AVX2、AVX-512
#include <omp.h>

using namespace std;

static const int N = 1000;
static const int threadNum = 4;

float tmp_M[N][N];
float M[N][N];

void m_reset(float tmp_M[][N])
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            tmp_M[i][j] = 0;
        }
        tmp_M[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            tmp_M[i][j] = rand() % 100;
    }
    for (int k = 0; k < N; k++)
    {
        for (int i = k + 1; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                tmp_M[i][j] += tmp_M[k][j];
            }
        }
    }
}




void newM(float M[][N], float tmp_M[][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[i][j] = tmp_M[i][j];
}



//串行
void Serial() {
    newM(M, tmp_M);
    timeval t_start;
    timeval t_end;
    gettimeofday(&t_start, NULL);

    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            M[k][j] = M[k][j] * 1.0 / M[k][k];
        }
        M[k][k] = 1.0;

        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                M[i][j] = M[i][j] - M[i][k] * M[k][j];
            }
            M[i][k] = 0;
        }
    }
    gettimeofday(&t_end, NULL);
    cout << "Serial time: "
        << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
}


//LU_block分解 rank为当前进程的编号 num为进程总数
void LU_block(float M[][N], int rank, int num)
{
    int block = N / num;//根据进程数将矩阵划分成若干块
    int remain = N % num;

    int begin = rank * block;//起始行
    int end = rank != num - 1 ? begin + block : begin + block + remain;//终止行

    for (int k = 0; k < N; k++)
    {
        //若该行属于当前进程，完成任务并向后发送。
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;

            for (int j = 0; j < num; j++)
                if (j != rank)
                    MPI_Send(&M[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);

        }
        //否则，该进程需要接收位于它前面其他进程发送的消息，并在本地进行计算
        else
        {
            int cur_j = k / block;
            MPI_Recv(&M[k], N, MPI_FLOAT, cur_j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }

        //消去部分
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    M[i][j] = M[i][j] - M[i][k] * M[k][j];
                M[i][k] = 0.0;
            }
        }
    }
}
void LU_block_pro(float A[][N], int rank, int num)
{
    int block = N / num;//根据进程数将矩阵划分成若干块
    int remain = N % num;

    int begin = rank * block;//起始行
    int end = rank != num - 1 ? begin + block : begin + block + remain;//终止行

    for (int k = 0; k < N; k++)
    {
        //当前行是自己进程的任务——进行消去
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int j = rank + 1; j < num; j++)
                MPI_Send(&A[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        //否则，该进程需要接收位于它前面其他进程发送的消息，并在本地进行计算
        else
        {
            int cur_j = k / block;
            if (cur_j < rank)
                MPI_Recv(&A[k], N, MPI_FLOAT, cur_j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        //消去
        for (int i = begin; i < end && i < N; i++)
        {
            if (i >= k + 1)
            {
                for (int j = k + 1; j < N; j++)
                    A[i][j] = A[i][j] - A[i][k] * A[k][j];
                A[i][k] = 0.0;
            }
        }
    }
}


void MPI_Block()
{

    timeval t_start;
    timeval t_end;

    int num;//进程数
    int rank;//进程号

    MPI_Comm_size(MPI_COMM_WORLD, &num);//获取进程的数量
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//获取进程的编号

    int block = N / num;
    int remain = N % num;

    //0号进程
    if (rank == 0)
    {
        newM(M, tmp_M);
        gettimeofday(&t_start, NULL);
        //任务划分
        for (int i = 1; i < num; i++)
        {
            if (i != num - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Send(&M[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Send(&M[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }
        }
        LU_block(M, rank, num);
        //接收其他进程处理之后的结果
        for (int i = 1; i < num; i++)
        {
            if (i != num - 1)
            {
                for (int j = 0; j < block; j++)
                    MPI_Recv(&M[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            else
            {
                for (int j = 0; j < block + remain; j++)
                    MPI_Recv(&M[i * block + j], N, MPI_FLOAT, i, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        gettimeofday(&t_end, NULL);
        cout << "MPI_Block time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }

    //其他进程
    else
    {
        //其他进程接收任务
        if (rank != num - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Recv(&M[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Recv(&M[rank * block + j], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU_block(M, rank, num);
        //其他进程完成任务后，结果传回0号进程
        if (rank != num - 1)
        {
            for (int j = 0; j < block; j++)
                MPI_Send(&M[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
        else
        {
            for (int j = 0; j < block + remain; j++)
                MPI_Send(&M[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}


void LU_cir(float M[][N], int rank, int num)
{
    for (int k = 0; k < N; k++)
    {
        //当前行是自己进程的任务则进行除法操作
        if (int(k % num) == rank)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;
            //向其他进程发送消息
            for (int p = 0; p < num; p++)
                if (p != rank)
                    MPI_Send(&M[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        //不是自己进程则接收消息
        else
        {
            MPI_Recv(&M[k], N, MPI_FLOAT, int(k % num), 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int(i % num) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    M[i][j] = M[i][j] - M[i][k] * M[k][j];
                M[i][k] = 0.0;
            }
        }
    }
}

void MPI_circulate()
{
    timeval t_start;
    timeval t_end;

    int num;//进程数
    int rank;//进程号

    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //0号进程——任务划分
    if (rank == 0)
    {
        newM(M, tmp_M);
        gettimeofday(&t_start, NULL);
        //任务划分
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Send(&M[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        LU_cir(M, rank, num);
        //接收其他进程的结果
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&M[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&t_end, NULL);
        cout << "MPI_circulate time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }
    //其他进程
    else
    {
        //接收任务
        for (int i = rank; i < N; i += num)
        {
            MPI_Recv(&M[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //执行任务
        LU_cir(M, rank, num);
        //传回给0号进程
        for (int i = rank; i < N; i += num)
        {
            MPI_Send(&M[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

void LU_pipeline(float M[][N], int rank, int num)
{
    int pre_proc = (rank + (num - 1)) % num;//上一个进程
    int next_proc = (rank + 1) % num;//下一个进程
    for (int k = 0; k < N; k++)
    {
        //判断当前的行是否为自己的任务
        if (int(k % num) == rank)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;
            //完成后向下一进程发送消息
            MPI_Send(&M[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        else
        {
            //若当前行不是当前进程的任务，则接收上一进程的消息
            MPI_Recv(&M[k], N, MPI_FLOAT, pre_proc, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //若当前行不是下一进程的任务，则发送消息
            if (int(k % num) != next_proc)
                MPI_Send(&M[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        for (int i = k + 1; i < N; i++)
        {
            if (int(i % num) == rank)
            {
                for (int j = k + 1; j < N; j++)
                    M[i][j] = M[i][j] - M[i][k] * M[k][j];
                M[i][k] = 0.0;
            }
        }
    }
}


void MPI_pipeline()
{
    timeval t_start;
    timeval t_end;

    int num;
    int rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        newM(M, tmp_M);
        gettimeofday(&t_start, NULL);
        //        在0号进程进行任务划分
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Send(&M[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        LU_pipeline(M, rank, num);
        //处理完0号进程自己的任务后需接收其他进程处理之后的结果
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Recv(&M[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        gettimeofday(&t_end, NULL);
        cout << "MPI_pipeline time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }
    else
    {
        //非0号进程先接收任务
        for (int i = rank; i < N; i += num)
        {
            MPI_Recv(&M[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU_pipeline(M, rank, num);
        //非0号进程完成任务之后，将结果传回到0号进程
        for (int i = rank; i < N; i += num)
        {
            MPI_Send(&M[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}


int main()
{
    m_reset(tmp_M);

    //Serial();

    MPI_Init(NULL, NULL);//令MPI_Block进行初始化工作

    MPI_Block();
    //MPI_block_SSE_OpenMP();

    MPI_circulate();

    MPI_pipeline();

    MPI_Finalize();//终止MPI

}