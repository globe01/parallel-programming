#include <iostream>
#include <mpi.h>
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

    for (int i = 0; i < N; i++)
    {
        int k1 = rand() % N;
        int k2 = rand() % N;
        for (int j = 0; j < N; j++)
        {
            tmp_M[i][j] += tmp_M[0][j];
            tmp_M[k1][j] += tmp_M[k2][j];
        }
    }
}

void newM(float M[][N], float tmp_M[][N])
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            M[i][j] = tmp_M[i][j];
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



//void LU_SSE(float M[][N], int rank, int num)
//{
//    __m128 v1, v2, v3;
//    int pre_proc = (rank + (num - 1)) % num;
//    int next_proc = (rank + 1) % num;
//#pragma omp parallel num_threads(threadNum)
//    for (int k = 0; k < N; k++)
//    {
//        if (int(k % num) == rank)
//        {
//            float temp1[4] = { M[k][k], M[k][k], M[k][k], M[k][k] };
//            v1 = _mm_loadu_ps(temp1);
//
//#pragma omp for schedule(dynamic, 20)
//            for (int j = k + 1; j < N - 3; j += 4)
//            {
//                v2 = _mm_loadu_ps(M[k] + j);//将四个单精度浮点数从内存加载到向量寄存器
//                v3 = _mm_div_ps(v2, v1);//向量对位相除
//                _mm_storeu_ps(M[k] + j, v3);//将四个单精度浮点数从向量寄存器存储到内存
//            }
//            for (int j = N - N % 4; j < N; j++)
//            {
//                M[k][j] = M[k][j] / M[k][k];
//            }
//            M[k][k] = 1.0;
//            MPI_Send(&M[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
//        }
//        else
//        {
//            MPI_Recv(&M[k], N, MPI_FLOAT, pre_proc, 2,
//                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            if (int(k % num) != next_proc)
//                MPI_Send(&M[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
//        }
//        for (int i = k + 1; i < N; i++)
//        {
//            if (int(i % num) == rank)
//            {
//                float temp2[4] = { M[i][k], M[i][k], M[i][k], M[i][k] };
//                v1 = _mm_loadu_ps(temp2);
//#pragma omp for schedule(dynamic, 20)
//                for (int j = k + 1; j <= N - 3; j += 4)
//                {
//                    v2 = _mm_loadu_ps(M[i] + j);
//                    v3 = _mm_loadu_ps(M[k] + j);
//                    v3 = _mm_mul_ps(v1, v3);
//                    v2 = _mm_sub_ps(v2, v3);
//                    _mm_storeu_ps(M[i] + j, v2);
//                }
//                for (int j = N - N % 4; j < N; j++)
//                    M[i][j] = M[i][j] - M[i][k] * M[k][j];
//                M[i][k] = 0;
//            }
//        }
//    }
//}
//
//void MPI_pipeline_SSE()
//{
//    timeval t_start;
//    timeval t_end;
//
//    int num;
//    int rank;
//
//    MPI_Comm_size(MPI_COMM_WORLD, &num);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    if (rank == 0)
//    {
//        newM(M, tmp_M);
//        gettimeofday(&t_start, NULL);
//        for (int i = 0; i < N; i++)
//        {
//            int flag = i % num;
//            if (flag == rank)
//                continue;
//            else
//                MPI_Send(&M[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
//        }
//        LU_SSE(M, rank, num);
//        for (int i = 0; i < N; i++)
//        {
//            int flag = i % num;
//            if (flag == rank)
//                continue;
//            else
//                MPI_Recv(&M[i], N, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//        gettimeofday(&t_end, NULL);
//        cout << "Pipeline MPI LU_pipeline with SSE and OpenMP time cost: "
//            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
//            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
//    }
//    else
//    {
//        for (int i = rank; i < N; i += num)
//        {
//            MPI_Recv(&M[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//        LU_SSE(M, rank, num);
//        for (int i = rank; i < N; i += num)
//        {
//            MPI_Send(&M[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
//        }
//    }
//}


int main()
{
    m_reset(tmp_M);

    MPI_Init(NULL, NULL);//令MPI_Block进行初始化工作

    MPI_pipeline();
    //MPI_pipeline_SSE();
    MPI_Finalize();//终止MPI



}