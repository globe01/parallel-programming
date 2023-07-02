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

static const int N = 3000;
static const int threadNum = 8;

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
    cout << "Serial time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
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


//void LU_SSE(float M[][N], int rank, int num)
//{
//    __m128 v1, v2, v3;
//    int block = N / num;
//    int remain = N % num;
//    int begin = rank * block;
//    int end = rank != num - 1 ? begin + block : begin + block + remain;
//#pragma omp parallel num_threads(threadNum),private(v1, v2, v3)
//    for (int k = 0; k < N; k++)
//    {
//        if (k >= begin && k < end)
//        {
//            float temp1[4] = { M[k][k], M[k][k], M[k][k], M[k][k] };
//            v1 = _mm_loadu_ps(temp1);
//#pragma omp for schedule(static)
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
//            for (int p = rank + 1; p < num; p++)
//                MPI_Send(&M[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
//        }
//        else
//        {
//            int cur_j = k / block;
//            if (cur_j < rank)
//                MPI_Recv(&M[k], N, MPI_FLOAT, cur_j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//        for (int i = begin; i < end && i < N; i++)
//        {
//            if (i >= k + 1)
//            {
//                float temp2[4] = { M[i][k], M[i][k], M[i][k], M[i][k] };
//                v1 = _mm_loadu_ps(temp2);
//#pragma omp for schedule(static)
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
//
//void MPI_block_SSE_OpenMP()
//{
//
//    timeval t_start;
//    timeval t_end;
//
//    int num;//进程数
//    int rank;//识别调用进程的rank，值从0~size-1
//
//    MPI_Comm_size(MPI_COMM_WORLD, &num);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    int block = N / num;
//    int remain = N % num;
//
//    //0号进程——任务划分
//    if (rank == 0)
//    {
//        newM(M, tmp_M);
//        gettimeofday(&t_start, NULL);
//        //任务划分
//        for (int i = 1; i < num; i++)
//        {
//            if (i != num - 1)
//            {
//                for (int j = 0; j < block; j++)
//                    MPI_Send(&M[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
//            }
//            else
//            {
//                for (int j = 0; j < block + remain; j++)
//                    MPI_Send(&M[i * block + j], N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
//            }
//        }
//        LU_SSE(M, rank, num);
//        //处理完0号进程自己的任务后需接收其他进程处理之后的结果
//        for (int i = 1; i < num; i++)
//        {
//            if (i != num - 1)
//            {
//                for (int j = 0; j < block; j++)
//                    MPI_Recv(&M[i * block + j], N, MPI_FLOAT, i, 1,
//                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            }
//            else
//            {
//                for (int j = 0; j < block + remain; j++)
//                    MPI_Recv(&M[i * block + j], N, MPI_FLOAT, i, 1,
//                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            }
//        }
//        gettimeofday(&t_end, NULL);
//        cout << "MPI_Block_SSE_OpenMP time: "
//            << 1000 * (t_end.tv_sec - t_start.tv_sec) +
//            0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
//    
//    }
//
//    //其他进程
//    else
//    {
//        //非0号进程先接收任务
//        if (rank != num - 1)
//        {
//            for (int j = 0; j < block; j++)
//                MPI_Recv(&M[rank * block + j], N, MPI_FLOAT, 0, 0,
//                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//        else
//        {
//            for (int j = 0; j < block + remain; j++)
//                MPI_Recv(&M[rank * block + j], N, MPI_FLOAT, 0, 0,
//                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//        LU_SSE(M, rank, num);
//        //处理完后向零号进程返回结果
//        if (rank != num - 1)
//        {
//            for (int j = 0; j < block; j++)
//                MPI_Send(&M[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
//        }
//        else
//        {
//            for (int j = 0; j < block + remain; j++)
//                MPI_Send(&M[rank * block + j], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
//        }
//    }
//}


int main()
{
    m_reset(tmp_M);

    Serial();

    MPI_Init(NULL, NULL);//令MPI_Block进行初始化工作

    MPI_Block();
    //MPI_block_SSE_OpenMP();
    MPI_Finalize();//终止MPI

}