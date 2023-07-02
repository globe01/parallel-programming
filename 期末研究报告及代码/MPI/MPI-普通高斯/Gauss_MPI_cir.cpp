//ѭ������


#include <iostream>
#include <mpi.h>
#include <sys/time.h>
//#include <xmmintrin.h> //SSE
//#include <emmintrin.h> //SSE2
//#include <pmmintrin.h> //SSE3
//#include <tmmintrin.h> //SSSE3
//#include <smmintrin.h> //SSE4.1
//#include <nmmintrin.h> //SSSE4.2
//#include <immintrin.h> //AVX��AVX2��AVX-512
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



void LU_cir(float M[][N], int rank, int num)
{
    for (int k = 0; k < N; k++)
    {
        //��ǰ�����Լ����̵���������г�������
        if (int(k % num) == rank)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;
            //���������̷�����Ϣ
            for (int p = 0; p < num; p++)
                if (p != rank)
                    MPI_Send(&M[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
        }
        //�����Լ������������Ϣ
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

    int num;//������
    int rank;//���̺�

    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //0�Ž��̡������񻮷�
    if (rank == 0)
    {
        newM(M, tmp_M);
        gettimeofday(&t_start, NULL);
        //���񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Send(&M[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        LU_cir(M, rank, num);
        //�����������̵Ľ��
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
    //��������
    else
    {
        //��������
        for (int i = rank; i < N; i += num)
        {
            MPI_Recv(&M[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        //ִ������
        LU_cir(M, rank, num);
        //���ظ�0�Ž���
        for (int i = rank; i < N; i += num)
        {
            MPI_Send(&M[i], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }
}

//void LU_SSE(float M[][N], int rank, int num)
//{
//    __m128 v1, v2, v3;
//#pragma omp parallel num_threads(threadNum)
//    for (int k = 0; k < N; k++)
//    {
//        if (int(k % num) == rank)
//        {
//            float temp1[4] = { M[k][k], M[k][k], M[k][k], M[k][k] };
//            v1 = _mm_loadu_ps(temp1);
//#pragma omp for schedule(dynamic, 20)
//            for (int j = k + 1; j < N - 3; j += 4)
//            {
//                v2 = _mm_loadu_ps(M[k] + j);//���ĸ������ȸ��������ڴ���ص������Ĵ���
//                v3 = _mm_div_ps(v2, v1);//������λ���
//                _mm_storeu_ps(M[k] + j, v3);//���ĸ������ȸ������������Ĵ����洢���ڴ�
//            }
//            for (int j = N - N % 4; j < N; j++)
//            {
//                M[k][j] = M[k][j] / M[k][k];
//            }
//            M[k][k] = 1.0;
//
//            for (int p = 0; p < num; p++)
//                if (p != rank)
//                    MPI_Send(&M[k], N, MPI_FLOAT, p, 2, MPI_COMM_WORLD);
//        }
//        else
//        {
//            MPI_Recv(&M[k], N, MPI_FLOAT, int(k % num), 2,
//                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
//void MPI_cir_SSE()
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
//        cout << "MPI_circulate_SSE_OpenMP time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
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

    MPI_Init(NULL, NULL);//��MPI_Block���г�ʼ������

    MPI_circulate();
    //MPI_cir_SSE();

    MPI_Finalize();//��ֹMPI

}