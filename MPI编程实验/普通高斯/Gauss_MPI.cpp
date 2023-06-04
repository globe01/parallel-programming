//�黮�� + ���SIMD OpenMP

#include <iostream>
#include <mpi.h> // mpiͷ�ļ�
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



//����
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


//LU_block�ֽ� rankΪ��ǰ���̵ı�� numΪ��������
void LU_block(float M[][N], int rank, int num)
{
    int block = N / num;//���ݽ����������󻮷ֳ����ɿ�
    int remain = N % num;

    int begin = rank * block;//��ʼ��
    int end = rank != num - 1 ? begin + block : begin + block + remain;//��ֹ��

    for (int k = 0; k < N; k++)
    {
        //���������ڵ�ǰ���̣������������͡�
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;

            for (int j = 0; j < num; j++)
                if (j != rank)
                    MPI_Send(&M[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);

        }
        //���򣬸ý�����Ҫ����λ����ǰ���������̷��͵���Ϣ�����ڱ��ؽ��м���
        else
        {
            int cur_j = k / block;
            MPI_Recv(&M[k], N, MPI_FLOAT, cur_j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }

        //��ȥ����
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
    int block = N / num;//���ݽ����������󻮷ֳ����ɿ�
    int remain = N % num;

    int begin = rank * block;//��ʼ��
    int end = rank != num - 1 ? begin + block : begin + block + remain;//��ֹ��

    for (int k = 0; k < N; k++)
    {
        //��ǰ�����Լ����̵����񡪡�������ȥ
        if (k >= begin && k < end)
        {
            for (int j = k + 1; j < N; j++)
                A[k][j] = A[k][j] / A[k][k];
            A[k][k] = 1.0;
            for (int j = rank + 1; j < num; j++)
                MPI_Send(&A[k], N, MPI_FLOAT, j, 2, MPI_COMM_WORLD);
        }
        //���򣬸ý�����Ҫ����λ����ǰ���������̷��͵���Ϣ�����ڱ��ؽ��м���
        else
        {
            int cur_j = k / block;
            if (cur_j < rank)
                MPI_Recv(&A[k], N, MPI_FLOAT, cur_j, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        //��ȥ
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

    int num;//������
    int rank;//���̺�

    MPI_Comm_size(MPI_COMM_WORLD, &num);//��ȡ���̵�����
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);//��ȡ���̵ı��

    int block = N / num;
    int remain = N % num;

    //0�Ž���
    if (rank == 0)
    {
        newM(M, tmp_M);
        gettimeofday(&t_start, NULL);
        //���񻮷�
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
        //�����������̴���֮��Ľ��
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

    //��������
    else
    {
        //�������̽�������
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
        //���������������󣬽������0�Ž���
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

void LU_pipeline(float M[][N], int rank, int num)
{
    int pre_proc = (rank + (num - 1)) % num;//��һ������
    int next_proc = (rank + 1) % num;//��һ������
    for (int k = 0; k < N; k++)
    {
        //�жϵ�ǰ�����Ƿ�Ϊ�Լ�������
        if (int(k % num) == rank)
        {
            for (int j = k + 1; j < N; j++)
                M[k][j] = M[k][j] / M[k][k];
            M[k][k] = 1.0;
            //��ɺ�����һ���̷�����Ϣ
            MPI_Send(&M[k], N, MPI_FLOAT, next_proc, 2, MPI_COMM_WORLD);
        }
        else
        {
            //����ǰ�в��ǵ�ǰ���̵������������һ���̵���Ϣ
            MPI_Recv(&M[k], N, MPI_FLOAT, pre_proc, 2,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            //����ǰ�в�����һ���̵�����������Ϣ
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
        //        ��0�Ž��̽������񻮷�
        for (int i = 0; i < N; i++)
        {
            int flag = i % num;
            if (flag == rank)
                continue;
            else
                MPI_Send(&M[i], N, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
        }
        LU_pipeline(M, rank, num);
        //������0�Ž����Լ��������������������̴���֮��Ľ��
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
        //��0�Ž����Ƚ�������
        for (int i = rank; i < N; i += num)
        {
            MPI_Recv(&M[i], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        LU_pipeline(M, rank, num);
        //��0�Ž����������֮�󣬽�������ص�0�Ž���
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

    MPI_Init(NULL, NULL);//��MPI_Block���г�ʼ������

    MPI_Block();
    //MPI_block_SSE_OpenMP();

    MPI_circulate();

    MPI_pipeline();

    MPI_Finalize();//��ֹMPI

}