#include <iostream>
#include <sstream>
#include <fstream>
#include <mpi.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2��AVX-512
#include <omp.h>
#include <sys/time.h>
using namespace std;

static const int threadNum = 4;


unsigned int Core[37960][1188] = { 0 };
unsigned int Rows[37960][1188] = { 0 };

const int N = 1187;
const int Rows_num = 14291;
const int column = 37960;
const int each_col = 10;//ÿ�ִ�����Ԫ�ӵ�����


//��ȡ��Ԫ�Ӳ���ʼ��
void Core_reset()
{
    unsigned int a;
    ifstream infile("Core_9.txt");
    char fin[10000] = { 0 };
    int index;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                index = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Core[index][N - 1 - j] += temp;
            Core[index][N] = 1;//1�ǿգ�0��
        }
    }
}

//��ȡ����Ԫ�в���ʼ��
void Rows_reset()
{
    unsigned int a;
    ifstream infile("Rows_9.txt");
    char fin[10000] = { 0 };
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                Rows[index][N] = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Rows[index][N - 1 - j] += temp;
        }
        index++;
    }
}

//�����㷨
void Serial()
{
    timeval t_start;
    timeval t_end;
    gettimeofday(&t_start, NULL);
    for (int i = column - 1; i - each_col >= -1; i -= each_col)//������Ԫ�ӵ��У�ÿ�ֵ���each_col����Ԫ��
    {
        for (int j = 0; j < Rows_num; j++)//��������Ԫ�У��ҵ���0Ԫ��
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col + 1)//����0Ԫ���ڵ�ǰ��Ԫ�Ӵ���Χ��
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)//��Ԫ�ӷǿ�ʱ������ȥ����������֮�����µ��׸���0Ԫ��
                {
                    for (int k = 0; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//��һѭ�����ֺ����ɽ���SIMD�Ż�
                    }
                    int t = 0;
                    for (int m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += (column - m - 1) * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;

                }
                else//��Ԫ��Ϊ�գ�����Ԫ������Ϊ��Ԫ��
                {
                    for (int k = 0; k < N; k++)
                    {
                        Core[curr][k] = Rows[j][k];
                    }
                    Core[curr][N] = 1;
                    break;
                }
            }
        }
    }
    gettimeofday(&t_end, NULL);
    cout << "Serial time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
}


void LU_Block(int rank, int num)
{
#pragma omp parallel num_threads(thread_count) 
    for (int i = column - 1; i - each_col >= -1; i -= each_col)
    {
#pragma omp for schedule(dynamic,20)     
        for (int j = 0; j < Rows_num; j++)
        {
            //��ǰ��Ϊ�Լ����̵�������ִ����ȥ����
            if (int(j % num) == rank)
            {
                while (Rows[j][N] <= i && Rows[j][N] >= i - 7)
                {
                    int index = Rows[j][N];

                    if (Core[index][N] == 1)//��Ԫ�ӷǿ�
                    {
                        int k;
                        __m128 vhang, vzi;
                        for (k = 0; k + 4 <= N; k += 4)
                        {
                            vzi = _mm_loadu_ps((float*)&(Core[index][k]));
                            vhang = _mm_loadu_ps((float*)&(Rows[j][k]));
                            vhang = _mm_xor_ps(vhang, vzi);
                            _mm_store_ss((float*)&(Rows[j][k]), vhang);
                        }

                        for (; k < N; k++)
                        {
                            Rows[j][k] = Rows[j][k] ^ Core[index][k];
                        }
                        int num = 0, S_num = 0;
                        for (num = 0; num < N; num++)
                        {
                            if (Rows[j][num] != 0)
                            {
                                unsigned int temp = Rows[j][num];
                                while (temp != 0)
                                {
                                    temp = temp >> 1;
                                    S_num++;
                                }
                                S_num += num * 32;
                                break;
                            }
                        }
                        Rows[j][N] = S_num - 1;
                    }
                    else//��Ԫ��Ϊ��
                    {
                        break;
                    }
                }
            }
        }
    }
}

void MPI_Block()
{
    int num;//������
    int rank;//���̺�
    MPI_Comm_size(MPI_COMM_WORLD, &num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //0�Ž���
    if (rank == 0)
    {
        timeval t_start;
        timeval t_end;
        gettimeofday(&t_start, NULL);
        int sign;
        do
        {
            //��������
            for (int i = 0; i < Rows_num; i++)
            {
                int flag = i % num;
                if (flag == rank)
                    continue;
                else
                    MPI_Send(&Rows[i], N + 1, MPI_FLOAT, flag, 0, MPI_COMM_WORLD);
            }
            LU_Block(rank, num);
            //������������������̵Ľ��
            for (int i = 0; i < Rows_num; i++)
            {
                int flag = i % num;
                if (flag == rank)
                    continue;
                else
                    MPI_Recv(&Rows[i], N + 1, MPI_FLOAT, flag, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            //������Ԫ�Ӳ��ж��Ƿ����
            sign = 0;
            for (int i = 0; i < Rows_num; i++)
            {
                //�ҵ���i������Ԫ�е�����
                int temp = Rows[i][N];
                if (temp == -1)
                {
                    continue;//˵��������Ϊ��Ԫ��
                }
                //�����Ӧ��Ԫ����Ϊ�գ�����
                if (Core[temp][N] == 0)
                {
                    //������Ԫ��
                    for (int k = 0; k < N; k++)
                        Core[temp][k] = Rows[i][k];
                    //����Ԫ������
                    Rows[i][N] = -1;
                    //��־��Ϊtrue������
                    sign = 1;
                }
            }
            for (int r = 1; r < num; r++)
            {
                MPI_Send(&sign, 1, MPI_INT, r, 2, MPI_COMM_WORLD);
            }
        } while (sign == 1);

        gettimeofday(&t_end, NULL);
        cout << "MPI_Block time: " << 1000 * (t_end.tv_sec - t_start.tv_sec) + 0.001 * (t_end.tv_usec - t_start.tv_usec) << "ms" << endl;
    }
    //��������
    else
    {
        int sign;
        do
        {
            //�������̽�������
            for (int i = rank; i < Rows_num; i += num)
            {
                MPI_Recv(&Rows[i], N + 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            //ִ������
            LU_Block(rank, num);
            //���ظ�0�Ž���
            for (int i = rank; i < Rows_num; i += num)
            {
                MPI_Send(&Rows[i], N + 1, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
            }

            MPI_Recv(&sign, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } while (sign == 1);
    }
}


int main()
{
    Core_reset();
    Rows_reset();
    Serial();

    Core_reset();
    Rows_reset();
    MPI_Init(NULL, NULL);
    MPI_Block();
    MPI_Finalize();

}