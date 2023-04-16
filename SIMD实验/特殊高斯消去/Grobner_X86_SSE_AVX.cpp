#include <iostream>
#include <windows.h>
#include <sstream>
#include <fstream>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2��AVX-512

using namespace std;

//����9
//unsigned int Core[37960][1188] = {0};//����Ԫ���Ƿ�Ϊ0
//unsigned int Rows[37960][1188] = {0};//����Ԫ��
//const int Rows_num = 14291;//����Ԫ����
//const int N = 1187;
//const int column = 37960;//��������
//const int each_col = 10;//ÿ�ִ�����Ԫ�ӵ�����


//����8
unsigned int Core[23075][722] = {0};//����Ԫ���Ƿ�Ϊ0
unsigned int Rows[23075][722] = {0};//����Ԫ��
const int Rows_num = 14325;//����Ԫ����
const int N = 721;
const int column = 23075;//��������
const int each_col = 10;//ÿ�ִ�����Ԫ�ӵ�����

//��ȡ��Ԫ�Ӳ���ʼ��
void Core_reset()
{
    unsigned int a;
    ifstream infile("Core_8.txt");
    char fin[10000] = { 0 };
    int curr;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                curr = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Core[curr][N - 1 - j] += temp;
            Core[curr][N] = 1;//1�ǿ� 0��
        }
    }
}

//��ȡ����Ԫ�в���ʼ��
void Rows_reset()
{
    unsigned int a;
    ifstream infile("Rows_8.txt");
    char fin[10000] = { 0 };
    int curr = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int p = 0;
        while (line >> a)
        {
            if (p == 0)
            {
                Rows[curr][N] = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;
            int temp = 1 << k;
            Rows[curr][N - 1 - j] += temp;
        }
        curr++;
    }
}


//�����㷨
void Serial()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)//������Ԫ�ӵ��У�ÿ�ֵ���each_col����Ԫ��
    {
        for (int j = 0; j < Rows_num; j++)//��������Ԫ�У��ҵ���0Ԫ��
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)//����0Ԫ���ڵ�ǰ��Ԫ�Ӵ���Χ��
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
                            t += (column-m-1) * 32;
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
}


//SSE�Ż�
__m128 v1_Core;
__m128 v1_Rows;
void SSE()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)
    {
        for (int j = 0; j < Rows_num; j++)
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)
                {
                    //������ѭ������SSE�Ż�
                    int k;
                    for (k = 0; k + each_col/2 <= N; k += each_col/2)
                    {
                        v1_Core = _mm_loadu_ps((float*)&(Core[curr][k]));//��Ԫ�Ӽ��ص������Ĵ���
                        v1_Rows = _mm_loadu_ps((float*)&(Rows[j][k]));//����Ԫ�м��ص������Ĵ���
                        v1_Rows = _mm_xor_ps(v1_Rows, v1_Core);//�������
                        _mm_store_ss((float*)&(Rows[j][k]), v1_Rows);//�洢���ڴ�
                    }
                    for (; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//��β���м�����δ����
                    }
                    int m = 0, t = 0;
                    for (m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += m * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;
                }
                else
                {
                    for (int k = 0; k < N; k++) {
                        Core[curr][k] = Rows[j][k];
                    }
                    Core[curr][N] = 1;
                    break;
                }
            }
        }
    }
}


//AVX�Ż�
__m256 v2_Core;
__m256 v2_Rows;
void AVX()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)
    {
        for (int j = 0; j < Rows_num; j++)
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)
                {
                    //������ѭ������SSE�Ż�
                    int k;
                    for (k = 0; k + each_col <= N; k += each_col)
                    {
                        v2_Core = _mm256_loadu_ps((float*)&(Core[curr][k]));//��Ԫ�Ӽ��ص������Ĵ���
                        v2_Rows = _mm256_loadu_ps((float*)&(Rows[j][k]));//����Ԫ�м��ص������Ĵ���
                        v2_Rows = _mm256_xor_ps(v2_Rows, v2_Core);//�������
                        _mm256_storeu_ps((float*)&(Rows[j][k]), v2_Rows);//�洢���ڴ�
                    }
                    for (; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];
                    }
                    int m = 0, t = 0;
                    for (m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += m * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;
                }
                else
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
}


//AVX-512�Ż�
__m512 v3_Core;
__m512 v3_Rows;
void AVX_512()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)
    {
        for (int j = 0; j < Rows_num; j++)
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)
                {
                    //������ѭ������SSE�Ż�
                    int k;
                    for (k = 0; k + 2*each_col <= N; k += 2*each_col)
                    {
                        v3_Core = _mm512_loadu_ps((float*)&(Core[curr][k]));//��Ԫ�Ӽ��ص������Ĵ���
                        v3_Rows = _mm512_loadu_ps((float*)&(Rows[j][k]));//����Ԫ�м��ص������Ĵ���
                        v3_Rows = _mm512_xor_ps(v3_Rows, v3_Core);//�������
                        _mm512_storeu_ps((float*)&(Rows[j][k]), v3_Rows);//�洢���ڴ�
                    }
                    for (; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];
                    }
                    int m = 0, t = 0;
                    for (m = 0; m < N; m++)
                    {
                        if (Rows[j][m] != 0)
                        {
                            unsigned int temp = Rows[j][m];
                            while (temp != 0)
                            {
                                temp = temp >> 1;
                                t++;
                            }
                            t += m * 32;
                            break;
                        }
                    }
                    Rows[j][N] = t - 1;
                }
                else
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
}




int main(){
    double seconds;
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    //����
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Serial();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//��λ ms
    cout << "���У�" << seconds << " ms" << endl;


    //SSE
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SSE();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//��λ ms
    cout << "SSE��" << seconds << " ms" << endl;


    //AVX
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    AVX();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//��λ ms
    cout << "AVX��" << seconds << " ms" << endl;


    //AVX-512
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    AVX_512();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//��λ ms
    cout << "AVX-512��" << seconds << " ms" << endl;
}

