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
#include <immintrin.h> //AVX、AVX2、AVX-512

using namespace std;

//样例9
//unsigned int Core[37960][1188] = {0};//存消元子是否为0
//unsigned int Rows[37960][1188] = {0};//被消元行
//const int Rows_num = 14291;//被消元行数
//const int N = 1187;
//const int column = 37960;//矩阵列数
//const int each_col = 10;//每轮处理消元子的列数


//样例8
unsigned int Core[23075][722] = {0};//存消元子是否为0
unsigned int Rows[23075][722] = {0};//被消元行
const int Rows_num = 14325;//被消元行数
const int N = 721;
const int column = 23075;//矩阵列数
const int each_col = 10;//每轮处理消元子的列数

//读取消元子并初始化
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
            Core[curr][N] = 1;//1非空 0空
        }
    }
}

//读取被消元行并初始化
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


//串行算法
void Serial()
{
    for (int i = column - 1; i - each_col >= -1; i -= each_col)//遍历消元子的列，每轮导入each_col个消元子
    {
        for (int j = 0; j < Rows_num; j++)//遍历被消元行，找到非0元素
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - each_col+1)//若非0元素在当前消元子处理范围内
            {
                int curr = Rows[j][N];
                if (Core[curr][N] == 1)//消元子非空时进行消去，即异或，异或之后找新的首个非0元素
                {
                    for (int k = 0; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//这一循环部分后续可进行SIMD优化
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
                else//消元子为空，被消元行升格为消元子
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


//SSE优化
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
                    //异或步骤的循环进行SSE优化
                    int k;
                    for (k = 0; k + each_col/2 <= N; k += each_col/2)
                    {
                        v1_Core = _mm_loadu_ps((float*)&(Core[curr][k]));//消元子加载到向量寄存器
                        v1_Rows = _mm_loadu_ps((float*)&(Rows[j][k]));//被消元行加载到向量寄存器
                        v1_Rows = _mm_xor_ps(v1_Rows, v1_Core);//向量异或
                        _mm_store_ss((float*)&(Rows[j][k]), v1_Rows);//存储到内存
                    }
                    for (; k < N; k++)
                    {
                        Rows[j][k] = Rows[j][k] ^ Core[curr][k];//结尾处有几个还未计算
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


//AVX优化
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
                    //异或步骤的循环进行SSE优化
                    int k;
                    for (k = 0; k + each_col <= N; k += each_col)
                    {
                        v2_Core = _mm256_loadu_ps((float*)&(Core[curr][k]));//消元子加载到向量寄存器
                        v2_Rows = _mm256_loadu_ps((float*)&(Rows[j][k]));//被消元行加载到向量寄存器
                        v2_Rows = _mm256_xor_ps(v2_Rows, v2_Core);//向量异或
                        _mm256_storeu_ps((float*)&(Rows[j][k]), v2_Rows);//存储到内存
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


//AVX-512优化
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
                    //异或步骤的循环进行SSE优化
                    int k;
                    for (k = 0; k + 2*each_col <= N; k += 2*each_col)
                    {
                        v3_Core = _mm512_loadu_ps((float*)&(Core[curr][k]));//消元子加载到向量寄存器
                        v3_Rows = _mm512_loadu_ps((float*)&(Rows[j][k]));//被消元行加载到向量寄存器
                        v3_Rows = _mm512_xor_ps(v3_Rows, v3_Core);//向量异或
                        _mm512_storeu_ps((float*)&(Rows[j][k]), v3_Rows);//存储到内存
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

    //串行
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    Serial();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//单位 ms
    cout << "串行：" << seconds << " ms" << endl;


    //SSE
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    SSE();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//单位 ms
    cout << "SSE：" << seconds << " ms" << endl;


    //AVX
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    AVX();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//单位 ms
    cout << "AVX：" << seconds << " ms" << endl;


    //AVX-512
    Core_reset();
    Rows_reset();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    AVX_512();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    seconds = (tail - head) * 1000.0 / freq;//单位 ms
    cout << "AVX-512：" << seconds << " ms" << endl;
}

