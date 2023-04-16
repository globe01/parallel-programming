#include <iostream>
#include <sstream>
#include <fstream>
#include <arm_neon.h>
#include <sys/time.h>

using namespace std;


//样例9
unsigned int Core[37960][1188] = { 0 };//存消元子是否为0
unsigned int Rows[37960][1188] = { 0 };//被消元行
const int Rows_num = 14291;//被消元行数
const int N = 1187;
const int column = 37960;//矩阵列数
const int each_col = 10;//每轮处理消元子的列数


//样例8
//unsigned int Core[23075][722] = { 0 };//存消元子是否为0
//unsigned int Rows[23075][722] = { 0 };//被消元行
//const int Rows_num = 14325;//被消元行数
//const int N = 721;
//const int column = 23075;//矩阵列数
//const int each_col = 10;//每轮处理消元子的列数


////读取消元子并初始化
void Core_reset()
{
    unsigned int a;
    ifstream infile("Core_9.txt");
    char fin[100000] = { 0 };
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
            Core[curr][N-1 - j] += temp;
            Core[curr][N] = 1;//1非看，0空
        }
    }
}


//读取被消元行并初始化
void Rows_reset()
{
    unsigned int a;
    ifstream infile("Rows_9.txt");
    char fin[100000] = { 0 };
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
            Rows[curr][N-1 - j] += temp;
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
                            t += m * 32;
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


//Neon
void Neon()
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
                    //异或步骤的循环进行Neon优化
                    int k;
                    for (k = 0; k + each_col/2 <= N; k += each_col/2)
                    {
                        uint32x4_t vzi = vld1q_u32(&(Core[curr][k]));//消元子加载到向量寄存器
                        uint32x4_t vhang = vld1q_u32(&(Rows[j][k]));//被消元行加载到向量寄存器
                        vhang = veorq_u32(vhang, vzi);//向量异或
                        vst1q_u32(&(Rows[j][k]), vhang);//存储到内存
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
    struct timeval head, tail;
    
    //串行
    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Serial();
    gettimeofday(&tail, NULL);
    double seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Serial: " << seconds << " ms" << endl;

    //Neon
    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Neon();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Neon: " << seconds << " ms" << endl;

}