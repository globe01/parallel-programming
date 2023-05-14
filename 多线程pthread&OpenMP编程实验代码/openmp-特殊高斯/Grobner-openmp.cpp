#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <semaphore.h>
#include <sys/time.h>
# include <arm_neon.h> 
using namespace std;


unsigned int Core[37960][1188] = { 0 };
unsigned int Rows[37960][1188] = { 0 };

const int N = 1187;
const int Rows_num = 14291;
const int column = 37960;


const int NUM_THREADS = 5; //线程数


//全局变量定义，用于判断接下来是否进入下一轮
bool sign;

struct threadParam_t
{
    int t_id; // 线程 id
};



//读取消元子并初始化
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
                //取每行第一个数字为行标
                index = a;
                p = 1;
            }
            int k = a % 32;
            int j = a / 32;

            int temp = 1 << k;
            Core[index][N - 1 - j] += temp;
            Core[index][N] = 1;//设置该位置记录消元子该行是否为空，为空则是0，否则为1
        }
    }
}

//读取被消元行并初始化
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


void Serial()
{
    int i;
    for (i = column - 1; i >= 0; i--)
    {
        for (int j = 0; j < Rows_num; j++)
        {
            while (Rows[j][N] <= i && Rows[j][N] >= i - 7)
            {
                int index = Rows[j][N];
                if (Core[index][N] == 1)
                {
                    for (int k = 0; k < N; k++)
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
                else
                {
                    for (int k = 0; k < N; k++)
                    {
                        Core[index][k] = Rows[j][k];
                    }                       
                    Core[index][N] = 1;
                    break;
                }
            }
        }
    }
}

void openmp()
{
    uint32x4_t vhang = vmovq_n_u32(0);
    uint32x4_t vzi = vmovq_n_u32(0);
    bool sign;
#pragma omp parallel num_threads(NUM_THREADS), private(vhang, vzi)
    do
    {
        for (int i = column - 1; i >= 0; i--)
        {
#pragma omp for schedule(static)
            for (int j = 0; j < Rows_num; j++)
            {
                while (Rows[j][N] == i)
                {
                    if (Core[i][N] == 1)
                    {
                        int k;
                        for (k = 0; k + 4 <= N; k += 4)
                        {
                            vzi = vld1q_u32(&(Core[i][k]));
                            vhang = vld1q_u32(&(Rows[j][k]));
                            vhang = veorq_u32(vhang, vzi);
                            vst1q_u32(&(Rows[j][k]), vhang);
                        }

                        for (; k < N; k++)
                        {
                            Rows[j][k] = Rows[j][k] ^ Core[i][k];
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
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

#pragma omp single
        {
            sign = false;
            for (int i = 0; i < Rows_num; i++)
            {
                int temp = Rows[i][N];
                if (temp == -1)
                {
                    continue;
                }
                if (Core[temp][N] == 0)
                {
                    for (int k = 0; k < N; k++)
                        Core[temp][k] = Rows[i][k];
                    Rows[i][N] = -1;
                    sign = true;
                }
            }
        }
    } while (sign == true);

}


int main()
{
    struct timeval head, tail;
    double seconds;

    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Serial();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Serial：" << seconds << "ms" << endl;

    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    openmp();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "openmp：" << seconds << "ms" << endl;

    return 0;
}

