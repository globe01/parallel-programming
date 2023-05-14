#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
# include <arm_neon.h>
using namespace std;



//unsigned int Core[8399][264] = { 0 };
//unsigned int Rows[8399][264] = { 0 };
//
//const int N = 263;
//const int Rows_num = 4535;
//const int column = 8399;





//unsigned int Core[23045][722] = { 0 };
//unsigned int Rows[23045][722] = { 0 };
//
//const int N = 721;
//const int Rows_num = 14325;
//const int column = 23075;



unsigned int Core[37960][1188] = { 0 };
unsigned int Rows[37960][1188] = { 0 };

const int N = 1187;
const int Rows_num = 14291;
const int column = 37960;



//unsigned int Core[43577][1363] = { 0 };
//unsigned int Rows[54274][1363] = { 0 };
//
//const int N = 1362;
//const int Rows_num = 54274;
//const int column = 43577;


//读取消元子并初始化
void Core_reset()
{
    unsigned int a;
    ifstream infile("Core_9.txt");
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
            Core[curr][N] = 1;//1非空，0空
        }
    }
}

void Rows_reset()
{
    unsigned int a;
    ifstream infile("Rows_9.txt");
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

void Neon()
{
    uint32x4_t vhang = vmovq_n_u32(0);
    uint32x4_t vzi = vmovq_n_u32(0);
    bool sign;//判断是否结束的标志
    do
    {
        for (int i = column - 1; i >= 0; i--)
        {
            //每轮1个消元子
            for (int j = 0; j < Rows_num; j++)
            {
                //有无首项等于i
                while (Rows[j][N] == i)
                {
                    //若消元子不为空则异或
                    if (Core[i][N] == 1)
                    {
                        int k;
                        for (k = 0; k + 4 <= N; k += 4)
                        {
                            vzi = vld1q_u32(&(Core[i][k]));//消元子加载到向量寄存器
                            vhang = vld1q_u32(&(Rows[j][k]));//被消元行加载到向量寄存器                            
                            vhang = veorq_u32(vhang, vzi);//向量异或
                            vst1q_u32(&(Rows[j][k]), vhang);//存储到内存
                        }

                        for (; k < N; k++)
                        {
                            Rows[j][k] = Rows[j][k] ^ Core[i][k];//结尾处有几个还未计算
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
                        break;
                    }
                }
            }
        }

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
        cout << "p";

    } while (sign == true);
}


int main()
{
    struct timeval head, tail;
    double seconds;
    Core_reset();
    Rows_reset();
    gettimeofday(&head, NULL);
    Neon();
    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Grobner-neon:" << seconds << "ms" << endl;

    return 0;
}

