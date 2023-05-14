#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/time.h>
# include <arm_neon.h>
#include <pthread.h>
#include <semaphore.h>//信号量
using namespace std;


/*
unsigned int Core[8399][264] = { 0 };
unsigned int Rows[8399][264] = { 0 };

const int N = 263;
const int Rows_num = 4535;
const int column = 8399;
*/



/*
unsigned int Core[23045][722] = { 0 };
unsigned int Rows[23045][722] = { 0 };

const int N = 721;
const int Rows_num = 14325;
const int column = 23045;
*/

/*
unsigned int Core[37960][1188] = { 0 };
unsigned int Rows[37960][1188] = { 0 };

const int N = 1187;
const int Rows_num = 14291;
const int column = 37960;
*/


unsigned int Core[43577][1363] = { 0 };
unsigned int Rows[54274][1363] = { 0 };

const int N = 1362;
const int Rows_num = 54274;
const int column = 43577;


int NUM_THREADS = 5;//线程数定义
//信号量定义
sem_t sem_leader;
sem_t* sem_Next = new sem_t[NUM_THREADS - 1]; // 每个线程有自己专属的信号量

bool sign;//判断结束的标志

struct threadParam_t
{
    int t_id; // 线程 id
};

//读取消元子并初始化
void init_A()
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
            Core[index][N] = 1;//1非空，0空
        }
    }
}

//读取被消元行并初始化
void Rows_reset()
{
    unsigned int a;
    ifstream infile("pas3.txt");
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


void* threadFunc(void* param)
{
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    uint32x4_t vhang = vmovq_n_u32(0);
    uint32x4_t vzi = vmovq_n_u32(0);
    do
    {
        int i;
        for (i = column - 1; i - 8 >= -1; i -= 8)
        {
            for (int j = t_id; j < Rows_num; j += NUM_THREADS)
            {
                while (Rows[j][N] <= i && Rows[j][N] >= i - 7)
                {
                    int index = Rows[j][N];
                    if (Core[index][N] == 1)//消元子不为空
                    {
                        int k;
                        for (k = 0; k + 4 <= N; k += 4)
                        {
                            vzi = vld1q_u32(&(Core[index][k]));
                            vhang = vld1q_u32(&(Rows[j][k]));
                            vhang = veorq_u32(vhang, vzi);
                            vst1q_u32(&(Rows[j][k]), vhang);
                        }

                        for (; k < N; k++)
                        {
                            Rows[j][k] = Rows[j][k] ^ Core[index][k];
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

        for (i = i + 8; i >= 0; i--)
        {
            for (int j = t_id; j < Rows_num; j += NUM_THREADS)
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
                    else//消元子为空
                    {
                        break;
                    }
                }
            }
        }

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_wait(&sem_leader); // 等待其它 worker 完成处理被消元行

        }
        else
        {
            //其他线程完成处理后，通知线程1已完成,然后进入睡眠，待1线程完成升格后进入下一轮
            sem_post(&sem_leader);// 通知 leader, 已完成处理被消元行
            sem_wait(&sem_Next[t_id - 1]); // 等待通知，进入下一轮
        }
        if (t_id == 0)
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
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; ++i)
                sem_post(&sem_Next[i]); // 通知其它 worker 进入下一轮
        }


    } while (sign == true);
    pthread_exit(NULL);
    return 0;
}

int main()
{
    init_A();
    Rows_reset();

    struct timeval head, tail;
    double seconds;

    gettimeofday(&head, NULL);

    //创建线程
    pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
    threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    //销毁所有信号量
    sem_destroy(&sem_leader);
    sem_destroy(sem_Next);

    gettimeofday(&tail, NULL);
    seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
    cout << "Grobner-pthread:" << seconds << "ms" << endl;
    return 0;
}