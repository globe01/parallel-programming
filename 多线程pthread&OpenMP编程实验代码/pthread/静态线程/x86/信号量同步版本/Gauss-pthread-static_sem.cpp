#include <iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX、AVX2、AVX-512
#include <pthread.h>//pthread
#include <semaphore.h>//信号量
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;

const int N = 1000;//问题规模500,1000,1500
float M[N][N];


//测试用例生成
void m_reset()
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			M[i][j] = 0;
		}
		M[i][i] = 1.0;
		for (int j = i + 1; j < N; j++)
			M[i][j] = rand();
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				M[i][j] += M[k][j];
			}
		}
	}
}

//串行算法
void Serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//除法步骤，整行除以第一个的系数
		}
		M[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];//消去步骤
			}
			M[i][k] = 0;//左下角化为0了
		}
	}
}

//静态线程+信号量同步

//线程数据结构定义
struct threadParam_t
{
	int t_id; //线程id
};
int NUM_THREADS = 5;
//信号量定义
sem_t sem_main;
sem_t* sem_workerstart = new sem_t[NUM_THREADS];// 每个线程有自己专属的信号量
sem_t* sem_workerend = new sem_t[NUM_THREADS];

//线程函数定义
void* threadFunc(void* param){
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]);// 阻塞，等待主线完成除法操作（操作自己专属的信号量）

		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			//消去
			for (int j = k + 1; j < N; ++j) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0.0;
		}
		sem_post(&sem_main); // 唤醒主线程
		sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
	}
	pthread_exit(NULL);
	return 0;
}

int main(){
	double seconds;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	//测量串行时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Serial();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "串行：" << seconds << "毫秒" << endl;

	//测量pthread静态线程+信号量同步时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	//初始化信号量
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//创建线程
	pthread_t* handles = new pthread_t[NUM_THREADS];// 创建对应的 Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// 创建对应的线程数据结构
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}

	for (int k = 0; k < N; ++k)
	{
		//主线程做除法操作
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] / M[k][k];
		}
		M[k][k] = 1.0;

		//开始唤醒工作线程
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//主线程睡眠（等待所有的工作线程完成此轮消去任务）
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// 主线程再次唤醒工作线程进入下一轮次的消去任务
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}

	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);


	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;

	cout << "pthread静态线程+信号量同步版本: " << seconds << "毫秒" << endl;
	return 0;
}
