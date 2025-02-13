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

const int N = 1000;
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

//静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数 + SSE

struct threadParam_t
{
	int t_id; //线程 id
};
int NUM_THREADS = 5;

//信号量定义
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1]; // 每个线程有自己专属的信号量
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];

//线程函数定义
void* threadFunc(void* param)
{
	__m128 va, vt, vx, vaij, vaik, vakj;
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		//串行算法中二重循环的优化
		vt = _mm_set_ps(M[k][k], M[k][k], M[k][k], M[k][k]);

		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				va = _mm_loadu_ps(&(M[k][j]));//将四个单精度浮点数从内存加载到向量寄存器
				va = _mm_div_ps(va, vt);//向量对位相除
				_mm_store_ps(&(M[k][j]), va);//将四个单精度浮点数从向量寄存器存储到内存
			}

			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
			}
			M[k][k] = 1.0;
		}
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}

		//串行算法中三重循环的优化
		//循环划分任务
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			vaik = _mm_set_ps(M[i][k], M[i][k], M[i][k], M[i][k]);
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = _mm_loadu_ps(&(M[k][j]));
				vaij = _mm_loadu_ps(&(M[i][j]));
				vx = _mm_mul_ps(vakj, vaik);
				vaij = _mm_sub_ps(vaij, vx);
				_mm_store_ps(&M[i][j], vaij);
			}
			for (; j < N; j++)
				M[i][j] = M[i][j] - M[i][k] * M[k][j];

			M[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // 等待其它 worker 完成消去

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // 通知其它 worker 进入下一轮
		}
		else
		{
			sem_post(&sem_leader);// 通知 leader, 已完成消去任务
			sem_wait(&sem_Elimination[t_id - 1]); // 等待通知，进入下一轮
		}
	}
	pthread_exit(NULL);
	return 0;
}

int main(){
	double seconds;
	long long head, tail, freq, noww;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	//初始化信号量
	sem_init(&sem_leader, 0, 0);

	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

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
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;

	cout << "pthread静态线程 + 信号量同步版本 + 三重循环全部纳入线程函数: " << seconds << "毫秒" << endl;
}
