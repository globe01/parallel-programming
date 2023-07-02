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

//静态线程 +barrier 同步

//线程数据结构定义
struct threadParam_t
{
	int t_id; //线程id
};

int NUM_THREADS = 5;

//barrier 定义
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//线程函数定义
void* threadFunc(void* param)
{
	__m256 AVXva, AVXvt, AVXvx, AVXvaij, AVXvaik, AVXvakj;
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		//串行算法中二重循环的优化
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				AVXva = _mm256_loadu_ps(&(M[k][j]));//将8个单精度浮点数从内存加载到向量寄存器
				AVXva = _mm256_div_ps(AVXva, AVXvt);//向量对位相除
				_mm256_store_ps(&(M[k][j]), AVXva);//将8个单精度浮点数从向量寄存器存储到内存
			}
			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];//该行结尾处有几个元素还未计算
			}
			M[k][k] = 1.0;
		}
		//串行算法中三重循环的优化
		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);

		//循环划分任务（同学们可以尝试多种任务划分方式）
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			//消去
			AVXvaik = _mm256_set_ps(M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k], M[i][k]);
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				AVXvakj = _mm256_loadu_ps(&(M[k][j]));
				AVXvaij = _mm256_loadu_ps(&(M[i][j]));
				AVXvx = _mm256_mul_ps(AVXvakj, AVXvaik);
				AVXvaij = _mm256_sub_ps(AVXvaij, AVXvx);
				_mm256_store_ps(&M[i][j], AVXvaij);
			}
			for (; j < N; j++)
				M[i][j] = M[i][j] - M[i][k] * M[k][j];

			M[i][k] = 0;
		}
		// 第二个同步点
		pthread_barrier_wait(&barrier_Elimination);

	}
	pthread_exit(NULL);
	return 0;
}




int main()
{
	m_reset();
	long long counter;// 记录次数
	double seconds;
	long long head, tail, freq, noww;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//开始计时


	//初始化barrier
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);


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

	//销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//结束计时
	seconds = (tail - head) * 1000.0 / freq;//单位 ms

	cout << "pthread静态线程 +barrier 同步+AVX: " << seconds << " ms" << endl;


}
