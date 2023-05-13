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
#pragma comment(lib, "pthreadVC2.lib")

using namespace std;

const int N = 1500;//问题规模500,1000,1500
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

//定义线程数据结构及线程函数
struct threadParam_t
{
	int k; //消去的轮次
	int t_id; // 线程 id
};
int worker_count = 5; //工作线程数量
void* threadFunc(void* param){
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //消去的轮次
	int t_id = p->t_id; //线程编号
	int i = k + t_id + 1; //获取自己的计算任务
	__m256 va, vt, vx, vaij, vaik, vakj;
	for (int m = k + 1 + t_id; m < N; m += worker_count)
	{
		vaik = _mm256_set_ps(M[m][k], M[m][k], M[m][k], M[m][k], M[m][k], M[m][k], M[m][k], M[m][k]);
		int j;
		for (j = k + 1; j + 8 <= N; j += 8)
		{
			vakj = _mm256_loadu_ps(&(M[k][j]));
			vaij = _mm256_loadu_ps(&(M[m][j]));
			vx = _mm256_mul_ps(vakj, vaik);
			vaij = _mm256_sub_ps(vaij, vx);
			_mm256_store_ps(&M[i][j], vaij);
		}
		for (; j < N; j++)
			M[m][j] = M[m][j] - M[m][k] * M[k][j];

		M[m][k] = 0;
	}


	pthread_exit(NULL);
	return 0;
}


int main(){
	double seconds;//总时间
	long long head, tail, freq;
	__m256 AVXva, AVXvt, AVXvx, AVXvaij, AVXvaik, AVXvakj;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//测量串行时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Serial();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "串行：" << seconds << "毫秒" << endl;

	//测量pthread动态线程+SSE时间
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	for (int k = 0; k < N; k++)
	{
		//主线程做除法操作
		//串行算法中二重循环的优化
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
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
		//串行算法中三重循环的优化
		//各行之间互不影响，可采用多线程执行
		
		//创建工作线程，进行消去操作
		//工作线程数量worker_count
		pthread_t* handles = new pthread_t[worker_count];// 创建对应的 Handle
		threadParam_t* param = new threadParam_t[worker_count];// 创建对应的线程数据结构

		//分配任务
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//创建线程
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
		}
			
		//主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_join(handles[t_id], NULL);
		}	
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "pthread动态线程+AVX: " << seconds << "毫秒" << endl;

}
