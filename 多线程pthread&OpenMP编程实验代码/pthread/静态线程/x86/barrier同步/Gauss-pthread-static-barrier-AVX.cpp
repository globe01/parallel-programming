#include <iostream>
#include <windows.h>
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE2
#include <pmmintrin.h> //SSE3
#include <tmmintrin.h> //SSSE3
#include <smmintrin.h> //SSE4.1
#include <nmmintrin.h> //SSSE4.2
#include <immintrin.h> //AVX��AVX2��AVX-512
#include <pthread.h>//pthread
#include <semaphore.h>//�ź���
#pragma comment(lib, "pthreadVC2.lib")
using namespace std;

const int N = 1000;
float M[N][N];

//������������
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

//�����㷨
void Serial()
{
	for (int k = 0; k < N; k++)
	{
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] * 1.0 / M[k][k];//�������裬���г��Ե�һ����ϵ��
		}
		M[k][k] = 1.0;
		for (int i = k + 1; i < N; i++)
		{
			for (int j = k + 1; j < N; j++)
			{
				M[i][j] = M[i][j] - M[i][k] * M[k][j];//��ȥ����
			}
			M[i][k] = 0;//���½ǻ�Ϊ0��
		}
	}
}

//��̬�߳� +barrier ͬ��

//�߳����ݽṹ����
struct threadParam_t
{
	int t_id; //�߳�id
};

int NUM_THREADS = 5;

//barrier ����
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//�̺߳�������
void* threadFunc(void* param)
{
	__m256 AVXva, AVXvt, AVXvx, AVXvaij, AVXvaik, AVXvakj;
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{
		//�����㷨�ж���ѭ�����Ż�
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 8 <= N; j += 8)
			{
				AVXva = _mm256_loadu_ps(&(M[k][j]));//��8�������ȸ��������ڴ���ص������Ĵ���
				AVXva = _mm256_div_ps(AVXva, AVXvt);//������λ���
				_mm256_store_ps(&(M[k][j]), AVXva);//��8�������ȸ������������Ĵ����洢���ڴ�
			}
			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];//���н�β���м���Ԫ�ػ�δ����
			}
			M[k][k] = 1.0;
		}
		//�����㷨������ѭ�����Ż�
		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);

		//ѭ����������ͬѧ�ǿ��Գ��Զ������񻮷ַ�ʽ��
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			//��ȥ
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
		// �ڶ���ͬ����
		pthread_barrier_wait(&barrier_Elimination);

	}
	pthread_exit(NULL);
	return 0;
}




int main()
{
	m_reset();
	long long counter;// ��¼����
	double seconds;
	long long head, tail, freq, noww;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	QueryPerformanceCounter((LARGE_INTEGER*)&head);//��ʼ��ʱ


	//��ʼ��barrier
	pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);


	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//�������е� barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);

	QueryPerformanceCounter((LARGE_INTEGER*)&tail);//������ʱ
	seconds = (tail - head) * 1000.0 / freq;//��λ ms

	cout << "pthread��̬�߳� +barrier ͬ��+AVX: " << seconds << " ms" << endl;


}
