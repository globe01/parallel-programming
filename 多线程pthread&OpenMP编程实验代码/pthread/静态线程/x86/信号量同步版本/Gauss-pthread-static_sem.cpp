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

const int N = 1000;//�����ģ500,1000,1500
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

//��̬�߳�+�ź���ͬ��

//�߳����ݽṹ����
struct threadParam_t
{
	int t_id; //�߳�id
};
int NUM_THREADS = 5;
//�ź�������
sem_t sem_main;
sem_t* sem_workerstart = new sem_t[NUM_THREADS];// ÿ���߳����Լ�ר�����ź���
sem_t* sem_workerend = new sem_t[NUM_THREADS];

//�̺߳�������
void* threadFunc(void* param){
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;
	for (int k = 0; k < N; ++k)
	{
		sem_wait(&sem_workerstart[t_id]);// �������ȴ�������ɳ��������������Լ�ר�����ź�����

		//ѭ����������
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			//��ȥ
			for (int j = k + 1; j < N; ++j) {
				M[i][j] = M[i][j] - M[i][k] * M[k][j];
			}
			M[i][k] = 0.0;
		}
		sem_post(&sem_main); // �������߳�
		sem_wait(&sem_workerend[t_id]); //�������ȴ����̻߳��ѽ�����һ��
	}
	pthread_exit(NULL);
	return 0;
}

int main(){
	double seconds;
	long long head, tail, freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	//��������ʱ��
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Serial();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "���У�" << seconds << "����" << endl;

	//����pthread��̬�߳�+�ź���ͬ��ʱ��
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	//��ʼ���ź���
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREADS; ++i)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}

	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}

	for (int k = 0; k < N; ++k)
	{
		//���߳�����������
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] / M[k][k];
		}
		M[k][k] = 1.0;

		//��ʼ���ѹ����߳�
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_post(&sem_workerstart[t_id]);
		}

		//���߳�˯�ߣ��ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_wait(&sem_main);
		}

		// ���߳��ٴλ��ѹ����߳̽�����һ�ִε���ȥ����
		for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
		{
			sem_post(&sem_workerend[t_id]);
		}

	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//���������ź���
	sem_destroy(&sem_main);
	sem_destroy(sem_workerstart);
	sem_destroy(sem_workerend);


	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;

	cout << "pthread��̬�߳�+�ź���ͬ���汾: " << seconds << "����" << endl;
	return 0;
}
