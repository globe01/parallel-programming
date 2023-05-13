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
#pragma comment(lib, "pthreadVC2.lib")

using namespace std;

const int N = 1500;//�����ģ500,1000,1500
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

//�����߳����ݽṹ���̺߳���
struct threadParam_t
{
	int k; //��ȥ���ִ�
	int t_id; // �߳� id
};
int worker_count = 5; //�����߳�����
void* threadFunc(void* param){
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
	int i = k + t_id + 1; //��ȡ�Լ��ļ�������
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
	double seconds;//��ʱ��
	long long head, tail, freq;
	__m256 AVXva, AVXvt, AVXvx, AVXvaij, AVXvaik, AVXvakj;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	//��������ʱ��
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	Serial();
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "���У�" << seconds << "����" << endl;

	//����pthread��̬�߳�+SSEʱ��
	m_reset();
	QueryPerformanceCounter((LARGE_INTEGER*)&head);

	for (int k = 0; k < N; k++)
	{
		//���߳�����������
		//�����㷨�ж���ѭ�����Ż�
		AVXvt = _mm256_set_ps(M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k], M[k][k]);
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
		//�����㷨������ѭ�����Ż�
		//����֮�以��Ӱ�죬�ɲ��ö��߳�ִ��
		
		//���������̣߳�������ȥ����
		//�����߳�����worker_count
		pthread_t* handles = new pthread_t[worker_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[worker_count];// ������Ӧ���߳����ݽṹ

		//��������
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
		}
			
		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_join(handles[t_id], NULL);
		}	
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	seconds = (tail - head) * 1000.0 / freq;
	cout << "pthread��̬�߳�+AVX: " << seconds << "����" << endl;

}
