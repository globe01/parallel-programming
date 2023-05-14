#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>//�ź���
using namespace std;

const int N = 500;
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

//����
void serial()
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


struct threadParam_t
{
	int t_id; //�߳� id
};
int NUM_THREADS = 5;
//barrier ����
pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;


//�̺߳�������
void* threadFunc(void* param)
{
	float32x4_t va = vmovq_n_f32(0);
	float32x4_t vx = vmovq_n_f32(0);
	float32x4_t vaij = vmovq_n_f32(0);
	float32x4_t vaik = vmovq_n_f32(0);
	float32x4_t vakj = vmovq_n_f32(0);
	float32x4_t vt = vmovq_n_f32(0);


	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k)
	{

		vt = vmovq_n_f32(M[k][k]);
		if (t_id == 0)
		{
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				va = vld1q_f32(&(M[k][j]));
				va = vdivq_f32(va, vt);
				vst1q_f32(&(M[k][j]), va);
			}

			for (; j < N; j++)
			{
				M[k][j] = M[k][j] * 1.0 / M[k][k];
			}
			M[k][k] = 1.0;
		}

		//��һ��ͬ����
		pthread_barrier_wait(&barrier_Divsion);

		//ѭ����������
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
		{
			//��ȥ
			vaik = vmovq_n_f32(M[i][k]);
			int j;
			for (j = k + 1; j + 4 <= N; j += 4)
			{
				vakj = vld1q_f32(&(M[k][j]));
				vaij = vld1q_f32(&(M[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&M[i][j], vaij);
			}
			for (; j < N; j++)
				M[i][j] = M[i][j] - M[i][k] * M[k][j];

			M[i][k] = 0.0;
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
	struct timeval head, tail;
	double seconds;
	gettimeofday(&head, NULL);

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


	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-ststic-barrier:" << seconds << "ms" << endl;
	return 0;
}
