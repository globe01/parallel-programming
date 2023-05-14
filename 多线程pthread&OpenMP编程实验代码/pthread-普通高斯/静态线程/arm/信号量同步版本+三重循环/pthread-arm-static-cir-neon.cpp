#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>//�ź���
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
	int t_id; //�߳�id
};
int NUM_THREADS = 5;

//�ź�������
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1];
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];



//ˮƽ�黮��
void* threadFuncHor(void* param)
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
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}



		int temp = (N - k - 1) / NUM_THREADS;
		int end = 0;
		if (t_id == NUM_THREADS - 1)
		{
			end = N;
		}
		else
		{
			end = k + 1 + temp * (t_id + 1);
		}

		//ѭ����������
		for (int i = k + 1 + t_id * temp; i < end; i++)
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

			M[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
	return 0;
}


//��ֱ�黮��
void* threadFuncVer(void* param)
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
		else
		{
			sem_wait(&sem_Divsion[t_id - 1]); // �������ȴ���ɳ�������
		}

		// t_id Ϊ 0 ���̻߳������������̣߳�������ȥ����
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//ѭ����������
		for (int i = k + 1; i < N; i++)
		{
			//��ȥ
			vaik = vmovq_n_f32(M[i][k]);
			int j;

			int temp = (N - k - 1) / NUM_THREADS;
			int end = 0;
			if (t_id == NUM_THREADS - 1)
			{
				end = N;
			}
			else
			{
				end = k + 1 + temp * (t_id + 1);
			}


			for (j = k + 1 + t_id * temp; j + 4 <= end; j += 4)
			{
				vakj = vld1q_f32(&(M[k][j]));
				vaij = vld1q_f32(&(M[i][j]));
				vx = vmulq_f32(vakj, vaik);
				vaij = vsubq_f32(vaij, vx);
				vst1q_f32(&M[i][j], vaij);
			}
			for (; j < end; j++)
				M[i][j] = M[i][j] - M[i][k] * M[k][j];

			M[i][k] = 0;
		}


		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_wait(&sem_leader); // �ȴ����� worker �����ȥ

			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Elimination[i]); // ֪ͨ���� worker ������һ��
		}
		else
		{
			sem_post(&sem_leader);// ֪ͨ leader, �������ȥ����
			sem_wait(&sem_Elimination[t_id - 1]); // �ȴ�֪ͨ��������һ��
		}
	}
	pthread_exit(NULL);
	return 0;
}


int main() {
	struct timeval head, tail;
	double seconds;




	//����ˮƽ
	m_reset();
	gettimeofday(&head, NULL);

	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);

	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFuncHor, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//���������ź���
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-static-sem-cir-neon-hor" << seconds << "ms" << endl;





	//������ֱ
	m_reset();
	gettimeofday(&head, NULL);

	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);

	for (int i = 0; i < NUM_THREADS - 1; ++i)
	{
		sem_init(&sem_Divsion[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}

	//�����߳�
	pthread_t* handles = new pthread_t[NUM_THREADS];// ������Ӧ�� Handle
	threadParam_t* param = new threadParam_t[NUM_THREADS];// ������Ӧ���߳����ݽṹ
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFuncVer, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//���������ź���
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-static-sem-cir-neon-ver" << seconds << "ms" << endl;


	return 0;
}
