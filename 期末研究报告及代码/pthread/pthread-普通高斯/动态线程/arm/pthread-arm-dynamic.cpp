#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>
#include <pthread.h>

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
	int k; //��ȥ���ִ�
	int t_id; // �߳� id
};
int worker_count = 5; //�����߳�����
void* threadFunc(void* param)
{
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; //��ȥ���ִ�
	int t_id = p->t_id; //�̱߳��
	int i = k + t_id + 1; //��ȡ�Լ��ļ�������
	for (int j = k; j < N; ++j) {
		M[i][j] = M[i][j] - M[i][k] * M[k][j];
	}
		M[i][k] = 0;
	pthread_exit(NULL);
	return 0;
}


int main()
{
	double seconds;//��ʱ��
	struct timeval head, tail;

	//��������ʱ��
	m_reset();
	gettimeofday(&head, NULL);
	//��ʱ��ʼ
	serial();
	//��ʱ����
	gettimeofday(&tail, NULL);
	double seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "Serial: " << seconds << " ms" << endl;

	//����pthread��̬�߳�ʱ��
	m_reset();
	gettimeofday(&head, NULL);
	for (int k = 0; k < N; k++)
	{
		//���߳�����������
		for (int j = k + 1; j < N; j++)
		{
			M[k][j] = M[k][j] / M[k][k];
		}
		M[k][k] = 1.0;

		//�����㷨������ѭ�����Ż�
		//����֮�以��Ӱ�죬�ɲ��ö��߳�ִ��
		//���������̣߳�������ȥ����
		pthread_t* handles = new pthread_t[worker_count];// ������Ӧ�� Handle
		threadParam_t* param = new threadParam_t[worker_count];// ������Ӧ���߳����ݽṹ

		//��������
		for (int t_id = 0; t_id < worker_count; t_id++)
		{
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}
		//�����߳�
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);

		//���̹߳���ȴ����еĹ����߳���ɴ�����ȥ����
		for (int t_id = 0; t_id < worker_count; t_id++)
			pthread_join(handles[t_id], NULL);

	}
	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-dynamic: " << seconds << "ms" << endl;
	return 0;
}
