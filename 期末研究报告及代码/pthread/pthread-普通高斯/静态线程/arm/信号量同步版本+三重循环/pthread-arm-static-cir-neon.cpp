#include <iostream>
#include <arm_neon.h>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>//信号量
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

//串行
void serial()
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


struct threadParam_t
{
	int t_id; //线程id
};
int NUM_THREADS = 5;

//信号量定义
sem_t sem_leader;
sem_t* sem_Divsion = new sem_t[NUM_THREADS - 1];
sem_t* sem_Elimination = new sem_t[NUM_THREADS - 1];



//水平块划分
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
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
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

		//循环划分任务
		for (int i = k + 1 + t_id * temp; i < end; i++)
		{
			//消去
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


//垂直块划分
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
			sem_wait(&sem_Divsion[t_id - 1]); // 阻塞，等待完成除法操作
		}

		// t_id 为 0 的线程唤醒其它工作线程，进行消去操作
		if (t_id == 0)
		{
			for (int i = 0; i < NUM_THREADS - 1; ++i)
				sem_post(&sem_Divsion[i]);
		}


		//循环划分任务
		for (int i = k + 1; i < N; i++)
		{
			//消去
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


int main() {
	struct timeval head, tail;
	double seconds;




	//测量水平
	m_reset();
	gettimeofday(&head, NULL);

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
		pthread_create(&handles[t_id], NULL, threadFuncHor, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-static-sem-cir-neon-hor" << seconds << "ms" << endl;





	//测量垂直
	m_reset();
	gettimeofday(&head, NULL);

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
		pthread_create(&handles[t_id], NULL, threadFuncVer, (void*)&param[t_id]);
	}


	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
		pthread_join(handles[t_id], NULL);

	//销毁所有信号量
	sem_destroy(&sem_leader);
	sem_destroy(sem_Divsion);
	sem_destroy(sem_Elimination);

	gettimeofday(&tail, NULL);
	seconds = ((tail.tv_sec - head.tv_sec) * 1000000 + (tail.tv_usec - head.tv_usec)) / 1000.0;
	cout << "pthread-static-sem-cir-neon-ver" << seconds << "ms" << endl;


	return 0;
}
